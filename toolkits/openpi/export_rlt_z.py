# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Export Stage 1 RLT vectors for a local LeRobot dataset in source order."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples/sft/config"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
export ROOT=/root/autodl-tmp/workspace/RLinf-dev
python toolkits/openpi/export_rlt_z.py \
  --config-name ur10e_rlt_stage1_sft_openpi_pi05 \
  --dataset-root $ROOT/data/lerobot_data/plug_deploy_v1_merge_vid \
  --checkpoint $ROOT/logs/20260922-18:16:40-ur10e_rlt_stage1_sft_openpi_pi05/ur10e_plug_rlt_stage1_20260922_181649/checkpoints/global_step_21000 \
  --norm-stats $ROOT/data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000/assets/plug_v2_merge_crop_vid/norm_stats.json \
  --output-dir $ROOT/results/ur10e_rlt_stage1_z_smoke_$(date +%Y%m%d_%H%M%S) \
  --batch-size 16 \
  --limit 100
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="ur10e_rlt_stage1_sft_openpi_pi05")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_config(
    config_name: str, checkpoint: Path, norm_stats: Path
) -> tuple[Any, str, str]:
    """Compose the training YAML, preserving its inputs except runtime overrides."""
    os.environ["EMBODIED_PATH"] = str(REPO_ROOT / "examples/sft")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=[f"root_dir={REPO_ROOT}"])
    base_model_path = str(cfg.actor.model.model_path)
    compute_dtype = str(cfg.actor.fsdp_config.mixed_precision.param_dtype).lower()
    training_stats = Path(str(cfg.actor.model.openpi_data.norm_stats_path)).resolve()
    if training_stats != norm_stats.resolve():
        raise ValueError(
            "--norm-stats must match the training YAML statistics path: "
            f"expected {training_stats}, got {norm_stats}"
        )
    cfg.actor.model.model_path = str(checkpoint)
    cfg.actor.model.openpi_data.norm_stats_path = str(norm_stats)
    cfg.actor.model.openpi.task = "eval"
    cfg.actor.model.openpi.rlt_freeze_vla = False
    return cfg.actor.model, base_model_path, compute_dtype


def resolve_checkpoint(path: Path) -> Path:
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        resolve_full_weights,
    )

    resolved = resolve_full_weights(path)
    if resolved is None or resolved.name != "full_weights.pt":
        raise FileNotFoundError(f"Stage 1 full_weights.pt not found under {path}")
    with zipfile.ZipFile(resolved) as archive:
        if not archive.namelist():
            raise ValueError(f"Checkpoint ZIP is empty: {resolved}")
    return resolved.resolve()


def _tensor(value: Any, index: int, key: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Sample {index} missing {key}")
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def make_batch(dataset: Any, indices: range) -> dict[str, Any]:
    """Map indexed UR10e LeRobot samples to the eval model's input contract."""
    states, bases, wrists, tasks = [], [], [], []
    for index in indices:
        sample = dataset[index]
        if "index" in sample and int(sample["index"]) != index:
            raise ValueError(
                f"Sample {index} reports dataset index {int(sample['index'])}"
            )
        required = (
            "observation.state",
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "task",
        )
        for key in required:
            if key not in sample or sample[key] is None:
                raise KeyError(f"Sample {index} missing {key}")
        states.append(_tensor(sample[required[0]], index, required[0]))
        bases.append(_tensor(sample[required[1]], index, required[1]))
        wrists.append(_tensor(sample[required[2]], index, required[2]))
        tasks.append(str(sample["task"]))
    return {
        "states": torch.stack(states),
        "main_images": torch.stack(bases),
        "extra_view_images": torch.stack(wrists).unsqueeze(1),
        "task_descriptions": tasks,
    }


def validate_loaded_checkpoint(model: Any, weights: Path) -> None:
    """Check that the factory loaded every model and RLT tensor from Stage 1."""
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        _normalize_wrapper_state_dict,
    )
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    loaded = torch.load(str(weights), map_location="cpu", weights_only=False, mmap=True)
    state = _normalize_wrapper_state_dict(as_state_dict(loaded))
    expected = model.state_dict()
    missing = sorted(set(expected) - state.keys())
    unexpected = sorted(set(state) - expected.keys())
    mismatched = [
        key
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Checkpoint {weights} does not fully match the eval model: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={mismatched[:20]}"
        )
    if not any(key.startswith("rlt_module.") for key in state):
        raise RuntimeError(f"Checkpoint {weights} has no rlt_module weights")


def extract_rlt_z(model: Any, env_obs: dict[str, Any]) -> torch.Tensor:
    """Mirror Stage 2's prefix/RLT path without decoding reference actions."""
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model

    model._require_rlt()
    repacked = model._repack_env_obs(env_obs)
    processed = model.input_transform(repacked, transpose=False)
    observation = model._observation_dict_to_device(processed)
    prepared = pi0_model.preprocess_observation(observation, train=False)
    prefix_output, prefix_mask, _ = model.model.build_prefix_cache(prepared)
    selected_output, selected_mask = model._select_rlt_prefix_embeddings(
        prefix_output, prefix_mask, prepared.tokenized_prompt
    )
    return model._encode_rlt_flat(selected_output, selected_mask).to(torch.float32)


def export(
    dataset: Any,
    model: Any,
    output_dir: Path,
    count: int,
    batch_size: int,
    metadata: dict[str, Any],
) -> None:
    """Write verified arrays, publishing metadata only after successful completion."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    z_tmp = output_dir / "z_rl.pending.npy"
    indices_tmp = output_dir / "indices.pending.npy"
    z = np.lib.format.open_memmap(
        z_tmp, mode="w+", dtype=np.float32, shape=(count, metadata["vector_dim"])
    )
    indices = np.lib.format.open_memmap(
        indices_tmp, mode="w+", dtype=np.int64, shape=(count,)
    )
    with torch.inference_mode(), tqdm(
        total=count, desc="Exporting z_rl", unit="frame", dynamic_ncols=True
    ) as progress:
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            values = extract_rlt_z(model, make_batch(dataset, range(start, end)))
            batch = values.detach().to(device="cpu", dtype=torch.float32).numpy()
            if batch.shape != (end - start, metadata["vector_dim"]):
                raise ValueError(
                    f"Rows {start}:{end}: unexpected z shape {batch.shape}"
                )
            if not np.isfinite(batch).all():
                raise ValueError(f"Rows {start}:{end}: z contains nonfinite values")
            z[start:end] = batch
            indices[start:end] = np.arange(start, end, dtype=np.int64)
            progress.update(end - start)
    z.flush()
    indices.flush()
    del z, indices
    verified_z = np.load(z_tmp, mmap_mode="r")
    verified_indices = np.load(indices_tmp, mmap_mode="r")
    if (
        verified_z.shape != (count, metadata["vector_dim"])
        or verified_z.dtype != np.float32
    ):
        raise ValueError("Written z array has unexpected shape or dtype")
    if not np.array_equal(verified_indices, np.arange(count, dtype=np.int64)):
        raise ValueError("Written dataset indices are misaligned")
    if not np.isfinite(verified_z).all():
        raise ValueError("Written z array contains nonfinite values")
    del verified_z, verified_indices
    os.replace(z_tmp, output_dir / "z_rl.npy")
    os.replace(indices_tmp, output_dir / "indices.npy")
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    meta_tmp = output_dir / "metadata.pending.json"
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(meta_tmp, output_dir / "metadata.json")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--batch-size and --limit must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json missing under {dataset_root}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Normalization statistics missing: {norm_stats}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    weights = resolve_checkpoint(checkpoint)
    model_cfg, base_model_path, compute_dtype = load_config(
        args.config_name, checkpoint, norm_stats
    )
    if not bool(model_cfg.openpi.use_rlt):
        raise ValueError("Training config does not enable RLT")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.models.embodiment.openpi_rlinf import get_model

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset = LeRobotDataset(str(model_cfg.openpi_data.repo_id), root=str(dataset_root))
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if count == 0:
        raise ValueError("Dataset has no samples")
    model = get_model(model_cfg)
    validate_loaded_checkpoint(model, weights)
    model = model.to(device).eval()
    dim = int(model_cfg.openpi.rlt_embed_dim)
    match = re.search(r"global_step_(\d+)", str(weights))
    metadata = {
        "config_name": args.config_name,
        "dataset_root": str(dataset_root),
        "checkpoint": str(weights),
        "checkpoint_step": int(match.group(1)) if match else None,
        "norm_stats": str(norm_stats),
        "base_model_path_in_training_config": base_model_path,
        "total_samples": len(dataset),
        "output_samples": count,
        "vector_dim": dim,
        "dtype": "float32",
        "model_precision": str(model_cfg.precision),
        "compute_dtype": compute_dtype,
        "rlt_config": {
            key: OmegaConf.select(model_cfg.openpi, key)
            for key in model_cfg.openpi
            if key.startswith("rlt_")
        },
        "runtime_overrides": {
            "root_dir": str(REPO_ROOT),
            "model_path": str(checkpoint),
            "openpi_data.norm_stats_path": str(norm_stats),
            "openpi.task": "eval",
            "openpi.rlt_freeze_vla": False,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    # Stage 1 uses bf16 FSDP parameter precision even though its stored model
    # precision is fp32. Autocast reproduces that compute path for standalone eval.
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if compute_dtype == "bf16"
        else contextlib.nullcontext()
    )
    with autocast:
        export(dataset, model, output_dir, count, args.batch_size, metadata)
    print(f"Completed {count} vectors in {output_dir}")


if __name__ == "__main__":
    main()
