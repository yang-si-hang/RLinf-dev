# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare persistent UR10e offline replay for Stage 2 critic training."""

import hashlib
import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.openpi.dataconfig.ur_dataconfig import (
    absolute_actions_to_relative,
    relative_actions_to_absolute,
)


def npv(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_stats(p):
    d = json.loads(Path(p).read_text())
    a = [v["actions"] for v in d.values() if isinstance(v, dict) and "actions" in v]
    if not a and "actions" in d:
        a = [d["actions"]]
    if len(a) != 1:
        raise ValueError("Expected one actions entry in norm_stats")
    lo = np.asarray(a[0]["q01"], np.float32)
    hi = np.asarray(a[0]["q99"], np.float32)
    if lo.shape != (10,) or hi.shape != (10,):
        raise ValueError("Expected 10D action statistics")
    return lo, hi


def transform(state, actions, lo, hi):
    rel = absolute_actions_to_relative(
        state.astype(np.float32), actions.astype(np.float32)
    )
    if not np.allclose(
        relative_actions_to_absolute(state.astype(np.float32), rel),
        actions,
        atol=2e-5,
        rtol=2e-5,
    ):
        raise ValueError("OpenPI action roundtrip failed")
    return ((rel - lo) / (hi - lo + 1e-6) * 2 - 1).astype(np.float32)


def import_data(cfg, root, zdir, success, lo, hi):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(str(cfg.data.lerobot_repo_id), root=str(root))
    meta = json.loads((Path(zdir) / "metadata.json").read_text())
    if int(meta.get("checkpoint_step", -1)) != int(cfg.data.stage1_checkpoint_step):
        raise ValueError("z checkpoint step mismatch")
    z = np.load(Path(zdir) / "z_rl.npy", mmap_mode="r")
    idx = np.load(Path(zdir) / "indices.npy", mmap_mode="r")
    if (
        z.ndim != 2
        or len(z) != len(ds)
        or not np.array_equal(idx, np.arange(len(ds)))
        or not np.isfinite(z).all()
    ):
        raise ValueError("z vectors are invalid or misaligned")
    ep = ds.meta.episodes
    starts = npv(ep["dataset_from_index"]).reshape(-1)
    ends = npv(ep["dataset_to_index"]).reshape(-1)
    out = []
    H = int(cfg.data.executed_chunk_length)
    for start, end in zip(starts, ends, strict=True):
        start, end = int(start), int(end)
        rows = [ds.hf_dataset[i] for i in range(start, end)]
        eid = int(rows[0]["episode_index"])
        ss = []
        aa = []
        for j, r in enumerate(rows):
            if (
                int(r["index"]) != start + j
                or int(r["episode_index"]) != eid
                or int(r["frame_index"]) != j
            ):
                raise ValueError("LeRobot indices are not aligned")
            ss.append(npv(r["observation.state"]).astype(np.float32).reshape(-1))
            aa.append(npv(r["action"]).astype(np.float32).reshape(-1))
        ss, aa = np.stack(ss), np.stack(aa)
        if (
            ss.shape != (len(rows), 10)
            or aa.shape != (len(rows), 10)
            or not np.isfinite(ss).all()
            or not np.isfinite(aa).all()
        ):
            raise ValueError("Invalid UR state/action")
        count = len(rows) - 1
        drop = count % H
        if drop and not cfg.data.drop_incomplete_suffix:
            raise ValueError(f"Episode {eid} needs suffix trim")
        usable = count - drop
        ts = []
        for t in range(0, usable, H):
            stop = t + H
            done = stop == usable
            r = np.full(
                H,
                float(
                    cfg.data.success_step_reward
                    if success
                    else cfg.data.failure_step_reward
                ),
                np.float32,
            )
            if done and not success:
                r[-1] = float(cfg.data.failure_terminal_reward)
            na = (
                np.zeros((H, 10), np.float32)
                if done
                else transform(ss[stop], aa[stop : stop + H], lo, hi)
            )
            ts.append(
                {
                    "z": np.array(z[start + t], dtype=np.float32, copy=True),
                    "s": ss[t],
                    "a": transform(ss[t], aa[t:stop], lo, hi),
                    "r": r,
                    "d": done,
                    "nz": np.array(z[start + stop], dtype=np.float32, copy=True),
                    "ns": ss[stop],
                    "na": na,
                    "kind": ("success_terminal" if success else "failure_terminal")
                    if done
                    else "ordinary",
                }
            )
        if ts:
            out.append(
                {
                    "label": "success" if success else "failure",
                    "eid": eid,
                    "drop": drop,
                    "ts": ts,
                }
            )
    return out


def split_data(eps, seed, fraction):
    rng = random.Random(seed)
    tr = []
    va = []
    m = {"seed": seed, "train_episode_ids": {}, "validation_episode_ids": {}}
    for label in ("success", "failure"):
        g = [e for e in eps if e["label"] == label]
        rng.shuffle(g)
        n = max(1, round(len(g) * fraction)) if len(g) > 1 else 0
        va += g[:n]
        tr += g[n:]
        m["train_episode_ids"][label] = [e["eid"] for e in g[n:]]
        m["validation_episode_ids"][label] = [e["eid"] for e in g[:n]]
    if not tr or not va:
        raise ValueError("Train/validation split is empty")
    return tr, va, m


def to_traj(t, i):
    return Trajectory(
        max_episode_length=1,
        model_weights_id=str(i),
        actions=torch.from_numpy(t["a"])[None, None],
        rewards=torch.tensor([[float(t["r"].sum())]]),
        dones=torch.tensor([[t["d"]]]),
        terminations=torch.tensor([[t["d"]]]),
        curr_obs={
            "z_rl": torch.from_numpy(t["z"])[None, None],
            "proprio": torch.from_numpy(t["s"])[None, None],
        },
        next_obs={
            "z_rl": torch.from_numpy(t["nz"])[None, None],
            "proprio": torch.from_numpy(t["ns"])[None, None],
            "behavior_actions": torch.from_numpy(t["na"].ravel())[None, None],
        },
        forward_inputs={"step_rewards": torch.from_numpy(t["r"])[None, None]},
    )


def make_replay(cfg, eps):
    result = {}
    for kind in ("ordinary", "success_terminal", "failure_terminal"):
        b = TrajectoryReplayBuffer(
            seed=int(cfg.algorithm.seed),
            enable_cache=True,
            cache_size=int(cfg.algorithm.replay_buffer.cache_size),
            sample_window_size=int(cfg.algorithm.replay_buffer.sample_window_size),
            auto_save=bool(cfg.algorithm.replay_buffer.auto_save),
            auto_save_path=str(Path(cfg.trainer.output_dir) / "replay" / kind),
            trajectory_format=str(cfg.algorithm.replay_buffer.trajectory_format),
        )
        items = []
        i = 0
        for e in eps:
            for t in e["ts"]:
                if t["kind"] == kind:
                    items.append(to_traj(t, i))
                    i += 1
        if items:
            b.add_trajectories(items)
            result[kind] = b
        else:
            b.close(wait=True)
    return result


@hydra.main(
    version_base="1.2",
    config_path="../../examples/embodiment/config",
    config_name="ur10e_rlt_stage2_critic_only",
)
def main(cfg: DictConfig) -> None:
    if not bool(cfg.data.action_quantile_norm):
        raise ValueError("Only original OpenPI q01/q99 normalization is supported")
    roots = [
        Path(cfg.data.success_dataset_path),
        Path(cfg.data.failure_dataset_path),
        Path(cfg.data.success_z_path),
        Path(cfg.data.failure_z_path),
        Path(cfg.data.action_norm_stats_path),
    ]
    for item in roots:
        if not item.exists():
            raise FileNotFoundError(item)
    out = Path(cfg.trainer.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(out)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < int(cfg.trainer.min_free_bytes):
        raise OSError(f"Insufficient disk space: {free_bytes} bytes")
    lo, hi = load_stats(roots[-1])
    episodes = import_data(cfg, roots[0], roots[2], True, lo, hi)
    episodes += import_data(cfg, roots[1], roots[3], False, lo, hi)
    train_eps, validation_eps, split = split_data(
        episodes, int(cfg.algorithm.seed), float(cfg.data.validation_fraction)
    )
    manifest = {
        "stage1_checkpoint_step": int(cfg.data.stage1_checkpoint_step),
        "success_dataset": str(roots[0]),
        "failure_dataset": str(roots[1]),
        "success_z": str(roots[2]),
        "failure_z": str(roots[3]),
        "action_representation": str(cfg.data.action_representation),
        "chunk_length": int(cfg.data.executed_chunk_length),
        "vla_prediction_horizon": int(cfg.data.vla_prediction_horizon),
        "norm_stats_sha256": digest(roots[-1]),
        "openpi_transform_version": "rlinf.ur_dataconfig.absolute_actions_to_relative:v1",
        "success_z_metadata_sha256": digest(roots[2] / "metadata.json"),
        "failure_z_metadata_sha256": digest(roots[3] / "metadata.json"),
        "reward": [
            float(cfg.data.success_step_reward),
            float(cfg.data.failure_step_reward),
            float(cfg.data.failure_terminal_reward),
        ],
        "split": split,
        "drop_back": [[ep["label"], ep["eid"], ep["drop"]] for ep in episodes],
    }
    manifest_path = out / "replay_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Existing replay manifest differs; choose a fresh output directory"
            )
        required = [
            Path(cfg.data.replay_path) / kind / "metadata.json"
            for kind in ("ordinary", "success_terminal", "failure_terminal")
        ]
        if all(item.exists() for item in required):
            print(f"Replay already prepared at {cfg.data.replay_path}")
            return
        raise FileNotFoundError("Manifest exists but replay index is incomplete")
    buffers = make_replay(cfg, train_eps)
    for buffer in buffers.values():
        buffer.close(wait=True)
    torch.save(
        {"q01": lo, "q99": hi, "sha256": manifest["norm_stats_sha256"]},
        out / "action_norm_stats.pt",
    )
    torch.save(validation_eps, out / "validation_transitions.pt")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    counts = {
        kind: sum(t["kind"] == kind for ep in train_eps for t in ep["ts"])
        for kind in ("ordinary", "success_terminal", "failure_terminal")
    }
    print(
        json.dumps(
            {
                "replay_path": str(cfg.data.replay_path),
                "train_episodes": len(train_eps),
                "validation_episodes": len(validation_eps),
                "train_transitions": sum(len(ep["ts"]) for ep in train_eps),
                "validation_transitions": sum(len(ep["ts"]) for ep in validation_eps),
                "train_transition_types": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
