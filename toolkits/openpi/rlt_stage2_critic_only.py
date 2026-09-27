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

"""Offline UR10e Stage 2 critic-only training."""

import hashlib
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import TwinQCritic
from rlinf.utils.metric_logger import MetricLogger


def td_target(step_rewards, dones, next_q, gamma):
    horizon = step_rewards.shape[-1]
    discounts = gamma ** torch.arange(
        horizon, device=step_rewards.device, dtype=step_rewards.dtype
    )
    returns = (step_rewards * discounts).sum(-1)
    return returns + (~dones).to(step_rewards.dtype) * gamma**horizon * next_q


def sample_replay(buffers, n):
    weights = {
        k: v
        for k, v in (
            ("ordinary", 0.6),
            ("success_terminal", 0.2),
            ("failure_terminal", 0.2),
        )
        if k in buffers
    }
    den = sum(weights.values())
    parts = [
        buffers[k].sample_chunks(max(1, round(n * v / den))) for k, v in weights.items()
    ]
    out = {}
    for key in ("actions", "dones", "forward_inputs", "curr_obs", "next_obs"):
        vals = [x[key] for x in parts if key in x]
        if vals and isinstance(vals[0], dict):
            out[key] = {k: torch.cat([v[k] for v in vals]) for k in vals[0]}
        elif vals:
            out[key] = torch.cat(vals)
    return out


@torch.no_grad()
def validate(q, qt, eps, device, gamma):
    ts = [t for e in eps for t in e["ts"]]
    s = torch.tensor(np.stack([np.r_[t["z"], t["s"]] for t in ts]), device=device)
    a = torch.tensor(np.stack([t["a"].ravel() for t in ts]), device=device)
    ns = torch.tensor(np.stack([np.r_[t["nz"], t["ns"]] for t in ts]), device=device)
    r = torch.tensor(np.stack([t["r"] for t in ts]), device=device)
    done = torch.tensor([t["d"] for t in ts], device=device)
    next_q = torch.zeros(len(ts), device=device)
    valid = ~done
    if valid.any():
        next_actions = torch.tensor(
            np.stack([t["na"] for t in ts if not t["d"]]), device=device
        ).flatten(1)
        next_q[valid] = qt(ns[valid], next_actions).min(-1).values
    y = td_target(r, done, next_q, gamma)
    pred = q(s, a)
    return {
        "val_loss": float(F.mse_loss(pred[:, 0], y) + F.mse_loss(pred[:, 1], y)),
        "val_td_abs": float((pred.mean(-1) - y).abs().mean()),
        "val_q1": float(pred[:, 0].mean()),
        "val_q2": float(pred[:, 1].mean()),
        "val_samples": len(ts),
    }


def prune_checkpoints(output_dir: Path, keep_period: int, latest_step: int) -> None:
    # Keep periodic checkpoints and the latest saved checkpoint.
    for checkpoint in output_dir.glob("critic_step_*.pt"):
        step = int(checkpoint.stem.rsplit("_", 1)[1])
        if step != latest_step and step % keep_period != 0:
            checkpoint.unlink()


@hydra.main(
    version_base="1.2",
    config_path="../../examples/embodiment/config",
    config_name="ur10e_rlt_stage2_critic_only",
)
def main(c: DictConfig):
    if (
        not c.algorithm.critic_only
        or int(c.algorithm.actor_updates) != 0
        or c.algorithm.target_action_source != "recorded_vla_action"
    ):
        raise ValueError(
            "critic-only mode requires recorded actions and zero actor updates"
        )
    if not c.data.action_quantile_norm:
        raise ValueError("Use the original OpenPI quantile action normalization")
    dev = torch.device(str(c.trainer.device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    out = Path(c.trainer.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(out)
    if stat.f_bavail * stat.f_frsize < int(c.trainer.min_free_bytes):
        raise OSError("Insufficient disk space for critic checkpoints")
    manifest_path = Path(c.data.replay_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Replay manifest missing at {manifest_path}; run prepare_rlt_stage2_replay.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    stats_path = Path(c.data.action_norm_stats_path)
    expected = {
        "stage1_checkpoint_step": int(c.data.stage1_checkpoint_step),
        "success_dataset": str(Path(c.data.success_dataset_path)),
        "failure_dataset": str(Path(c.data.failure_dataset_path)),
        "success_z": str(Path(c.data.success_z_path)),
        "failure_z": str(Path(c.data.failure_z_path)),
        "action_representation": str(c.data.action_representation),
        "chunk_length": int(c.data.executed_chunk_length),
        "vla_prediction_horizon": int(c.data.vla_prediction_horizon),
        "norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "success_z_metadata_sha256": hashlib.sha256(
            (Path(c.data.success_z_path) / "metadata.json").read_bytes()
        ).hexdigest(),
        "failure_z_metadata_sha256": hashlib.sha256(
            (Path(c.data.failure_z_path) / "metadata.json").read_bytes()
        ).hexdigest(),
        "reward": [
            float(c.data.success_step_reward),
            float(c.data.failure_step_reward),
            float(c.data.failure_terminal_reward),
        ],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Replay/config mismatch for {key}")
    stats_blob = torch.load(
        out / "action_norm_stats.pt", map_location="cpu", weights_only=False
    )
    lo, hi = stats_blob["q01"], stats_blob["q99"]
    validation_eps = torch.load(
        out / "validation_transitions.pt", map_location="cpu", weights_only=False
    )
    replay = {}
    for kind in ("ordinary", "success_terminal", "failure_terminal"):
        replay_dir = Path(c.data.replay_path) / kind
        if not (replay_dir / "metadata.json").is_file():
            continue
        buffer = TrajectoryReplayBuffer(
            seed=int(c.algorithm.seed),
            enable_cache=True,
            cache_size=int(c.algorithm.replay_buffer.cache_size),
            sample_window_size=int(c.algorithm.replay_buffer.sample_window_size),
            auto_save=bool(c.algorithm.replay_buffer.auto_save),
            auto_save_path=str(replay_dir),
            trajectory_format=str(c.algorithm.replay_buffer.trajectory_format),
        )
        buffer.load_checkpoint(str(replay_dir))
        if buffer.total_samples <= 0:
            raise ValueError(f"Replay category {kind} is empty")
        replay[kind] = buffer
    if not replay:
        raise ValueError(f"No replay buffers found under {c.data.replay_path}")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    m = c.actor.model
    sd = int(m.z_dim + m.proprio_dim)
    ad = int(m.action_dim * m.num_action_chunks)
    q = TwinQCritic(sd, ad, int(c.trainer.hidden_dim), int(c.trainer.hidden_layers)).to(
        dev
    )
    qt = TwinQCritic(
        sd, ad, int(c.trainer.hidden_dim), int(c.trainer.hidden_layers)
    ).to(dev)
    qt.load_state_dict(q.state_dict())
    for p in qt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=float(c.trainer.learning_rate))
    start = 0
    if c.trainer.resume_checkpoint:
        ck = torch.load(
            c.trainer.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if ck["manifest_hash"] != manifest_hash:
            raise ValueError("Checkpoint manifest mismatch")
        q.load_state_dict(ck["critic"])
        qt.load_state_dict(ck["target"])
        opt.load_state_dict(ck["optimizer"])
        start = int(ck["step"])
    log_interval = int(c.trainer.log_interval)
    save_interval = int(c.trainer.save_interval)
    keep_period = int(c.trainer.keep_period)
    if log_interval <= 0 or save_interval <= 0 or keep_period <= 0:
        raise ValueError(
            "log_interval, save_interval, and keep_period must be positive"
        )
    if keep_period % save_interval != 0:
        raise ValueError("keep_period must be a multiple of save_interval")
    gamma = float(c.algorithm.gamma)
    tau = float(c.algorithm.tau)
    metric_logger = MetricLogger(c)
    try:
        for step in range(start, int(c.trainer.total_steps)):
            b = sample_replay(replay, int(c.trainer.batch_size))
            o = b["curr_obs"]
            n = b["next_obs"]
            s = torch.cat([o["z_rl"].float(), o["proprio"].float()], -1).to(dev)
            ns = torch.cat([n["z_rl"].float(), n["proprio"].float()], -1).to(dev)
            a = b["actions"].float().flatten(1).to(dev)
            r = b["forward_inputs"]["step_rewards"].float().squeeze(1).to(dev)
            d = b["dones"].bool().reshape(-1).to(dev)
            next_q = torch.zeros_like(d, dtype=r.dtype, device=dev)
            with torch.no_grad():
                valid = ~d
                if valid.any():
                    na = n["behavior_actions"].float().to(dev)[valid].flatten(1)
                    next_q[valid] = qt(ns[valid], na).min(-1).values
            y = td_target(r, d, next_q, gamma)
            pred = q(s, a)
            loss = F.mse_loss(pred[:, 0], y) + F.mse_loss(pred[:, 1], y)
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(pred).all()
                or not torch.isfinite(y).all()
            ):
                raise FloatingPointError(
                    "Critic loss, prediction, or target is non-finite"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                q.parameters(), float(c.trainer.max_grad_norm)
            )
            if not torch.isfinite(gn):
                raise FloatingPointError("Critic gradient norm is non-finite")
            opt.step()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), qt.parameters(), strict=True):
                    tp.lerp_(p, tau)
            if (step + 1) % log_interval == 0:
                met = {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "q1": float(pred[:, 0].mean().detach()),
                    "q2": float(pred[:, 1].mean().detach()),
                    "td_abs": float((pred.mean(-1) - y).abs().mean().detach()),
                    "grad_norm": float(gn),
                }
                met.update(validate(q, qt, validation_eps, dev, gamma))
                print(json.dumps(met), flush=True)
                metric_logger.log(
                    {
                        f"{('val' if key.startswith('val_') else 'train')}/{key.removeprefix('val_')}": value
                        for key, value in met.items()
                        if key != "step"
                    },
                    step=step + 1,
                )
            if (step + 1) % save_interval == 0 or step + 1 == int(
                c.trainer.total_steps
            ):
                torch.save(
                    {
                        "step": step + 1,
                        "critic": q.state_dict(),
                        "target": qt.state_dict(),
                        "optimizer": opt.state_dict(),
                        "manifest_hash": manifest_hash,
                        "config": OmegaConf.to_container(c, resolve=True),
                        "q01": lo,
                        "q99": hi,
                    },
                    out / f"critic_step_{step + 1}.pt",
                )
                prune_checkpoints(out, keep_period, step + 1)
    finally:
        metric_logger.finish()
    for b in replay.values():
        b.close(wait=True)


if __name__ == "__main__":
    main()
