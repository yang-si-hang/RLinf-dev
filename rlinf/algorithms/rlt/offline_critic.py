# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Validated import of UR10e offline twin-Q weights into online TD3."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from rlinf.envs.realworld.ur10e.action_space import UR10eActionSpace


def load_ur10e_offline_critic(
    model: torch.nn.Module,
    checkpoint_path: str,
    manifest_path: str,
    stats_path: str,
    stage1_success_metadata_path: str | None = None,
    stage1_failure_metadata_path: str | None = None,
) -> None:
    """Load only Q weights after verifying data and architecture provenance."""
    action_space = UR10eActionSpace(
        stats_path,
        manifest_path,
        stage1_success_metadata_path=stage1_success_metadata_path,
        stage1_failure_metadata_path=stage1_failure_metadata_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if checkpoint.get("manifest_hash") != manifest_hash:
        raise ValueError("Offline critic checkpoint manifest SHA256 mismatch")
    if not np.array_equal(
        checkpoint.get("q01"), action_space.q01
    ) or not np.array_equal(checkpoint.get("q99"), action_space.q99):
        raise ValueError("Offline critic action statistics mismatch")
    expected = {
        "z_dim": 2048,
        "proprio_dim": 10,
        "action_dim": 10,
        "num_action_chunks": 15,
        "critic_type": "twin_q",
    }
    if checkpoint.get("config", {}).get("actor", {}).get("model") != expected:
        raise ValueError("Offline critic model configuration mismatch")
    state = model.q_head.state_dict()
    weights = checkpoint.get("critic")
    if (
        not isinstance(weights, dict)
        or state.keys() != weights.keys()
        or any(state[key].shape != weights[key].shape for key in state)
    ):
        raise ValueError("Offline critic weight keys or tensor shapes mismatch")
    model.q_head.load_state_dict(weights, strict=True)
