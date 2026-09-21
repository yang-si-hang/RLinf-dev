# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Gym registration for the external-service UR10e environment."""

from __future__ import annotations

from typing import Any, Mapping

from gymnasium.envs.registration import register

from .ur10e_env import UR10eEnv


def create_ur10e_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> UR10eEnv:
    """Build a chunk-native UR10e environment."""
    del env_cfg
    return UR10eEnv(override_cfg, worker_info, hardware_info, env_idx)


register(
    id="UR10eEnv-v1", entry_point="rlinf.envs.realworld.ur10e.tasks:create_ur10e_env"
)
