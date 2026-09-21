# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Chunk-native Gym environment for an external UR10e device service."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rlinf.scheduler import UR10eHWInfo, WorkerInfo

from .device_client import UR10eDeviceClient, UR10eObservation


@dataclass
class UR10eEnvConfig:
    """Deployment-only configuration for :class:`UR10eEnv`."""

    endpoint: str | None = None
    timeout_ms: int = 120_000
    require_reset_confirmation: bool = True
    reset_confirmation_text: str = "start"
    camera_max_age_ms: int = 500
    max_num_steps: int = 240

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0 or self.camera_max_age_ms < 0:
            raise ValueError(
                "timeout_ms must be positive and camera_max_age_ms non-negative."
            )
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive.")
        if self.require_reset_confirmation and not self.reset_confirmation_text:
            raise ValueError("reset_confirmation_text must not be empty.")


class UR10eEnv(gym.Env):
    """Send complete action chunks to an externally managed UR10e service."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[UR10eHWInfo],
        env_idx: int,
        **_: Any,
    ):
        del worker_info, env_idx
        self.config = UR10eEnvConfig(**override_cfg)
        if hardware_info is not None and not isinstance(hardware_info, UR10eHWInfo):
            raise TypeError(f"Expected UR10eHWInfo, got {type(hardware_info)}.")
        endpoint = self.config.endpoint
        if endpoint is None and hardware_info is not None:
            endpoint = hardware_info.config.endpoint
        if not endpoint:
            raise ValueError("UR10e device service endpoint is required.")

        self._task_description = ""
        self._client = UR10eDeviceClient(endpoint, timeout_ms=self.config.timeout_ms)
        self._next_chunk_id = 0
        self._executed_steps = 0
        self._has_reset = False
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        image_space = gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "ur_state": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {"base_0_rgb": image_space, "left_wrist_0_rgb": image_space}
                ),
            }
        )

    @property
    def task_description(self) -> str:
        """Return the language instruction supplied to OpenPI."""
        return self._task_description

    @staticmethod
    def _run(coro):
        """Run one async client call from Gym's synchronous environment API."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("UR10eEnv synchronous API cannot run inside an event loop.")

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"{name} must be an HWC uint8 RGB image, got {image.shape}."
            )
        if image.shape != (224, 224, 3):
            raise ValueError(
                f"{name} must be 224x224; RLinf does not crop or resize, got {image.shape}."
            )
        return np.ascontiguousarray(image)

    def _validate_observation(self, observation: UR10eObservation) -> dict[str, Any]:
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (10,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"UR10e state must contain 10 finite values, got {state.shape}."
            )
        now = time.time()
        age_ms = (now - observation.capture_timestamp) * 1000.0
        if age_ms < 0 or age_ms > self.config.camera_max_age_ms:
            raise ValueError(
                f"UR10e camera frame age {age_ms:.1f}ms is invalid or stale."
            )
        if not np.isfinite(observation.robot_timestamp):
            raise ValueError("UR10e robot timestamp must be finite.")
        if not observation.task_description:
            raise ValueError("Device observation must include a task_description.")
        self._task_description = observation.task_description
        return {
            "state": {"ur_state": state},
            "frames": {
                "base_0_rgb": self._validate_image(
                    observation.base_image, "base_image"
                ),
                "left_wrist_0_rgb": self._validate_image(
                    observation.wrist_image, "wrist_image"
                ),
            },
        }

    def reset(self, *, seed=None, options=None):
        """Wait for manual reset confirmation and read without commanding devices."""
        super().reset(seed=seed)
        del options
        if self.config.require_reset_confirmation:
            expected = self.config.reset_confirmation_text.strip().lower()
            response = input(
                f"Manually reset UR10e, then type {self.config.reset_confirmation_text!r} "
                "to start inference: "
            )
            if response.strip().lower() != expected:
                raise KeyboardInterrupt("UR10e manual reset was cancelled.")
        observation = self._run(self._client.get_initial_observation())
        self._executed_steps = 0
        self._has_reset = True
        return self._validate_observation(observation), {}

    def step(self, action):
        """Reject scalar stepping because this device consumes native chunks."""
        del action
        raise RuntimeError("UR10eEnv requires execute_action_chunk(), not step().")

    def execute_action_chunk(self, actions: np.ndarray):
        """Submit one action chunk and return only its final observation."""
        if not self._has_reset:
            raise RuntimeError("UR10eEnv must be reset before executing a chunk.")
        actions = np.asarray(actions, dtype=np.float32)
        result = self._run(self._client.execute_chunk(actions, self._next_chunk_id))
        self._next_chunk_id += 1
        # Partial consumption is valid: the external service owns the execution
        # horizon and returns after the number of policy steps it selected.
        self._executed_steps += result.executed_steps
        truncated = (
            result.truncated or self._executed_steps >= self.config.max_num_steps
        )
        info = {
            "chunk_id": result.chunk_id,
            "executed_steps": result.executed_steps,
            "message": result.message,
        }
        return (
            self._validate_observation(result.observation),
            0.0,
            result.terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close local RPC resources without sending a device lifecycle command."""
        self._client.close()
