# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""OpenPI quantile action contract for UR10e online control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ACTION_REPRESENTATION = "openpi_tcp_relative_xyz_rot6d_absolute_gripper"


class UR10eActionSpace:
    """Use the exact quantiles and epsilon used by the offline importer."""

    def __init__(
        self,
        stats_path: str,
        manifest_path: str | None = None,
        *,
        stage1_step: int = 18000,
        stage1_success_metadata_path: str | None = None,
        stage1_failure_metadata_path: str | None = None,
    ):
        path = Path(stats_path).expanduser()
        blob = json.loads(path.read_text())
        entries = [
            v["actions"]
            for v in blob.values()
            if isinstance(v, dict) and "actions" in v
        ]
        if not entries and "actions" in blob:
            entries = [blob["actions"]]
        if len(entries) != 1:
            raise ValueError("Expected exactly one OpenPI actions statistics entry")
        self.q01 = np.asarray(entries[0]["q01"], dtype=np.float32)
        self.q99 = np.asarray(entries[0]["q99"], dtype=np.float32)
        if (
            self.q01.shape != (10,)
            or self.q99.shape != (10,)
            or not np.isfinite(self.q01).all()
            or not np.isfinite(self.q99).all()
            or np.any(self.q99 <= self.q01)
        ):
            raise ValueError("Invalid 10D OpenPI action quantiles")
        if manifest_path is not None:
            manifest = json.loads(Path(manifest_path).expanduser().read_text())
            expected = {
                "norm_stats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "action_representation": ACTION_REPRESENTATION,
                "chunk_length": 15,
                "vla_prediction_horizon": 30,
                "stage1_checkpoint_step": stage1_step,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError(f"Replay manifest mismatch for {key}")
            for label, configured_path in (
                ("success", stage1_success_metadata_path),
                ("failure", stage1_failure_metadata_path),
            ):
                sha_key = f"{label}_z_metadata_sha256"
                if sha_key not in manifest:
                    continue
                metadata_path = (
                    Path(configured_path)
                    if configured_path is not None
                    else Path(manifest[f"{label}_z"]) / "metadata.json"
                )
                if (
                    hashlib.sha256(metadata_path.read_bytes()).hexdigest()
                    != manifest[sha_key]
                ):
                    raise ValueError(f"Replay manifest mismatch for {sha_key}")
                metadata = json.loads(metadata_path.read_text())
                if int(metadata.get("checkpoint_step", -1)) != stage1_step:
                    raise ValueError(
                        f"Stage 1 {label} metadata checkpoint step mismatch"
                    )

    def normalize(self, physical: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(physical, dtype=np.float32) - self.q01)
            / (self.q99 - self.q01 + 1e-6)
            * 2
            - 1
        ).astype(np.float32)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(normalized, dtype=np.float32) + 1)
            / 2
            * (self.q99 - self.q01 + 1e-6)
            + self.q01
        ).astype(np.float32)
