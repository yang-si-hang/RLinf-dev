# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Scheduler resource descriptor for an externally controlled UR10e."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class UR10eHWInfo(HardwareInfo):
    """Connection information injected into a UR10e environment worker."""

    config: "UR10eConfig"


@Hardware.register()
class UR10eRobot(Hardware):
    """Declare external UR10e device services as schedulable resources."""

    HW_TYPE = "UR10e"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["UR10eConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Return configured UR10e services assigned to ``node_rank``."""
        if configs is None:
            raise ValueError("UR10e hardware requires explicit service configs.")
        infos = [
            UR10eHWInfo(type=cls.HW_TYPE, model=cfg.device_id, config=cfg)
            for cfg in configs
            if isinstance(cfg, UR10eConfig) and cfg.node_rank == node_rank
        ]
        return HardwareResource(type=cls.HW_TYPE, infos=infos) if infos else None


@NodeHardwareConfig.register_hardware_config(UR10eRobot.HW_TYPE)
@dataclass
class UR10eConfig(HardwareConfig):
    """Address and identity of an external UR10e device service."""

    endpoint: str = "tcp://127.0.0.1:5555"
    device_id: str = "ur10e"

    def __post_init__(self) -> None:
        """Validate the external service identity and endpoint."""
        super().__post_init__()
        if not self.endpoint.startswith(("tcp://", "ipc://")):
            raise ValueError("UR10e endpoint must use tcp:// or ipc://.")
        if not self.device_id:
            raise ValueError("UR10e device_id must not be empty.")
