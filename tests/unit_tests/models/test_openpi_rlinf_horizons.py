# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Tests for OpenPI_RLinf full prediction-chunk handling."""

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_rlinf import _resolve_action_horizon
from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)


def test_rlinf_chunk_matches_prediction_horizon():
    cfg = OmegaConf.create(
        {"num_action_chunks": 20, "openpi": {"model_action_horizon": 20}}
    )

    assert _resolve_action_horizon(cfg, cfg.openpi) == 20


def test_prediction_horizon_defaults_to_configured_chunk_horizon():
    cfg = OmegaConf.create({"num_action_chunks": 7, "openpi": {}})

    assert _resolve_action_horizon(cfg, cfg.openpi) == 7


def test_rlinf_cannot_truncate_prediction_horizon():
    cfg = OmegaConf.create(
        {"num_action_chunks": 10, "openpi": {"model_action_horizon": 20}}
    )

    with pytest.raises(ValueError, match="must equal"):
        _resolve_action_horizon(cfg, cfg.openpi)


def test_ur_repack_uses_upstream_dotted_keys():
    model = object.__new__(OpenPiPytorchEvalActionModel)
    model.config_name = "pi05_ur10e_lora_finetune"
    model.state_indices = None
    env_obs = {
        "main_images": torch.zeros(1, 3, 8, 8),
        "extra_view_images": torch.zeros(1, 1, 3, 8, 8),
        "states": torch.zeros(1, 10),
        "task_descriptions": ["pick"],
    }

    repacked = model._repack_env_obs(env_obs)

    assert "observation.state" in repacked
    assert "observation/state" not in repacked
    assert "observation.images.base_0_rgb" in repacked
    assert "observation.images.left_wrist_0_rgb" in repacked
