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

"""Tests for freezing the OpenPI core in RLT-only Stage 1 SFT."""

import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.openpi_rlinf import get_model
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import build_rlt_config


class _TinyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rlt_module = nn.Linear(2, 2)


def _config(checkpoint, **openpi_overrides):
    openpi = {
        "task": "sft",
        "use_rlt": True,
        "rlt_alpha": 0.0,
        "model_action_dim": 2,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
    }
    openpi.update(openpi_overrides)
    return OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "precision": "fp32",
            "num_action_chunks": 2,
            "num_steps": 1,
            "action_dim": 2,
            "openpi": openpi,
        }
    )


def test_freeze_defaults_to_false():
    assert build_rlt_config(OmegaConf.create({})).rlt_freeze_vla is False
    assert build_rlt_config(OmegaConf.create({"rlt_freeze_vla": True})).rlt_freeze_vla


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "rl"},
        {"task": "eval"},
        {"use_rlt": False},
        {"rlt_alpha": 1.0},
        {"rlt_alpha": 1e-12},
    ],
)
def test_invalid_freeze_request_fails_before_checkpoint_loading(tmp_path, overrides):
    cfg = _config(tmp_path / "missing", rlt_freeze_vla=True, **overrides)
    with pytest.raises(
        ValueError,
        match="rlt_freeze_vla requires task='sft', use_rlt=True, and rlt_alpha=0.0",
    ):
        get_model(cfg)


@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("full_checkpoint", [False, True])
def test_freeze_applied_after_weights_load(
    tmp_path, monkeypatch, freeze, full_checkpoint
):
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_rlinf.utils import model_builders

    checkpoint = tmp_path / (
        "full_weights.pt" if full_checkpoint else "model.safetensors"
    )
    checkpoint.touch()
    monkeypatch.setattr(Pi0Config, "create", lambda self: nn.Linear(2, 2))
    monkeypatch.setattr(
        model_builders,
        "_build_sft_model",
        lambda model_cfg, model, **kwargs: _TinyWrapper(model),
    )
    monkeypatch.setattr(
        "rlinf.models.embodiment.openpi_rlinf.load_base_safetensors",
        lambda *args: None,
    )

    def load_full(wrapper, *args, **kwargs):
        # Loading happens before the final trainability flags are applied.
        wrapper.model.requires_grad_(True)
        wrapper.rlt_module.requires_grad_(False)

    monkeypatch.setattr(
        "rlinf.models.embodiment.openpi_rlinf.load_full_wrapper_weights", load_full
    )
    wrapper = get_model(_config(checkpoint, rlt_freeze_vla=freeze))
    assert all(p.requires_grad == (not freeze) for p in wrapper.model.parameters())
    assert all(
        p.requires_grad == (freeze or not full_checkpoint)
        for p in wrapper.rlt_module.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP requires CUDA")
def test_no_shard_fsdp_adamw_contains_only_rlt_parameters(tmp_path):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    if dist.is_initialized():
        pytest.skip("test requires an isolated process group")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'dist_init'}", rank=0, world_size=1
    )
    try:
        torch.manual_seed(7)
        wrapper = _TinyWrapper(nn.Linear(2, 2)).cuda()
        baseline = copy.deepcopy(wrapper)
        wrapper.model.requires_grad_(False)
        wrapper.rlt_module.requires_grad_(True)
        wrapped = FSDP(
            wrapper,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
        )
        named = dict(wrapped.named_parameters())
        rlt_ids = {id(p) for name, p in named.items() if "rlt_module." in name}
        vla_ids = {id(p) for name, p in named.items() if "model." in name}
        manager = object.__new__(FSDPModelManager)
        manager._cfg = OmegaConf.create(
            {
                "optim": {"lr": 1e-2, "adam_beta1": 0.9, "adam_beta2": 0.999},
                "fsdp_config": {"sharding_strategy": "no_shard"},
            }
        )
        manager.store_requires_grad_param_name = []
        optimizer = manager.build_optimizer(wrapped)
        optimizer_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        assert rlt_ids and vla_ids
        assert optimizer_ids == rlt_ids
        assert optimizer_ids.isdisjoint(vla_ids)

        x = torch.ones(2, 2, device="cuda")
        before_vla = [p.detach().clone() for p in wrapper.model.parameters()]
        before_rlt = [p.detach().clone() for p in wrapper.rlt_module.parameters()]
        # This small loss exercises the same detached-prefix, zero-alpha graph
        # structure as Stage 1 without constructing the multi-billion-param Pi0.
        vla_loss = wrapper.model(x).square().mean()
        rlt_loss = wrapper.rlt_module(wrapper.model(x).detach()).square().mean()
        loss = rlt_loss + 0.0 * vla_loss
        expected_vla = baseline.model(x).square().mean()
        expected_rlt = baseline.rlt_module(baseline.model(x).detach()).square().mean()
        torch.testing.assert_close(vla_loss, expected_vla)
        torch.testing.assert_close(rlt_loss, expected_rlt)
        torch.testing.assert_close(loss, expected_rlt)
        loss.backward()
        assert all(p.grad is None for p in wrapper.model.parameters())
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in wrapper.rlt_module.parameters()
        )
        optimizer.step()
        assert all(
            torch.equal(a, b) for a, b in zip(before_vla, wrapper.model.parameters())
        )
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before_rlt, wrapper.rlt_module.parameters())
        )
    finally:
        dist.destroy_process_group()
