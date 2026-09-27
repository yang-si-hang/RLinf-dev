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

import json
import sys
import types

import numpy as np
import torch
from omegaconf import OmegaConf

from toolkits.openpi.prepare_rlt_stage2_replay import import_data, transform
from toolkits.openpi.rlt_stage2_critic_only import prune_checkpoints, td_target


def _dataset(length):
    state = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    rows = []
    for i in range(length):
        action = state.copy()
        action[0] = i / 1000
        action[9] = float(i % 2)
        rows.append(
            {
                "index": torch.tensor(i),
                "episode_index": torch.tensor(7),
                "frame_index": torch.tensor(i),
                "observation.state": torch.tensor(state),
                "action": torch.tensor(action),
            }
        )
    rows[-1]["action"][0] = 100.0  # Last action was never executed.

    class Meta:
        episodes = {
            "dataset_from_index": torch.tensor([0]),
            "dataset_to_index": torch.tensor([length]),
        }

    class Dataset:
        meta = Meta()
        hf_dataset = None

        def __init__(self):
            self.hf_dataset = self

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

    return Dataset()


def _install_fake_lerobot(monkeypatch, dataset):
    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = lambda repo_id, root: dataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)


def _fixture(tmp_path, monkeypatch, length=31):
    dataset = _dataset(length)
    _install_fake_lerobot(monkeypatch, dataset)
    zdir = tmp_path / "z"
    zdir.mkdir()
    np.save(
        zdir / "z_rl.npy", np.arange(length * 4, dtype=np.float32).reshape(length, 4)
    )
    np.save(zdir / "indices.npy", np.arange(length, dtype=np.int64))
    (zdir / "metadata.json").write_text(json.dumps({"checkpoint_step": 18000}))
    cfg = OmegaConf.create(
        {
            "data": {
                "lerobot_repo_id": "fixture",
                "stage1_checkpoint_step": 18000,
                "executed_chunk_length": 15,
                "drop_incomplete_suffix": True,
                "success_step_reward": -1.0,
                "failure_step_reward": -1.0,
                "failure_terminal_reward": -750.0,
            }
        }
    )
    lo = np.zeros(10, dtype=np.float32)
    hi = np.ones(10, dtype=np.float32)
    return cfg, dataset, zdir, lo, hi


def test_transform_roundtrip_preserves_gripper_and_unclipped_quantile_values():
    state = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32)
    actions = np.tile(state, (2, 1))
    actions[:, 0] += 2.0
    actions[:, 9] = 0.5
    transformed = transform(state, actions, np.zeros(10), np.ones(10))
    assert transformed.shape == (2, 10)
    assert transformed[0, 0] > 1.0
    assert abs(float(transformed[0, 9])) < 2e-6


def test_import_drops_final_unexecuted_action_and_aligns_terminal_observation(
    tmp_path, monkeypatch
):
    cfg, _, zdir, lo, hi = _fixture(tmp_path, monkeypatch, length=31)
    episodes = import_data(cfg, tmp_path, zdir, True, lo, hi)
    transitions = episodes[0]["ts"]
    assert len(transitions) == 2
    assert transitions[0]["d"] is False
    assert transitions[1]["d"] is True
    assert transitions[0]["a"].shape == (15, 10)
    assert transitions[1]["a"][-1, 0] < 1.0
    assert np.array_equal(transitions[1]["nz"], np.load(zdir / "z_rl.npy")[30])
    assert np.count_nonzero(transitions[1]["na"]) == 0
    assert np.all(transitions[1]["r"] == -1.0)


def test_suffix_trim_and_failure_terminal_reward(tmp_path, monkeypatch):
    cfg, _, zdir, lo, hi = _fixture(tmp_path, monkeypatch, length=35)
    episodes = import_data(cfg, tmp_path, zdir, False, lo, hi)
    assert episodes[0]["drop"] == 4
    transitions = episodes[0]["ts"]
    assert len(transitions) == 2
    assert np.array_equal(transitions[0]["z"], np.load(zdir / "z_rl.npy")[0])
    assert np.array_equal(transitions[-1]["nz"], np.load(zdir / "z_rl.npy")[30])
    assert transitions[-1]["r"][-1] == -750.0
    assert np.all(transitions[-1]["r"][:-1] == -1.0)


def test_td_target_uses_control_step_discount_and_terminal_mask():
    gamma = 0.9
    rewards = torch.full((2, 15), -1.0)
    dones = torch.tensor([False, True])
    next_q = torch.tensor([10.0, 999.0])
    target = td_target(rewards, dones, next_q, gamma)
    expected_return = -sum(gamma**k for k in range(15))
    assert torch.allclose(target[0], torch.tensor(expected_return + gamma**15 * 10.0))
    assert torch.allclose(target[1], torch.tensor(expected_return))


def test_prune_checkpoints_keeps_periodic_and_latest(tmp_path):
    for step in (1000, 2000, 3000, 4000, 5000, 6000):
        (tmp_path / f"critic_step_{step}.pt").touch()
    prune_checkpoints(tmp_path, keep_period=5000, latest_step=6000)
    assert {p.name for p in tmp_path.glob("critic_step_*.pt")} == {
        "critic_step_5000.pt",
        "critic_step_6000.pt",
    }
    (tmp_path / "critic_step_7000.pt").touch()
    prune_checkpoints(tmp_path, keep_period=5000, latest_step=7000)
    assert {p.name for p in tmp_path.glob("critic_step_*.pt")} == {
        "critic_step_5000.pt",
        "critic_step_7000.pt",
    }
