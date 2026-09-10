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
"""OpenPI data configuration for Universal Robots arms."""

import dataclasses
import pathlib
from collections.abc import Sequence

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.policies import ur_policy
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from scipy.spatial.transform import Rotation
from typing_extensions import override

UR_ACTION_DIM = 10


def rotation_to_rot6d(rotation: Rotation) -> np.ndarray:
    """Encode a single rotation using the first two columns of its matrix."""
    matrix = rotation.as_matrix()
    if matrix.shape != (3, 3):
        raise ValueError("Expected a single rotation")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rot6d_to_rotation(rot6d: Sequence[float]) -> Rotation:
    """Decode a 6D rotation with Gram-Schmidt orthonormalization."""
    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Rot6D must contain 6 values, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rot6D values must be finite")
    first = values[:3]
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:
        raise ValueError("Rot6D first direction must be non-zero")
    first = first / norm
    second = values[3:] - np.dot(first, values[3:]) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-8:
        raise ValueError("Rot6D directions must not be parallel")
    second = second / norm
    return Rotation.from_matrix(
        np.column_stack((first, second, np.cross(first, second)))
    )


def _validate(state, actions):
    state_values = np.asarray(state, dtype=np.float64)
    action_array = np.asarray(actions)
    action_values = np.asarray(actions, dtype=np.float64)
    if state_values.shape != (UR_ACTION_DIM,):
        raise ValueError(
            f"UR state must have shape ({UR_ACTION_DIM},), got {state_values.shape}"
        )
    if action_values.ndim == 0 or action_values.shape[-1] != UR_ACTION_DIM:
        raise ValueError(
            f"UR actions must have last dimension {UR_ACTION_DIM}, got {action_values.shape}"
        )
    if not np.all(np.isfinite(state_values)) or not np.all(np.isfinite(action_values)):
        raise ValueError("UR state and action values must be finite")
    dtype = (
        action_array.dtype
        if np.issubdtype(action_array.dtype, np.floating)
        else np.float32
    )
    return state_values, action_values, dtype


def absolute_actions_to_relative(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Express absolute TCP targets relative to the current TCP frame."""
    state, actions, dtype = _validate(state, actions)
    position, rotation = state[:3], rot6d_to_rotation(state[3:9])
    inverse = rotation.inv()
    result = actions.copy()
    for source, target in zip(
        actions.reshape(-1, 10), result.reshape(-1, 10), strict=True
    ):
        target[:3] = inverse.apply(source[:3] - position)
        target[3:9] = rotation_to_rot6d(inverse * rot6d_to_rotation(source[3:9]))
    return result.astype(dtype, copy=False)


def relative_actions_to_absolute(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Convert TCP-frame relative actions to absolute TCP targets."""
    state, actions, dtype = _validate(state, actions)
    position, rotation = state[:3], rot6d_to_rotation(state[3:9])
    result = actions.copy()
    for source, target in zip(
        actions.reshape(-1, 10), result.reshape(-1, 10), strict=True
    ):
        target[:3] = position + rotation.apply(source[:3])
        target[3:9] = rotation_to_rot6d(rotation * rot6d_to_rotation(source[3:9]))
    return result.astype(dtype, copy=False)


def make_ur_example() -> dict:
    """Create an example observation following the UR input contract."""
    return {
        "observation.images.base_0_rgb": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation.images.left_wrist_0_rgb": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation.state": np.concatenate(
            (np.zeros(3), rotation_to_rot6d(Rotation.identity()), np.zeros(1))
        ).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"UR images must be rank 3, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"UR images must have three channels, got shape {image.shape}")
    return image


# @dataclasses.dataclass(frozen=True)
# class URInputs(transforms.DataTransformFn):
#     model_type: _model.ModelType

#     def __call__(self, data: dict) -> dict:
#         if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
#             raise ValueError(f"Unsupported model type for UR policy: {self.model_type}")
#         state = np.asarray(data["observation.state"])
#         if state.shape != (10,) or not np.all(np.isfinite(state)):
#             raise ValueError("UR state must contain 10 finite values")
#         base = _parse_image(data["observation.images.base_0_rgb"])
#         wrist = _parse_image(data["observation.images.left_wrist_0_rgb"])
#         result = {
#             "state": state,
#             "image": {
#                 "base_0_rgb": base,
#                 "left_wrist_0_rgb": wrist,
#                 "right_wrist_0_rgb": np.zeros_like(base),
#             },
#             "image_mask": {
#                 "base_0_rgb": np.True_,
#                 "left_wrist_0_rgb": np.True_,
#                 "right_wrist_0_rgb": np.False_,
#             },
#         }
#         if "actions" in data:
#             actions = np.asarray(data["actions"])
#             if (
#                 actions.ndim == 0
#                 or actions.shape[-1] != 10
#                 or not np.all(np.isfinite(actions))
#             ):
#                 raise ValueError(
#                     "UR actions must have 10 finite values in the last dimension"
#                 )
#             result["actions"] = actions
#         if "prompt" in data:
#             prompt = data["prompt"]
#             result["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
#         return result


# @dataclasses.dataclass(frozen=True)
# class AbsoluteTCPActionsToRelative(transforms.DataTransformFn):
#     def __call__(self, data: dict) -> dict:
#         if "actions" not in data:
#             return data
#         return {
#             **data,
#             "actions": absolute_actions_to_relative(data["state"], data["actions"]),
#         }


# @dataclasses.dataclass(frozen=True)
# class RelativeTCPActionsToAbsolute(transforms.DataTransformFn):
#     """Recover absolute TCP actions from model-predicted relative actions."""

#     def __call__(self, data: dict) -> dict:
#         return {
#             **data,
#             "actions": relative_actions_to_absolute(data["state"], data["actions"]),
#         }


# @dataclasses.dataclass(frozen=True)
# class UROutputs(transforms.DataTransformFn):
#     def __call__(self, data: dict) -> dict:
#         return {"actions": np.asarray(data["actions"][..., :UR_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class LeRobotURDataConfig(DataConfigFactory):
    """Data configuration for 20 Hz UR TCP-pose datasets in LeRobot format."""

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> DataConfig:
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation.images.base_0_rgb": "observation.images.base_0_rgb",
                        "observation.images.left_wrist_0_rgb": "observation.images.left_wrist_0_rgb",
                        "observation.state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = transforms.Group(
            inputs=[
                ur_policy.URInputs(model_type=model_config.model_type),
                ur_policy.AbsoluteTCPActionsToRelative(),
            ],
            outputs=[
                ur_policy.UROutputs(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )
