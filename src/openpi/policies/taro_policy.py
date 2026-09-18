"""Shared offline/online Taro transforms and snapshot-anchored decoding."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.shared import taro_geometry as geometry


@dataclasses.dataclass(frozen=True)
class TaroInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (38,) or not np.isfinite(state).all():
            raise ValueError("Taro observation must have 38 finite coordinates")
        result = {
            "state": state,
            "image": data["image"],
            "image_mask": dict.fromkeys(data["image"], np.True_),
            "prompt": data["prompt"],
        }
        if "taro_pool_id" in data:
            result["taro_pool_id"] = data["taro_pool_id"]
        if "actions" in data:
            result["actions"], result["action_loss_mask"] = geometry.relative_actions(
                state, data["actions"], data["action_loss_mask"]
            )
        return result


@dataclasses.dataclass(frozen=True)
class TaroOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :29]}


@dataclasses.dataclass(frozen=True)
class SanitizeMaskedActions(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "action_loss_mask" in data:
            mask = data["action_loss_mask"]
            if not np.any(mask):
                raise ValueError("Unsupervised training window")
            data["actions"] = np.where(mask, data["actions"], 0).astype(np.float32)
        return data


class TaroPolicy:
    """Expose absolute physical row-rot6d predictions from openpi's relative policy."""

    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation: dict, *, noise: np.ndarray | None = None) -> dict:
        anchor = np.asarray(observation["state"], dtype=np.float32).copy()
        output = self.policy.infer(observation, noise=noise)
        return {**output, "actions": geometry.absolute_actions(anchor, output["actions"])}
