# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical action slots shared by serving and sequential replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

Action = Mapping[str, tuple[float, ...]]


def execution_tick(source_tick: int, target_offset: int, action_offset: int, row: int) -> int:
    """Map an original model row to a request-driven execution tick."""
    if row < action_offset:
        raise ValueError("row is before the selected action offset")
    return source_tick + target_offset + row - action_offset


def freeze_action(action: Mapping[str, list[float]]) -> Action:
    return MappingProxyType({name: tuple(values) for name, values in action.items()})


def rotation_matrix(value: tuple[float, ...] | list[float]) -> np.ndarray:
    axes = np.asarray(value, dtype=np.float64).reshape(2, 3)
    first_norm = np.linalg.norm(axes[0])
    if not np.isfinite(first_norm) or first_norm < 1e-10:
        raise ValueError("degenerate column-rot6d first axis")
    first = axes[0] / first_norm
    second = axes[1] - np.dot(first, axes[1]) * first
    second_norm = np.linalg.norm(second)
    if not np.isfinite(second_norm) or second_norm < 1e-10:
        raise ValueError("degenerate column-rot6d second axis")
    second /= second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def _rot6d(matrix: np.ndarray) -> list[float]:
    return matrix[:, :2].T.reshape(-1).tolist()


def blend_actions(
    actions: list[Action], weights: np.ndarray, rotation_fields: tuple[str, ...]
) -> dict[str, list[float]]:
    """DEFT-equivalent quaternion scatter mean; native hand values stay continuous."""
    output = {}
    for name in actions[0]:
        values = np.asarray([action[name] for action in actions], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite physical action field {name}")
        if name in rotation_fields and len(actions) == 1:
            output[name] = _rot6d(rotation_matrix(values[0]))
        elif name in rotation_fields:
            quats = Rotation.from_matrix(np.stack([rotation_matrix(v) for v in values])).as_quat()
            eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("i,ij,ik->jk", weights, quats, quats))
            if eigenvalues[-1] - eigenvalues[-2] <= 1e-6:
                raise ValueError("ill-conditioned weighted rotation mean")
            output[name] = _rot6d(Rotation.from_quat(eigenvectors[:, -1]).as_matrix())
        else:
            output[name] = np.sum(values * weights[:, None], axis=0).tolist()
    return output


def interpolate_actions(
    previous: Action, current: Action, alpha: float, rotation_fields: tuple[str, ...]
) -> dict[str, list[float]]:
    output = {}
    for name in current:
        if name in rotation_fields:
            rotations = Rotation.from_matrix(
                np.stack([rotation_matrix(previous[name]), rotation_matrix(current[name])])
            )
            output[name] = _rot6d(Slerp([0, 1], rotations)(alpha).as_matrix())
        else:
            output[name] = ((1 - alpha) * np.asarray(previous[name]) + alpha * np.asarray(current[name])).tolist()
    return output


@dataclass(frozen=True)
class Contribution:
    action: Action
    source_tick: int
    model_row: int
    received_at_s: float
    deadline_s: float

    def provenance(self, weight: float) -> dict:
        return {
            "source_tick": self.source_tick,
            "model_row": self.model_row,
            "received_at_s": self.received_at_s,
            "weight": float(weight),
        }


@dataclass(frozen=True)
class TransitionAnchor:
    action: Action
    deadline_s: float
    alpha: float
    provenance: tuple[dict, ...]


@dataclass(frozen=True)
class ExecutionSlot:
    contributions: tuple[Contribution, ...]
    anchor: TransitionAnchor | None = None

    def live(self, now_s: float) -> ExecutionSlot:
        return ExecutionSlot(
            tuple(c for c in self.contributions if now_s <= c.deadline_s),
            self.anchor if self.anchor is not None and now_s <= self.anchor.deadline_s else None,
        )

    def resolve(
        self,
        *,
        strategy: str,
        coefficient: float,
        control_hz: float,
        rotation_fields: tuple[str, ...],
    ) -> tuple[dict[str, list[float]], dict]:
        if not self.contributions:
            raise ValueError("empty execution slot")
        contributions = self.contributions[-1:] if strategy == "latest_only" else self.contributions
        newest = max(c.received_at_s for c in contributions)
        weights = np.exp([-coefficient * (newest - c.received_at_s) * control_hz for c in contributions])
        weights /= weights.sum()
        output = blend_actions([c.action for c in contributions], weights, rotation_fields)
        provenance = {
            "contributions": [c.provenance(w) for c, w in zip(contributions, weights, strict=True)],
            "transition": None,
        }
        if self.anchor is not None:
            output = interpolate_actions(self.anchor.action, output, self.anchor.alpha, rotation_fields)
            provenance["transition"] = {
                "alpha": self.anchor.alpha,
                "deadline_s": self.anchor.deadline_s if math.isfinite(self.anchor.deadline_s) else None,
                "sources": list(self.anchor.provenance),
            }
        return output, provenance

    def anchor_deadline(self) -> float:
        return min(
            *(c.deadline_s for c in self.contributions),
            self.anchor.deadline_s if self.anchor is not None else math.inf,
        )
