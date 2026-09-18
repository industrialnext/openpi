# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Industrial Next wire-to-GR00T semihumanoid adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any


@dataclass(frozen=True)
class CachedImage:
    payload: bytes
    metadata: Mapping[str, Any]
    updated_timestep: int
    received_at_s: float = field(default_factory=lambda: time.monotonic())


@dataclass(frozen=True)
class ObservationSnapshot:
    state: Mapping[str, tuple[float, ...]]
    images: Mapping[str, CachedImage]
    task_uuid: str
    task_text: str
    source_timestep: int
    generation: int
    received_at_s: float = field(default_factory=lambda: time.monotonic())


@dataclass(frozen=True)
class ObservationAdmission:
    snapshot: ObservationSnapshot | None
    image_ages: Mapping[str, int | None]
    missing_images: tuple[str, ...]
    stale_images: tuple[str, ...]
    ignored_depth_fields: int

    @property
    def ready(self) -> bool:
        return self.snapshot is not None


def snapshot_is_fresh(
    snapshot: ObservationSnapshot,
    *,
    current_timestep: int,
    active_generation: int,
    max_staleness_steps: int,
    now_s: float | None = None,
) -> bool:
    """Return whether pending work is still current enough to launch."""
    if snapshot.generation != active_generation:
        return False
    if current_timestep - snapshot.source_timestep > max_staleness_steps:
        return False
    now_s = time.monotonic() if now_s is None else now_s
    if now_s - snapshot.received_at_s > max(1, max_staleness_steps) / 50.0:
        return False
    return all(
        now_s - image.received_at_s <= max(1, max_staleness_steps) / 50.0
        and current_timestep - image.updated_timestep <= max_staleness_steps
        for image in snapshot.images.values()
    )
