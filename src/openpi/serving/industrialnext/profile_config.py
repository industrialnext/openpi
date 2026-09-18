# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven wire/model profile for Industrial Next GR00T serving."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from dataclasses import fields
import time
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from openpi.shared import taro_contract
from openpi.shared import taro_geometry

from .adapter import CachedImage
from .adapter import ObservationAdmission
from .adapter import ObservationSnapshot
from .task_catalog import task_catalog_from_mapping


@dataclass(frozen=True)
class ProfileLayout:
    key: str
    fields: tuple[str, ...]
    widths: tuple[int, ...]
    rot6d: str | None
    rot6d_index: int | None
    rep: str | None = None
    action_type: str | None = None
    action_format: str | None = None
    state_key: str | None = None

    @property
    def width(self) -> int:
        return sum(self.widths)


class ConfigDrivenIndustrialNextProfile:
    def __init__(self, config_name: str):
        self.contract = taro_contract.load_profile(config_name)
        p, s = self.contract, self.contract["serving"]
        self.name = p["name"]
        self.profile_name = s["profile"]
        self.action_horizon = p["action_horizon"]
        self.action_start_offset_steps = p["target_offset"]
        self.image_height, self.image_width = s["image_size"]
        self.field_lengths = s["field_lengths"]
        self.field_units = s["field_units"]
        self.eef_frame = s["eef_frame"]
        self.neck_joint_names = ()
        self.ignored_observation_keys = frozenset(s["ignored_observation_keys"])
        self.gripper_action_keys = tuple(s["gripper_action_keys"])
        self.task_catalog = task_catalog_from_mapping("taro", p["task_catalog"])
        self.supported_rtc_modes = ("off",)
        for key in (
            "host",
            "port",
            "control_hz",
            "rtc_mode",
            "action_offset",
            "ensemble_strategy",
            "ensemble_coeff",
            "max_ensemble_chunks",
            "chunk_transition_frames",
            "control_clock_mode",
            "max_action_lateness_s",
            "max_control_clock_drift_s",
            "max_command_gap_s",
        ):
            setattr(self, key, s[key])
        model_cameras = {"head": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}
        self.wire_image_to_model = {wire: model_cameras[key] for key, wire in s["cameras"].items()}

        def layouts(items):
            return tuple(
                ProfileLayout(
                    key=x["key"],
                    fields=tuple(x["fields"]),
                    widths=tuple(self.field_lengths[f] for f in x["fields"]),
                    rot6d=x.get("rot6d"),
                    rot6d_index=1 if x.get("rot6d") else None,
                    rep=x.get("rep"),
                    action_type=x.get("type"),
                    action_format=x.get("format"),
                    state_key=x.get("state_key"),
                )
                for x in items
            )

        self.state_layouts = layouts(p["state"])
        self.action_layouts = layouts(p["action"])

    def _state_vector(self, fields):
        return np.concatenate(
            [
                taro_geometry.change_convention(np.asarray(fields[f]), to_columns=False)
                if f in self.rotation_state_fields
                else np.asarray(fields[f], dtype=np.float32)
                for f in self.state_fields
            ]
        ).astype(np.float32)

    def build_model_observation(self, snapshot):
        images = {}
        for wire, model in self.wire_image_to_model.items():
            decoded = cv2.imdecode(np.frombuffer(snapshot.images[wire].payload, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (self.image_height, self.image_width, 3):
                raise ValueError(f"Invalid decoded RGB image: {wire}")
            images[model] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return {"state": self._state_vector(snapshot.state), "image": images, "prompt": snapshot.task_text}

    def build_synthetic_model_observation(self, task_text):
        values = {f: np.zeros(self.field_lengths[f]) for f in self.state_fields}
        for f in self.rotation_state_fields:
            values[f] = np.array([1, 0, 0, 0, 1, 0])
        return {
            "state": self._state_vector(values),
            "image": {
                k: np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                for k in self.wire_image_to_model.values()
            },
            "prompt": task_text,
        }

    def map_action_chunk(self, action):
        action = np.asarray(action)
        if action.shape != (self.action_horizon, 29) or not np.isfinite(action).all():
            raise ValueError("Expected finite absolute physical actions (40,29)")
        rows = []
        for values in action:
            row, cursor = {}, 0
            for f in self.action_fields:
                width = self.field_lengths[f]
                part = values[cursor : cursor + width]
                if f in self.rotation_action_fields:
                    part = taro_geometry.change_convention(part, to_columns=True)
                row[f] = part.astype(float).tolist()
                cursor += width
            rows.append(row)
        return tuple(rows)

    @property
    def state_fields(self) -> tuple[str, ...]:
        return tuple(field for layout in self.state_layouts for field in layout.fields)

    @property
    def action_fields(self) -> tuple[str, ...]:
        return tuple(field for layout in self.action_layouts for field in layout.fields)

    @property
    def rotation_state_fields(self) -> tuple[str, ...]:
        return tuple(
            layout.fields[layout.rot6d_index] for layout in self.state_layouts if layout.rot6d_index is not None
        )

    @property
    def position_action_fields(self) -> tuple[str, ...]:
        return tuple(layout.fields[0] for layout in self.action_layouts if layout.action_type == "EEF")

    @property
    def rotation_action_fields(self) -> tuple[str, ...]:
        return tuple(
            layout.fields[layout.rot6d_index] for layout in self.action_layouts if layout.rot6d_index is not None
        )

    @property
    def auxiliary_action_fields(self) -> tuple[str, ...]:
        pose_fields = set(self.position_action_fields) | set(self.rotation_action_fields)
        return tuple(field for field in self.action_fields if field not in pose_fields)

    def admit_observation(
        self,
        observation: Mapping[str, Any],
        *,
        image_cache: MutableMapping[str, CachedImage],
        timestep: int,
        task_uuid: str,
        task_text: str,
        generation: int,
        max_image_staleness_steps: int,
        now_s: float | None = None,
    ) -> ObservationAdmission:
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")
        now_s = time.monotonic() if now_s is None else now_s
        state = {
            field: _finite_tuple(observation.get(field), self.field_lengths[field], field)
            for field in self.state_fields
        }
        if observation.get("task_uuid", task_uuid) != task_uuid:
            raise ValueError("observation task_uuid does not match the registered session")
        if observation.get("task_text", task_text) != task_text:
            raise ValueError("observation task_text does not match the registered session")
        raw_metadata = observation.get("images_meta", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("images_meta must be a mapping")
        allowed = set(self.state_fields) | {"task_uuid", "task_text", "images_meta"}
        updates: dict[str, CachedImage] = {}
        ignored = 0
        for name, value in observation.items():
            if name in allowed:
                continue
            if name in self.ignored_observation_keys:
                ignored += 1
                continue
            if name not in self.wire_image_to_model:
                raise ValueError(f"unexpected image or observation field {name!r}")
            if not isinstance(value, bytes | bytearray | memoryview) or not value:
                raise ValueError(f"{name} must contain encoded image bytes")
            metadata = raw_metadata.get(name)
            if not isinstance(metadata, Mapping):
                raise ValueError(f"images_meta.{name} must be provided as a mapping")
            updates[name] = CachedImage(
                payload=bytes(value),
                metadata=_validate_rgb_metadata(metadata, name, self.image_width, self.image_height),
                updated_timestep=timestep,
                received_at_s=now_s,
            )
        for name in raw_metadata:
            if name not in updates and name not in self.ignored_observation_keys:
                raise ValueError(f"orphan or unexpected image metadata {name!r}")
        image_cache.update(updates)
        ages = {
            name: None if name not in image_cache else timestep - image_cache[name].updated_timestep
            for name in self.wire_image_to_model
        }
        missing = tuple(name for name, age in ages.items() if age is None)
        stale = tuple(
            name
            for name, age in ages.items()
            if age is not None
            and (
                age > max_image_staleness_steps
                or now_s - image_cache[name].received_at_s > max(1, max_image_staleness_steps) / self.control_hz
            )
        )
        snapshot = None
        if not missing and not stale:
            snapshot = ObservationSnapshot(
                state=MappingProxyType(state),
                images=MappingProxyType(dict(image_cache)),
                task_uuid=task_uuid,
                task_text=task_text,
                source_timestep=timestep,
                generation=generation,
                received_at_s=now_s,
            )
        return ObservationAdmission(
            snapshot=snapshot,
            image_ages=MappingProxyType(ages),
            missing_images=missing,
            stale_images=stale,
            ignored_depth_fields=ignored,
        )

    def _field_metadata(self, layouts: tuple[ProfileLayout, ...]) -> list[dict[str, Any]]:
        fields = []
        offset = 0
        for layout in layouts:
            for index, (name, width) in enumerate(zip(layout.fields, layout.widths, strict=True)):
                rotation = index == layout.rot6d_index
                role = "rotation" if rotation else "position" if name.endswith("_pose_pos") else "scalar"
                fields.append(
                    {
                        "name": name,
                        "length": width,
                        "slice": [offset, offset + width],
                        "role": role,
                        "group": layout.key,
                        "rotation_mode": "rot6d" if rotation else None,
                        "rotation_convention": "columns" if rotation else None,
                        "units": self.field_units[name],
                        "frame": self.eef_frame if rotation or role == "position" else None,
                        "representation": "absolute",
                    }
                )
                offset += width
        return fields

    def service_metadata(self) -> dict[str, Any]:
        return {
            "state_fields": self._field_metadata(self.state_layouts),
            "action_fields": self._field_metadata(self.action_layouts),
            "internal_action_fields": self._field_metadata(self.action_layouts),
            "state_dim": sum(layout.width for layout in self.state_layouts),
            "action_dim": sum(layout.width for layout in self.action_layouts),
            "internal_action_dim": sum(layout.width for layout in self.action_layouts),
            "action_fields_scope": "model_predicted_fields",
            "vision_modalities": ["rgb"],
            "profile": self.profile_name,
            "expert_camera_height": self.image_height,
            "expert_camera_width": self.image_width,
            "video_keys": list(self.wire_image_to_model.values()),
            "wire_image_keys": list(self.wire_image_to_model),
            "state_keys": [layout.key for layout in self.state_layouts],
            "action_keys": [layout.key for layout in self.action_layouts],
            "wire_state_fields": list(self.state_fields),
            "wire_action_fields": list(self.action_fields),
            "position_action_fields": list(self.position_action_fields),
            "rotation_action_fields": list(self.rotation_action_fields),
            "auxiliary_action_fields": list(self.auxiliary_action_fields),
            "action_horizon": self.action_horizon,
            "action_start_offset_steps": self.action_start_offset_steps,
            "field_lengths": dict(self.field_lengths),
            "field_units": dict(self.field_units),
            "eef_frame": self.eef_frame,
            "neck": {
                scope: {
                    "name": "neck_joint_pos",
                    "length": len(self.neck_joint_names),
                    "joint_names": list(self.neck_joint_names),
                    "units": "rad",
                    "representation": "absolute",
                    **({"predicted": True} if scope == "action" else {}),
                }
                for scope, fields in (("state", self.state_fields), ("action", self.action_fields))
                if "neck_joint_pos" in fields and self.neck_joint_names
            },
        }

    def monitoring_gripper_values(self, action: Mapping[str, list[float]] | None) -> dict[str, list[float]]:
        if action is None:
            return {}
        return {key: list(action[key]) for key in self.gripper_action_keys}


def _validate_rgb_metadata(metadata: Mapping[str, Any], name: str, width: int, height: int) -> Mapping[str, Any]:
    expected = {
        "format": "jpeg",
        "dtype": "uint8",
        "channels": 3,
        "height": height,
        "width": width,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"images_meta.{name}.{key} must be {expected_value!r}, got {metadata.get(key)!r}")
    quality = metadata.get("quality")
    if quality is not None and (not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100):
        raise ValueError(f"images_meta.{name}.quality must be an integer in [1, 100]")
    return MappingProxyType(dict(metadata))


def _finite_tuple(value: Any, width: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple | np.ndarray):
        raise ValueError(f"{name} must be a numeric sequence")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {width} finite values")
    return tuple(float(item) for item in array)


def resolve_serving_config(profile, **overrides):
    from .async_server import IndustrialNextServingConfig

    names = {f.name for f in fields(IndustrialNextServingConfig)}
    values = {k: v for k, v in profile.contract["serving"].items() if k in names}
    values["action_horizon"] = profile.action_horizon
    values.update({k: v for k, v in overrides.items() if v is not None})
    return IndustrialNextServingConfig(**values)
