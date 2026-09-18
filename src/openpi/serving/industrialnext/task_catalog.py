# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict task-catalog loading for the Industrial Next serving protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TOP_LEVEL_KEYS = frozenset({"schema_version", "task_family", "catalog_version", "tasks"})
_REQUIRED_TOP_LEVEL_KEYS = frozenset({"schema_version", "task_family", "tasks"})
_TASK_KEYS = frozenset({"task_uuid", "task_text", "display_name"})
_SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TaskCatalogEntry:
    """One immutable UUID-to-prompt catalog entry."""

    task_uuid: str
    task_text: str
    display_name: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "task_uuid": self.task_uuid,
            "task_text": self.task_text,
            "display_name": self.display_name,
        }


@dataclass(frozen=True)
class TaskCatalog:
    """Validated task catalog used by metadata and session registration."""

    schema_version: int
    task_family: str
    catalog_version: str | None
    tasks: tuple[TaskCatalogEntry, ...]

    @property
    def task_uuid_to_text(self) -> dict[str, str]:
        return {entry.task_uuid: entry.task_text for entry in self.tasks}

    def to_metadata(self) -> list[dict[str, str]]:
        return [entry.to_metadata() for entry in self.tasks]

    def resolve(self, task_uuid: str, task_text: str) -> str:
        task_uuid = _exact_nonempty_string(task_uuid, "task_uuid")
        task_text = _exact_nonempty_string(task_text, "task_text")
        expected = self.task_uuid_to_text.get(task_uuid)
        if expected is None:
            raise ValueError(f"unknown task_uuid {task_uuid!r}")
        if task_text != expected:
            raise ValueError(f"task_text mismatch for {task_uuid!r}: expected {expected!r}, got {task_text!r}")
        return expected


def load_task_catalog(path: str | Path) -> TaskCatalog:
    """Load a YAML catalog and reject schema drift or ambiguous task entries."""
    catalog_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to load task catalog {catalog_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("task catalog must be a mapping")

    raw_keys = frozenset(raw)
    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - raw_keys)
    unknown = sorted(raw_keys - _TOP_LEVEL_KEYS)
    if missing:
        raise ValueError(f"task catalog is missing keys: {missing}")
    if unknown:
        raise ValueError(f"task catalog has unknown keys: {unknown}")

    schema_version = raw["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError("schema_version must be an integer")
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {schema_version}; expected {_SUPPORTED_SCHEMA_VERSION}")
    task_family = _exact_nonempty_string(raw["task_family"], "task_family")
    catalog_version_raw = raw.get("catalog_version")
    catalog_version = (
        None if catalog_version_raw is None else _exact_nonempty_string(catalog_version_raw, "catalog_version")
    )

    raw_tasks = raw["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a non-empty list")
    tasks: list[TaskCatalogEntry] = []
    seen_uuids: set[str] = set()
    for index, raw_entry in enumerate(raw_tasks):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"tasks[{index}] must be a mapping")
        entry_keys = frozenset(raw_entry)
        if entry_keys != _TASK_KEYS:
            missing_entry = sorted(_TASK_KEYS - entry_keys)
            unknown_entry = sorted(entry_keys - _TASK_KEYS)
            raise ValueError(f"tasks[{index}] schema mismatch: missing={missing_entry}, unknown={unknown_entry}")
        task_uuid = _exact_nonempty_string(raw_entry["task_uuid"], f"tasks[{index}].task_uuid")
        if task_uuid in seen_uuids:
            raise ValueError(f"duplicate task_uuid {task_uuid!r}")
        seen_uuids.add(task_uuid)
        tasks.append(
            TaskCatalogEntry(
                task_uuid=task_uuid,
                task_text=_exact_nonempty_string(raw_entry["task_text"], f"tasks[{index}].task_text"),
                display_name=_exact_nonempty_string(raw_entry["display_name"], f"tasks[{index}].display_name"),
            )
        )
    return TaskCatalog(
        schema_version=schema_version,
        task_family=task_family,
        catalog_version=catalog_version,
        tasks=tuple(tasks),
    )


def task_catalog_from_mapping(
    task_family: str,
    tasks: Mapping[str, str],
    *,
    display_names: Mapping[str, str] | None = None,
) -> TaskCatalog:
    """Build the same validated catalog directly from an embodiment config."""
    task_family = _exact_nonempty_string(task_family, "task_family")
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("tasks must be a non-empty mapping")
    if display_names is not None and not isinstance(display_names, Mapping):
        raise ValueError("display_names must be a mapping")
    display_names = {} if display_names is None else display_names
    entries: list[TaskCatalogEntry] = []
    for task_uuid, task_text in tasks.items():
        uuid_value = _exact_nonempty_string(task_uuid, "task_uuid")
        entries.append(
            TaskCatalogEntry(
                task_uuid=uuid_value,
                task_text=_exact_nonempty_string(task_text, f"tasks.{uuid_value}"),
                display_name=_exact_nonempty_string(
                    display_names.get(uuid_value, uuid_value.replace("_", " ").title()),
                    f"display_names.{uuid_value}",
                ),
            )
        )
    return TaskCatalog(
        schema_version=_SUPPORTED_SCHEMA_VERSION,
        task_family=task_family,
        catalog_version=None,
        tasks=tuple(entries),
    )


def _exact_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if value != normalized:
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value
