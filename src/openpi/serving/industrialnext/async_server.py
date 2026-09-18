# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking Industrial Next direct-RPC server for a PI0.5 policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass
from dataclasses import field
import logging
import math
import time
from typing import Any, Protocol
import uuid

from industrialnext_rpc.direct.metadata import Metadata
import numpy as np

from .adapter import CachedImage
from .adapter import ObservationAdmission
from .adapter import ObservationSnapshot
from .adapter import snapshot_is_fresh
from .execution import Contribution
from .execution import ExecutionSlot
from .execution import TransitionAnchor
from .execution import execution_tick
from .execution import freeze_action
from .execution import rotation_matrix
from .profile_config import ConfigDrivenIndustrialNextProfile

logger = logging.getLogger(__name__)

ASYNC_PROTOCOL_VERSION = 2
ASYNC_CAPABILITIES = [
    "error_envelope_v2",
    "monitoring_in_step",
    "server_owned_gripper_snap",
]


def _source_rot6d_matrix(value: list[float]) -> np.ndarray:
    return rotation_matrix(value)


def _rotation_error_rad(left: list[float], right: list[float]) -> float:
    delta = _source_rot6d_matrix(left).T @ _source_rot6d_matrix(right)
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _prefix_errors(
    expected: tuple[dict[str, list[float]], ...],
    actual: tuple[dict[str, list[float]], ...],
    count: int,
    profile: ConfigDrivenIndustrialNextProfile,
) -> tuple[float, float, float]:
    position_error = 0.0
    orientation_error = 0.0
    gripper_error = 0.0
    for expected_row, actual_row in zip(expected[:count], actual[:count], strict=False):
        for field_name in profile.position_action_fields:
            left = np.asarray(actual_row[field_name], dtype=np.float64)
            right = np.asarray(expected_row[field_name], dtype=np.float64)
            position_error = max(position_error, float(np.linalg.norm(left - right)))
        for field_name in profile.rotation_action_fields:
            orientation_error = max(
                orientation_error,
                _rotation_error_rad(actual_row[field_name], expected_row[field_name]),
            )
        for field_name in profile.auxiliary_action_fields:
            left = np.asarray(actual_row[field_name], dtype=np.float64)
            right = np.asarray(expected_row[field_name], dtype=np.float64)
            gripper_error = max(gripper_error, float(np.max(np.abs(left - right))))
    return position_error, orientation_error, gripper_error


def _trajectory_dynamics(
    rows: tuple[dict[str, list[float]], ...],
    profile: ConfigDrivenIndustrialNextProfile,
) -> tuple[float, float, float, float, float]:
    max_position_step = 0.0
    max_orientation_step = 0.0
    max_gripper_step = 0.0
    max_position_second_difference = 0.0
    max_gripper_second_difference = 0.0
    for field_name in profile.position_action_fields:
        values = np.asarray([row[field_name] for row in rows], dtype=np.float64)
        if len(rows) >= 2:
            max_position_step = max(max_position_step, float(np.linalg.norm(np.diff(values, axis=0), axis=1).max()))
        if len(rows) >= 3:
            max_position_second_difference = max(
                max_position_second_difference,
                float(np.linalg.norm(np.diff(values, n=2, axis=0), axis=1).max()),
            )
    for field_name in profile.rotation_action_fields:
        if len(rows) >= 2:
            max_orientation_step = max(
                max_orientation_step,
                *(
                    _rotation_error_rad(rows[index][field_name], rows[index - 1][field_name])
                    for index in range(1, len(rows))
                ),
            )
    for field_name in profile.auxiliary_action_fields:
        values = np.asarray([row[field_name] for row in rows], dtype=np.float64)
        if len(rows) >= 2:
            max_gripper_step = max(max_gripper_step, float(np.abs(np.diff(values, axis=0)).max()))
        if len(rows) >= 3:
            max_gripper_second_difference = max(
                max_gripper_second_difference, float(np.abs(np.diff(values, n=2, axis=0)).max())
            )
    return (
        max_position_step,
        max_orientation_step,
        max_gripper_step,
        max_position_second_difference,
        max_gripper_second_difference,
    )


class Policy(Protocol):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class IndustrialNextServingConfig:
    """Runtime settings for the single-session PI0.5 server."""

    control_hz: float = 50.0
    action_horizon: int = 40
    max_image_staleness_steps: int = 5
    min_usable_action_steps: int = 1
    idle_session_timeout_s: float = 300.0
    stats_log_interval_steps: int = 250
    action_offset: int = 0
    ensemble_strategy: str = "temporal_exponential"
    ensemble_coeff: float = 0.1
    max_ensemble_chunks: int = 3
    chunk_transition_frames: int = 4
    max_action_lateness_s: float = 0.04
    max_control_clock_drift_s: float = 0.1
    control_clock_mode: str = "accepted_requests"
    max_command_gap_s: float | None = None
    rtc_mode: str = "off"
    max_position_step_m: float | None = None
    max_orientation_step_rad: float | None = None
    max_gripper_step: float | None = None
    max_position_second_difference_m: float | None = None
    max_gripper_second_difference: float | None = None

    def __post_init__(self) -> None:
        if self.control_clock_mode not in {"accepted_requests", "elapsed_time"}:
            raise ValueError("control_clock_mode must be accepted_requests or elapsed_time")
        if self.max_command_gap_s is not None and (
            isinstance(self.max_command_gap_s, bool)
            or not math.isfinite(self.max_command_gap_s)
            or self.max_command_gap_s <= 0
        ):
            raise ValueError("max_command_gap_s must be finite and positive")
        if not math.isfinite(self.control_hz) or self.control_hz != 50.0:
            raise ValueError("control_hz must be exactly 50.0")
        if (
            isinstance(self.action_horizon, bool)
            or not isinstance(self.action_horizon, int)
            or self.action_horizon <= 0
        ):
            raise ValueError("action_horizon must be a positive integer")
        if (
            not isinstance(self.max_image_staleness_steps, int)
            or isinstance(self.max_image_staleness_steps, bool)
            or self.max_image_staleness_steps < 0
        ):
            raise ValueError("max_image_staleness_steps must be a non-negative integer")
        if (
            not isinstance(self.min_usable_action_steps, int)
            or isinstance(self.min_usable_action_steps, bool)
            or not 1 <= self.min_usable_action_steps <= self.action_horizon
        ):
            raise ValueError(f"min_usable_action_steps must be within [1, {self.action_horizon}]")
        if not math.isfinite(self.idle_session_timeout_s) or self.idle_session_timeout_s <= 0:
            raise ValueError("idle_session_timeout_s must be finite and positive")
        if (
            not isinstance(self.stats_log_interval_steps, int)
            or isinstance(self.stats_log_interval_steps, bool)
            or self.stats_log_interval_steps < 0
        ):
            raise ValueError("stats_log_interval_steps must be a non-negative integer")
        for name in ("action_offset", "max_ensemble_chunks", "chunk_transition_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.action_offset > self.action_horizon - self.min_usable_action_steps:
            raise ValueError("action_offset must leave min_usable_action_steps")
        if self.max_ensemble_chunks < 1:
            raise ValueError("max_ensemble_chunks must be positive")
        if self.chunk_transition_frames > self.action_horizon - self.action_offset:
            raise ValueError("chunk_transition_frames exceeds selectable horizon")
        if self.ensemble_strategy not in {"temporal_exponential", "latest_only"}:
            raise ValueError("ensemble_strategy must be temporal_exponential or latest_only")
        if isinstance(self.ensemble_coeff, bool) or not math.isfinite(self.ensemble_coeff) or self.ensemble_coeff < 0:
            raise ValueError("ensemble_coeff must be finite and nonnegative")
        for name in ("max_action_lateness_s", "max_control_clock_drift_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rtc_mode != "off":
            raise ValueError("PI0.5 supports rtc_mode=off only")
        optional_limits = {
            "max_position_step_m": self.max_position_step_m,
            "max_orientation_step_rad": self.max_orientation_step_rad,
            "max_gripper_step": self.max_gripper_step,
            "max_position_second_difference_m": self.max_position_second_difference_m,
            "max_gripper_second_difference": self.max_gripper_second_difference,
        }
        for name, value in optional_limits.items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be None or finite and positive")


@dataclass(frozen=True)
class InferenceRequest:
    snapshot: ObservationSnapshot
    rtc_mode: str
    prefix_rows: tuple[dict[str, list[float]], ...] = ()
    predicted_frozen_steps: int = 0
    overlap_steps: int = 0


@dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    rows: tuple[dict[str, list[float]], ...]
    inference_latency_ms: float
    image_decode_latency_ms: float
    policy_info: Mapping[str, Any]


@dataclass
class SessionStats:
    missing_rgb_steps: int = 0
    stale_rgb_steps: int = 0
    ignored_depth_fields: int = 0
    inference_failures: int = 0
    rejected_tails: int = 0
    rejected_delay_underestimates: int = 0
    rejected_prefix_mismatches: int = 0
    missing_prefixes: int = 0
    rejected_dynamics: int = 0
    expired_rows: int = 0
    stale_pending_snapshots: int = 0
    null_reasons: dict[str, int] = field(default_factory=dict)
    gripper_min: float | None = None
    gripper_max: float | None = None

    def record_null(self, reason: str) -> None:
        self.null_reasons[reason] = self.null_reasons.get(reason, 0) + 1

    def record_fields(self, rows: tuple[dict[str, list[float]], ...], fields: tuple[str, ...]) -> None:
        values = [float(value) for row in rows for field_name in fields for value in row[field_name]]
        if not values:
            return
        observed_min = min(values)
        observed_max = max(values)
        self.gripper_min = observed_min if self.gripper_min is None else min(self.gripper_min, observed_min)
        self.gripper_max = observed_max if self.gripper_max is None else max(self.gripper_max, observed_max)


@dataclass
class ActiveSession:
    session_id: str
    generation: int
    task_uuid: str
    task_text: str
    control_hz: float
    created_at_s: float
    last_activity_s: float
    timestep: int = -1
    monitoring_timestep: int = -1
    image_cache: dict[str, CachedImage] = field(default_factory=dict)
    timeline: dict[int, ExecutionSlot] = field(default_factory=dict)
    served_history: dict[int, dict[str, list[float]]] = field(default_factory=dict)
    observed_delays: list[int] = field(default_factory=list)
    inference_status: str = "idle"
    inference_latency_ms: float = 0.0
    image_decode_latency_ms: float = 0.0
    source_to_result_latency_ms: float = 0.0
    total_inferences: int = 0
    total_actions_served: int = 0
    latest_inference_error: str | None = None
    latest_source_timestep: int | None = None
    latest_image_ages: dict[str, int | None] = field(default_factory=dict)
    latest_null_reason: str = "startup"
    latest_rtc_mode: str = "off"
    latest_predicted_delay_steps: int = 0
    latest_actual_delay_steps: int = 0
    latest_overlap_steps: int = 0
    latest_available_prefix_steps: int = 0
    latest_new_tail_steps: int = 0
    latest_prefix_position_error: float = 0.0
    latest_prefix_orientation_error_rad: float = 0.0
    latest_prefix_gripper_error: float = 0.0
    latest_max_position_step_m: float = 0.0
    latest_max_orientation_step_rad: float = 0.0
    latest_max_gripper_step: float = 0.0
    latest_max_position_second_difference_m: float = 0.0
    latest_max_gripper_second_difference: float = 0.0
    latest_first_admitted_position_seam_m: float = 0.0
    latest_first_admitted_orientation_seam_rad: float = 0.0
    latest_first_admitted_gripper_seam: float = 0.0
    epoch_time_s: float | None = None
    last_emitted_at_s: float | None = None
    latest_emitted_provenance: dict[str, Any] = field(default_factory=dict)
    latest_emitted_dynamics: tuple[float, ...] = ()
    terminal_reason: str | None = None
    requires_reregistration: bool = False
    stats: SessionStats = field(default_factory=SessionStats)


class IndustrialNextAsyncServer:
    """Synchronous DirectServer handler with one asynchronous inference worker."""

    def __init__(
        self,
        *,
        policy: Policy,
        executor: Executor,
        config: IndustrialNextServingConfig,
        service_provenance: Mapping[str, Any],
        embodiment_tag: str,
        profile: ConfigDrivenIndustrialNextProfile,
        owns_executor: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if config.action_horizon != profile.action_horizon:
            raise ValueError("serving config action_horizon differs from the profile")
        if config.rtc_mode not in profile.supported_rtc_modes:
            raise ValueError(f"rtc_mode {config.rtc_mode!r} is not supported by the profile")
        self.clock = time.monotonic if clock is None else clock
        self.policy = policy
        self.executor = executor
        self.task_catalog = profile.task_catalog
        self.config = config
        self.service_provenance = dict(service_provenance)
        self.embodiment_tag = embodiment_tag
        self.profile = profile
        self.owns_executor = owns_executor
        self.server_instance_id = str(uuid.uuid4())

        self._generation = 0
        self._active_session: ActiveSession | None = None
        self._inference_future: asyncio.Future[InferenceResult] | None = None
        self._pending_snapshot: ObservationSnapshot | None = None
        self._idle_timer: asyncio.TimerHandle | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def get_metadata(self) -> Metadata:
        """Return the static direct-RPC and service metadata."""
        service_metadata = self._service_metadata()
        return {
            "server_info": {
                "server_name": "pi05_industrialnext_async_server",
                "service_type": "rl_inference",
            },
            "service_metadata": service_metadata,
            "request_format": {
                "register_session": {
                    "type": str,
                    "control_hz": float,
                    "task_uuid": str,
                    "task_text": str,
                },
                "step": {
                    "type": str,
                    "session_id": str,
                    "observation": dict,
                },
                "close_session": {
                    "type": str,
                    "session_id": str,
                },
            },
            "response_format": {
                "register_session": {
                    "session_id": str,
                    "metadata": dict,
                    "error": str,
                },
                "step": {
                    "action": object,
                    "timestep": int,
                    "queue_len": int,
                    "inference_status": str,
                    "inference_latency_ms": float,
                    "inference_batch_size": int,
                    "obs_buffer_len": int,
                    "session_uptime_s": float,
                    "total_inferences": int,
                    "total_actions_served": int,
                    "server_step_ms": float,
                    "monitoring": dict,
                    "monitoring_timestep": int,
                    "monitoring_gripper_values": dict,
                    "error": str,
                },
                "close_session": {
                    "ok": bool,
                    "error": str,
                },
            },
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one validated request without blocking on model inference."""
        if self._closed:
            return self._error_response(str(request.get("type")), "server_closed")
        self._bind_event_loop()
        request_type = request.get("type")
        try:
            if request_type == "register_session":
                return self._register_session(request)
            if request_type == "step":
                return self._step(request)
            if request_type == "close_session":
                return self._close_session(request)
            raise ValueError(f"unsupported request type {request_type!r}")
        except Exception as exc:
            logger.warning("Industrial Next request %r failed: %s", request_type, exc)
            return self._error_response(str(request_type), str(exc))

    async def shutdown(self) -> None:
        """Stop admission, invalidate sessions, drain inference, and close the worker."""
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._active_session = None
        self._pending_snapshot = None
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        future = self._inference_future
        try:
            if future is not None:
                await asyncio.shield(future)
        except Exception:
            logger.exception("In-flight PI0.5 inference failed during shutdown")
        finally:
            self._inference_future = None
            if self.owns_executor:
                self.executor.shutdown(wait=True, cancel_futures=True)

    def _bind_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._event_loop is None:
            self._event_loop = loop
        elif self._event_loop is not loop:
            raise RuntimeError("server handler was called from a different event loop")
        return loop

    def _service_metadata(self) -> dict[str, Any]:
        task_catalog = self.task_catalog.to_metadata()
        return {
            **self.service_provenance,
            "execution": {
                "mode": "fixed_offset",
                "action_offset": self.config.action_offset,
                "target_source_offset": self.profile.action_start_offset_steps,
                "output_hz": self.config.control_hz,
                "speed_factor": 1.0,
                "timestamp_clock": "server_monotonic",
                "tick_clock": self.config.control_clock_mode,
                "max_action_lateness_s": self.config.max_action_lateness_s,
                "max_control_clock_drift_s": self.config.max_control_clock_drift_s,
                "max_command_gap_s": self._command_gap_limit_s,
            },
            "default_ensemble": {
                "strategy": self.config.ensemble_strategy,
                "temporal_coeff": self.config.ensemble_coeff,
                "max_ensemble_chunks": self.config.max_ensemble_chunks,
                "chunk_transition_frames": self.config.chunk_transition_frames,
                "gripper_blend_mode": "continuous",
            },
            "async_serving": True,
            "async_protocol_version": ASYNC_PROTOCOL_VERSION,
            "async_capabilities": list(ASYNC_CAPABILITIES),
            **self.profile.service_metadata(),
            "effective_gripper_snap_config": {
                "enabled": False,
                "gripper_action_fields": list(self.profile.gripper_action_keys),
                "snap_range": None,
                "signal_range": None,
            },
            "task_conditioning": {
                "enabled": True,
                "mode": "text",
                "prompt_field_name": "task_text",
                "task_catalog_version": self.task_catalog.catalog_version,
                "task_catalog": task_catalog,
                "task_uuid_to_text": self.task_catalog.task_uuid_to_text,
            },
            "server_instance_id": self.server_instance_id,
            "embodiment_tag": self.embodiment_tag,
            "action_semantics": "absolute",
            "action_rotation_representation": "rot6d",
            "max_sessions": 1,
            "default_control_hz": self.config.control_hz,
            "max_image_staleness_steps": self.config.max_image_staleness_steps,
            "min_usable_action_steps": self.config.min_usable_action_steps,
            "idle_session_timeout_s": self.config.idle_session_timeout_s,
            "rtc": {
                "mode": self.config.rtc_mode,
                "supported_modes": ["off"],
                "action_limits": {
                    "max_position_step_m": self.config.max_position_step_m,
                    "max_orientation_step_rad": self.config.max_orientation_step_rad,
                    "max_gripper_step": self.config.max_gripper_step,
                    "max_position_second_difference_m": (self.config.max_position_second_difference_m),
                    "max_gripper_second_difference": (self.config.max_gripper_second_difference),
                },
            },
        }

    def _register_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        control_hz = request.get("control_hz")
        if isinstance(control_hz, bool) or not isinstance(control_hz, int | float):
            raise ValueError("control_hz must be numeric")
        control_hz = float(control_hz)
        if not math.isfinite(control_hz) or control_hz != self.config.control_hz:
            raise ValueError(f"control_hz must be exactly {self.config.control_hz}")
        task_uuid = request.get("task_uuid")
        task_text = request.get("task_text")
        if not isinstance(task_uuid, str) or not isinstance(task_text, str):
            raise ValueError("task_uuid and task_text must be strings")
        pinned_task_text = self.task_catalog.resolve(task_uuid, task_text)

        previous_session_id = None if self._active_session is None else self._active_session.session_id
        self._generation += 1
        now_s = self.clock()
        session = ActiveSession(
            session_id=str(uuid.uuid4()),
            generation=self._generation,
            task_uuid=task_uuid,
            task_text=pinned_task_text,
            control_hz=control_hz,
            created_at_s=now_s,
            last_activity_s=now_s,
        )
        self._active_session = session
        self._pending_snapshot = None
        self._schedule_idle_expiry(session)
        if previous_session_id is not None:
            logger.warning(
                "Replacing Industrial Next session old=%s new=%s",
                previous_session_id,
                session.session_id,
            )
        else:
            logger.info("Registered Industrial Next session %s", session.session_id)
        return {
            "session_id": session.session_id,
            "metadata": self._service_metadata(),
        }

    def _close_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id", ""))
        if self._active_session is not None and self._active_session.session_id == session_id:
            self._generation += 1
            self._active_session = None
            self._pending_snapshot = None
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            logger.info("Closed Industrial Next session %s", session_id)
        return {"ok": True}

    def _step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        step_started_at = time.perf_counter()
        session = self._session(str(request.get("session_id", "")))
        if session.requires_reregistration:
            return self._terminal_response(session)
        candidate_timestep = session.timestep + 1
        now_s = self.clock()
        if self.config.control_clock_mode == "elapsed_time" and session.epoch_time_s is not None:
            # Advance past missed control slots rather than stretching old predictions.
            elapsed_tick = int((now_s - session.epoch_time_s) * self.config.control_hz + 0.5)
            candidate_timestep = max(candidate_timestep, elapsed_tick)
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")
        admission = self.profile.admit_observation(
            observation,
            image_cache=session.image_cache,
            timestep=candidate_timestep,
            task_uuid=session.task_uuid,
            task_text=session.task_text,
            generation=session.generation,
            max_image_staleness_steps=self.config.max_image_staleness_steps,
            now_s=now_s,
        )

        if session.epoch_time_s is None:
            session.epoch_time_s = now_s
        drift_s = now_s - session.epoch_time_s - candidate_timestep / self.config.control_hz
        if abs(drift_s) > self.config.max_control_clock_drift_s:
            self._terminate(session, f"control_clock_drift: {drift_s:.6f}s")
            return self._terminal_response(session)
        if self._command_gap_expired(session, now_s):
            self._terminate(session, "command_gap_expired")
            return self._terminal_response(session)
        session.timestep = candidate_timestep
        session.monitoring_timestep += 1
        session.last_activity_s = self.clock()
        session.latest_image_ages = dict(admission.image_ages)
        session.stats.missing_rgb_steps += int(bool(admission.missing_images))
        session.stats.stale_rgb_steps += int(bool(admission.stale_images))
        session.stats.ignored_depth_fields += admission.ignored_depth_fields
        self._schedule_idle_expiry(session)

        self._expire_timeline(session, now_s, candidate_timestep)
        slot = session.timeline.pop(candidate_timestep, None)
        action = None
        session.latest_emitted_provenance = {}
        if slot is not None:
            try:
                action, provenance = self._resolve_slot(slot)
                previous = list(session.served_history.values())[-2:]
                dynamics = _trajectory_dynamics((*previous, action), self.profile)
                if self._dynamics_exceeded(dynamics):
                    raise ValueError(f"emitted_action_dynamics_limit: {dynamics}")
                seam = _prefix_errors((previous[-1],), (action,), 1, self.profile) if previous else (0.0, 0.0, 0.0)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                self._terminate(session, str(exc))
                return self._terminal_response(session)
            session.latest_emitted_dynamics = dynamics
            (
                session.latest_first_admitted_position_seam_m,
                session.latest_first_admitted_orientation_seam_rad,
                session.latest_first_admitted_gripper_seam,
            ) = seam
            session.latest_emitted_provenance = {"execution_tick": candidate_timestep, **provenance}
            session.last_emitted_at_s = now_s
            session.served_history[candidate_timestep] = action
            oldest_history = candidate_timestep - self.profile.action_horizon + 1
            session.served_history = {
                target: row for target, row in session.served_history.items() if target >= oldest_history
            }
            session.total_actions_served += 1
            session.latest_null_reason = ""

        if admission.snapshot is not None:
            if session.requires_reregistration:
                session.inference_status = "reregistration_required"
            elif self._inference_future is None:
                self._launch_inference(admission.snapshot)
            else:
                self._pending_snapshot = admission.snapshot
        elif self._inference_future is None and not session.timeline:
            session.inference_status = "waiting_for_images"

        if action is None:
            null_reason = self._null_reason(session, admission)
            session.latest_null_reason = null_reason
            session.stats.record_null(null_reason)

        if session.requires_reregistration:
            return self._terminal_response(session)
        server_step_ms = (time.perf_counter() - step_started_at) * 1000.0
        response = self._step_success_response(
            session=session,
            action=action,
            server_step_ms=server_step_ms,
        )
        self._maybe_log_stats(session, server_step_ms=server_step_ms)
        return response

    def _resolve_slot(self, slot: ExecutionSlot) -> tuple[dict[str, list[float]], dict]:
        return slot.resolve(
            strategy=self.config.ensemble_strategy,
            coefficient=self.config.ensemble_coeff,
            control_hz=self.config.control_hz,
            rotation_fields=self.profile.rotation_action_fields,
        )

    def _expire_timeline(self, session: ActiveSession, now_s: float, first_tick: int) -> None:
        for tick, slot in list(session.timeline.items()):
            live = slot.live(now_s)
            if tick < first_tick or not live.contributions:
                del session.timeline[tick]
                session.stats.expired_rows += 1
            else:
                session.timeline[tick] = live

    @property
    def _command_gap_limit_s(self) -> float:
        if self.config.max_command_gap_s is not None:
            return self.config.max_command_gap_s
        return 1 / self.config.control_hz + self.config.max_action_lateness_s

    def _command_gap_expired(self, session: ActiveSession, now_s: float) -> bool:
        return session.last_emitted_at_s is not None and now_s > session.last_emitted_at_s + self._command_gap_limit_s

    def _dynamics_exceeded(self, dynamics: tuple[float, ...]) -> bool:
        limits = (
            self.config.max_position_step_m,
            self.config.max_orientation_step_rad,
            self.config.max_gripper_step,
            self.config.max_position_second_difference_m,
            self.config.max_gripper_second_difference,
        )
        return any(limit is not None and value > limit for value, limit in zip(dynamics, limits, strict=True))

    def _terminate(self, session: ActiveSession, reason: str) -> None:
        if session.terminal_reason is None:
            session.terminal_reason = reason[:512]
            logger.error(
                "PI0.5 session terminated session=%s step=%d reason=%s queue=%d "
                "inference_ms=%.1f source_to_result_ms=%.1f command_gap_s=%s "
                "command_gap_limit_s=%.3f clock_mode=%s",
                session.session_id,
                session.timestep,
                session.terminal_reason,
                len(session.timeline),
                session.inference_latency_ms,
                session.source_to_result_latency_ms,
                None if session.last_emitted_at_s is None else self.clock() - session.last_emitted_at_s,
                self._command_gap_limit_s,
                self.config.control_clock_mode,
            )
        session.requires_reregistration = True
        session.inference_status = "session_unusable"
        session.timeline.clear()
        if self._pending_snapshot is not None and self._pending_snapshot.generation == session.generation:
            self._pending_snapshot = None

    def _terminal_response(self, session: ActiveSession) -> dict[str, Any]:
        return {
            **self._error_response("step", "session_unusable"),
            "reason": session.terminal_reason,
            "timestep": session.timestep,
        }

    def _session(self, session_id: str) -> ActiveSession:
        session = self._active_session
        if session is None or session.session_id != session_id:
            raise SessionNotFoundError("session_not_found")
        return session

    def _build_inference_request(self, session, snapshot):
        return InferenceRequest(snapshot=snapshot, rtc_mode="off")

    def _launch_inference(self, snapshot: ObservationSnapshot) -> None:
        session = self._active_session
        if (
            session is None
            or session.requires_reregistration
            or not snapshot_is_fresh(
                snapshot,
                current_timestep=session.timestep,
                active_generation=session.generation,
                max_staleness_steps=self.config.max_image_staleness_steps,
                now_s=self.clock(),
            )
        ):
            if session is not None:
                session.stats.stale_pending_snapshots += 1
            return
        if self._inference_future is not None:
            self._pending_snapshot = snapshot
            return
        request = self._build_inference_request(session, snapshot)
        if request is None:
            return
        loop = self._bind_event_loop()
        session.inference_status = "running"
        session.latest_inference_error = None
        session.latest_rtc_mode = request.rtc_mode
        session.latest_overlap_steps = request.overlap_steps
        session.latest_new_tail_steps = self.profile.action_horizon - request.overlap_steps
        future = loop.run_in_executor(self.executor, self._run_inference, request)
        self._inference_future = future
        future.add_done_callback(lambda completed: self._complete_inference(request, completed))

    def _run_inference(self, request: InferenceRequest) -> InferenceResult:
        started_at = time.perf_counter()
        model_observation = self.profile.build_model_observation(request.snapshot)
        decode_ms = (time.perf_counter() - started_at) * 1000.0
        output = self.policy.infer(model_observation)
        action, policy_info = output["actions"], output.get("policy_timing", {})
        rows = self.profile.map_action_chunk(action)
        return InferenceResult(
            request=request,
            rows=rows,
            inference_latency_ms=(time.perf_counter() - started_at) * 1000.0,
            image_decode_latency_ms=decode_ms,
            policy_info=policy_info,
        )

    def _complete_inference(
        self,
        request: InferenceRequest,
        future: asyncio.Future[InferenceResult],
    ) -> None:
        if future is not self._inference_future:
            return
        self._inference_future = None
        session = self._active_session
        try:
            result = future.result()
        except Exception as exc:
            if (
                session is not None
                and not session.requires_reregistration
                and session.generation == request.snapshot.generation
            ):
                session.inference_status = "error"
                session.latest_inference_error = f"{type(exc).__name__}: {exc}"
                session.stats.inference_failures += 1
                self._expire_timeline(session, self.clock(), session.timestep + 1)
                if not session.timeline:
                    self._terminate(session, session.latest_inference_error or session.inference_status)
                logger.error(
                    "PI0.5 inference failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        else:
            if (
                not self._closed
                and session is not None
                and not session.requires_reregistration
                and session.generation == result.request.snapshot.generation
            ):
                try:
                    self._expire_timeline(session, self.clock(), session.timestep + 1)
                    self._admit_inference_result(session, result)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    session.latest_inference_error = f"postprocessing_error: {exc}"
                    session.inference_status = "postprocessing_error"
                    if not session.timeline:
                        self._terminate(session, session.latest_inference_error)
        self._launch_pending_if_valid()

    def _admit_inference_result(self, session: ActiveSession, result: InferenceResult) -> None:
        if session.requires_reregistration:
            return
        if self._command_gap_expired(session, self.clock()):
            self._terminate(session, "command_gap_expired")
            return
        request = result.request
        target_start = request.snapshot.source_timestep + self.profile.action_start_offset_steps
        actual_delay = max(0, session.timestep - target_start + 1)
        session.latest_actual_delay_steps = actual_delay
        session.observed_delays.append(actual_delay)
        session.observed_delays = session.observed_delays[-20:]
        session.inference_latency_ms = result.inference_latency_ms
        session.image_decode_latency_ms = result.image_decode_latency_ms
        session.source_to_result_latency_ms = (self.clock() - request.snapshot.received_at_s) * 1000
        session.total_inferences += 1
        session.latest_source_timestep = request.snapshot.source_timestep
        session.stats.record_fields(result.rows, self.profile.gripper_action_keys)

        dynamics = _trajectory_dynamics(result.rows, self.profile)
        (
            session.latest_max_position_step_m,
            session.latest_max_orientation_step_rad,
            session.latest_max_gripper_step,
            session.latest_max_position_second_difference_m,
            session.latest_max_gripper_second_difference,
        ) = dynamics
        limits = (
            self.config.max_position_step_m,
            self.config.max_orientation_step_rad,
            self.config.max_gripper_step,
            self.config.max_position_second_difference_m,
            self.config.max_gripper_second_difference,
        )
        violated = [
            (value, limit) for value, limit in zip(dynamics, limits, strict=True) if limit is not None and value > limit
        ]
        if violated:
            session.stats.rejected_dynamics += 1
            session.latest_inference_error = f"action_dynamics_limit: values={dynamics}, limits={limits}"
            session.inference_status = "action_dynamics_limit"
            if not session.timeline:
                self._terminate(session, session.latest_inference_error or session.inference_status)
            return

        now_s = self.clock()
        self._expire_timeline(session, now_s, session.timestep + 1)
        future_rows = {}
        for index in range(self.config.action_offset, len(result.rows)):
            target = execution_tick(
                request.snapshot.source_timestep,
                self.profile.action_start_offset_steps,
                self.config.action_offset,
                index,
            )
            deadline = (
                request.snapshot.received_at_s
                + (target - request.snapshot.source_timestep) / self.config.control_hz
                + self.config.max_action_lateness_s
            )
            if target > session.timestep and now_s <= deadline:
                future_rows[target] = Contribution(
                    freeze_action(result.rows[index]),
                    request.snapshot.source_timestep,
                    index,
                    request.snapshot.received_at_s,
                    deadline,
                )
        if len(future_rows) < self.config.min_usable_action_steps:
            session.stats.rejected_tails += 1
            session.latest_inference_error = (
                f"insufficient_usable_tail: {len(future_rows)} < {self.config.min_usable_action_steps}"
            )
            session.inference_status = "insufficient_tail"
            if not session.timeline:
                self._terminate(session, session.latest_inference_error)
            return
        # Resolve before mutation: an invalid rotation must not partly install a chunk.
        updated = dict(session.timeline)
        anchor_ticks = set(list(future_rows)[: self.config.chunk_transition_frames])
        if session.timeline:
            anchor_ticks.add(max(session.timeline))
        old_outputs = {
            tick: self._resolve_slot(session.timeline[tick]) for tick in anchor_ticks if tick in session.timeline
        }
        old_tail = max(session.timeline) if session.timeline else None
        append = old_outputs[old_tail] if old_tail is not None else None
        append_deadline = session.timeline[old_tail].anchor_deadline() if old_tail is not None else math.inf
        if append is None and session.served_history:
            append = (
                list(session.served_history.values())[-1],
                {"contributions": [], "transition": None},
            )
        for frame_index, (target, contribution) in enumerate(future_rows.items()):
            previous_slot = session.timeline.get(target)
            contributions = () if previous_slot is None else previous_slot.contributions
            contributions = (
                *tuple(c for c in contributions if c.source_tick != contribution.source_tick),
                contribution,
            )
            contributions = tuple(sorted(contributions, key=lambda c: (c.received_at_s, c.source_tick)))[
                -self.config.max_ensemble_chunks :
            ]
            if self.config.ensemble_strategy == "latest_only":
                contributions = contributions[-1:]
            slot = ExecutionSlot(contributions)
            prior = old_outputs.get(target, append)
            frames = self.config.chunk_transition_frames
            if frame_index < frames and prior is not None:
                new_output, _ = self._resolve_slot(slot)
                old_action, provenance = prior
                deadline = previous_slot.anchor_deadline() if previous_slot is not None else append_deadline
                if now_s <= deadline and any(
                    not np.allclose(old_action[key], new_output[key], atol=1e-6) for key in new_output
                ):
                    u = (frame_index + 1) / frames
                    sources = list(provenance["contributions"])
                    if provenance["transition"] is not None:
                        sources.extend(provenance["transition"]["sources"])
                    sources = list({(v["source_tick"], v["model_row"]): v for v in sources}.values())
                    slot = ExecutionSlot(
                        contributions,
                        TransitionAnchor(
                            freeze_action(old_action),
                            deadline,
                            3 * u * u - 2 * u * u * u,
                            tuple(sources),
                        ),
                    )
            self._resolve_slot(slot)
            updated[target] = slot
        session.timeline = updated
        session.latest_inference_error = None
        session.inference_status = "ready"

    def _launch_pending_if_valid(self) -> None:
        pending = self._pending_snapshot
        self._pending_snapshot = None
        if self._closed or pending is None:
            return
        session = self._active_session
        if session is None or not snapshot_is_fresh(
            pending,
            current_timestep=session.timestep,
            active_generation=session.generation,
            max_staleness_steps=self.config.max_image_staleness_steps,
            now_s=self.clock(),
        ):
            if session is not None:
                session.stats.stale_pending_snapshots += 1
            return
        self._launch_inference(pending)

    def _schedule_idle_expiry(self, session: ActiveSession) -> None:
        loop = self._bind_event_loop()
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = loop.call_later(
            self.config.idle_session_timeout_s,
            self._expire_session_if_idle,
            session.session_id,
            session.generation,
        )

    def _expire_session_if_idle(self, session_id: str, generation: int) -> None:
        self._idle_timer = None
        session = self._active_session
        if session is None or session.session_id != session_id or session.generation != generation:
            return
        idle_s = self.clock() - session.last_activity_s
        remaining_s = self.config.idle_session_timeout_s - idle_s
        if remaining_s > 0:
            loop = self._event_loop
            if loop is not None:
                self._idle_timer = loop.call_later(
                    remaining_s,
                    self._expire_session_if_idle,
                    session_id,
                    generation,
                )
            return
        logger.warning("Expiring idle Industrial Next session %s", session_id)
        self._generation += 1
        self._active_session = None
        self._pending_snapshot = None

    def _null_reason(self, session: ActiveSession, admission: ObservationAdmission) -> str:
        if admission.missing_images:
            return "missing_images"
        if admission.stale_images:
            return "stale_images"
        if session.requires_reregistration:
            return "reregistration_required"
        if session.inference_status == "error":
            return "inference_error"
        if session.inference_status == "insufficient_tail":
            return "insufficient_tail"
        if session.inference_status in {
            "delay_underestimate",
            "prefix_mismatch",
            "missing_prefix",
            "prefix_out_of_range",
            "action_dynamics_limit",
        }:
            return session.inference_status
        if self._inference_future is not None:
            return "inference_running"
        if session.total_inferences == 0:
            return "startup"
        return "timeline_gap"

    def _step_success_response(
        self,
        *,
        session: ActiveSession,
        action: dict[str, list[float]] | None,
        server_step_ms: float,
    ) -> dict[str, Any]:
        now_s = self.clock()
        sources = session.latest_emitted_provenance.get("contributions", [])
        action_age = max((session.timestep - source["source_tick"] for source in sources), default=None)
        monitoring = {
            "command_gap_s": None if session.last_emitted_at_s is None else now_s - session.last_emitted_at_s,
            "control_clock_drift_s": None
            if session.epoch_time_s is None
            else now_s - session.epoch_time_s - session.timestep / self.config.control_hz,
            "source_to_result_latency_ms": session.source_to_result_latency_ms,
            "action_source_age_s": [now_s - source["received_at_s"] for source in sources],
            "emitted_action": session.latest_emitted_provenance,
            "emitted_dynamics": session.latest_emitted_dynamics,
            "raw_chunk_dynamics": [
                session.latest_max_position_step_m,
                session.latest_max_orientation_step_rad,
                session.latest_max_gripper_step,
                session.latest_max_position_second_difference_m,
                session.latest_max_gripper_second_difference,
            ],
            "image_age_s": {name: now_s - cached.received_at_s for name, cached in session.image_cache.items()},
            "progress": 0.0,
            "classification": "unknown",
            "scene_valid": True,
            "confidence": 0.0,
            "latest_inference_error": session.latest_inference_error,
            "action_source_age_steps": action_age,
            "usable_tail_count": len(session.timeline),
            "image_age_steps": dict(session.latest_image_ages),
            "image_decode_latency_ms": session.image_decode_latency_ms,
            "null_reason": session.latest_null_reason,
            "rtc_mode": session.latest_rtc_mode,
            "rtc_predicted_delay_steps": session.latest_predicted_delay_steps,
            "rtc_actual_delay_steps": session.latest_actual_delay_steps,
            "rtc_overlap_steps": session.latest_overlap_steps,
            "rtc_available_prefix_steps": session.latest_available_prefix_steps,
            "rtc_new_tail_steps": session.latest_new_tail_steps,
            "rtc_prefix_position_error": session.latest_prefix_position_error,
            "rtc_prefix_orientation_error_rad": session.latest_prefix_orientation_error_rad,
            "rtc_prefix_gripper_error": session.latest_prefix_gripper_error,
            "rtc_delay_window": list(session.observed_delays),
            "max_position_step_m": session.latest_max_position_step_m,
            "max_orientation_step_rad": session.latest_max_orientation_step_rad,
            "max_gripper_step": session.latest_max_gripper_step,
            "max_position_second_difference_m": (session.latest_max_position_second_difference_m),
            "max_gripper_second_difference": session.latest_max_gripper_second_difference,
            "first_admitted_position_seam_m": (session.latest_first_admitted_position_seam_m),
            "first_admitted_orientation_seam_rad": (session.latest_first_admitted_orientation_seam_rad),
            "first_admitted_gripper_seam": session.latest_first_admitted_gripper_seam,
            "requires_reregistration": session.requires_reregistration,
        }
        monitoring_grippers = self.profile.monitoring_gripper_values(action)
        return {
            "action": action,
            "timestep": session.timestep,
            "queue_len": len(session.timeline),
            "inference_status": session.inference_status,
            "inference_latency_ms": float(session.inference_latency_ms),
            "inference_batch_size": 1,
            "obs_buffer_len": 1,
            "session_uptime_s": float(now_s - session.created_at_s),
            "total_inferences": session.total_inferences,
            "total_actions_served": session.total_actions_served,
            "server_step_ms": float(server_step_ms),
            "monitoring": monitoring,
            "monitoring_timestep": session.monitoring_timestep,
            "monitoring_gripper_values": monitoring_grippers,
        }

    def _error_response(self, request_type: str, error: str) -> dict[str, Any]:
        if request_type == "register_session":
            return {"session_id": "", "metadata": {}, "error": error}
        if request_type == "close_session":
            return {"ok": False, "error": error}
        return {
            "action": None,
            "timestep": -1,
            "queue_len": 0,
            "inference_status": "error",
            "inference_latency_ms": 0.0,
            "inference_batch_size": 0,
            "obs_buffer_len": 0,
            "session_uptime_s": 0.0,
            "total_inferences": 0,
            "total_actions_served": 0,
            "server_step_ms": 0.0,
            "monitoring": {},
            "monitoring_timestep": -1,
            "monitoring_gripper_values": {},
            "error": error,
        }

    def _maybe_log_stats(self, session: ActiveSession, *, server_step_ms: float) -> None:
        interval = self.config.stats_log_interval_steps
        if interval <= 0 or (session.monitoring_timestep + 1) % interval != 0:
            return
        logger.info(
            "PI0.5 async stats session=%s step=%d server_step_ms=%.3f "
            "inference_ms=%.1f decode_ms=%.1f queue=%d missing_rgb=%d stale_rgb=%d "
            "ignored_depth=%d inference_failures=%d expired_rows=%d rejected_tails=%d "
            "stale_pending=%d rtc_mode=%s delay=(%d,%d) overlap=%d available_prefix=%d "
            "new_tail=%d rejected_delay=%d rejected_prefix=%d rejected_dynamics=%d "
            "missing_prefix=%d "
            "null_reasons=%s gripper_range=(%s,%s)",
            session.session_id,
            session.timestep,
            server_step_ms,
            session.inference_latency_ms,
            session.image_decode_latency_ms,
            len(session.timeline),
            session.stats.missing_rgb_steps,
            session.stats.stale_rgb_steps,
            session.stats.ignored_depth_fields,
            session.stats.inference_failures,
            session.stats.expired_rows,
            session.stats.rejected_tails,
            session.stats.stale_pending_snapshots,
            session.latest_rtc_mode,
            session.latest_predicted_delay_steps,
            session.latest_actual_delay_steps,
            session.latest_overlap_steps,
            session.latest_available_prefix_steps,
            session.latest_new_tail_steps,
            session.stats.rejected_delay_underestimates,
            session.stats.rejected_prefix_mismatches,
            session.stats.rejected_dynamics,
            session.stats.missing_prefixes,
            session.stats.null_reasons,
            session.stats.gripper_min,
            session.stats.gripper_max,
        )


class SessionNotFoundError(ValueError):
    """Exact sentinel error required by the ROS reconnect path."""
