"""Shared encoded observations and deterministic production-server replay."""

import asyncio
from concurrent.futures import Executor
from concurrent.futures import Future
import dataclasses
import time

import cv2
import numpy as np

from openpi.shared import taro_geometry

from .async_server import IndustrialNextAsyncServer
from .profile_config import resolve_serving_config

RECIPES = {
    "control": {"action_offset": 0, "ensemble_strategy": "latest_only", "chunk_transition_frames": 0},
    "offset_only": {"action_offset": 2, "ensemble_strategy": "latest_only", "chunk_transition_frames": 0},
    "ensemble_only": {"action_offset": 0, "ensemble_strategy": "temporal_exponential", "chunk_transition_frames": 0},
    "transition_only": {"action_offset": 0, "ensemble_strategy": "latest_only", "chunk_transition_frames": 4},
    "postprocessing_k0": {
        "action_offset": 0,
        "ensemble_strategy": "temporal_exponential",
        "chunk_transition_frames": 4,
    },
    "production": {},
}


def wire_observation(profile, observation):
    state = np.asarray(observation["state"])
    result = {}
    cursor = 0
    for field in profile.state_fields:
        width = profile.field_lengths[field]
        value = state[cursor : cursor + width]
        cursor += width
        if field in profile.rotation_state_fields:
            value = taro_geometry.change_convention(value, to_columns=True)
        result[field] = value.astype(float).tolist()
    result["images_meta"] = {}
    for wire, model in profile.wire_image_to_model.items():
        rgb = observation["image"][model]
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("JPEG encoding failed")
        result[wire] = encoded.tobytes()
        result["images_meta"][wire] = {
            "format": "jpeg",
            "quality": 90,
            "dtype": "uint8",
            "channels": 3,
            "height": rgb.shape[0],
            "width": rgb.shape[1],
        }
    return result


def physical_row(profile, wire):
    parts = []
    for field in profile.action_fields:
        value = np.asarray(wire[field])
        if field in profile.rotation_action_fields:
            value = taro_geometry.change_convention(value, to_columns=False)
        parts.append(value)
    return np.concatenate(parts)


class ReplayExecutor(Executor):
    """Complete one inference at measured virtual delay; cache matched real-model chunks."""

    def __init__(self, clock, cache, policy, profile, noise, delay_s):
        self.clock, self.cache, self.policy, self.profile, self.noise, self.delay_s = (
            clock,
            cache,
            policy,
            profile,
            noise,
            delay_s,
        )
        self.pending = None

    def submit(self, function, request):
        from .async_server import InferenceResult

        if self.pending is not None:
            raise RuntimeError("More than one worker")
        tick = request.snapshot.source_timestep
        if tick not in self.cache:
            started = time.perf_counter()
            observation = self.profile.build_model_observation(request.snapshot)
            output = self.policy.infer(observation, noise=self.noise)
            rows = self.profile.map_action_chunk(output["actions"])
            self.cache[tick] = (rows, (time.perf_counter() - started) * 1000)
        rows, measured_ms = self.cache[tick]
        result = InferenceResult(
            request=request, rows=rows, inference_latency_ms=measured_ms, image_decode_latency_ms=0, policy_info={}
        )
        future = Future()
        self.pending = (self.clock() + self.delay_s, future, result)
        return future

    def complete(self):
        if self.pending is not None and self.clock() >= self.pending[0]:
            _, future, result = self.pending
            self.pending = None
            future.set_result(result)

    def shutdown(self, wait=True, *, cancel_futures=False):  # noqa: FBT002 - Executor API
        if self.pending is not None:
            self.pending[1].cancel()
            self.pending = None


async def replay(profile, policy, observations, *, delay_s, noise, missed_indices=()):
    """observations maps original source row to encoded request; gaps remain gaps."""
    cache = {}
    reports = {}
    first = min(observations)
    last = max(observations)
    production = dataclasses.asdict(resolve_serving_config(profile))
    for name, overrides in RECIPES.items():
        now = [0.0]

        def clock(now=now):
            return now[0]

        config = resolve_serving_config(profile, **overrides)
        assert config.control_clock_mode == production["control_clock_mode"]
        assert config.max_command_gap_s == production["max_command_gap_s"]
        executor = ReplayExecutor(clock, cache, policy, profile, noise, delay_s)
        server = IndustrialNextAsyncServer(
            policy=policy,
            executor=executor,
            config=config,
            service_provenance={},
            embodiment_tag="taro",
            profile=profile,
            clock=clock,
        )
        task = profile.task_catalog.tasks[0]
        outputs = []
        session = server.handle_request(
            {"type": "register_session", "control_hz": 50.0, "task_uuid": task.task_uuid, "task_text": task.task_text}
        )["session_id"]
        try:
            for row in range(first, last + 1):
                now[0] = (row - first) / 50
                executor.complete()
                # Deliver concurrent-future result and asyncio done callbacks before the next request.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                if row not in observations or row in missed_indices:
                    continue
                response = server.handle_request(
                    {"type": "step", "session_id": session, "observation": observations[row]}
                )
                outputs.append({"source_row": row, "elapsed_s": now[0], "response": response})
                if response.get("error"):
                    break
            reports[name] = {
                "config": dataclasses.asdict(config),
                "delay_s": delay_s,
                "missed_source_rows": list(missed_indices),
                "outputs": outputs,
                "null_count": sum(x["response"].get("action") is None for x in outputs),
                "terminal_count": sum(x["response"].get("error") == "session_unusable" for x in outputs),
            }
        finally:
            if executor.pending is not None:
                now[0] = executor.pending[0]
                executor.complete()
            await server.shutdown()
    return reports
