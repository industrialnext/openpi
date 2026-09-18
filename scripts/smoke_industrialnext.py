"""Isolated real RPC/ROS transport smoke; never imports or publishes a ROS node."""

import json
from pathlib import Path
import sys
import time

from industrialnext_rpc.direct.client import DirectClient
import numpy as np
import tyro

from openpi.serving.industrialnext.profile_config import ConfigDrivenIndustrialNextProfile
from openpi.serving.industrialnext.replay import wire_observation
from openpi.shared.taro_contract import atomic_json
from openpi.shared.taro_contract import create_output_directory


def exercise(client, profile):
    client.connect()
    try:
        metadata = client.get_metadata()
        if hasattr(client, "get_service_metadata"):
            client.get_service_metadata()
        task = profile.task_catalog.tasks[0]
        registration = client.request(
            {"type": "register_session", "control_hz": 50.0, "task_uuid": task.task_uuid, "task_text": task.task_text}
        )
        sid = registration["session_id"]
        responses = []
        latencies = []
        # Missing images must return a temporary null without launching inference.
        observation = wire_observation(profile, profile.build_synthetic_model_observation(task.task_text))
        no_images = {k: v for k, v in observation.items() if k in profile.state_fields}
        null = client.request({"type": "step", "session_id": sid, "observation": no_images})
        if null.get("action") is not None or null.get("error"):
            raise ValueError("Missing-image null contract")
        start = time.monotonic()
        for i in range(250):
            time.sleep(max(0, start + (i + 1) / 50 - time.monotonic()))
            before = time.perf_counter()
            response = client.request({"type": "step", "session_id": sid, "observation": observation})
            latencies.append((time.perf_counter() - before) * 1000)
            responses.append(response)
            if response.get("error"):
                break
        actions = [r["action"] for r in responses if r.get("action") is not None]
        for action in actions:
            if set(action) != set(profile.action_fields) or len(action["right_hand"]) != 20:
                raise ValueError("Physical wire action contract mismatch")
        # Once a command was emitted, a deliberate command gap must latch terminal state.
        time.sleep(0.25)
        terminal = client.request({"type": "step", "session_id": sid, "observation": observation})
        repeat = client.request({"type": "step", "session_id": sid, "observation": observation})
        if actions and (terminal.get("error") != "session_unusable" or repeat.get("error") != "session_unusable"):
            raise ValueError("Terminal command gap was not latched")
        close = client.request({"type": "close_session", "session_id": sid})
        return {
            "metadata": json.loads(
                json.dumps(metadata, default=lambda value: value.__name__ if isinstance(value, type) else str(value))
            ),
            "initial_null": null,
            "responses": responses,
            "terminal": terminal,
            "terminal_repeat": repeat,
            "close": close,
            "actions": len(actions),
            "rtt_ms": dict(zip(("p50", "p95", "p99"), np.percentile(latencies, [50, 95, 99]).tolist(), strict=True)),
            "passed": bool(actions) and terminal.get("error") == "session_unusable",
        }
    finally:
        client.close()


def main(
    config_name: str,
    output_dir: Path,
    host: str = "127.0.0.1",
    port: int = 10014,
    ros_client_source_dir: Path | None = None,
):
    create_output_directory(output_dir)
    profile = ConfigDrivenIndustrialNextProfile(config_name)
    reports = {"direct_rpc": exercise(DirectClient(host, port), profile)}
    if ros_client_source_dir is not None:
        sys.path.insert(0, str(ros_client_source_dir))
        from industrialnext_operator_policy_client.rpc_client import RobustDirectClient

        reports["ros_transport"] = exercise(
            RobustDirectClient(host, port, open_timeout_s=5, request_timeout_s=5), profile
        )
    atomic_json(output_dir / "report.json", reports)
    print(
        json.dumps(
            {k: {"passed": v["passed"], "actions": v["actions"], "rtt_ms": v["rtt_ms"]} for k, v in reports.items()},
            indent=2,
        )
    )
    if not all(r["passed"] for r in reports.values()):
        raise RuntimeError("Transport ran, but sustained real-model action coverage failed; see report")


if __name__ == "__main__":
    tyro.cli(main)
