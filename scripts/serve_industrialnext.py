"""Serve a validated, self-contained Taro PI0.5 checkpoint over Industrial Next RPC."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import logging
from pathlib import Path
import signal

from industrialnext_rpc.direct.server import DirectServer
import tyro

from openpi.policies.policy_config import create_trained_policy
from openpi.serving.industrialnext.async_server import IndustrialNextAsyncServer
from openpi.serving.industrialnext.profile_config import ConfigDrivenIndustrialNextProfile
from openpi.serving.industrialnext.profile_config import resolve_serving_config
from openpi.shared.taro_contract import sha256
from openpi.training.config import get_config


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    config_name: str
    checkpoint_dir: Path
    host: str | None = None
    port: int | None = None
    action_offset: int | None = None
    ensemble_strategy: str | None = None
    ensemble_coeff: float | None = None
    max_ensemble_chunks: int | None = None
    chunk_transition_frames: int | None = None
    control_clock_mode: str | None = None
    max_action_lateness_s: float | None = None
    max_control_clock_drift_s: float | None = None
    max_command_gap_s: float | None = None
    min_usable_action_steps: int | None = None
    rtc_mode: str | None = None
    num_steps: int = 10


async def serve(config: ServerConfig):
    config_name, checkpoint_dir, host, port = config.config_name, config.checkpoint_dir, config.host, config.port
    if config.num_steps < 1:
        raise ValueError("num_steps must be positive")
    profile = ConfigDrivenIndustrialNextProfile(config_name)
    runtime = resolve_serving_config(
        profile,
        **{
            k: v
            for k, v in dataclasses.asdict(config).items()
            if k not in {"config_name", "checkpoint_dir", "host", "port", "num_steps"}
        },
    )
    policy = create_trained_policy(
        get_config(config_name), checkpoint_dir, sample_kwargs={"num_steps": config.num_steps}
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-inference")
    handler = None
    try:
        observation = profile.build_synthetic_model_observation(profile.task_catalog.tasks[0].task_text)
        output = await asyncio.get_running_loop().run_in_executor(executor, policy.infer, observation)
        profile.map_action_chunk(output["actions"])
        receipt = checkpoint_dir / "assets" / config_name / "taro_assets.json"
        handler = IndustrialNextAsyncServer(
            policy=policy,
            executor=executor,
            config=runtime,
            service_provenance={
                "checkpoint": str(checkpoint_dir.resolve()),
                "assets_sha256": sha256(receipt),
                "flow_sampling_steps": config.num_steps,
            },
            embodiment_tag="taro",
            profile=profile,
        )
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        async with DirectServer(host or profile.host, port or profile.port, handler, max_size_bytes=16 * 1024 * 1024):
            logging.info("Ready: ws://%s:%d", host or profile.host, port or profile.port)
            await stop.wait()
    finally:
        if handler is not None:
            await handler.shutdown()
        else:
            executor.shutdown(wait=True, cancel_futures=True)


def main(config: ServerConfig):
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(serve(config))


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
