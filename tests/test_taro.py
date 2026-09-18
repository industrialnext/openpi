# ruff: noqa: SLF001
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.policies.taro_policy import TaroPolicy
from openpi.serving.industrialnext.async_server import IndustrialNextAsyncServer
from openpi.serving.industrialnext.async_server import IndustrialNextServingConfig
from openpi.serving.industrialnext.execution import execution_tick
from openpi.serving.industrialnext.profile_config import ConfigDrivenIndustrialNextProfile
from openpi.serving.industrialnext.profile_config import resolve_serving_config
from openpi.shared import taro_geometry as geometry
from openpi.training.taro_dataset import eligible_starts
from openpi.training.taro_stats import MaskedStats


def test_nonidentity_se3_roundtrip_and_wire_convention():
    rng = np.random.default_rng(1)
    state = rng.normal(size=38)
    state[3:9] = geometry.rot6d(Rotation.random(random_state=rng).as_matrix())
    state[12:18] = geometry.rot6d(Rotation.random(random_state=rng).as_matrix())
    actions = rng.normal(size=(40, 29))
    actions[:, 3:9] = geometry.rot6d(Rotation.random(40, random_state=rng).as_matrix())
    relative, mask = geometry.relative_actions(state, actions, np.ones_like(actions, bool))
    np.testing.assert_allclose(geometry.absolute_actions(state, relative), actions, atol=5e-7)
    wire = geometry.change_convention(actions[:, 3:9], to_columns=True)
    np.testing.assert_allclose(geometry.change_convention(wire, to_columns=False), actions[:, 3:9], atol=2e-7)
    np.testing.assert_array_equal(relative[:, 9:], actions[:, 9:].astype(np.float32))
    mask[0, 0] = False
    actions[0, 0] = np.nan
    actions[1, 10] = np.nan
    mask[1, 10] = False
    relative, mask = geometry.relative_actions(state, actions, mask)
    assert not mask[0, :9].any()
    assert not mask[1, 10]
    assert np.isfinite(relative).all()
    assert relative[1, 10] == 0


def test_snapshot_anchor_not_later_state():
    state = np.zeros(38)
    state[3:9] = state[12:18] = [1, 0, 0, 0, 1, 0]
    state[9] = 4

    class Policy:
        def infer(self, obs, noise=None):
            obs["state"][9] = 50
            actions = np.zeros((40, 29))
            actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
            return {"actions": actions}

    assert np.all(TaroPolicy(Policy()).infer({"state": state})["actions"][:, 0] == 4)


def test_index_excludes_boundaries_invalid_observations_and_unsupervised_windows():
    n = 90
    arrays = {
        "observation.state": np.zeros((n, 38)),
        "action": np.zeros((n, 29)),
        "observation.state_mask": np.ones((n, 38), bool),
        "action_mask": np.zeros((n, 29), bool),
        "observation.valid": np.ones(n, bool),
        "frame_index": np.arange(n),
        "timestamp": np.arange(n) / 50,
    }
    arrays["action_mask"][40, 9] = True
    arrays["observation.valid"][5] = False
    expected = np.array([i for i in range(1, 41) if i != 5])
    np.testing.assert_array_equal(eligible_starts(arrays), expected)
    arrays["frame_index"][-1] += 1
    with pytest.raises(ValueError, match="Non-contiguous"):
        eligible_starts(arrays)


def test_masked_stats_ignore_missing_native_hand():
    values = np.array([[1.0, 10.0], [3.0, float("nan")], [5.0, 30.0]])
    mask = np.array([[True, True], [True, False], [True, True]])
    stats = MaskedStats(2)
    stats.update(values, mask)
    stats.update(values, mask, histogram=True)
    np.testing.assert_allclose(stats.finish().mean, [3, 20])
    np.testing.assert_array_equal(stats.count, [3, 2])


def test_production_configuration_and_rtc_rejection():
    profile = ConfigDrivenIndustrialNextProfile("pi05_taro_exp_100")
    config = resolve_serving_config(profile)
    assert (config.action_offset, config.control_clock_mode, config.max_command_gap_s) == (2, "elapsed_time", 0.2)
    assert profile.service_metadata()["action_dim"] == 29
    assert execution_tick(10, 0, 2, 2) == 10
    with pytest.raises(ValueError, match="off only"):
        IndustrialNextServingConfig(rtc_mode="native")


class FakePolicy:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, obs):
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("unreleased fake")
        actions = np.zeros((40, 29))
        actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
        actions[:, 9:] = np.arange(20) / 20
        return {"actions": actions}


def wire_observation(profile, *, images=True):
    obs = {f: [0.0] * profile.field_lengths[f] for f in profile.state_fields}
    for f in profile.rotation_state_fields:
        obs[f] = [1, 0, 0, 0, 1, 0]
    if images:
        ok, image = cv2.imencode(".jpg", np.zeros((256, 256, 3), np.uint8))
        assert ok
        obs["images_meta"] = {}
        for wire in profile.wire_image_to_model:
            obs[wire] = image.tobytes()
            obs["images_meta"][wire] = {
                "format": "jpeg",
                "quality": 90,
                "dtype": "uint8",
                "channels": 3,
                "height": 256,
                "width": 256,
            }
    return obs


async def wait_for(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise TimeoutError("predicate")


def test_async_generation_deadlines_terminal_and_native_hand():
    async def scenario():
        profile = ConfigDrivenIndustrialNextProfile("pi05_taro_exp_100")
        policy = FakePolicy()
        now = [10.0]
        server = IndustrialNextAsyncServer(
            policy=policy,
            executor=ThreadPoolExecutor(1),
            config=resolve_serving_config(profile),
            service_provenance={},
            embodiment_tag="taro",
            profile=profile,
            clock=lambda: now[0],
        )
        task = profile.task_catalog.tasks[0]

        def register():
            return server.handle_request(
                {
                    "type": "register_session",
                    "control_hz": 50.0,
                    "task_uuid": task.task_uuid,
                    "task_text": task.task_text,
                }
            )["session_id"]

        def step(sid, *, images=True):
            return server.handle_request(
                {"type": "step", "session_id": sid, "observation": wire_observation(profile, images=images)}
            )

        try:
            sid = register()
            r = step(sid)
            assert r["action"] is None
            await wait_for(policy.started.is_set)
            sid2 = register()
            policy.release.set()
            await wait_for(lambda: server._inference_future is None)
            assert server._active_session.total_inferences == 0
            now[0] += 0.02
            assert step(sid2)["action"] is None
            await wait_for(lambda: server._active_session.total_inferences == 1)
            now[0] += 0.02
            r = step(sid2)
            assert len(r["action"]["right_hand"]) == 20
            assert set(r["action"]) == set(profile.action_fields)
            now[0] += 0.22
            r = step(sid2)
            assert r["error"] == "session_unusable"
            now[0] += 0.02
            assert step(sid2)["error"] == "session_unusable"
            assert server._active_session.terminal_reason == "command_gap_expired"
        finally:
            policy.release.set()
            await server.shutdown()

    asyncio.run(scenario())


def test_masked_flow_gradient_and_dense_compatibility():
    import jax
    import jax.numpy as jnp

    from openpi.models.action_loss import masked_flow_loss

    prediction = jnp.ones((2, 40, 32))
    target = jnp.zeros_like(prediction)
    mask = np.zeros((2, 40, 32), bool)
    mask[0, :, :9] = True
    mask[1, :, 9:29] = True

    def loss(p):
        return jnp.mean(masked_flow_loss(p, target, jnp.asarray(mask)))

    assert float(loss(prediction)) == pytest.approx(1.0)
    gradients = np.asarray(jax.grad(loss)(prediction))
    assert np.all(gradients[~mask] == 0)
    assert np.all(gradients[mask] > 0)
    changed = jnp.where(mask, prediction, 1e10)
    assert float(loss(changed)) == pytest.approx(1.0)
    np.testing.assert_array_equal(masked_flow_loss(prediction, target), jnp.mean((prediction - target) ** 2, axis=-1))


def test_six_recipes_share_clock_and_preserve_missed_rows():
    from openpi.serving.industrialnext.replay import replay

    class Policy:
        def infer(self, obs, noise=None):
            actions = np.zeros((40, 29))
            actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
            return {"actions": actions}

    profile = ConfigDrivenIndustrialNextProfile("pi05_taro_exp_100")
    observations = {i: wire_observation(profile) for i in range(100, 160)}
    report = asyncio.run(
        replay(profile, Policy(), observations, delay_s=0.08, noise=np.zeros((40, 32)), missed_indices=(110, 111))
    )
    assert len(report) == 6
    for item in report.values():
        assert item["config"]["control_clock_mode"] == "elapsed_time"
        assert item["config"]["max_command_gap_s"] == 0.2
        assert not {110, 111}.intersection(x["source_row"] for x in item["outputs"])
        assert any(x["response"].get("action") for x in item["outputs"])


def test_comparison_export_preserves_assets_without_optimizer(tmp_path):
    from openpi.training.taro_receipts import archive_comparison

    source = tmp_path / "5000"
    for item in ("params", "assets", "train_state"):
        (source / item).mkdir(parents=True)
        (source / item / "payload").write_bytes(item.encode())
    export = archive_comparison(tmp_path, 5000)
    assert sorted(p.name for p in export.iterdir()) == ["assets", "params"]
    assert (source / "params/payload").stat().st_ino == (export / "params/payload").stat().st_ino
    assert archive_comparison(tmp_path, 5000) == export


def test_checkpoint_assets_reject_other_variant_and_modified_stats(tmp_path):
    from openpi.shared import taro_contract
    from openpi.training.taro_receipts import verify_assets

    profile = taro_contract.load_profile("pi05_taro_exp_100")
    (tmp_path / "norm_stats.json").write_text("{}")
    (tmp_path / "paligemma_tokenizer.model").write_bytes(b"test tokenizer")
    receipt = {
        "contract": taro_contract.training_contract(profile),
        "norm_stats_sha256": taro_contract.sha256(tmp_path / "norm_stats.json"),
        "tokenizer_sha256": taro_contract.sha256(tmp_path / "paligemma_tokenizer.model"),
    }
    taro_contract.atomic_json(tmp_path / "taro_assets.json", receipt)
    assert verify_assets(tmp_path, profile) == receipt
    with pytest.raises(ValueError, match="contract mismatch"):
        verify_assets(tmp_path, taro_contract.load_profile("pi05_taro_exp_full"))
    (tmp_path / "norm_stats.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="asset changed"):
        verify_assets(tmp_path, profile)
