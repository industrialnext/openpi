"""Matched physical holdout metrics and six production-timeline replay recipes."""

import asyncio
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import tyro

from openpi.policies.policy_config import create_trained_policy
from openpi.serving.industrialnext.profile_config import ConfigDrivenIndustrialNextProfile
from openpi.serving.industrialnext.replay import physical_row
from openpi.serving.industrialnext.replay import replay
from openpi.serving.industrialnext.replay import wire_observation
from openpi.shared import taro_contract
from openpi.shared import taro_geometry
from openpi.training.config import get_config
from openpi.training.taro_dataset import TaroDataset
from openpi.training.taro_dataset import read_episode


def errors(prediction, target, mask):
    pose = mask[:, :9].all(axis=-1)
    translation = np.linalg.norm(prediction[pose, :3] - target[pose, :3], axis=-1)
    relative = taro_geometry.rotation_matrix(prediction[pose, 3:9]) @ np.swapaxes(
        taro_geometry.rotation_matrix(target[pose, 3:9]), -1, -2
    )
    orientation = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    hands = np.abs(prediction[:, 9:] - target[:, 9:])
    return {
        "translation_sum_m": float(translation.sum()),
        "orientation_sum_rad": float(orientation.sum()),
        "pose_count": int(pose.sum()),
        "hand_abs_sum": np.where(mask[:, 9:], hands, 0).sum(axis=0).tolist(),
        "hand_count": mask[:, 9:].sum(axis=0).tolist(),
    }


def aggregate(rows):
    poses = sum(r["pose_count"] for r in rows)
    hand_count = np.sum([r["hand_count"] for r in rows], axis=0)
    hand_sum = np.sum([r["hand_abs_sum"] for r in rows], axis=0)
    return {
        "windows": len(rows),
        "valid_pose_targets": poses,
        "translation_mae_m": sum(r["translation_sum_m"] for r in rows) / poses if poses else None,
        "orientation_mae_rad": sum(r["orientation_sum_rad"] for r in rows) / poses if poses else None,
        "hand_coordinate_mae": [float(s / n) if n else None for s, n in zip(hand_sum, hand_count, strict=True)],
        "hand_coordinate_counts": hand_count.tolist(),
    }


def main(
    config_name: str,
    checkpoint_dir: Path,
    output_dir: Path,
    starts_per_episode: int = 1,
    replay_frames: int = 100,
    seed: int = 42,
):
    if starts_per_episode < 1 or replay_frames < 1:
        raise ValueError("Positive subset sizes required")
    taro_contract.create_output_directory(output_dir)
    profile = ConfigDrivenIndustrialNextProfile(config_name)
    dataset = TaroDataset(config_name, split="val")
    started = time.perf_counter()
    policy = create_trained_policy(get_config(config_name), checkpoint_dir)
    noise = np.random.default_rng(seed).normal(size=(40, 32)).astype(np.float32)
    policy.infer({k: v for k, v in dataset[0].items() if k not in {"actions", "action_loss_mask"}}, noise=noise)
    cold_s = time.perf_counter() - started
    timings = []
    rows = []
    predictions = []
    targets = []
    masks = []
    by_pool = defaultdict(list)
    selected = []
    for eid in np.unique(dataset.index[:, 0]):
        indices = np.flatnonzero(dataset.index[:, 0] == eid)
        selected.extend(
            indices[np.linspace(0, len(indices) - 1, min(starts_per_episode, len(indices)), dtype=int)].tolist()
        )
    for index in selected:
        sample = dataset[index]
        eid, start = map(int, dataset.index[index])
        episode = dataset.episodes[eid]
        observation = {k: v for k, v in sample.items() if k not in {"actions", "action_loss_mask"}}
        begin = time.perf_counter()
        prediction = policy.infer(observation, noise=noise)["actions"]
        timings.append((time.perf_counter() - begin) * 1000)
        mask = sample["action_loss_mask"].copy()
        mask[:, :9] = mask[:, :9].all(axis=-1, keepdims=True)
        metrics = errors(prediction, sample["actions"], mask)
        row = {
            "episode_id": eid,
            "start": start,
            "target_rows": list(range(start, start + 40)),
            "pool": episode["pool"],
            "source": episode["source"],
            "metrics": metrics,
        }
        rows.append(row)
        by_pool[episode["pool"]].append(metrics)
        predictions.append(prediction)
        targets.append(sample["actions"])
        masks.append(mask)
    np.savez_compressed(
        output_dir / "physical_chunks.npz", prediction=predictions, target=targets, mask=masks, noise=noise
    )
    report = {
        "config_name": config_name,
        "checkpoint": str(checkpoint_dir.resolve()),
        "checkpoint_step": checkpoint_dir.name,
        "assets_sha256": taro_contract.sha256(checkpoint_dir / "assets" / config_name / "taro_assets.json"),
        "preparation_sha256": taro_contract.sha256(taro_contract.prepared_root(dataset.profile) / "manifest.json"),
        "seed": seed,
        "subset": "evenly_spaced_eligible_starts_per_episode",
        "starts_per_episode": starts_per_episode,
        "eligible_holdout_windows": len(dataset),
        "holdout_episodes": sum(e["split"] == "val" for e in dataset.episodes),
        "eligible_holdout_episodes": len(np.unique(dataset.index[:, 0])),
        "excluded_holdout_entries": [
            e for i, e in enumerate(dataset.episodes) if e["split"] == "val" and i not in set(dataset.index[:, 0])
        ],
        "cold_load_compile_s": cold_s,
        "synchronized_warm_ms": dict(
            zip(("p50", "p95", "p99"), np.percentile(timings, [50, 95, 99]).tolist(), strict=True)
        ),
        "overall": aggregate([r["metrics"] for r in rows]),
        "by_pool": {k: aggregate(v) for k, v in by_pool.items()},
        "windows": rows,
    }
    taro_contract.atomic_json(output_dir / "direct_metrics.json", report)
    # One explicitly identified episode, preserving all original row gaps and missed requests.
    eid = int(dataset.index[0, 0])
    indices = np.flatnonzero(dataset.index[:, 0] == eid)
    first = int(dataset.index[indices[0], 1])
    last = first + replay_frames - 1
    observations = {
        int(dataset.index[i, 1]): wire_observation(profile, dataset[int(i)])
        for i in indices
        if int(dataset.index[i, 1]) <= last
    }
    delay = max(float(np.percentile(timings, 95)) / 1000, 0.001)
    missed = tuple(i for i in range(first, last + 1) if (i - first) % 23 in (11, 12))
    results = asyncio.run(replay(profile, policy, observations, delay_s=delay, noise=noise, missed_indices=missed))
    episode = dataset.episodes[eid]
    arrays = read_episode(episode["parquet"])
    for result in results.values():
        emitted = []
        same = []
        lookahead = []
        for entry in result["outputs"]:
            response = entry["response"]
            action = response.get("action")
            if action is None:
                continue
            physical = physical_row(profile, action)
            emitted.append(physical)
            row = entry["source_row"]
            target_row = row + result["config"]["action_offset"]
            mask = arrays["action_mask"][row : row + 1].astype(bool)
            same.append(errors(physical[None], arrays["action"][row : row + 1], mask))
            mask = arrays["action_mask"][target_row : target_row + 1].astype(bool)
            lookahead.append(errors(physical[None], arrays["action"][target_row : target_row + 1], mask))
        result["same_time_metrics"] = aggregate(same) if same else None
        result["selected_lookahead_metrics"] = aggregate(lookahead) if lookahead else None
        result["emitted_command_count"] = len(emitted)
        if len(emitted) > 1:
            emitted = np.asarray(emitted)
            result["emitted_dynamics"] = {
                "max_translation_jump_m": float(np.linalg.norm(np.diff(emitted[:, :3], axis=0), axis=-1).max()),
                "max_native_hand_jump": float(np.abs(np.diff(emitted[:, 9:], axis=0)).max()),
            }
    taro_contract.atomic_json(
        output_dir / "replay.json",
        {
            "episode": episode,
            "source_first_row": first,
            "delay_trace": "fixed measured warm p95, identical across recipes",
            "production_server_replay": results,
        },
    )
    print(json.dumps({k: report[k] for k in ("overall", "synchronized_warm_ms")}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
