"""Verify immutable GR00T artifacts and build an openpi-owned eligible-window index."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import filelock
import numpy as np
import tyro
import yaml

from openpi.shared import taro_contract as contract
from openpi.training.taro_dataset import eligible_starts
from openpi.training.taro_dataset import read_episode


def verify_record(record: dict) -> dict:
    path = Path(record["path"])
    stat = path.stat()
    if stat.st_size != record["size_bytes"] or contract.sha256(path) != record["sha256"]:
        raise ValueError(f"Frozen artifact changed: {path}")
    return {"path": str(path), "stat": [stat.st_size, stat.st_mtime_ns]}


def audit(profile: dict) -> tuple[dict, list[dict]]:
    data = profile["data"]
    root = Path(data["converted_root"])
    marker = root / "_frozen_corpus_manifest.json"
    if contract.sha256(marker) != data["frozen_sha256"]:
        raise ValueError("Frozen marker differs from the accepted experiment")
    frozen = json.loads(marker.read_text())
    historical = subprocess.check_output(
        ["git", "-C", data["historical_repository"], "show", f"{data['historical_commit']}:{data['historical_config']}"]
    )
    import hashlib

    if hashlib.sha256(historical).hexdigest() != frozen["pipeline_config"]["sha256"]:
        raise ValueError("Historical YAML does not match frozen provenance")
    before = yaml.safe_load(historical)
    after = yaml.safe_load(Path(frozen["pipeline_config"]["path"]).read_text())
    before.pop("serving")
    after.pop("serving")
    if before != after:
        raise ValueError("GR00T non-serving configuration changed")
    source = Path(data["source_root"])
    pointer = json.loads((source / "meta/merged_episode_manifest_v1.latest.json").read_text())
    release = source / "meta" / pointer["filename"]
    if pointer["sha256"] != data["release_sha256"] or contract.sha256(release) != data["release_sha256"]:
        raise ValueError("Release manifest changed")
    if contract.sha256(Path(data["split_path"])) != data["split_sha256"]:
        raise ValueError("Split identity changed")
    if contract.sha256(source / "meta/projection.yaml") != data["projection_sha256"]:
        raise ValueError("Projection identity changed")
    release_rows = [json.loads(line) for line in release.read_text().splitlines()]
    expected = {str((source / row["path"]).resolve()) for row in release_rows}
    actual = {str(p.resolve()) for p in (source / "data").glob("*/*/episode.h5")}
    if expected != actual or len(expected) != len(release_rows):
        raise ValueError("Released source membership changed")
    split = json.loads(Path(data["split_path"]).read_text())
    if split["right_hand_coordinate_names"] != profile["hand_coordinate_names"]:
        raise ValueError("Native hand order changed")
    records = [frozen["modality_module"], *frozen["converted_artifact_inventory"]]
    print(f"Verifying {len(records)} frozen artifacts for {profile['name']}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        guards = list(pool.map(verify_record, records))
    for record in frozen["source_stat_inventory"]:
        stat = Path(record["path"]).stat()
        if [stat.st_size, stat.st_mtime_ns] != [record["size_bytes"], record["mtime_ns"]]:
            raise ValueError(f"Source guard changed: {record['path']}")
        guards.append({"path": record["path"], "stat": [stat.st_size, stat.st_mtime_ns]})
    for path in (marker, release, source / "meta/merged_episode_manifest_v1.latest.json"):
        stat = path.stat()
        guards.append({"path": str(path), "stat": [stat.st_size, stat.st_mtime_ns]})
    return {"historical_yaml": historical.decode(), "frozen_sha256": data["frozen_sha256"]}, guards


def video_count(path: str) -> int:
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path,
        ]
    )
    return int(result.strip())


def main(config_name: str, *, check_video_counts: bool = True):
    profile = contract.load_profile(config_name)
    output = contract.prepared_root(profile)
    output.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(output / ".prepare.lock")):
        sources = (
            [profile] if profile["name"] == "taro_exp_full" else [profile, contract.load_profile("pi05_taro_exp_full")]
        )
        audits, guards = {}, []
        for source_profile in sources:
            receipt, new_guards = audit(source_profile)
            audits[source_profile["name"]] = receipt
            guards.extend(new_guards)
        episodes, indices = [], {"train": [], "val": []}
        counts = {}
        for source_profile in sources:
            root = Path(source_profile["data"]["converted_root"])
            for ledger_path in sorted((root / "_ledgers").glob("*.json")):
                ledger = json.loads(ledger_path.read_text())
                for relative, row in ledger["sources"].items():
                    if row["status"] != "complete":
                        raise ValueError("Frozen source is not completely converted")
                    for segment in row["segments"]:
                        split = segment["split"]
                        if source_profile["name"] != profile["name"] and split != "val":
                            continue
                        dataset = root / segment["dataset"]
                        info = json.loads((dataset / "meta/info.json").read_text())
                        layout = json.loads((dataset / "_layout.json").read_text())
                        if (
                            info["fps"] != 50
                            or not info["masked_supervision"]
                            or layout["image_shape"] != [256, 256, 3]
                        ):
                            raise ValueError("Incompatible Taro artifact layout")
                        if layout["state_slices"] != {"left_eef": [0, 9], "right_eef": [9, 18], "right_hand": [18, 38]}:
                            raise ValueError("State layout changed")
                        if layout["action_slices"] != {"right_eef": [0, 9], "right_hand": [9, 29]}:
                            raise ValueError("Action layout changed")
                        task_rows = [
                            json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()
                        ]
                        tasks = {x["task_index"]: x["task"] for x in task_rows}
                        task = tasks[segment["task_index"]]
                        if task != profile["task_catalog"][row["task_id"]]:
                            raise ValueError("Task identity changed")
                        number = segment["episode_index"]
                        fmt = {"episode_index": number, "episode_chunk": number // info["chunks_size"]}
                        parquet = dataset / info["data_path"].format(**fmt)
                        arrays = read_episode(str(parquet))
                        if len(arrays["action"]) != segment["length"]:
                            raise ValueError("Segment length mismatch")
                        starts = eligible_starts(arrays)
                        videos = {
                            cam: str(dataset / info["video_path"].format(**fmt, video_key=f"observation.images.{cam}"))
                            for cam in profile["cameras"]
                        }
                        if check_video_counts:
                            with ThreadPoolExecutor(max_workers=3) as pool:
                                if any(n != segment["length"] for n in pool.map(video_count, videos.values())):
                                    raise ValueError("Video frame count mismatch")
                        episode = {
                            "parquet": str(parquet),
                            "videos": videos,
                            "length": segment["length"],
                            "source": str(
                                Path(source_profile["data"]["source_root"]) / "data" / ledger_path.stem / relative
                            ),
                            "source_start": segment["source_start"],
                            "pool": ledger_path.stem,
                            "split": split,
                            "task": task,
                            "task_uuid": row["task_id"],
                            "dataset": str(dataset),
                            "episode_index": number,
                        }
                        indices[split].extend((len(episodes), int(t)) for t in starts)
                        episodes.append(episode)
                        key = f"{split}/{ledger_path.stem}"
                        counts[key] = counts.get(key, 0) + len(starts)
        hashes = {}
        for split, rows in indices.items():
            path = output / f"{split}.npz"
            temporary = output / f"{split}.tmp.npz"
            np.savez_compressed(temporary, index=np.asarray(rows, dtype=np.int32).reshape(-1, 2))
            temporary.replace(path)
            hashes[split] = contract.sha256(path)
        manifest = {
            "schema_version": 1,
            "contract_sha256": contract.identity(contract.training_contract(profile)),
            "profile_data_sha256": contract.identity(profile["data"]),
            "index_sha256": hashes,
            "audits": audits,
            "guards": list({g["path"]: g for g in guards}.values()),
            "episodes": episodes,
            "window_counts": counts,
            "video_counts_verified": check_video_counts,
        }
        contract.atomic_json(output / "manifest.json", manifest)
        print(json.dumps({"manifest": str(output / "manifest.json"), "windows": counts}, indent=2), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
