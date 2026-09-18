"""Read-only adapter for the frozen Taro LeRobot v2.1 Parquet/video surface."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import polars as pl

from openpi.shared import taro_contract as contract

POOLS = (
    "clean_with_ext_cam",
    "clean_without_ext_cam",
    "original_d405",
    "original_fisheye_with_ext_cam",
    "original_fisheye_without_ext_cam",
)

CAMERAS = {"head": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}


@lru_cache(maxsize=8)
def read_episode(path: str) -> dict[str, np.ndarray]:
    frame = pl.read_parquet(path)
    result = {}
    for key in ("observation.state", "action", "action_mask", "observation.state_mask"):
        result[key] = np.stack(frame[key].to_list())
    for key in ("observation.valid", "timestamp", "frame_index", "task_index"):
        result[key] = frame[key].to_numpy()
    return result


def eligible_starts(arrays: dict, horizon: int = 40) -> np.ndarray:
    state, action = arrays["observation.state"], arrays["action"]
    sm, am = arrays["observation.state_mask"], arrays["action_mask"]
    length = len(state)
    if (
        state.shape != (length, 38)
        or action.shape != (length, 29)
        or sm.shape != state.shape
        or am.shape != action.shape
    ):
        raise ValueError("Incompatible Taro dimensions")
    for value in (sm, am, arrays["observation.valid"]):
        if not np.isin(value, [0, 1]).all():
            raise ValueError("Non-binary Taro mask")
    sm, am = sm.astype(bool), am.astype(bool)
    if not np.isfinite(state[sm]).all() or not np.isfinite(action[am]).all():
        raise ValueError("Nonfinite valid Taro coordinate")
    if not np.array_equal(arrays["frame_index"], np.arange(length)):
        raise ValueError("Non-contiguous converted frame index")
    if not np.allclose(arrays["timestamp"], np.arange(length) / 50, atol=1e-4):
        raise ValueError("Converted timestamps differ from 50 Hz")
    starts = np.arange(max(0, length - horizon + 1))
    am = am.copy()
    am[:, :9] = am[:, :9].all(axis=-1, keepdims=True)
    supervised = am.any(axis=-1).astype(np.int64)
    cumulative = np.concatenate(([0], np.cumsum(supervised)))
    valid = arrays["observation.valid"].astype(bool) & sm.all(axis=-1)
    return starts[valid[starts] & ((cumulative[starts + horizon] - cumulative[starts]) > 0)]


def load_preparation(profile: dict, *, verify_guards: bool = True) -> dict:
    path = contract.prepared_root(profile) / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["contract_sha256"] != contract.identity(contract.training_contract(profile)):
        raise ValueError("Taro preparation contract changed")
    if manifest["profile_data_sha256"] != contract.identity(profile["data"]):
        raise ValueError("Taro data identity changed")
    for split in ("train", "val"):
        if contract.sha256(path.parent / f"{split}.npz") != manifest["index_sha256"][split]:
            raise ValueError("Taro window index changed")
    if verify_guards:
        for item in manifest["guards"]:
            stat = Path(item["path"]).stat()
            if [stat.st_size, stat.st_mtime_ns] != item["stat"]:
                raise ValueError(f"Prepared input changed: {item['path']}; rerun preparation verification")
    return manifest


class TaroDataset:
    """Index only complete eligible chunks; no LeRobot endpoint clamping."""

    def __init__(self, config_name: str, split: str = "train", *, decode_images: bool = True):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.profile = contract.load_profile(config_name)
        self.manifest = load_preparation(self.profile)
        self.episodes = self.manifest["episodes"]
        self.index = np.load(contract.prepared_root(self.profile) / f"{split}.npz")["index"]
        self.decode_images = decode_images

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        episode_index, start = map(int, self.index[index])
        episode = self.episodes[episode_index]
        arrays = read_episode(episode["parquet"])
        result = {
            "state": arrays["observation.state"][start].astype(np.float32),
            "actions": arrays["action"][start : start + 40].astype(np.float32),
            "action_loss_mask": arrays["action_mask"][start : start + 40].astype(bool),
            "prompt": episode["task"],
            "taro_pool_id": np.int32(POOLS.index(episode["pool"])),
        }
        if self.decode_images:
            # Use the pinned LeRobot video reader without metadata migrations or downloads.
            from lerobot.common.datasets.video_utils import decode_video_frames

            result["image"] = {}
            for camera, model_key in CAMERAS.items():
                image = decode_video_frames(episode["videos"][camera], [start / 50], 1e-4, "pyav")[0]
                pixels = np.rint(image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                if pixels.shape != (256, 256, 3):
                    raise ValueError("Unexpected Taro video dimensions")
                result["image"][model_key] = pixels
        return result
