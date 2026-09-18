"""Versioned Taro profile and content identities shared by training and serving."""

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_profile(config_name: str) -> dict:
    if config_name not in {"pi05_taro_exp_100", "pi05_taro_exp_full"}:
        raise ValueError(f"Unknown Taro config: {config_name}")
    path = REPOSITORY_ROOT / "configs/industrialnext" / f"{config_name.removeprefix('pi05_')}.json"
    profile = json.loads(path.read_text())
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    expected = {
        "schema_version": 1,
        "state_dim": 38,
        "action_dim": 29,
        "model_action_dim": 32,
        "action_horizon": 40,
        "target_offset": 0,
        "fps": 50,
        "representation_version": "taro_relative_se3_rows_20d_v1",
    }
    for key, value in expected.items():
        if profile[key] != value:
            raise ValueError(f"Unsupported Taro {key}: {profile[key]}")
    if len(profile["hand_coordinate_names"]) != 20 or len(set(profile["hand_coordinate_names"])) != 20:
        raise ValueError("Taro requires 20 unique hand coordinate names")
    if profile["cameras"] != {"head": "view0_rgb", "left_wrist": "view1_rgb", "right_wrist": "view2_rgb"}:
        raise ValueError("Taro requires exactly the three admitted RGB cameras")


def training_contract(profile: dict) -> dict:
    """Runtime timing/address overrides do not change the learned data contract."""
    return {key: value for key, value in profile.items() if key not in {"data", "serving"}} | {
        "wire": {
            key: profile["serving"][key]
            for key in ("cameras", "field_lengths", "field_units", "eef_frame", "image_size")
        }
    }


def prepared_root(profile: dict) -> Path:
    return REPOSITORY_ROOT / profile["data"]["prepared_root"]


def create_output_directory(path: Path) -> None:
    resolved = path.resolve()
    roots = [(REPOSITORY_ROOT / name).resolve() for name in ("data", "checkpoints")]
    if not any(resolved.is_relative_to(root) and resolved != root for root in roots):
        raise ValueError("Evaluation output must be a new directory under ignored data/ or checkpoints/")
    path.mkdir(parents=True, exist_ok=False)
