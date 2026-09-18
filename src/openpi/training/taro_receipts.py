"""Bind training/resume and self-contained inference assets to the frozen contract."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

from openpi.shared import taro_contract as contract
from openpi.training.taro_dataset import load_preparation


def verify_assets(directory: Path, profile: dict) -> dict:
    receipt = json.loads((directory / "taro_assets.json").read_text())
    if receipt["contract"] != contract.training_contract(profile):
        raise ValueError("Checkpoint/assets Taro contract mismatch")
    for name, key in [("norm_stats.json", "norm_stats_sha256"), ("paligemma_tokenizer.model", "tokenizer_sha256")]:
        if contract.sha256(directory / name) != receipt[key]:
            raise ValueError(f"Taro asset changed: {name}")
    return receipt


def prepare_run(config):
    profile = contract.load_profile(config.name)
    expected = {
        "pi05": True,
        "discrete_state_input": True,
        "state_dim": 38,
        "action_dim": 32,
        "action_horizon": 40,
        "max_token_len": 256,
        "strict_token_length": True,
    }
    if any(getattr(config.model, k) != v for k, v in expected.items()):
        raise ValueError("Taro model dimensions/tokenization differ from the physical contract")
    load_preparation(profile)
    assets = config.assets_dirs / config.data.assets.asset_id
    receipt = verify_assets(assets, profile)
    manifest_hash = contract.sha256(contract.prepared_root(profile) / "manifest.json")
    if receipt["preparation_sha256"] != manifest_hash:
        raise ValueError("Statistics and prepared window index differ")
    donor = Path(config.weight_loader.params_path)
    donor_files = {str(p.relative_to(donor)): contract.sha256(p) for p in sorted(donor.rglob("*")) if p.is_file()}
    if not donor_files:
        raise ValueError("Empty initialization checkpoint")
    source_files = sorted(
        [
            *contract.REPOSITORY_ROOT.glob("src/openpi/**/*.py"),
            *contract.REPOSITORY_ROOT.glob("scripts/*.py"),
            *contract.REPOSITORY_ROOT.glob("configs/industrialnext/*.json"),
            contract.REPOSITORY_ROOT / "uv.lock",
        ]
    )
    implementation = {str(p.relative_to(contract.REPOSITORY_ROOT)): contract.sha256(p) for p in source_files}
    identity = {
        "implementation": implementation,
        "config_name": config.name,
        "model": dataclasses.asdict(config.model),
        "assets_sha256": contract.sha256(assets / "taro_assets.json"),
        "preparation_sha256": manifest_hash,
        "donor_files": donor_files,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "fsdp_devices": config.fsdp_devices,
        "optimizer": dataclasses.asdict(config.optimizer),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "ema_decay": config.ema_decay,
        "num_train_steps": config.num_train_steps,
    }
    path = config.checkpoint_dir / "run_receipt.json"
    if path.exists() and not config.overwrite:
        if not config.resume or json.loads(path.read_text())["identity"] != identity:
            raise ValueError("Refusing changed Taro training identity; use a new experiment name")
    elif config.resume:
        raise ValueError("Cannot resume without the original run receipt")
    return {
        "schema_version": 1,
        "runtime": {
            "backend": "jax",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "torch", "lerobot")
            },
        },
        "identity": identity,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
    }


def checkpoint_assets(directory: Path, config_name: str, run_dir: Path, completed_updates: int):
    from openpi.training.config import get_config

    config = get_config(config_name)
    source = config.assets_dirs / config.data.assets.asset_id
    profile = contract.load_profile(config_name)
    verify_assets(source, profile)
    target = directory / config.data.assets.asset_id
    for filename in ("taro_assets.json", "paligemma_tokenizer.model"):
        shutil.copyfile(source / filename, target / filename)
    contract.atomic_json(target / "profile.json", profile)
    shutil.copyfile(run_dir / "run_receipt.json", target / "run_receipt.json")
    contract.atomic_json(
        target / "checkpoint.json",
        {
            "completed_updates": completed_updates,
            "parameter_kind": "ema",
            "run_receipt_sha256": contract.sha256(target / "run_receipt.json"),
        },
    )


def archive_comparison(run_dir: Path, step: int) -> Path:
    """Keep self-contained EMA exports while retaining only the latest optimizer state."""
    source = run_dir / str(step)
    target = run_dir / "exports" / str(step)
    if target.exists():
        return target
    temporary = target.with_name(f".{step}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    for item in ("params", "assets"):
        shutil.copytree(source / item, temporary / item, copy_function=os.link)
    temporary.rename(target)
    return target
