"""Verify both ablation identities and representative real decoded model inputs."""

import json
from pathlib import Path

import numpy as np
import tyro

from openpi.shared import taro_contract
from openpi.training.config import get_config
from openpi.training.data_loader import transform_dataset
from openpi.training.taro_dataset import TaroDataset
from openpi.training.taro_dataset import eligible_starts
from openpi.training.taro_dataset import read_episode
from openpi.training.taro_receipts import verify_assets


def main(output_dir: Path):
    taro_contract.create_output_directory(output_dir)
    results = {}
    holdouts = []
    for variant in ("100", "full"):
        name = f"pi05_taro_exp_{variant}"
        config = get_config(name)
        dataset = TaroDataset(name)
        assets = verify_assets(config.assets_dirs / config.data.assets.asset_id, dataset.profile)
        admitted = dataset.index
        # Every stored start must still match the effective physical supervision rule.
        for eid in np.unique(admitted[:, 0]):
            expected = eligible_starts(read_episode(dataset.episodes[int(eid)]["parquet"]))
            np.testing.assert_array_equal(admitted[admitted[:, 0] == eid, 1], expected)
        transformed = transform_dataset(dataset, config.data.create(config.assets_dirs, config.model))
        samples = []
        for pool in sorted({e["pool"] for e in dataset.episodes if e["split"] == "train"}):
            ids = {i for i, e in enumerate(dataset.episodes) if e["pool"] == pool and e["split"] == "train"}
            indices = np.flatnonzero(np.isin(admitted[:, 0], list(ids)))
            for i in indices[np.linspace(0, len(indices) - 1, 3, dtype=int)]:
                x = transformed[int(i)]
                assert x["state"].shape == (38,)
                assert x["actions"].shape == (40, 32)
                assert not x["action_loss_mask"][:, 29:].any()
                assert (x["actions"][~x["action_loss_mask"]] == 0).all()
                assert np.isfinite(x["actions"]).all()
                assert set(x["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
                assert all(v.shape == (224, 224, 3) for v in x["image"].values())
                samples.append(
                    {
                        "pool": pool,
                        "index": int(i),
                        "tokens": int(x["tokenized_prompt_mask"].sum()),
                        "valid_targets": int(x["action_loss_mask"].sum()),
                    }
                )
        val = TaroDataset(name, split="val", decode_images=False)
        holdout = [(val.episodes[int(eid)]["parquet"], int(start)) for eid, start in val.index]
        holdouts.append(holdout)
        results[name] = {
            "training_corpus_entries": sum(e["split"] == "train" for e in dataset.episodes),
            "train_episodes": len(np.unique(admitted[:, 0])),
            "excluded_training_entries": [
                e for i, e in enumerate(dataset.episodes) if e["split"] == "train" and i not in set(admitted[:, 0])
            ],
            "train_windows": len(dataset),
            "holdout_corpus_entries": sum(e["split"] == "val" for e in dataset.episodes),
            "val_episodes": len(np.unique(val.index[:, 0])),
            "excluded_holdout_entries": [
                e for i, e in enumerate(val.episodes) if e["split"] == "val" and i not in set(val.index[:, 0])
            ],
            "val_windows": len(val),
            "samples": samples,
            "physical_action_stat_counts": assets["counts"]["actions"],
            "assets_sha256": taro_contract.sha256(
                config.assets_dirs / config.data.assets.asset_id / "taro_assets.json"
            ),
        }
    if holdouts[0] != holdouts[1]:
        raise ValueError("Ablation holdout membership/order differs")
    results["matched_holdout_sha256"] = taro_contract.identity(holdouts[0])
    taro_contract.atomic_json(output_dir / "report.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
