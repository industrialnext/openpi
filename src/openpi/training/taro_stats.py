"""Deterministic, bounded-memory masked statistics over admitted Taro windows."""

from pathlib import Path
import shutil

import numpy as np

from openpi.shared import normalize
from openpi.shared import taro_contract as contract
from openpi.shared.taro_geometry import relative_actions
from openpi.training.taro_dataset import TaroDataset
from openpi.training.taro_dataset import read_episode


class MaskedStats:
    """Two-pass histograms avoid repeated approximate histogram rebinning."""

    def __init__(self, width: int, bins: int = 20000):
        self.bins = bins
        self.count = np.zeros(width, dtype=np.int64)
        self.total = np.zeros(width)
        self.squares = np.zeros(width)
        self.low = np.full(width, np.inf)
        self.high = np.full(width, -np.inf)
        self.hist = np.zeros((width, bins), dtype=np.int64)

    def update(self, values: np.ndarray, mask: np.ndarray, *, histogram: bool = False):
        values = values.reshape(-1, values.shape[-1])
        mask = mask.reshape(values.shape)
        for dim in range(values.shape[-1]):
            valid = values[:, dim][mask[:, dim]].astype(np.float64)
            if not len(valid):
                continue
            if not np.isfinite(valid).all():
                raise ValueError("Nonfinite valid statistics input")
            if histogram:
                span = self.high[dim] - self.low[dim]
                ix = (
                    np.zeros(len(valid), dtype=np.int64)
                    if span == 0
                    else np.minimum(((valid - self.low[dim]) / span * self.bins).astype(np.int64), self.bins - 1)
                )
                self.hist[dim] += np.bincount(ix, minlength=self.bins)
            else:
                self.count[dim] += len(valid)
                self.total[dim] += valid.sum()
                self.squares[dim] += np.square(valid).sum()
                self.low[dim] = min(self.low[dim], valid.min())
                self.high[dim] = max(self.high[dim], valid.max())

    def finish(self) -> normalize.NormStats:
        if (self.count == 0).any() or not np.array_equal(self.hist.sum(axis=1), self.count):
            raise ValueError("Empty physical statistics coordinate or inconsistent histogram")
        mean = self.total / self.count
        std = np.sqrt(np.maximum(0, self.squares / self.count - mean**2))
        quantiles = []
        for q in (0.01, 0.99):
            ix = np.array([np.searchsorted(np.cumsum(h), q * n) for h, n in zip(self.hist, self.count, strict=True)])
            quantiles.append(self.low + (ix + 0.5) / self.bins * (self.high - self.low))
        low, high = quantiles
        # A one-native-unit symmetric interval keeps constant/near-constant dimensions stable and invertible.
        constant = high - low < 1e-4
        low = np.where(constant, mean - 0.5, low)
        high = np.where(constant, mean + 0.5, high)
        return normalize.NormStats(mean=mean, std=std, q01=low, q99=high)


def batches(dataset: TaroDataset):
    index = dataset.index
    for episode_id in np.unique(index[:, 0]):
        episode = dataset.episodes[int(episode_id)]
        starts = index[index[:, 0] == episode_id, 1]
        data = read_episode(episode["parquet"])
        for begin in range(0, len(starts), 1024):
            rows = starts[begin : begin + 1024]
            target_rows = rows[:, None] + np.arange(40)
            states = data["observation.state"][rows]
            actions, mask = relative_actions(states, data["action"][target_rows], data["action_mask"][target_rows])
            yield states, actions, mask


def compute(config):
    dataset = TaroDataset(config.name, decode_images=False)
    state, action = MaskedStats(38), MaskedStats(29)
    for histogram in (False, True):
        for states, actions, mask in batches(dataset):
            state.update(states, np.ones(states.shape, dtype=bool), histogram=histogram)
            action.update(actions, mask, histogram=histogram)
        print(f"Taro statistics pass {2 if histogram else 1}/2 complete", flush=True)
    output = config.assets_dirs / config.data.assets.asset_id
    normalize.save(output, {"state": state.finish(), "actions": action.finish()})
    tokenizer = Path(config.model.tokenizer_path)
    shutil.copyfile(tokenizer, output / "paligemma_tokenizer.model")
    receipt = {
        "schema_version": 1,
        "config_name": config.name,
        "contract": contract.training_contract(dataset.profile),
        "preparation_sha256": contract.sha256(contract.prepared_root(dataset.profile) / "manifest.json"),
        "norm_stats_sha256": contract.sha256(output / "norm_stats.json"),
        "tokenizer_sha256": contract.sha256(output / "paligemma_tokenizer.model"),
        "counts": {"state": state.count.tolist(), "actions": action.count.tolist()},
        "normalization": "quantile_hist20000_constant_span1_v1",
        "sampling": "uniform_eligible_start; state once/start; action overlapping windows",
    }
    contract.atomic_json(output / "taro_assets.json", receipt)
    print(output, flush=True)
