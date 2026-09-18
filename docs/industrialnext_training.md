# Industrial Next Taro PI0.5 training

This fork trains independent Taro 100 and full PI0.5 models from the same local
PI0.5 base. Industrial Next serving is described in
[industrialnext_serving.md](industrialnext_serving.md).

## Environment and frozen data

Use Python 3.11 and the repository lockfile, including the pinned RPC submodule:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra industrialnext --extra taro-audit
```

The two JSON profiles in `configs/industrialnext/` bind the frozen GR00T-converted
Parquet/video artifacts, original configuration revision, release, projection and
capture-group split. Preparation never reconverts or modifies those artifacts.
The local adapter avoids LeRobot metadata migration and Hub lookups while reusing
its pinned PyAV reader. `ffprobe` must be available for video frame-count checks.
On the training machine, `data`, `assets`, `checkpoints` and `.venv` resolve to NVMe.
Keep donor `gs://openpi-assets/checkpoints/pi05_base/params` under
`assets/pi05_base/params`, and the PaliGemma tokenizer under
`assets/taro/paligemma_tokenizer.model`. Startup records their content identities.

```bash
JAX_PLATFORMS=cpu uv run --no-sync scripts/prepare_taro.py --config-name pi05_taro_exp_100
JAX_PLATFORMS=cpu uv run --no-sync scripts/prepare_taro.py --config-name pi05_taro_exp_full
JAX_PLATFORMS=cpu uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_taro_exp_100
JAX_PLATFORMS=cpu uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_taro_exp_full
JAX_PLATFORMS=cpu uv run --no-sync scripts/check_taro.py --output-dir checkpoints/NEW_PREFLIGHT
```

Never rerun preparation/statistics underneath an active run. Startup verifies
bound source stat guards and index/statistics identities. A changed source needs
a fresh verified preparation and a new run. The preflight report enumerates
entries with zero eligible windows; corpus membership is not a usable-window count.

| Population | Released entries | Entries with eligible starts | Eligible starts |
|---|---:|---:|---:|
| Taro 100 training | 100 | 93 | 56,917 |
| Full training | 1,429 | 1,332 | 849,044 |
| Shared holdout | 81 | 77 | 51,880 |

The 100/full source variants intentionally preserve cleaned/original versions.
Neither training set includes the common holdout. Invalid current state/RGB is
excluded; partially supervised future actions remain masked. Full 40-frame windows
cannot cross converted segments or repeat an episode endpoint.

## Representation and optimization

State contains both EEF poses and all 20 native right-hand coordinates: 38 total.
PI0.5 receives that state through discrete tokens; its 32-wide pretrained action
head remains unchanged. Physical outputs contain right EEF position plus row
rot6d and 20 absolute native-hand coordinates, 29 total. EEF actions are trained
as full SE(3) relative transforms against the current right EEF. Invalid EEF
blocks, independently invalid hand coordinates and padding receive zero loss
weight. Input placeholders are zeroed after normalization, before noise injection.
Loss is averaged by each sample's valid physical-coordinate count.

Each variant fits its own train-only statistics over the admitted sampling
population. State is counted once per start and action targets across overlapping
windows. Quantiles use a deterministic two-pass 20,000-bin histogram; near-constant
coordinates use an invertible unit-width interval. Counts and hashes are saved.

Both configurations use full JAX fine-tuning, global batch 64, two-device FSDP,
15,000 updates (960,000 sampled windows), seed 42, EMA 0.999, AdamW with clipping 1.0, weight decay 1e-10,
750-step warmup (5% of the budget) and cosine learning rate 2.5e-5 to 2.5e-6. Sampling is uniform
across eligible starts. `metrics.json` records actual pool exposure since the
current process started, including the resumed-at update. Loss is not a physical
robot-quality metric.

```bash
# User-assigned tmux pane 1:1.0: physical GPUs 0,1
./inx_train.sh 100 --exp-name taro_100_YYYYMMDD
# User-assigned tmux pane 1:1.1: physical GPUs 2,3
./inx_train.sh full --exp-name taro_full_YYYYMMDD
```

First qualify batch 64, finite gradients, checkpoint reload and transport. The
launcher fixes GPU assignment and offline behavior; use a unique experiment name.
The run receipt binds runtime code (including uncommitted source), donor, assets,
optimizer, schedule, topology, seed and model dimensions. `--resume` requires the
same identity. It restores optimizer and EMA, but does not restore sampler position;
resumed samples are not bit-identical to an uninterrupted run.

## Checkpoint selection and evaluation

For Taro, save cadence uses **completed updates**. Orbax directory names retain
the upstream zero-based loop index: directory `4999` means 5,000 updates and
`14999` means 15,000. These correspond to GR00T `checkpoint-5000` and
`checkpoint-15000`, respectively, if those checkpoints are available. The existing
GR00T final checkpoints completed 40,000 updates; comparing them to the new PI0.5
final exports is a comparison with different training budgets. Label it accordingly.

The latest full optimizer checkpoint is retained for resume. Every 5,000 completed
updates and at completion, `exports/<loop-index>/` retains self-contained EMA
parameters and assets using local hardlinks. Old optimizer states are pruned.
An export is for inference/evaluation, not optimizer resume. This keeps matched
comparison steps without retaining roughly 42 GiB of optimizer state per point.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/eval_taro.py \
  --config-name pi05_taro_exp_100 \
  --checkpoint-dir checkpoints/pi05_taro_exp_100/RUN/exports/14999 \
  --output-dir checkpoints/NEW_EVALUATION --starts-per-episode 3
```

The default is one deterministic eligible start per eligible holdout entry;
raise the count for a denser comparison. Both variants must use the same count,
seed, source rows and completed-update checkpoint. Reports preserve physical
predictions, targets/masks, source rows, counts, per-pool metrics and synchronized
latency. They separately report translation metres, SO(3) radians and per-coordinate
native-hand error. Six virtual-time replay recipes use the production resolver,
fixed measured p95 delay, fixed noise and explicit missed requests. This replay
is not a deployment load test. Real RPC smoke is a separate measurement.

GR00T's projector/action-decoder tuning, frozen backbones, 20% state dropout,
256-pixel augmentation, learning rate/weight decay, saved processor normalization
and shard-based sampler differ from PI0.5's full fine-tuning, 224-pixel inputs,
discrete state, quantile normalization and uniform-window sampler. The two PI0.5 variants match each other at 15,000 × 64 windows; the existing GR00T
runs used 40,000 × 64. Equal corpus and batch do not imply equal training budget,
compute or optimization. Preserve these differences in real-robot result tables;
do not compare normalized loss.
