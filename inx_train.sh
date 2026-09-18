#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
variant="${1:?Usage: ./inx_train.sh 100|full --exp-name NAME [training options]}"
shift
case "$variant" in
  100) export CUDA_VISIBLE_DEVICES=0,1 ;;
  full) export CUDA_VISIBLE_DEVICES=2,3 ;;
  *) echo 'Variant must be 100 or full' >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
exec .venv/bin/python -u scripts/train.py "pi05_taro_exp_${variant}" "$@"
