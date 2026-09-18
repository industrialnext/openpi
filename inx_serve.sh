#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
variant="${1:?Usage: ./inx_serve.sh 100|full --checkpoint-dir PATH [--port PORT]}"
shift
case "$variant" in
  100|full) ;;
  *) echo 'Variant must be 100 or full' >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
exec .venv/bin/python scripts/serve_industrialnext.py --config-name "pi05_taro_exp_${variant}" "$@"
