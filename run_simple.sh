#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_port="${MASTER_PORT:-29500}" \
  train_gpt_simple.py \
  "$@"
