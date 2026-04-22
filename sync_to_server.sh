#!/usr/bin/env bash
set -euo pipefail

target="beaupeng@beaupeng-any3.devcloud.woa.com:/apdcephfs_zwfy2/share_303944931/beaupeng/modded-nanogpt"

exec rsync -av --rsync-path="sudo rsync" \
  --exclude '.git/' \
  --exclude 'archive/' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  -e "ssh -p 36000" \
  ./ \
  "$target"
