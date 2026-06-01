#!/usr/bin/env bash
# One-command cloud training launcher (Kaggle / Colab / Lightning AI).
# Installs deps, reports the GPU, and trains with auto-resume so a killed free-tier
# session continues from the latest checkpoint. Point CKPT_DIR at persistent
# storage (Drive / Kaggle Dataset / Lightning) so checkpoints survive a reset.
#
#   bash scripts/cloud_train.sh CONFIG DATA_DIR [CKPT_DIR]
#
# Colab proof-of-life, checkpoints on mounted Drive:
#   bash scripts/cloud_train.sh configs/colab_t4.yaml data/fineweb-edu /content/drive/MyDrive/hagi
# Ampere Stage 0 baseline:
#   bash scripts/cloud_train.sh configs/baseline.yaml data/fineweb-edu /workspace/ckpt
set -euo pipefail

CONFIG="${1:?usage: cloud_train.sh CONFIG DATA_DIR [CKPT_DIR]}"
DATA_DIR="${2:?usage: cloud_train.sh CONFIG DATA_DIR [CKPT_DIR]}"
CKPT_DIR="${3:-checkpoints}"

echo "== installing deps =="
pip install -q -r requirements.txt

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null \
  || echo "no GPU detected (CPU run — validation only)"

echo "== train (auto-resume into $CKPT_DIR) =="
python -m prototype.training.train \
  --config "$CONFIG" \
  --data "$DATA_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --resume auto
