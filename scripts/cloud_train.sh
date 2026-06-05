#!/usr/bin/env bash
# One-command cloud training launcher (Kaggle / Colab / Lightning AI / Azure).
# Installs deps, reports the GPU, and trains with auto-resume so a killed free-tier
# session continues from the latest checkpoint. Point CKPT_DIR at persistent
# storage (Drive / Kaggle Dataset / Lightning / Azure disk) so checkpoints survive.
#
#   bash scripts/cloud_train.sh CONFIG DATA_DIR [CKPT_DIR] [EXTRA_ARGS...]
#
# CONFIG may be a path, or "auto" to pick by GPU: Ampere+ (sm80, bf16/FA2) ->
# baseline.yaml; older/16GB (e.g. T4/V100) -> colab_t4.yaml (fp16). Handy on Azure
# where the GPU SKU you get (T4 / V100 / A10) is not known in advance.
#
# Examples:
#   bash scripts/cloud_train.sh auto data/fineweb-edu ~/storage
#   bash scripts/cloud_train.sh configs/baseline.yaml data/fineweb-edu /workspace/ckpt --steps 200
set -euo pipefail

CONFIG="${1:?usage: cloud_train.sh CONFIG DATA_DIR [CKPT_DIR] [EXTRA_ARGS...]}"
DATA_DIR="${2:?usage: cloud_train.sh CONFIG DATA_DIR [CKPT_DIR] [EXTRA_ARGS...]}"
CKPT_DIR="${3:-checkpoints}"
shift $(( $# < 3 ? $# : 3 ))   # remaining args ($@) pass through to train.py

echo "== installing deps =="
pip install -q -r requirements.txt

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null \
  || echo "no GPU detected (CPU run — validation only)"

if [[ "$CONFIG" == "auto" ]]; then
  CONFIG=$(python - <<'PY'
import torch
cfg = "configs/colab_t4.yaml"
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    sm = p.major * 10 + p.minor
    # Ampere+ with >=24GB -> canonical bf16 baseline; otherwise the 16GB fp16 profile.
    cfg = "configs/baseline.yaml" if (sm >= 80 and p.total_memory / 1e9 >= 22) else "configs/colab_t4.yaml"
print(cfg)
PY
)
  echo "== auto-selected config: $CONFIG =="
fi

echo "== train (auto-resume into $CKPT_DIR) =="
python -m prototype.training.train \
  --config "$CONFIG" \
  --data "$DATA_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --resume auto "$@"
