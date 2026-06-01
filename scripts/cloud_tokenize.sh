#!/usr/bin/env bash
# Tokenize a HuggingFace dataset into flat uint16 .bin shards for training.
# Runs on any Linux cloud box (Kaggle / Colab / Lightning AI).
#
#   bash scripts/cloud_tokenize.sh DATASET SUBSET OUTPUT_DIR [LIMIT]
#
# Full FineWeb-Edu 10BT sample:
#   bash scripts/cloud_tokenize.sh HuggingFaceFW/fineweb-edu sample-10BT data/fineweb-edu
# Quick smoke (first 200 docs):
#   bash scripts/cloud_tokenize.sh HuggingFaceFW/fineweb-edu sample-10BT data/smoke 200
set -euo pipefail

DATASET="${1:?usage: cloud_tokenize.sh DATASET SUBSET OUTPUT_DIR [LIMIT]}"
SUBSET="${2:?usage: cloud_tokenize.sh DATASET SUBSET OUTPUT_DIR [LIMIT]}"
OUTPUT="${3:?usage: cloud_tokenize.sh DATASET SUBSET OUTPUT_DIR [LIMIT]}"
LIMIT="${4:-}"

echo "== installing deps =="
pip install -q -r requirements.txt

ARGS=(--dataset "$DATASET" --subset "$SUBSET" --output "$OUTPUT" \
      --tokenizer HuggingFaceTB/SmolLM2-135M)
if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi

echo "== tokenize -> $OUTPUT =="
python -m prototype.data.tokenize "${ARGS[@]}"
