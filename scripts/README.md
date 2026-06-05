# scripts/

Utility + cloud-launch scripts. The cloud scripts target Linux notebooks
(Kaggle / Colab / Lightning AI). Full background: [../docs/CLOUD_TRAINING.md](../docs/CLOUD_TRAINING.md).

| Script | Purpose |
|--------|---------|
| `param_count.py` | Print parameter counts for a config |
| `cloud_tokenize.sh` | Stream a HF dataset → flat uint16 `.bin` shards |
| `cloud_train.sh` | Install deps, report GPU, train with `--resume auto` |

## Typical free-tier session (Colab / Kaggle)

```bash
# Clone the active branch (all current work lives on `experimental`, not main).
git clone -b experimental https://github.com/ShmidtS/HAGI.git && cd HAGI

# 1. Tokenize once (or pull pre-tokenized shards from persistent storage)
bash scripts/cloud_tokenize.sh HuggingFaceFW/fineweb-edu sample-10BT data/fineweb-edu

# 2. Proof-of-life on a 16GB T4. Checkpoints to persistent storage so a
#    killed 12h session resumes. Re-run the SAME line after a session dies —
#    --resume auto picks up the latest checkpoint.
bash scripts/cloud_train.sh configs/colab_t4.yaml data/fineweb-edu /content/drive/MyDrive/hagi
```

Notes:
- **Persistent storage matters.** Kaggle/Colab wipe local files on reset. Pass a
  Drive / Kaggle-Dataset / Lightning path as the 3rd arg so checkpoints survive.
- **Precision is per-GPU.** `colab_t4.yaml` is fp16 (T4 has no bf16 tensor cores);
  on Ampere use `baseline.yaml` (bf16) and the Muon ablation becomes possible.
- **Tokenize once, reuse.** Tokenization is CPU-bound; store the shards and skip
  step 1 on later sessions.
