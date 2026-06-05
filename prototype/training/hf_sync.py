"""Optional HuggingFace Hub sync for checkpoints — cross-platform persistence.

Lets one run continue across Colab/Kaggle sessions and machines: each saved
checkpoint is pushed to a HF model repo, and on resume with an empty local dir
the latest checkpoint is pulled back. So you can train a chunk on Colab, let the
session die, and continue on Kaggle from the same step.

Best-effort by design: upload/download failures are printed, never fatal to
training (a failed push just means the checkpoint is still safe locally).

Requires `huggingface_hub` (installed via transformers/datasets) and a **write**
token in the `HF_TOKEN` env var (or `huggingface-cli login`).
"""

from __future__ import annotations

from pathlib import Path


def push_checkpoint(path: str, repo_id: str, keep: int = 3) -> None:
    """Upload one checkpoint to repo_id (private model repo), then prune old ones.

    Each checkpoint is ~1.4GB (model + AdamW optimizer state), and only the latest
    is needed to resume, so we keep just the newest `keep` to bound repo size.
    Best-effort — failures never interrupt training.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=Path(path).name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"hf: pushed {Path(path).name} -> {repo_id}")

        ckpts = sorted(
            f
            for f in api.list_repo_files(repo_id, repo_type="model")
            if f.startswith("step-") and f.endswith(".pt")
        )
        for old in ckpts[:-keep]:
            try:
                api.delete_file(old, repo_id, repo_type="model")
                print(f"hf: pruned {old}")
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001 — never let a sync error kill training
        print(f"hf: push failed ({e}) — checkpoint is still saved locally")


def pull_latest_checkpoint(repo_id: str, ckpt_dir: str) -> str | None:
    """Download the highest-numbered step-*.pt from repo_id into ckpt_dir.

    Returns the local path, or None if the repo is empty/missing or on error
    (in which case training just starts fresh / from local).
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = [
            f
            for f in api.list_repo_files(repo_id, repo_type="model")
            if f.startswith("step-") and f.endswith(".pt")
        ]
        if not files:
            return None
        latest = sorted(files)[-1]
        out = Path(ckpt_dir)
        out.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(repo_id, latest, repo_type="model", local_dir=str(out))
        print(f"hf: pulled {latest} <- {repo_id}")
        return local
    except Exception as e:  # noqa: BLE001
        print(f"hf: pull failed ({e}) — starting from local/scratch")
        return None
