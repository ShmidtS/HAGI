"""Generate text from a HAGI checkpoint — minimal inference CLI.

Loads a checkpoint (local path or HF Hub repo) and samples a continuation. A 113M
model runs in well under 1GB — **CPU is fine, no GPU needed** — so anyone can try
it for free on a laptop or a free CPU notebook.

    # from the HF repo (downloads the latest checkpoint)
    python scripts/generate.py --hf-repo NAME0x0/hagi-stage0 --prompt "The moon is" --device cpu

    # from a local checkpoint
    python scripts/generate.py --ckpt checkpoints/stage0_a100/step-00003815.pt --prompt "..."
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `prototype` importable when run as `python scripts/generate.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description="Generate text from a HAGI checkpoint.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="local checkpoint path (step-*.pt)")
    src.add_argument("--hf-repo", help="HF model repo id; downloads the latest step-*.pt")
    ap.add_argument("--hf-file", default=None, help="specific file in the HF repo (default: latest)")
    ap.add_argument("--prompt", default="The sun is a star that")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    ckpt = args.ckpt
    if args.hf_repo:
        from huggingface_hub import HfApi, hf_hub_download

        fname = args.hf_file
        if fname is None:
            steps = sorted(
                f
                for f in HfApi().list_repo_files(args.hf_repo, repo_type="model")
                if f.startswith("step-") and f.endswith(".pt")
            )
            if not steps:
                raise SystemExit(f"no step-*.pt checkpoints in {args.hf_repo}")
            fname = steps[-1]
        ckpt = hf_hub_download(args.hf_repo, fname, repo_type="model")
        print(f"loaded {fname} from {args.hf_repo}")

    from prototype.data.tokenizer import load_tokenizer
    from prototype.training.loop import load_checkpoint

    model, step = load_checkpoint(ckpt, device=args.device)
    model.eval()
    tok = load_tokenizer(args.tokenizer)

    x = torch.tensor([tok.encode(args.prompt)], device=args.device)
    for _ in range(args.max_new_tokens):
        with torch.no_grad():
            logits = model(x)[0, -1].float()
        if args.top_k:
            kth = torch.topk(logits, args.top_k).values[-1]
            logits[logits < kth] = float("-inf")
        probs = F.softmax(logits / args.temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt.view(1, 1)], dim=1)

    print(f"\n[step {step}] {tok.decode(x[0].tolist())}")


if __name__ == "__main__":
    main()
