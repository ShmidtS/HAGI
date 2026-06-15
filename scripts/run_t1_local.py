"""Run the T1 adapter bake-off locally (frozen SmolLM2-360M + gradient checkpointing
fits a 4GB card). Mirrors notebooks/hagi_t1_adapter_bakeoff.ipynb exactly, but:

  - gradient checkpointing on (low-VRAM),
  - results + small adapter weights saved after EACH adapter (resumable / crash-safe),
  - skips adapters already recorded in the output results.json.

    python scripts/run_t1_local.py --out _local/t1_out

The whole point of this file: the user is out of Colab compute. This produces the
geo-vs-LoRA verdict on free local hardware and never loses partial progress.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prototype.model.clifford_adapters import inject_adapters

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
TARGETS = ("q_proj", "v_proj")
ADAPTERS = {"lora": dict(rank=16), "geo": dict(n_mv=2), "rotor": dict()}
PROMPT = "Question: {q}\nAnswer:"


def build_example(tok, q, a, seq):
    p = tok(PROMPT.format(q=q), add_special_tokens=True).input_ids
    ans = tok(" " + a + tok.eos_token, add_special_tokens=False).input_ids
    return (p + ans)[:seq], ([-100] * len(p) + ans)[:seq]


def collate(items, pad_id):
    m = max(len(i[0]) for i in items)
    ids = torch.full((len(items), m), pad_id, dtype=torch.long)
    lab = torch.full((len(items), m), -100, dtype=torch.long)
    for j, (i, l) in enumerate(items):
        ids[j, :len(i)] = torch.tensor(i)
        lab[j, :len(l)] = torch.tensor(l)
    return ids, lab


def extract_final(t):
    m = re.search(r"####\s*(-?[\d,]+)", t)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", t)
    return nums[-1].replace(",", "") if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_local/t1_out")
    ap.add_argument("--train-examples", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=320)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--eval-ce", type=int, default=300)
    ap.add_argument("--eval-em", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", nargs="+", default=None, help="subset of adapters, e.g. --only geo lora")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    done = {}
    if os.path.exists(results_path):
        done = {r["kind"]: r for r in json.load(open(results_path))}
        print(f"resuming; already done: {list(done)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cc = torch.cuda.get_device_capability()[0] if dev == "cuda" else 0
    dtype = torch.bfloat16 if cc >= 8 else torch.float32
    print(f"device {dev} cc{cc} dtype {dtype}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gsm = load_dataset("openai/gsm8k", "main")
    train_raw, test_raw = gsm["train"], gsm["test"]
    random.seed(args.seed)
    n_train = min(args.train_examples, len(train_raw))
    train_items = [build_example(tok, e["question"], e["answer"], args.seq)
                   for e in train_raw.select(range(n_train))]
    val_items = [build_example(tok, e["question"], e["answer"], args.seq)
                 for e in test_raw.select(range(args.eval_ce))]
    print(f"train {len(train_items)} | val(CE) {len(val_items)}")

    @torch.no_grad()
    def eval_ce(model):
        model.eval(); tot = 0.0; ntok = 0
        for k in range(0, len(val_items), args.batch):
            ids, lab = collate(val_items[k:k + args.batch], tok.pad_token_id)
            ids, lab = ids.to(dev), lab.to(dev)
            logits = model(input_ids=ids).logits[:, :-1].float()
            tgt = lab[:, 1:]
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                ignore_index=-100, reduction="sum")
            tot += l.item(); ntok += (tgt != -100).sum().item()
        return tot / max(ntok, 1)

    @torch.no_grad()
    def eval_em(model, n):
        model.eval(); correct = 0
        for e in test_raw.select(range(n)):
            enc = tok(PROMPT.format(q=e["question"]), return_tensors="pt").to(dev)
            gen = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
            comp = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            correct += (extract_final(comp) == extract_final(e["answer"]))
        return correct / n

    def run_adapter(kind, **knobs):
        torch.manual_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(dev)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # base frozen + grad ckpt needs this
        model.config.use_cache = False
        info = inject_adapters(model, kind, targets=TARGETS, **knobs)
        model.to(dev); model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

        def lr_at(s):
            if s < args.warmup:
                return s / max(args.warmup, 1)
            p = (s - args.warmup) / max(args.steps - args.warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * p))

        order = list(range(len(train_items))); random.shuffle(order)
        step = 0; ptr = 0; t0 = time.time(); last = 0.0
        while step < args.steps:
            if ptr + args.batch > len(order):
                random.shuffle(order); ptr = 0
            idxs = order[ptr:ptr + args.batch]; ptr += args.batch
            ids, lab = collate([train_items[j] for j in idxs], tok.pad_token_id)
            ids, lab = ids.to(dev), lab.to(dev)
            loss = model(input_ids=ids, labels=lab).loss
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); step += 1; last = loss.item()
            if step % 100 == 0 or step == 1:
                print(f"  [{kind}] step {step:4d}/{args.steps} loss {last:.3f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

        model.config.use_cache = True  # generation wants the cache
        ce = eval_ce(model)
        em = eval_em(model, args.eval_em)
        # Persist the small adapter weights (trainable params only) + result.
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(sd, os.path.join(args.out, f"{kind}_adapter.pt"))
        res = dict(kind=kind, trainable=info["trainable"], n_wrapped=info["n_wrapped"],
                   train_loss=round(last, 4), held_out_ce=round(ce, 4),
                   ppl=round(math.exp(ce), 3), test_em=round(em, 4))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{kind}] DONE  CE {ce:.4f} | ppl {math.exp(ce):.2f} | EM {em:.3f}", flush=True)
        return res

    kinds = args.only or list(ADAPTERS)
    results = list(done.values())
    for kind in kinds:
        if kind in done:
            print(f"=== {kind} (cached) ==="); continue
        print(f"=== {kind} ===", flush=True)
        res = run_adapter(kind, **ADAPTERS[kind])
        results = [r for r in results if r["kind"] != kind] + [res]
        json.dump(results, open(results_path, "w"), indent=2)  # incremental save

    print("\n================ T1 RESULTS ================")
    hdr = f"{'adapter':<8}{'trainable':>12}{'held-out CE':>14}{'ppl':>10}{'test EM':>10}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(results, key=lambda r: r["held_out_ce"]):
        print(f"{r['kind']:<8}{r['trainable']:>12,}{r['held_out_ce']:>14.4f}"
              f"{r['ppl']:>10.2f}{r['test_em']:>10.3f}")
    lora = next((r for r in results if r["kind"] == "lora"), None)
    if lora:
        print("\nvs LoRA (held-out CE; negative = Clifford adapter wins):")
        for r in results:
            if r["kind"] != "lora":
                d = r["held_out_ce"] - lora["held_out_ce"]
                print(f"  {r['kind']:<6} dCE = {d:+.4f}   dEM = {r['test_em']-lora['test_em']:+.3f}"
                      f"   ({'WINS' if d < 0 else 'loses'} vs LoRA)")
    print("\nsaved ->", results_path)


if __name__ == "__main__":
    main()
