"""Generates notebooks/hagi_t1_adapter_bakeoff.ipynb.

The adapter library (clifford.py primitives + clifford_adapters.py) is embedded
verbatim from the tested sources so the notebook is standalone (no repo clone) and
runs the exact code covered by prototype/tests/test_clifford_adapters.py.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _strip(src: str) -> str:
    out = []
    for line in src.splitlines(keepends=True):
        if line.startswith("from __future__ import annotations"):
            continue
        if re.match(r"\s*from \.clifford import", line):
            continue
        out.append(line)
    return "".join(out)


clifford = _strip((ROOT / "prototype/model/clifford.py").read_text(encoding="utf-8"))
adapters = _strip((ROOT / "prototype/model/clifford_adapters.py").read_text(encoding="utf-8"))

LIB_CELL = (
    "from __future__ import annotations\n"
    "# ====================================================================\n"
    "# Embedded verbatim from prototype/model/{clifford,clifford_adapters}.py\n"
    "# (covered by prototype/tests/test_clifford_adapters.py). Standalone here\n"
    "# so the notebook needs no repo clone.\n"
    "# ====================================================================\n"
    + clifford + "\n\n" + adapters
)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = []

cells.append(md(
"""# T1 — Adapter Bake-off: does Clifford structure beat LoRA at equal budget?

The pivot test. Three PEFT adapters on a **frozen** SmolLM2-360M, same GSM8K data,
same schedule, same `q_proj`/`v_proj` targets — only the adapter math differs:

| adapter | math | trainable |
|---|---|---|
| **LoRA** | low-rank delta `B(A(x))` | ~1.64M (0.45%) |
| **Geo** | ReZero geometric-product residual `α·proj(mv ⊗ w)` | ~1.64M (matched) |
| **Rotor** | orthogonal rotor sandwich `R·base(x)·R̃` | ~15k (frugal) |

`Geo` is the direct test of the one GDR ember that survived the gated re-test (the
geometric-product cross-term). Both Clifford adapters are **exact identities at init**
(α=0 / bivector=0) — the same ReZero discipline that made the gated test fair.

**Metric:** held-out CE on GSM8K answer tokens (dense, low-variance — primary) +
test exact-match accuracy (capability check). **Read:** if Geo/Rotor don't beat LoRA
at matched budget, Clifford has no edge even as an adapter — pivot is pure capability.
Free-T4 runtime ~1–1.5 h for all three. Set GPU runtime.
"""))

cells.append(code(
"""# Colab ships torch/transformers/datasets as a consistent set. Do NOT upgrade numpy
# (it breaks the prebuilt scipy/sklearn ABI). Install datasets only if missing.
import importlib, subprocess, sys
for pkg in ("transformers", "datasets"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], check=True)
import torch, transformers, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| cuda", torch.cuda.is_available())
"""))

cells.append(code(
"""# --- knobs ---
BASE_MODEL     = "HuggingFaceTB/SmolLM2-360M"
TARGETS        = ("q_proj", "v_proj")   # same insertion points for all three adapters
LORA_RANK      = 16                      # LoRA r ; geo n_mv chosen to match its param count
GEO_NMV        = 2
ADAPTERS       = {"lora": dict(rank=LORA_RANK), "geo": dict(n_mv=GEO_NMV), "rotor": dict()}

TRAIN_EXAMPLES = 4000     # GSM8K train subset to tokenize
TRAIN_STEPS    = 400
BATCH          = 8
SEQ            = 320      # GSM8K Q+A fits comfortably
LR             = 2e-4
WARMUP         = 30
EVAL_CE_N      = 300      # held-out test examples for CE (primary metric)
EVAL_EM_N      = 150      # test examples for generate+exact-match (slower)
SEED           = 1234
OUT_DIR        = "t1_out"  # results.json + small adapter weights saved here per adapter
GRAD_CKPT      = False     # set True for low-VRAM GPUs (frozen 360M fits 4GB with this on)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "set Colab runtime to GPU (T4)"

# IMPORTANT: SmolLM2 in fp16 overflows to NaN logits on some GSM8K sequences.
# Use bf16 on Ampere+ (cc>=8: A100/L4) for speed, fp32 on older cards (T4, cc 7.5)
# for correctness. Never fp16.
_CC = torch.cuda.get_device_capability()[0]
DTYPE = torch.bfloat16 if _CC >= 8 else torch.float32
print(f"GPU {torch.cuda.get_device_name()} (cc {_CC}.x) -> dtype {DTYPE}")
"""))

cells.append(md(
"""### (optional) Persist results to Google Drive

Colab's local disk dies with the session. Run this to save `results.json` + the small
adapter weights to Drive so a disconnect mid-run loses nothing (the bug that ate the
earlier A100 run). Skip it and `OUT_DIR` stays on local disk.
"""))

cells.append(code(
"""import os
# Uncomment to persist across disconnects:
# from google.colab import drive; drive.mount('/content/drive')
# OUT_DIR = '/content/drive/MyDrive/hagi_t1_out'
os.makedirs(OUT_DIR, exist_ok=True)
print('saving results + adapter weights to:', os.path.abspath(OUT_DIR))
"""))

cells.append(code(LIB_CELL))

cells.append(code(
"""# GSM8K -> prompt-masked SFT tensors (train on the answer tokens only).
import re, random
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

gsm = load_dataset("openai/gsm8k", "main")
train_raw, test_raw = gsm["train"], gsm["test"]
PROMPT = "Question: {q}\\nAnswer:"

def build_example(q, a):
    p = tok(PROMPT.format(q=q), add_special_tokens=True).input_ids
    ans = tok(" " + a + tok.eos_token, add_special_tokens=False).input_ids
    ids = (p + ans)[:SEQ]
    labels = ([-100] * len(p) + ans)[:SEQ]
    return ids, labels

random.seed(SEED)
n_train = min(TRAIN_EXAMPLES, len(train_raw))
train_items = [build_example(e["question"], e["answer"]) for e in train_raw.select(range(n_train))]
val_items   = [build_example(e["question"], e["answer"]) for e in test_raw.select(range(EVAL_CE_N))]
print(f"train items {len(train_items)} | val(CE) items {len(val_items)} | "
      f"median len {sorted(len(i[0]) for i in train_items)[len(train_items)//2]}")

def collate(items):
    m = max(len(i[0]) for i in items)
    ids = torch.full((len(items), m), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(items), m), -100, dtype=torch.long)
    for j, (i, l) in enumerate(items):
        ids[j, :len(i)] = torch.tensor(i)
        lab[j, :len(l)] = torch.tensor(l)
    return ids, lab

def extract_final(t):
    m = re.search(r"####\\s*(-?[\\d,]+)", t)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\\d[\\d,]*", t)
    return nums[-1].replace(",", "") if nums else None
"""))

cells.append(code(
"""# Train one adapter, then score held-out CE (answer tokens) + test exact-match.
import math, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def fresh_base():
    torch.manual_seed(SEED)
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE).to(DEVICE)
    if GRAD_CKPT:
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()   # base frozen + grad ckpt needs this
        m.config.use_cache = False
    return m

@torch.no_grad()
def eval_ce(model):
    model.eval(); tot = 0.0; ntok = 0
    for k in range(0, len(val_items), BATCH):
        ids, lab = collate(val_items[k:k + BATCH])
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
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
        enc = tok(PROMPT.format(q=e["question"]), return_tensors="pt").to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        comp = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        correct += (extract_final(comp) == extract_final(e["answer"]))
    return correct / n

def run_adapter(kind, **knobs):
    model = fresh_base()
    info = inject_adapters(model, kind, targets=TARGETS, **knobs)
    model.to(DEVICE); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    def lr_at(s):
        if s < WARMUP:
            return s / max(WARMUP, 1)
        p = (s - WARMUP) / max(TRAIN_STEPS - WARMUP, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    order = list(range(len(train_items))); random.shuffle(order)
    step = 0; ptr = 0; t0 = time.time(); last = 0.0
    while step < TRAIN_STEPS:
        if ptr + BATCH > len(order):
            random.shuffle(order); ptr = 0
        idxs = order[ptr:ptr + BATCH]; ptr += BATCH
        ids, lab = collate([train_items[j] for j in idxs])
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
        loss = model(input_ids=ids, labels=lab).loss
        for g in opt.param_groups:
            g["lr"] = LR * lr_at(step)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); step += 1; last = loss.item()
        if step % 100 == 0 or step == 1:
            print(f"  [{kind}] step {step:4d}/{TRAIN_STEPS} loss {last:.3f} "
                  f"({(time.time()-t0)/step:.2f}s/step)")
    if GRAD_CKPT:
        model.config.use_cache = True  # generation wants the cache back
    ce = eval_ce(model)
    em = eval_em(model, EVAL_EM_N)
    # Persist the small adapter weights (trainable params only) so a disconnect
    # after this adapter still leaves something on disk.
    import os
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sd, os.path.join(OUT_DIR, f"{kind}_adapter.pt"))
    res = dict(kind=kind, trainable=info["trainable"], n_wrapped=info["n_wrapped"],
               train_loss=round(last, 4), held_out_ce=round(ce, 4),
               ppl=round(math.exp(ce), 3), test_em=round(em, 4))
    del model
    torch.cuda.empty_cache()
    print(f"  [{kind}] DONE  CE {ce:.4f} | ppl {math.exp(ce):.2f} | EM {em:.3f}")
    return res
"""))

cells.append(code(
"""# Run the bake-off. Same data/schedule/targets; only the adapter math differs.
# results.json is rewritten after EACH adapter, so a disconnect keeps finished ones.
import json as _json, os
results = []
for kind, knobs in ADAPTERS.items():
    print(f"=== {kind} ===")
    results.append(run_adapter(kind, **knobs))
    _json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

print("\\n================ T1 RESULTS ================")
hdr = f"{'adapter':<8}{'trainable':>12}{'held-out CE':>14}{'ppl':>10}{'test EM':>10}"
print(hdr); print("-" * len(hdr))
for r in sorted(results, key=lambda r: r["held_out_ce"]):
    print(f"{r['kind']:<8}{r['trainable']:>12,}{r['held_out_ce']:>14.4f}"
          f"{r['ppl']:>10.2f}{r['test_em']:>10.3f}")

lora = next(r for r in results if r["kind"] == "lora")
print("\\nvs LoRA (held-out CE; negative = Clifford adapter wins):")
for r in results:
    if r["kind"] != "lora":
        d = r["held_out_ce"] - lora["held_out_ce"]
        print(f"  {r['kind']:<6} ΔCE = {d:+.4f}   ΔEM = {r['test_em']-lora['test_em']:+.3f}"
              f"   ({'WINS' if d < 0 else 'loses'} vs LoRA)")
import json as _json
print("\\nJSON " + _json.dumps(results))
"""))

cells.append(md(
"""## Read

- **Primary = held-out CE on answer tokens** (dense, low-variance). Test EM is the
  capability check but noisy at 360M + light SFT — trust CE for the ranking, EM for
  the headline.
- **Geo vs LoRA at matched 1.64M params** is the clean head-to-head: does the
  geometric product add anything a low-rank delta can't? `Geo ΔCE < 0` ⇒ Clifford
  structure has real value as an adapter → scale to SmolLM2-1.7B (T2) and port the
  fuller grade/loop variants. `Geo ΔCE ≥ 0` ⇒ the ember was noise; pivot is pure
  capability (plain LoRA), Clifford retired for good.
- **Rotor** is a separate budget class (~15k params, 100× leaner). If it rivals LoRA
  at 100× fewer params, that is a genuine efficiency finding worth its own note; if it
  trails, part of that is the budget gap — not a clean refutation.

Bump `TRAIN_STEPS` / `EVAL_EM_N` for a tighter read; drop them if T4 time is short.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

out = ROOT / "notebooks/hagi_t1_adapter_bakeoff.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)

import nbformat
nbformat.read(out, as_version=4)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print("valid + all code cells compile")
