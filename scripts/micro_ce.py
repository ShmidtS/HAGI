"""Microbench: isolate composite-loss (aux/iso/moe) overhead vs CE-only.
Single process only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import train as train_script  # noqa: E402
from hagi.model import HAGI  # noqa: E402
from hagi.train.config import config_from_dict  # noqa: E402
from hagi.train.loop import _resolve_loss, autocast_ctx  # noqa: E402
import yaml  # noqa: E402

cfg = yaml.safe_load(open("configs/rtx3070_canonical.yaml"))
model_cfg = config_from_dict(cfg["model"])
train_cfg = cfg["training"]
data_cfg = cfg["data"]
device = "cuda"
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

bs = train_cfg["batch_size"]; sl = data_cfg["max_seq_len"]; ga = train_cfg["grad_accum_steps"]
chunk = model_cfg.ce_chunk_size
model = HAGI(model_cfg).to(device).to(torch.bfloat16)
model.cfg.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", model_cfg.gradient_checkpointing))
model.train()
cw = dict(train_cfg["composite_loss"])

def bench(weights, n=10, warm=5):
    tm = weights is not None
    for _ in range(warm):
        x = torch.randint(0, 49152, (bs, sl), device=device); y = torch.randint(0, 49152, (bs, sl), device=device)
        with autocast_ctx("manual_bf16", device):
            out = model(x, targets=y, training_mode=tm, weights=weights)
            if tm:
                loss, _ = _resolve_loss(out, y, weights, chunk)
            else:
                # fused CE returns dict{training} else (logits, loss) tuple
                if isinstance(out, dict):
                    loss = out.get("loss")
                elif isinstance(out, tuple) and len(out) == 2:
                    loss = out[1]
                else:
                    loss = None
                if loss is None:
                    lg = out["logits"] if isinstance(out, dict) else out[0]
                    import torch.nn.functional as F
                    loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n):
        x = torch.randint(0, 49152, (bs, sl), device=device); y = torch.randint(0, 49152, (bs, sl), device=device)
        with autocast_ctx("manual_bf16", device):
            out = model(x, targets=y, training_mode=tm, weights=weights)
            if tm:
                loss, _ = _resolve_loss(out, y, weights, chunk)
            else:
                if isinstance(out, dict):
                    loss = out.get("loss")
                elif isinstance(out, tuple) and len(out) == 2:
                    loss = out[1]
                else:
                    loss = None
                if loss is None:
                    lg = out["logits"] if isinstance(out, dict) else out[0]
                    import torch.nn.functional as F
                    loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100)
        torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000

# CE-only (no aux/iso/moe weights active): pass weights=None
fwd_ce = bench(None)
# full composite (warmup already past, so weights are at final values)
fwd_full = bench(cw)
print(f"fwd+loss CE-only:  {fwd_ce:.1f} ms")
print(f"fwd+loss full:     {fwd_full:.1f} ms")
print(f"aux+iso overhead:  {fwd_full-fwd_ce:.1f} ms  ({(fwd_full-fwd_ce)/fwd_full*100:.0f}% of full)")
