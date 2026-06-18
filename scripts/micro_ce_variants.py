"""Compare CE strategies: fused-chunked vs plain vs no-chunk on realistic shapes.
Isolate the dominant cost in the loss path.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

device = "cuda"
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

B, T, H, V = 8, 1024, 768, 49152
N = B * T  # 8192
weight = torch.randn(V, H, device=device, dtype=torch.bfloat16) * 0.02
weight.requires_grad_(True)
h = torch.randn(N, H, device=device, dtype=torch.bfloat16, requires_grad=True)
tgt = torch.randint(0, V, (N,), device=device)
CHUNK = 2048

def fused_chunk(hp, wp, tp, cs=CHUNK):
    flat_h = hp.reshape(-1, hp.size(-1))
    flat_t = tp.reshape(-1)
    valid = (flat_t != -100).sum().clamp(min=1)
    total = torch.zeros((), dtype=torch.float32, device=hp.device)
    for i in range(0, flat_h.size(0), cs):
        hc = flat_h[i:i+cs]
        tc = flat_t[i:i+cs]
        def cl(hc, tc):
            return F.cross_entropy(F.linear(hc, wp), tc, ignore_index=-100, reduction="sum")
        total = total + ckpt(cl, hc, tc, use_reentrant=False)
    return total / valid

def plain_ce(hp, wp, tp):
    logits = F.linear(hp, wp)
    return F.cross_entropy(logits, tp, ignore_index=-100)

def fused_nockpt(hp, wp, tp, cs=CHUNK):
    flat_h = hp.reshape(-1, hp.size(-1))
    flat_t = tp.reshape(-1)
    valid = (flat_t != -100).sum().clamp(min=1)
    total = torch.zeros((), dtype=torch.float32, device=hp.device)
    for i in range(0, flat_h.size(0), cs):
        hc = flat_h[i:i+cs]
        tc = flat_t[i:i+cs]
        total = total + F.cross_entropy(F.linear(hc, wp), tc, ignore_index=-100, reduction="sum")
    return total / valid

def bench(fn, n=20, warm=10):
    for _ in range(warm):
        hh = h.detach().clone().requires_grad_(True)
        ww = weight.detach().clone().requires_grad_(True)
        loss = fn(hh, ww, tgt)
        loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        hh = h.detach().clone().requires_grad_(True)
        ww = weight.detach().clone().requires_grad_(True)
        loss = fn(hh, ww, tgt)
        loss.backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/n*1000
    mem = torch.cuda.max_memory_allocated()/1024**3
    return dt, mem

for name, fn in [("fused_chunk(ckpt,2048)", fused_chunk), ("fused_nockpt(2048)", fused_nockpt), ("plain(full logits)", plain_ce)]:
    dt, mem = bench(fn)
    print(f"{name:28s} {dt:6.1f} ms  peak={mem:.2f}GB")
