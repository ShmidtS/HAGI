"""Overfit sanity test — the cheapest proof the training loop is correct.

A tiny model must drive loss toward zero on a fixed toy corpus. If this fails,
the loop / gradients / optimizer are broken. Runs on CPU in seconds (small model,
few steps). Covers both AdamW (baseline) and the full GDR variant.
"""

import torch

from hagi.data.toy import make_toy_batch_fn
from hagi.model.gdr import GradeConfig
from hagi.model.hagi import HAGI, HAGIConfig
from hagi.model.transformer import TransformerConfig
from hagi.train.loop import LoopConfig, train
from hagi.train.optim import build_optimizer


def _tiny_cfg(use_loop: bool, use_gdr: bool) -> HAGIConfig:
    hidden = 64
    tcfg = TransformerConfig(
        hidden_size=hidden,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
    )
    # Grades sum to hidden=64; vector/bivector divisible by 8 (blade count).
    grades = GradeConfig(
        scalar=8,
        vector=16,
        bivector=16,
        trivector=8,
        residual=16,
        scalar_momentum=0.9,
        vector_momentum=0.5,
    )
    return HAGIConfig(
        vocab_size=64,
        hidden_size=hidden,
        perception_layers=1,
        reasoning_layers=1,
        expression_layers=1,
        loop_count=2,
        use_loop=use_loop,
        use_gdr=use_gdr,
        transformer=tcfg,
        grades=grades,
    )


def _run_overfit(use_loop: bool, use_gdr: bool, optimizer: str) -> tuple[float, float]:
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_loop, use_gdr)
    model = HAGI(cfg)
    get_batch = make_toy_batch_fn(
        vocab_size=64,
        block_size=32,
        batch_size=8,
        num_sequences=4,
        device="cpu",
        seed=0,
    )

    # Initial loss.
    x, y = get_batch()
    _, loss0 = model(x, targets=y)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loop = LoopConfig(
        max_steps=1000,
        warmup_steps=1,
        learning_rate=1e-2,
        grad_accum_steps=1,
        precision="fp32",
        eval_interval=0,
        ckpt_interval=0,
        log_interval=1000,
    )
    final = train(model, opt, get_batch, loop, device="cpu")
    return float(loss0.item()), final


def test_overfit_baseline_adamw():
    loss0, final = _run_overfit(use_loop=False, use_gdr=False, optimizer="adamw")
    assert final < loss0 * 0.9, f"loss did not drop: {loss0:.3f} -> {final:.3f}"


def test_overfit_gdr_adamw():
    loss0, final = _run_overfit(use_loop=True, use_gdr=True, optimizer="adamw")
    assert final < loss0 * 0.9, f"loss did not drop: {loss0:.3f} -> {final:.3f}"
