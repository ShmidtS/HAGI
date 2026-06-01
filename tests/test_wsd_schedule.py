"""Tests for the WSD (warmup-stable-decay) LR schedule (G005 §C)."""

from __future__ import annotations

import math

import pytest

from scripts.train import lr_at


def test_warmup_ramps_linearly_to_peak():
    lr = lr_at(0, max_steps=1000, warmup_steps=100, learning_rate=1.0, schedule="wsd")
    assert lr == pytest.approx(1.0 / 100.0)
    lr_mid = lr_at(50, max_steps=1000, warmup_steps=100, learning_rate=1.0, schedule="wsd")
    assert lr_mid == pytest.approx(51.0 / 100.0)
    lr_peak = lr_at(99, max_steps=1000, warmup_steps=100, learning_rate=1.0, schedule="wsd")
    assert lr_peak == pytest.approx(1.0)


def test_stable_phase_holds_peak_lr():
    peak = 1.0
    for step in (100, 500, 900):
        lr = lr_at(step, max_steps=1000, warmup_steps=100, learning_rate=peak, schedule="wsd")
        assert lr == pytest.approx(peak), f"step {step} should be stable"


def test_cooldown_decays_linearly_to_min_lr():
    min_ratio = 0.1
    peak = 1.0
    max_steps = 1000
    cooldown_frac = 0.05  # last 50 steps
    cooldown_start = int(max_steps * (1.0 - cooldown_frac))  # 950

    # At cooldown_start, LR == peak.
    lr_start = lr_at(
        cooldown_start,
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=peak,
        min_lr_ratio=min_ratio,
        schedule="wsd",
        cooldown_frac=cooldown_frac,
    )
    assert lr_start == pytest.approx(peak)

    # At the boundary (step == max_steps), LR == min_lr.
    lr_end = lr_at(
        max_steps,
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=peak,
        min_lr_ratio=min_ratio,
        schedule="wsd",
        cooldown_frac=cooldown_frac,
    )
    assert lr_end == pytest.approx(peak * min_ratio)

    # At the last training step (max_steps - 1) we are 98% through cooldown.
    lr_near_end = lr_at(
        max_steps - 1,
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=peak,
        min_lr_ratio=min_ratio,
        schedule="wsd",
        cooldown_frac=cooldown_frac,
    )
    assert peak * min_ratio < lr_near_end < peak


def test_wsd_default_cooldown_frac_is_five_percent():
    # Mid cooldown for default 5% window.
    max_steps = 1000
    cooldown_start = 950
    lr = lr_at(
        975,
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=1.0,
        min_lr_ratio=0.1,
        schedule="wsd",
    )
    expected = 1.0 * (0.1 + 0.9 * (1.0 - 0.5))
    assert lr == pytest.approx(expected, abs=1e-6)


def test_cosine_schedule_falls_back_to_cosine_shape():
    peak = 1.0
    max_steps = 1000
    warmup = 100
    # After warmup, cosine is symmetric about halfway.
    for frac in (0.0, 0.5, 1.0):
        step = warmup + int((max_steps - warmup) * frac)
        lr = lr_at(step, max_steps=max_steps, warmup_steps=warmup, learning_rate=peak, schedule="cosine")
        progress = (step - warmup) / max(1, max_steps - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        expected = peak * 0.1 + coeff * peak * (1.0 - 0.1)
        assert lr == pytest.approx(expected, abs=1e-6), f"frac={frac} lr={lr}"


def test_wsd_does_not_call_cooldown_before_cooldown_start():
    # Stable phase LR == peak across the entire non-warmup, non-cooldown region.
    for step in (200, 400, 800, 949):
        lr = lr_at(step, max_steps=1000, warmup_steps=100, learning_rate=1.0, schedule="wsd")
        assert lr == pytest.approx(1.0)


def test_wsd_is_monotonically_decreasing_through_cooldown():
    max_steps = 1000
    cooldown_start = 950
    prev = math.inf
    for step in range(cooldown_start, max_steps):
        lr = lr_at(
            step,
            max_steps=max_steps,
            warmup_steps=100,
            learning_rate=1.0,
            min_lr_ratio=0.1,
            schedule="wsd",
        )
        assert lr <= prev + 1e-9
        prev = lr
