"""T1 adapter unit tests. The two Clifford adapters must be EXACT identities at init
(ReZero discipline), pass gradients, and the rotor must preserve norm (orthogonality).
LoRA must start as a no-op delta. Injection must freeze the base and wrap the targets.
"""

import torch
from torch import nn

from prototype.model.clifford_adapters import (
    GeoProductAdapter,
    LoRAAdapter,
    RotorAdapter,
    WrappedLinear,
    count_trainable,
    inject_adapters,
)


def _x(d=64, b=2, t=5):
    return torch.randn(b, t, d)


def test_lora_is_noop_delta_at_init():
    a = LoRAAdapter(64, 48, rank=8)
    x, base = _x(64), torch.randn(2, 5, 48)
    assert torch.equal(a(x, base), base), "B init 0 -> LoRA delta must be 0 at init"


def test_geo_is_identity_at_init():
    a = GeoProductAdapter(64, 48, n_mv=2)
    x, base = _x(64), torch.randn(2, 5, 48)
    assert torch.equal(a(x, base), base), "alpha init 0 -> geo adapter must be identity"


def test_rotor_is_identity_at_init():
    a = RotorAdapter(48)
    base = torch.randn(2, 5, 48)
    out = a(torch.empty_like(base), base)
    assert torch.allclose(out, base, atol=1e-5), "bivector init 0 -> rotor R=1 (identity)"


def test_rotor_preserves_norm():
    # A nonzero rotor must rotate (change values) yet preserve per-multivector norm.
    a = RotorAdapter(48)
    with torch.no_grad():
        a.bivector.normal_(0, 0.5)
    base = torch.randn(2, 5, 48)
    out = a(torch.empty_like(base), base)
    assert not torch.allclose(out, base, atol=1e-4), "nonzero rotor must change the output"
    bv = base.reshape(2, 5, 6, 8).norm(dim=-1)
    ov = out.reshape(2, 5, 6, 8).norm(dim=-1)
    assert torch.allclose(bv, ov, atol=1e-4), "rotor sandwich must preserve multivector norm"


def test_geo_gradients_reach_gate_and_proj():
    a = GeoProductAdapter(64, 48, n_mv=2)
    with torch.no_grad():
        a.alpha.fill_(0.3)  # open the gate so proj weights get signal
    x, base = _x(64), torch.randn(2, 5, 48)
    a(x, base).pow(2).sum().backward()
    for name in ("alpha", "weight_mv"):
        g = getattr(a, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} no grad"


def test_rotor_gradients_flow():
    a = RotorAdapter(48)
    with torch.no_grad():
        a.bivector.normal_(0, 0.2)
    base = torch.randn(2, 5, 48)
    a(torch.empty_like(base), base).pow(2).sum().backward()
    g = a.bivector.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def _toy_model():
    # Two named Linears that look like attention projections.
    m = nn.Module()
    m.q_proj = nn.Linear(64, 48)
    m.v_proj = nn.Linear(64, 48)
    m.other = nn.Linear(64, 50)  # must NOT be wrapped
    m.forward = lambda x: m.v_proj(m.q_proj.weight.new_zeros(1))  # unused
    return m


def test_inject_wraps_targets_and_freezes_base():
    for kind in ("lora", "geo", "rotor"):
        m = _toy_model()
        info = inject_adapters(m, kind, targets=("q_proj", "v_proj"), rank=8, n_mv=1)
        assert info["n_wrapped"] == 2, f"{kind}: expected 2 wrapped, got {info['n_wrapped']}"
        assert isinstance(m.q_proj, WrappedLinear) and isinstance(m.v_proj, WrappedLinear)
        assert not isinstance(m.other, WrappedLinear), "non-target must stay untouched"
        # base frozen, adapter trainable
        assert all(not p.requires_grad for p in m.q_proj.base.parameters())
        assert any(p.requires_grad for p in m.q_proj.adapter.parameters())
        assert info["trainable"] == count_trainable(m) and info["trainable"] > 0


def test_param_budgets_are_reportable_and_close():
    # LoRA r=16 vs Geo n_mv=2 on the same shapes should land in the same ballpark.
    m1, m2 = _toy_model(), _toy_model()
    lo = inject_adapters(m1, "lora", rank=16)["trainable"]
    ge = inject_adapters(m2, "geo", n_mv=2)["trainable"]
    ratio = ge / lo
    assert 0.5 < ratio < 2.0, f"geo/lora param ratio {ratio:.2f} should be within 2x for matching"
