"""Gated (ReZero) GDR: the fair re-test of the grade machinery.

The legacy GDR had three optimization faults (half-open sigmoid gates, wholesale
bivector/trivector replacement, far-from-identity init). The gated variant must:
  1. be an EXACT identity at init (D_gated == B functionally at step 0),
  2. still pass gradients to the gates (so the model can opt in),
  3. leave the legacy path byte-identical (past results stay reproducible).
"""

import torch

from prototype.model.gdr import GradeConfig, GradeDecomposedRecurrence


def _cfg(**kw):
    return GradeConfig(scalar=8, vector=16, bivector=16, trivector=8, residual=16, **kw)


def test_gated_is_exact_identity_at_init():
    torch.manual_seed(0)
    m = GradeDecomposedRecurrence(_cfg(gated=True))
    h = torch.randn(2, 5, _cfg().hidden_size)
    out = m(h)
    assert torch.equal(out, h), "gated GDR must be a no-op at init"


def test_gated_gradients_reach_gates():
    torch.manual_seed(0)
    m = GradeDecomposedRecurrence(_cfg(gated=True))
    h = torch.randn(2, 5, _cfg().hidden_size, requires_grad=True)
    m(h).pow(2).sum().backward()
    for name in ("alpha_scalar", "alpha_vector", "alpha_bivector", "alpha_trivector",
                 "gate_scalar", "gate_bivector"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all(), f"{name}: no/invalid grad"
    # At init the MLP weights legitimately get zero grad (alpha=0 blocks them);
    # the gates carry the signal first - that is the ReZero mechanism.
    assert m.alpha_bivector.grad.abs().sum() > 0, "bivector gate grad should be nonzero"


def test_gated_opens_when_gates_move():
    torch.manual_seed(0)
    m = GradeDecomposedRecurrence(_cfg(gated=True))
    h = torch.randn(2, 5, _cfg().hidden_size)
    with torch.no_grad():
        m.alpha_bivector.fill_(0.5)
    assert not torch.equal(m(h), h), "non-zero gate must change the output"


def test_legacy_path_unchanged():
    torch.manual_seed(0)
    m = GradeDecomposedRecurrence(_cfg(gated=False))
    h = torch.randn(2, 5, _cfg().hidden_size)
    out = m(h)
    # Legacy is NOT identity (the documented faults) and must stay that way.
    assert not torch.allclose(out, h)
    # Residual slice passes through untouched in both modes.
    assert torch.equal(out[..., -16:], h[..., -16:])


def test_gated_param_count_close_to_legacy():
    legacy = sum(p.numel() for p in GradeDecomposedRecurrence(_cfg(gated=False)).parameters())
    gated = sum(p.numel() for p in GradeDecomposedRecurrence(_cfg(gated=True)).parameters())
    assert gated - legacy == 4, "gated adds exactly the 4 alpha scalars"
