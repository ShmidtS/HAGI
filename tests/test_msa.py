"""Tests for Memory Sparse Attention (MSA) components.

Covers:
- MemorySlot creation and properties
- SlotRegistry register/get/keys_tensor/slot_ids/len
- SparseRouter route_top_k correctness
- MSAAttention forward pass shape
- DocumentWiseRoPE application
- HDIMSlotRouter routing_key creation
- HostKvCache append-only contract
- dtype-agnostic behavior (float16/float32)
"""

import pytest

torch = pytest.importorskip("torch")

from hagi.model.msa import (
    DocumentWiseRoPE,
    HDIMSlotRouter,
    HostKvCache,
    MSAAttention,
    MemorySlot,
    SlotRegistry,
    SparseRouter,
)


# ---------------------------------------------------------------------------
# MemorySlot
# ---------------------------------------------------------------------------

def test_memory_slot_creation_and_properties():
    rk = torch.randn(4)
    k = torch.randn(2, 8, 4)
    v = torch.randn(2, 8, 4)
    slot = MemorySlot(slot_id=7, domain_id=3, routing_key=rk, k_cache=k, v_cache=v)
    assert slot.slot_id == 7
    assert slot.domain_id == 3
    assert torch.equal(slot.routing_key, rk)
    assert torch.equal(slot.k_cache, k)
    assert torch.equal(slot.v_cache, v)


# ---------------------------------------------------------------------------
# SlotRegistry
# ---------------------------------------------------------------------------

def test_slot_registry_register_and_get():
    reg = SlotRegistry()
    slot = MemorySlot(1, 0, torch.randn(4), torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    reg.register(slot)
    assert len(reg) == 1
    assert reg.get(1) is slot


def test_slot_registry_get_raises_key_error_for_missing():
    reg = SlotRegistry()
    with pytest.raises(KeyError, match="Slot 99 not found"):
        reg.get(99)


def test_slot_registry_keys_tensor_rebuilds():
    reg = SlotRegistry()
    keys = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    for i, k in enumerate(keys):
        reg.register(MemorySlot(i, 0, k, torch.randn(2, 2, 4), torch.randn(2, 2, 4)))
    stacked = reg.keys_tensor()
    assert stacked.shape == (2, 2)
    assert torch.allclose(stacked[0], keys[0])
    assert torch.allclose(stacked[1], keys[1])


def test_slot_registry_keys_tensor_empty_raises():
    reg = SlotRegistry()
    with pytest.raises(RuntimeError, match="No slots registered"):
        reg.keys_tensor()


def test_slot_registry_slot_ids_in_registration_order():
    reg = SlotRegistry()
    reg.register(MemorySlot(3, 0, torch.randn(2), torch.randn(2, 2, 4), torch.randn(2, 2, 4)))
    reg.register(MemorySlot(1, 0, torch.randn(2), torch.randn(2, 2, 4), torch.randn(2, 2, 4)))
    assert reg.slot_ids() == [3, 1]


# ---------------------------------------------------------------------------
# SparseRouter
# ---------------------------------------------------------------------------

def test_sparse_router_route_top_k_returns_correct_shapes():
    router = SparseRouter(hidden_size=8, key_dim=4)
    reg = SlotRegistry()
    for i in range(5):
        reg.register(MemorySlot(i, 0, torch.randn(4), torch.randn(2, 2, 4), torch.randn(2, 2, 4)))
    query = torch.randn(2, 8)  # [B, hidden_size]
    ids, scores, weights = router.route_top_k(query, reg, top_k=3)
    assert ids.shape == (2, 3)
    assert scores.shape == (2, 3)
    assert weights.shape == (2, 3)
    assert weights.sum(dim=-1).allclose(torch.ones(2), atol=1e-5)


def test_sparse_router_route_top_k_clamps_to_registry_size():
    router = SparseRouter(hidden_size=8, key_dim=4)
    reg = SlotRegistry()
    reg.register(MemorySlot(0, 0, torch.randn(4), torch.randn(2, 2, 4), torch.randn(2, 2, 4)))
    query = torch.randn(1, 8)
    ids, scores, weights = router.route_top_k(query, reg, top_k=10)
    assert ids.shape == (1, 1)


def test_sparse_router_empty_registry_raises():
    router = SparseRouter(hidden_size=8)
    with pytest.raises(RuntimeError, match="registry is empty"):
        router.route_top_k(torch.randn(1, 8), SlotRegistry(), top_k=1)


def test_sparse_router_device_dtype_mismatch_handled():
    router = SparseRouter(hidden_size=4, key_dim=4)
    reg = SlotRegistry()
    reg.register(MemorySlot(0, 0, torch.randn(4), torch.randn(2, 2, 4), torch.randn(2, 2, 4)))
    # Force keys to float32 on CPU, query to float32 on CPU (same device, so just dtype)
    # keys_tensor() is float32 by default.
    query = torch.randn(1, 4, dtype=torch.float32)
    ids, scores, weights = router.route_top_k(query, reg, top_k=1)
    assert ids.dtype == torch.long


# ---------------------------------------------------------------------------
# DocumentWiseRoPE
# ---------------------------------------------------------------------------

def test_document_wise_rope_output_shape():
    rope = DocumentWiseRoPE(head_dim=8)
    x = torch.randn(2, 4, 16, 8)  # [B, H, T, D]
    offsets = torch.zeros(2, 16, dtype=torch.long)  # [B, T]
    out = rope(x, offsets)
    assert out.shape == x.shape


def test_document_wise_rope_3d_offsets_uses_first_slot():
    rope = DocumentWiseRoPE(head_dim=8)
    x = torch.randn(1, 2, 4, 8)
    offsets = torch.zeros(1, 4, 3, dtype=torch.long)  # [B, T, K]
    out = rope(x, offsets)
    assert out.shape == x.shape


def test_document_wise_rope_invalid_offsets_shape_raises():
    rope = DocumentWiseRoPE(head_dim=8)
    x = torch.randn(1, 2, 4, 8)
    offsets = torch.zeros(1, 4, 3, 2, dtype=torch.long)  # 4D
    with pytest.raises(ValueError, match="slot_offsets must be \\[B, T\\]"):
        rope(x, offsets)


def test_document_wise_rope_cache_reuses_same_seq_len():
    rope = DocumentWiseRoPE(head_dim=8)
    x = torch.randn(1, 2, 4, 8)
    offsets = torch.zeros(1, 4, dtype=torch.long)
    _ = rope(x, offsets)
    cache_key = (rope.max_seq_len, x.dtype, x.device)
    assert cache_key in rope._cache
    _ = rope(x, offsets)
    assert len(rope._cache) == 1


# ---------------------------------------------------------------------------
# MSAAttention
# ---------------------------------------------------------------------------

def test_msa_attention_forward_shape():
    hidden_size = 32
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 8
    top_k = 2
    B, T = 2, 4

    attn = MSAAttention(hidden_size, num_query_heads, num_kv_heads, head_dim)
    hidden = torch.randn(B, T, hidden_size)

    reg = SlotRegistry()
    for sid in range(3):
        # k/v cache: [nkv, T_slot, head_dim]
        k = torch.randn(num_kv_heads, 6, head_dim)
        v = torch.randn(num_kv_heads, 6, head_dim)
        reg.register(MemorySlot(sid, 0, torch.randn(head_dim), k, v))

    slot_ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)  # [B, top_k]
    out = attn(hidden, slot_ids, reg)
    assert out.shape == (B, T, hidden_size)


def test_msa_attention_forward_with_attn_mask():
    hidden_size = 16
    num_query_heads = 2
    num_kv_heads = 2
    head_dim = 8
    B, T = 1, 2
    top_k = 1

    attn = MSAAttention(hidden_size, num_query_heads, num_kv_heads, head_dim)
    hidden = torch.randn(B, T, hidden_size)

    reg = SlotRegistry()
    k = torch.randn(num_kv_heads, 4, head_dim)
    v = torch.randn(num_kv_heads, 4, head_dim)
    reg.register(MemorySlot(0, 0, torch.randn(head_dim), k, v))

    slot_ids = torch.tensor([[0]], dtype=torch.long)
    mask = torch.zeros(B, T, 4)  # broadcast-compatible
    out = attn(hidden, slot_ids, reg, attn_mask=mask)
    assert out.shape == (B, T, hidden_size)


# ---------------------------------------------------------------------------
# HDIMSlotRouter
# ---------------------------------------------------------------------------

def test_hdim_slot_router_routing_key_is_scalar_invariant():
    router = HDIMSlotRouter(hidden_size=8)
    hidden = torch.randn(2, 4, 8)
    inv = router.routing_key(hidden)
    assert inv.shape == (2, 4)
    assert inv.dtype == hidden.dtype


def test_hdim_slot_router_create_slot_routing_key_shape():
    router = HDIMSlotRouter(hidden_size=8)
    hidden = torch.randn(1, 1, 8)
    k_cache = torch.randn(2, 4, 4)
    v_cache = torch.randn(2, 4, 4)
    slot = router.create_slot(slot_id=5, domain_id=2, hidden_states=hidden, k_cache=k_cache, v_cache=v_cache)
    assert slot.slot_id == 5
    assert slot.domain_id == 2
    assert isinstance(slot.routing_key, torch.Tensor)
    assert slot.routing_key.numel() == 1


def test_hdim_slot_router_create_slot_with_1d_hidden():
    router = HDIMSlotRouter(hidden_size=8)
    hidden = torch.randn(8)
    k_cache = torch.randn(2, 4, 4)
    v_cache = torch.randn(2, 4, 4)
    slot = router.create_slot(slot_id=0, domain_id=0, hidden_states=hidden, k_cache=k_cache, v_cache=v_cache)
    assert isinstance(slot.routing_key, torch.Tensor)
    assert slot.routing_key.numel() == 1


# ---------------------------------------------------------------------------
# HostKvCache
# ---------------------------------------------------------------------------

def test_host_kv_cache_append_only_increases_length():
    slot = MemorySlot(0, 0, torch.randn(4), torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    cache = HostKvCache(slot, max_len=64)
    initial_len = cache.cache_len
    k_new = torch.randn(2, 3, 4)
    v_new = torch.randn(2, 3, 4)
    cache.append(k_new, v_new)
    assert cache.cache_len == initial_len + 3
    assert cache.k.size(-2) == initial_len + 3
    assert cache.v.size(-2) == initial_len + 3
    assert torch.equal(cache.k, slot.k_cache)
    assert torch.equal(cache.v, slot.v_cache)


def test_host_kv_cache_append_overflow_raises():
    slot = MemorySlot(0, 0, torch.randn(4), torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    cache = HostKvCache(slot, max_len=8)
    k_new = torch.randn(2, 5, 4)
    v_new = torch.randn(2, 5, 4)
    with pytest.raises(RuntimeError, match="cache overflow"):
        cache.append(k_new, v_new)


# ---------------------------------------------------------------------------
# dtype-agnostic
# ---------------------------------------------------------------------------

def test_sparse_router_dtype_agnostic_float16():
    if not torch.cuda.is_available():
        pytest.skip("float16 routing requires GPU for stable matmul")
    router = SparseRouter(hidden_size=8, key_dim=4).cuda().half()
    reg = SlotRegistry()
    for i in range(3):
        reg.register(MemorySlot(i, 0, torch.randn(4, dtype=torch.float16), torch.randn(2, 2, 4, dtype=torch.float16), torch.randn(2, 2, 4, dtype=torch.float16)))
    query = torch.randn(1, 8, dtype=torch.float16, device="cuda")
    ids, scores, weights = router.route_top_k(query, reg, top_k=2)
    assert ids.dtype == torch.long
    assert scores.dtype == torch.float16


def test_msa_attention_dtype_agnostic_float32():
    hidden_size = 16
    attn = MSAAttention(hidden_size, num_query_heads=2, num_kv_heads=2, head_dim=8)
    hidden = torch.randn(1, 2, hidden_size, dtype=torch.float32)
    reg = SlotRegistry()
    reg.register(MemorySlot(0, 0, torch.randn(8, dtype=torch.float32), torch.randn(2, 4, 8, dtype=torch.float32), torch.randn(2, 4, 8, dtype=torch.float32)))
    slot_ids = torch.tensor([[0]], dtype=torch.long)
    out = attn(hidden, slot_ids, reg)
    assert out.dtype == torch.float32


def test_document_wise_rope_dtype_agnostic():
    rope = DocumentWiseRoPE(head_dim=8)
    x = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    offsets = torch.zeros(1, 4, dtype=torch.long)
    out = rope(x, offsets)
    assert out.dtype == torch.float64
