"""Memory Sparse Attention (MSA) for HAGI.

Provides slot-based sparse attention with document-wise RoPE and HDIM integration.
Each slot is an independent memory domain with a routing key derived from Clifford
invariants (scalar part of the geometric product).  K/V caches are append-only.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .binary_factorized import BinaryFactorizedLinear
from .clifford import BLADE_COUNT, geometric_product, inner_product
from .triton_kernels import TRITON_AVAILABLE, sparse_attention_triton


@dataclass
class MemorySlot:
    """A single memory slot with routing key and append-only K/V cache."""

    slot_id: int
    domain_id: int
    routing_key: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor


class SlotRegistry:
    """Registers memory slots and indexes their routing keys.

    Contract **RouteWithinSlots**: every slot ID returned by the router must exist
    in the registry.  Missing IDs raise ``KeyError``.
    """

    def __init__(self, max_slots: int = 10000) -> None:
        self._slots: Dict[int, MemorySlot] = {}
        self._routing_keys: torch.Tensor | None = None
        self._slot_ids: List[int] = []
        self._slot_ids_tensor: torch.Tensor | None = None
        self._max_slots = max_slots

    def _evict_if_full(self) -> None:
        """LRU eviction: remove oldest slot when at capacity."""
        if len(self._slots) >= self._max_slots and self._slot_ids:
            oldest = self._slot_ids[0]
            self._slots.pop(oldest, None)
            self._slot_ids.pop(0)

    def register(self, slot: MemorySlot) -> None:
        """Register a slot and invalidate the key tensor cache."""
        self._evict_if_full()
        self._slots[slot.slot_id] = slot
        self._routing_keys = None
        self._slot_ids_tensor = None
        if slot.slot_id in self._slot_ids:
            self._slot_ids.remove(slot.slot_id)
        self._slot_ids.append(slot.slot_id)

    def batch_register(self, slots: List[MemorySlot]) -> None:
        """Register multiple slots at once, invalidating caches only once."""
        ids_to_remove = {s.slot_id for s in slots}
        self._slot_ids = [sid for sid in self._slot_ids if sid not in ids_to_remove]
        for slot in slots:
            self._evict_if_full()
            self._slots[slot.slot_id] = slot
        self._slot_ids.extend(s.slot_id for s in slots)
        self._routing_keys = None
        self._slot_ids_tensor = None

    def get(self, slot_id: int) -> MemorySlot:
        """Retrieve a slot by ID."""
        if slot_id not in self._slots:
            raise KeyError(f"Slot {slot_id} not found in registry")
        return self._slots[slot_id]

    def batch_get(self, slot_ids: torch.Tensor) -> List[MemorySlot]:
        """Fetch multiple slots by ID.

        Args:
            slot_ids: 1-D tensor of slot IDs.

        Returns:
            List of ``MemorySlot`` matching the IDs.
        """
        ids_list = slot_ids.tolist()
        return [self._slots[sid] for sid in ids_list]

    def keys_tensor(self, device: str | None = None) -> torch.Tensor:
        """Return a stacked tensor of all routing keys [N, key_dim].

        Lazily rebuilds when new slots are registered.
        """
        if self._routing_keys is None:
            if not self._slots:
                raise RuntimeError("No slots registered")
            keys = [self._slots[sid].routing_key for sid in self._slot_ids]
            stacked = torch.stack(keys, dim=0)
            self._routing_keys = stacked.to(device) if device is not None else stacked
        elif device is not None and self._routing_keys.device.type != device.split(':')[0]:
            # Rebuild if device changed to avoid mixed-device cat
            keys = [self._slots[sid].routing_key for sid in self._slot_ids]
            stacked = torch.stack(keys, dim=0)
            self._routing_keys = stacked.to(device)
        elif device is not None:
            self._routing_keys = self._routing_keys.to(device)
        return self._routing_keys

    def slot_ids(self) -> List[int]:
        """Return registered slot IDs in registration order."""
        return list(self._slot_ids)

    def slot_ids_tensor(self, device: str | None = None) -> torch.Tensor:
        """Return a cached tensor of slot IDs."""
        if self._slot_ids_tensor is None:
            self._slot_ids_tensor = torch.tensor(self._slot_ids, dtype=torch.long).pin_memory()
        if device is not None:
            return self._slot_ids_tensor.to(device, non_blocking=True)
        return self._slot_ids_tensor

    def clear(self) -> None:
        """Remove all slots and invalidate caches."""
        self._slots.clear()
        self._slot_ids.clear()
        self._routing_keys = None
        self._slot_ids_tensor = None

    def __len__(self) -> int:
        return len(self._slots)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "slot_ids": self._slot_ids,
            "slots": {
                sid: {
                    "slot_id": slot.slot_id,
                    "domain_id": slot.domain_id,
                    "routing_key": slot.routing_key.cpu().tolist(),
                    "k_cache": slot.k_cache.cpu().tolist(),
                    "v_cache": slot.v_cache.cpu().tolist(),
                }
                for sid, slot in self._slots.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serializable state."""
        self._slot_ids = state.get("slot_ids", [])
        self._slots = {}
        for sid, slot_data in state.get("slots", {}).items():
            sid = int(sid)
            self._slots[sid] = MemorySlot(
                slot_id=sid,
                domain_id=slot_data["domain_id"],
                routing_key=torch.tensor(slot_data["routing_key"]),
                k_cache=torch.tensor(slot_data["k_cache"]),
                v_cache=torch.tensor(slot_data["v_cache"]),
            )
        self._routing_keys = None


class SparseRouter(nn.Module):
    """Routes queries to the top-k most relevant memory slots.

    Computes dot-product scores between the query hidden state and slot routing
    keys, then returns the top-k slot IDs, raw scores, and softmax-normalised
    weights.
    """

    def __init__(self, hidden_size: int, key_dim: int | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim if key_dim is not None else hidden_size
        self.query_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

    def route_top_k(
        self,
        query_hidden: torch.Tensor,
        registry: SlotRegistry,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top-k slot IDs, scores, and weights.

        Args:
            query_hidden: [B, T, hidden_size] or [B, hidden_size].
            registry: slot registry with pre-registered routing keys.
            top_k: number of slots to select.

        Returns:
            top_k_ids: [B, T, top_k] or [B, top_k] long tensor of slot IDs.
            scores:    same shape as top_k_ids, raw dot-product scores.
            weights:   same shape, softmax-normalised weights.
        """
        if len(registry) == 0:
            raise RuntimeError("Cannot route: registry is empty")

        query = self.query_proj(query_hidden)  # [B, T, key_dim] or [B, key_dim]
        keys = registry.keys_tensor(device=str(query.device))  # [N, key_dim]

        # Move keys to query device/dtype if needed
        if keys.device != query.device or keys.dtype != query.dtype:
            keys = keys.to(device=query.device, dtype=query.dtype)

        # Dot-product scores: [B, T, N] or [B, N]
        scores = torch.matmul(query, keys.transpose(-2, -1))

        top_k = min(top_k, scores.size(-1))
        top_k_weights, top_k_indices = torch.topk(scores, k=top_k, dim=-1, sorted=False)
        weights = F.softmax(top_k_weights, dim=-1)

        # Map indices back to actual slot IDs (cached tensor)
        slot_ids = registry.slot_ids_tensor(device=str(query.device))
        top_k_ids = slot_ids[top_k_indices]

        return top_k_ids, top_k_weights, weights


class DocumentWiseRoPE(nn.Module):
    """RoPE with per-slot document offsets.

    Each slot maintains its own position offset so that tokens from different
    memory domains do not share the same rotation angles.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._cache: OrderedDict[tuple[int, torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _get_cache(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, dtype, device)
        if key not in self._cache:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos().to(dtype)
            sin = freqs.sin().to(dtype)
            self._cache[key] = (cos, sin)
            if len(self._cache) > 100:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return self._cache[key]

    def forward(
        self,
        x: torch.Tensor,
        slot_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply document-wise RoPE.

        Args:
            x: [B, H, T, D].
            slot_offsets: [B, T] per-token offsets. If 3D [B, T, K] the first slot
                is used for compatibility with some gather paths.

        Returns:
            Rotated tensor with same shape as ``x``.
        """
        B, H, T, D = x.shape
        # Fixed-size cache: always use max_seq_len to avoid GPU sync from .item()
        # The cache is computed once and reused. Memory overhead is negligible.
        seq_len = self.max_seq_len
        cos, sin = self._get_cache(seq_len, x.dtype, x.device)

        # slot_offsets must be [B, T]
        if slot_offsets.dim() == 3:
            slot_offsets = slot_offsets[..., 0]
        if slot_offsets.dim() != 2:
            raise ValueError(f"slot_offsets must be [B, T], got {slot_offsets.shape}")

        positions = slot_offsets.unsqueeze(1)  # [B, 1, T]
        positions = positions + torch.arange(T, device=x.device).view(1, 1, T)
        pos_indices = positions.long().clamp(0, seq_len - 1)
        pos_indices = pos_indices.expand(B, H, T)  # [B, H, T]

        cos_idx = cos[pos_indices]  # [B, H, T, D//2]
        sin_idx = sin[pos_indices]  # [B, H, T, D//2]

        x1, x2 = x[..., 0::2], x[..., 1::2]
        rx1 = x1 * cos_idx - x2 * sin_idx
        rx2 = x1 * sin_idx + x2 * cos_idx
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out


class HostKvCache:
    """Append-only K/V cache for memory slots.

    Contract **CacheAppendOnly**: tokens are appended via ``append()``, never
    overwritten in-place.  Pre-allocates a buffer to avoid O(n²) memory copies.
    """

    def __init__(self, slot: MemorySlot, max_len: int = 4096) -> None:
        self.slot = slot
        self.max_len = max_len
        nkv, _, head_dim = slot.k_cache.shape
        self._k = torch.empty(
            nkv, max_len, head_dim,
            dtype=slot.k_cache.dtype, device=slot.k_cache.device,
        )
        self._v = torch.empty(
            nkv, max_len, head_dim,
            dtype=slot.v_cache.dtype, device=slot.v_cache.device,
        )
        self._len = slot.k_cache.size(-2)
        self._k[..., : self._len, :] = slot.k_cache
        self._v[..., : self._len, :] = slot.v_cache

    @property
    def k(self) -> torch.Tensor:
        return self._k[..., : self._len, :]

    @property
    def v(self) -> torch.Tensor:
        return self._v[..., : self._len, :]

    @property
    def cache_len(self) -> int:
        return self._len

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append new K/V tokens to the slot cache.

        Args:
            k_new: [..., new_len, head_dim] key tokens.
            v_new: [..., new_len, head_dim] value tokens.
        """
        new_len = k_new.size(-2)
        if self._len + new_len > self.max_len:
            raise RuntimeError(
                f"Slot {self.slot.slot_id} cache overflow: {self._len + new_len} > {self.max_len}"
            )
        self._k[..., self._len : self._len + new_len, :] = k_new
        self._v[..., self._len : self._len + new_len, :] = v_new
        self._len += new_len
        self.slot.k_cache = self.k
        self.slot.v_cache = self.v


class MSAAttention(nn.Module):
    """Sparse attention across selected memory slots.

    Accepts a query hidden state and fetched K/V tensors from selected slots,
    then computes grouped-query attention output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        max_seq_len: int = 4096,
        use_binary_factorized: bool = False,
        binary_factorized_rank: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_query_heads
        assert num_query_heads % num_kv_heads == 0, "query heads must be divisible by kv heads"

        _make = (
            lambda i, o: BinaryFactorizedLinear(i, o, binary_factorized_rank)
            if use_binary_factorized
            else nn.Linear(i, o, bias=False)
        )
        self.q_proj = _make(hidden_size, num_query_heads * self.head_dim)
        self.k_proj = _make(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = _make(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = _make(num_query_heads * self.head_dim, hidden_size)

        self.rope = DocumentWiseRoPE(self.head_dim, max_seq_len, rope_theta)

    def _fetch_kv_from_slots(
        self,
        slot_ids: torch.Tensor,
        registry: SlotRegistry,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch K, V, and offsets for the selected slots.

        Deduplicates unique slot IDs to avoid O(B*T*top_k) redundant copies.
        """
        B = slot_ids.size(0)
        top_k = slot_ids.size(-1)
        is_3d = slot_ids.dim() == 3
        T = slot_ids.size(1) if is_3d else 1

        device = slot_ids.device
        flat_ids = slot_ids.flatten().long()
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)

        # Fetch unique caches once using vectorised batch_get
        slots = registry.batch_get(unique_ids)
        unique_k = torch.stack([s.k_cache for s in slots], dim=0).to(device)
        unique_v = torch.stack([s.v_cache for s in slots], dim=0).to(device)
        unique_offsets = torch.tensor(
            [s.k_cache.size(-2) for s in slots],
            dtype=torch.long, device=device,
        )

        # Index to reconstruct original order
        k_all = unique_k[inverse]
        v_all = unique_v[inverse]
        offsets = unique_offsets[inverse]

        # Reshape
        if is_3d:
            k_all = k_all.view(B, T, top_k, *unique_k.shape[1:])
            v_all = v_all.view(B, T, top_k, *unique_v.shape[1:])
            offsets = offsets.view(B, T, top_k)
        else:
            k_all = k_all.view(B, top_k, *unique_k.shape[1:])
            v_all = v_all.view(B, top_k, *unique_v.shape[1:])
            offsets = offsets.view(B, top_k)
        return k_all, v_all, offsets

    def forward(
        self,
        hidden_states: torch.Tensor,
        slot_ids: torch.Tensor,
        registry: SlotRegistry,
        attn_mask: torch.Tensor | None = None,
        nars_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse attention over selected slots.

        Args:
            hidden_states: [B, T, hidden_size].
            slot_ids: [B, T, top_k] or [B, top_k] selected slot IDs.
            registry: slot registry.
            attn_mask: optional broadcast-compatible mask.
            nars_weights: optional [B, T, top_k] or [B, top_k] truth-weighted
                attention multipliers from NARS.

        Returns:
            [B, T, hidden_size] attention output.
        """
        B, T, _ = hidden_states.shape
        top_k = slot_ids.size(-1)
        is_3d = slot_ids.dim() == 3

        q = self.q_proj(hidden_states).view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)
        # q: [B, nq, T, head_dim]

        # Fetch K/V from slots
        k_all, v_all, offsets = self._fetch_kv_from_slots(slot_ids, registry)
        nkv = k_all.size(-3)
        rep = self.num_query_heads // nkv

        # Apply document-wise RoPE to slot K/V
        if is_3d:
            Bk = B * T * top_k
            k_all = k_all.view(Bk, nkv, -1, self.head_dim)
            v_all = v_all.view(Bk, nkv, -1, self.head_dim)
            offsets_expanded = offsets.view(Bk, 1).expand(Bk, k_all.size(-2))
            k_all = self.rope(k_all, offsets_expanded)
            v_all = self.rope(v_all, offsets_expanded)
            k_all = k_all.view(B, T, top_k, nkv, -1, self.head_dim)
            v_all = v_all.view(B, T, top_k, nkv, -1, self.head_dim)
        else:
            Bk = B * top_k
            k_all = k_all.view(Bk, nkv, -1, self.head_dim)
            v_all = v_all.view(Bk, nkv, -1, self.head_dim)
            offsets_expanded = offsets.view(Bk, 1).expand(Bk, k_all.size(-2))
            k_all = self.rope(k_all, offsets_expanded)
            v_all = self.rope(v_all, offsets_expanded)
            k_all = k_all.view(B, top_k, nkv, -1, self.head_dim)
            v_all = v_all.view(B, top_k, nkv, -1, self.head_dim)

        # Vectorized attention using matmul (avoids einsum bugs with singleton dims)
        scale = self.head_dim ** 0.5

        if is_3d:
            # GQA: broadcast KV heads instead of repeat_interleave (zero-copy)
            # k_all: [B, T, top_k, nkv, T_slot, head_dim] -> [B, nq, T, top_k, T_slot, head_dim]
            kk = k_all.unsqueeze(3).expand(-1, -1, -1, rep, -1, -1, -1).reshape(B, T, top_k, self.num_query_heads, -1, self.head_dim).permute(0, 3, 1, 2, 4, 5)
            # v_all: [B, T, top_k, nkv, T_slot, head_dim] -> [B, nq, T, top_k, T_slot, head_dim]
            vk = v_all.unsqueeze(3).expand(-1, -1, -1, rep, -1, -1, -1).reshape(B, T, top_k, self.num_query_heads, -1, self.head_dim).permute(0, 3, 1, 2, 4, 5)

            qk = q.unsqueeze(3)  # [B, nq, T, 1, head_dim]
            qk_mat = qk.reshape(B * self.num_query_heads * T, 1, self.head_dim)
            kk_mat = kk.reshape(B * self.num_query_heads * T, top_k * kk.size(4), self.head_dim)
            scores = torch.matmul(qk_mat, kk_mat.transpose(-2, -1)) / scale
            scores = scores.view(B, self.num_query_heads, T, top_k, kk.size(4))
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1).to(q.dtype)
            if nars_weights is not None:
                w = nars_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, top_k, 1]
                attn = attn * w
            attn_mat = attn.reshape(B * self.num_query_heads * T, 1, top_k * kk.size(4))
            vk_mat = vk.reshape(B * self.num_query_heads * T, top_k * vk.size(4), self.head_dim)
            out_k = torch.matmul(attn_mat, vk_mat)  # [B*nq*T, 1, head_dim]
            out_k = out_k.view(B, self.num_query_heads, T, self.head_dim)
        else:
            # GQA: broadcast KV heads instead of repeat_interleave
            # k_all: [B, top_k, nkv, T_slot, head_dim] -> [B, nq, top_k, T_slot, head_dim]
            kk = k_all.unsqueeze(2).expand(-1, -1, rep, -1, -1, -1).reshape(B, top_k, self.num_query_heads, -1, self.head_dim).permute(0, 2, 1, 3, 4)
            vk = v_all.unsqueeze(2).expand(-1, -1, rep, -1, -1, -1).reshape(B, top_k, self.num_query_heads, -1, self.head_dim).permute(0, 2, 1, 3, 4)

            qk = q.unsqueeze(2)  # [B, nq, 1, T, head_dim]
            qk_mat = qk.reshape(B * self.num_query_heads, T, self.head_dim)
            kk_mat = kk.reshape(B * self.num_query_heads, top_k * kk.size(3), self.head_dim)
            scores = torch.matmul(qk_mat, kk_mat.transpose(-2, -1)) / scale
            scores = scores.view(B, self.num_query_heads, T, top_k, kk.size(3))
            scores = scores.permute(0, 1, 3, 2, 4)  # [B, nq, top_k, T, T_slot]
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1).to(q.dtype)
            if nars_weights is not None:
                w = nars_weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, top_k, 1, 1]
                attn = attn * w
            attn_mat = attn.permute(0, 1, 3, 2, 4).reshape(B * self.num_query_heads * T, 1, top_k * kk.size(3))
            vk_mat = vk.reshape(B * self.num_query_heads, top_k * vk.size(3), self.head_dim).unsqueeze(1).expand(-1, T, -1, -1).reshape(B * self.num_query_heads * T, top_k * vk.size(3), self.head_dim)
            out_k = torch.matmul(attn_mat, vk_mat)  # [B*nq*T, 1, head_dim]
            out_k = out_k.squeeze(1)  # [B*nq*T, head_dim]
            out_k = out_k.view(B, self.num_query_heads, T, self.head_dim)

        out = out_k.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        return out


class HDIMSlotRouter(nn.Module):
    """HDIM-aware slot router that derives routing keys from Clifford invariants.

    Each slot corresponds to a separate HDIM domain; the routing key is the
    scalar (grade-0) invariant extracted via the inner product of the domain
    rotor with the hidden state.
    """

    def __init__(self, hidden_size: int, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.hidden_size = hidden_size
        self.blade_count = blade_count
        self.hidden_to_mv = nn.Linear(hidden_size, blade_count)

    def routing_key(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute scalar Clifford invariant from hidden states.

        Args:
            hidden_states: [B, T, hidden_size] or [hidden_size].

        Returns:
            [B, T] or scalar tensor of invariant values.
        """
        mv = self.hidden_to_mv(hidden_states)
        # Inner product of mv with itself gives the scalar norm-like invariant
        inv = inner_product(mv, mv)
        return inv

    def create_slot(
        self,
        slot_id: int,
        domain_id: int,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> MemorySlot:
        """Create a MemorySlot with routing key derived from HDIM invariants.

        Args:
            hidden_states: [B, T, hidden_size] or [hidden_size]; used to compute key.
            k_cache: [nkv, T_slot, head_dim].
            v_cache: [nkv, T_slot, head_dim].

        Returns:
            A ``MemorySlot`` with the Clifford invariant as its routing key.
        """
        inv = self.routing_key(hidden_states)
        # Flatten invariant to a 1-D key vector
        key = inv.view(-1)
        if key.numel() == 1:
            key = key.unsqueeze(0)
        return MemorySlot(
            slot_id=slot_id,
            domain_id=domain_id,
            routing_key=key,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    def batch_create_slots(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_id_base: int = 0,
        domain_id: int = 0,
    ) -> List[MemorySlot]:
        """Create ``MemorySlot`` objects for all positions in a batch.

        Vectorises ``routing_key`` computation over the full batch, then builds
        slots with a fast comprehension instead of nested Python loops.

        Args:
            hidden_states: [B, T, hidden_size].
            k_cache: [B, nkv, T, head_dim].
            v_cache: [B, nkv, T, head_dim].
            slot_id_base: starting slot ID.
            domain_id: domain ID for all slots.

        Returns:
            List of ``MemorySlot`` (length B*T).
        """
        B, T, _ = hidden_states.shape
        inv = self.routing_key(hidden_states)  # [B, T]
        inv_flat = inv.view(-1)
        total = B * T

        # Transpose to [B, T, nkv, head_dim] for per-token slicing
        k_t = k_cache.transpose(1, 2)
        v_t = v_cache.transpose(1, 2)

        # Flatten batch dimension and create per-token [nkv, 1, head_dim] slices
        k_flat = k_t.reshape(total, k_cache.size(1), k_cache.size(-1)).unsqueeze(2)
        v_flat = v_t.reshape(total, v_cache.size(1), v_cache.size(-1)).unsqueeze(2)

        # Unbind once to avoid per-iteration Python overhead
        inv_list = inv_flat.unsqueeze(1).unbind(0)
        k_list = k_flat.unbind(0)
        v_list = v_flat.unbind(0)

        slots = [
            MemorySlot(
                slot_id=slot_id_base + i,
                domain_id=domain_id,
                routing_key=inv_list[i],
                k_cache=k_list[i],
                v_cache=v_list[i],
            )
            for i in range(total)
        ]
        return slots
