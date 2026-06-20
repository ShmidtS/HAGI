"""Memory Sparse Attention (MSA) for HAGI.

Provides slot-based sparse attention with document-wise RoPE and HDIM integration.
Each slot is an independent memory domain with a routing key derived from Clifford
invariants (scalar part of the geometric product).  K/V caches are append-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .binary_factorized import BinaryFactorizedLinear
from .clifford import BLADE_COUNT


@dataclass(slots=True)
class MemorySlot:
    """A single memory slot with routing key and append-only K/V cache."""

    slot_id: int
    domain_id: int
    routing_key: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor


class SlotRegistry:
    """Registers memory slots and indexes their routing keys.

    Stores slots as batched tensors to avoid per-slot Python object overhead.
    Contract **RouteWithinSlots**: every slot ID returned by the router must exist
    in the registry.  Missing IDs raise ``KeyError``.
    """

    def __init__(self, max_slots: int = 10000) -> None:
        self._slot_ids: list[int] = []
        self._id_to_idx: dict[int, int] = {}
        self._routing_keys: torch.Tensor | None = None
        self._k_caches: torch.Tensor | None = None
        self._v_caches: torch.Tensor | None = None
        self._slot_ids_tensor: torch.Tensor | None = None
        self._max_slots = max_slots
        self._slots_compat: dict[int, MemorySlot] = {}
        self._id_to_idx_tensor = torch.full((max_slots * 1000,), -1, dtype=torch.long)
        # Per-slot real K/V length along the time axis. The dense stack pads
        # every slot to the max T_slot seen so far (generation mixes prefill
        # chunk_size slots with decode per-token slots), so the buffer width is
        # the max but each slot's valid range is its own length. MSAAttention
        # uses this to mask padded positions out of the softmax. None when empty.
        self._slot_lens: torch.Tensor | None = None

    # SlotRegistry is a stateful Python bookkeeping object (lists/dicts/CPU
    # tensors), not a differentiable nn.Module. Under torch.compile its methods
    # are traced into the graph and Dynamo specializes a guard on
    # len(self._slot_ids) — which changes every forward (clear -> 0 -> fill ->
    # N) — hitting config.recompile_limit (8) and falling back to eager. Mark
    # every entry point @torch.compiler.disable so Dynamo treats them as opaque
    # Python side-effects and stops guarding on registry length. The actual
    # tensor ops (matmul/topk) live in SparseRouter / MSAAttention nn.Modules
    # and remain compiled.
    @torch.compiler.disable
    def _evict_oldest(self, n: int) -> None:
        """Remove oldest n slots to make room."""
        # Clamp n to the registry size so the slice below stays aligned.
        n = max(0, min(n, len(self._slot_ids)))
        for _ in range(n):
            oldest = self._slot_ids.pop(0)
            self._id_to_idx.pop(oldest, None)
        # After popping the oldest n IDs, the surviving data is rows [n:].
        if self._routing_keys is not None:
            self._routing_keys = (
                self._routing_keys[n:] if n < self._routing_keys.size(0) else None
            )
        if self._k_caches is not None:
            self._k_caches = self._k_caches[n:] if n < self._k_caches.size(0) else None
        if self._v_caches is not None:
            self._v_caches = self._v_caches[n:] if n < self._v_caches.size(0) else None
        if self._slot_lens is not None:
            self._slot_lens = (
                self._slot_lens[n:] if n < self._slot_lens.size(0) else None
            )
        self._id_to_idx = {sid: i for i, sid in enumerate(self._slot_ids)}
        self._id_to_idx_tensor.fill_(-1)
        for i, sid in enumerate(self._slot_ids):
            self._id_to_idx_tensor[sid] = i
        self._slot_ids_tensor = None

    @torch.compiler.disable
    def prune_oldest(self, n: int) -> None:
        """Remove the oldest n slots from the registry."""
        if n <= 0:
            return
        n = max(0, min(n, len(self)))
        self._evict_oldest(n)

    @torch.compiler.disable
    def batch_register(
        self,
        slot_ids: torch.Tensor,
        routing_keys: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
    ) -> None:
        """Register a batch of slots from tensors.

        Args:
            slot_ids: [N] tensor of slot IDs.
            routing_keys: [N, key_dim] tensor of routing keys.
            k_caches: [N, nkv, T_slot, head_dim] tensor of K caches.
            v_caches: [N, nkv, T_slot, head_dim] tensor of V caches.
        """
        n = slot_ids.size(0)

        # Fast path: empty registry (the common per-step case, since HAGI's
        # forward clears the model-owned registry each call). Skip dedup/evict
        # and append directly — no Python loop, no set intersection, no
        # per-element _id_to_idx_tensor writes.
        if len(self._slot_ids) == 0:
            ids_list = slot_ids.tolist()
            self._slot_ids = ids_list
            self._id_to_idx = {sid: i for i, sid in enumerate(ids_list)}
            self._routing_keys = routing_keys
            self._k_caches = k_caches
            self._v_caches = v_caches
            self._slot_lens = torch.full(
                (n,), k_caches.shape[-2], dtype=torch.long, device=k_caches.device
            )
            # One vectorized scatter instead of N scalar writes.
            idx_t = self._id_to_idx_tensor
            if slot_ids.device != idx_t.device:
                slot_ids_cpu = slot_ids.to(idx_t.device)
            else:
                slot_ids_cpu = slot_ids
            idx_t[slot_ids_cpu] = torch.arange(
                n, dtype=idx_t.dtype, device=idx_t.device
            )
            self._slot_ids_tensor = None
            return

        ids_list = slot_ids.tolist()

        # Remove duplicates
        existing = set(ids_list) & set(self._slot_ids)
        if existing:
            keep = [i for i, sid in enumerate(self._slot_ids) if sid not in existing]
            self._slot_ids = [self._slot_ids[i] for i in keep]
            self._routing_keys = (
                self._routing_keys[keep] if self._routing_keys is not None else None
            )
            self._k_caches = (
                self._k_caches[keep] if self._k_caches is not None else None
            )
            self._v_caches = (
                self._v_caches[keep] if self._v_caches is not None else None
            )
            self._id_to_idx = {sid: i for i, sid in enumerate(self._slot_ids)}
            self._id_to_idx_tensor.fill_(-1)
            for i, sid in enumerate(self._slot_ids):
                self._id_to_idx_tensor[sid] = i

        # Evict if needed
        excess = len(self._slot_ids) + n - self._max_slots
        if excess > 0:
            self._evict_oldest(excess)

        # Append
        start = len(self._slot_ids)
        self._slot_ids.extend(ids_list)
        idx_t = self._id_to_idx_tensor
        if slot_ids.device != idx_t.device:
            slot_ids_cpu = slot_ids.to(idx_t.device)
        else:
            slot_ids_cpu = slot_ids
        new_indices = torch.arange(
            start, start + n, dtype=idx_t.dtype, device=idx_t.device
        )
        idx_t[slot_ids_cpu] = new_indices
        for i, sid in enumerate(ids_list):
            self._id_to_idx[sid] = start + i

        self._routing_keys = (
            routing_keys
            if self._routing_keys is None
            else torch.cat([self._routing_keys, routing_keys], dim=0)
        )
        # K/V caches are a dense [N, nkv, T_slot, head_dim] stack, so cat along
        # dim=0 needs identical T_slot (dim=-2). Generation mixes slots of
        # different T_slot (prefill chunk_size vs decode per-token fallback,
        # msa.py batch_create_slots), so pad the shorter cache's time axis to
        # the max T_slot. The pad tail is masked out of attention via the
        # per-slot real length in _slot_lens (MSAAttention valid-len mask).
        t_max = max(
            self._k_caches.shape[-2] if self._k_caches is not None else 0,
            k_caches.shape[-2],
        )
        self._k_caches = self._cat_kv_pad(self._k_caches, k_caches, t_max)
        self._v_caches = self._cat_kv_pad(self._v_caches, v_caches, t_max)
        incoming_lens = torch.full(
            (n,), k_caches.shape[-2], dtype=torch.long, device=k_caches.device
        )
        self._slot_lens = (
            incoming_lens
            if self._slot_lens is None
            else torch.cat([self._slot_lens, incoming_lens], dim=0)
        )
        self._slot_ids_tensor = None

    @staticmethod
    def _cat_kv_pad(
        existing: torch.Tensor | None, incoming: torch.Tensor, t_max: int
    ) -> torch.Tensor:
        """Cat two [N, nkv, T_slot, head_dim] caches, zero-padding T_slot to t_max."""
        if existing is None:
            return (
                incoming
                if incoming.shape[-2] == t_max
                else F.pad(incoming, (0, 0, 0, t_max - incoming.shape[-2]))
            )
        pad_existing = (
            existing
            if existing.shape[-2] == t_max
            else F.pad(existing, (0, 0, 0, t_max - existing.shape[-2]))
        )
        pad_incoming = (
            incoming
            if incoming.shape[-2] == t_max
            else F.pad(incoming, (0, 0, 0, t_max - incoming.shape[-2]))
        )
        return torch.cat([pad_existing, pad_incoming], dim=0)

    def register(self, slot: MemorySlot) -> None:
        """Register a single slot (backward compatibility)."""
        self._slots_compat[slot.slot_id] = slot
        self.batch_register(
            torch.tensor([slot.slot_id], dtype=torch.long),
            slot.routing_key.unsqueeze(0),
            slot.k_cache.unsqueeze(0),
            slot.v_cache.unsqueeze(0),
        )

    def get(self, slot_id: int) -> MemorySlot:
        """Retrieve a slot by ID (backward compatibility)."""
        if slot_id in self._slots_compat:
            return self._slots_compat[slot_id]
        idx = self._id_to_idx.get(slot_id)
        if idx is None:
            raise KeyError(f"Slot {slot_id} not found in registry")
        return MemorySlot(
            slot_id=slot_id,
            domain_id=0,
            routing_key=(
                self._routing_keys[idx]
                if self._routing_keys is not None
                else torch.tensor([])
            ),
            k_cache=(
                self._k_caches[idx] if self._k_caches is not None else torch.tensor([])
            ),
            v_cache=(
                self._v_caches[idx] if self._v_caches is not None else torch.tensor([])
            ),
        )

    @torch.compiler.disable
    def get_indices(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Return indices [N] into batched tensors for the given slot IDs."""
        flat = slot_ids.long().flatten()
        idx_tensor = self._id_to_idx_tensor.to(flat.device)
        indices = idx_tensor[flat]
        if (indices < 0).any():
            bad = flat[indices < 0]
            raise KeyError(f"Slots {bad.tolist()} not found in registry")
        return indices.view_as(slot_ids)

    @torch.compiler.disable
    def keys_tensor(self, device: str | None = None) -> torch.Tensor:
        """Return a stacked tensor of all routing keys [N, key_dim]."""
        if self._routing_keys is None:
            raise RuntimeError("No slots registered")
        if device is not None:
            return self._routing_keys.to(device)
        return self._routing_keys

    def slot_ids(self) -> list[int]:
        """Return registered slot IDs in registration order."""
        return list(self._slot_ids)

    @torch.compiler.disable
    def slot_ids_tensor(self, device: str | None = None) -> torch.Tensor:
        """Return a cached tensor of slot IDs."""
        if self._slot_ids_tensor is None:
            # Plain CPU tensor; route_top_k immediately .to(device, non_blocking)
            # it, which already overlaps the H2D copy. pin_memory() here would
            # add nothing and is unsupported inside torch.compile (aten._pin_memory
            # is NYI for fake tensors), so it is omitted.
            self._slot_ids_tensor = torch.tensor(self._slot_ids, dtype=torch.long)
        if device is not None:
            return self._slot_ids_tensor.to(device, non_blocking=True)
        return self._slot_ids_tensor

    @torch.compiler.disable
    def clear(self) -> None:
        """Remove all slots and invalidate caches."""
        self._slot_ids.clear()
        self._id_to_idx.clear()
        self._routing_keys = None
        self._k_caches = None
        self._v_caches = None
        self._slot_lens = None
        self._slot_ids_tensor = None
        self._id_to_idx_tensor.fill_(-1)

    @torch.compiler.disable
    def __len__(self) -> int:
        return len(self._slot_ids)

    @torch.compiler.disable
    def get_kv(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K and V caches for the given indices.

        Args:
            indices: [N] tensor of indices into the batched caches.

        Returns:
            (k, v) where k and v are [N, nkv, T_slot, head_dim].
        """
        if self._k_caches is None or self._v_caches is None:
            raise RuntimeError("No K/V caches in registry")
        return self._k_caches[indices], self._v_caches[indices]

    @torch.compiler.disable
    def get_offsets(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the real per-slot K/V length (time dim) for the given indices.

        Slots can carry different real lengths (prefill chunk_size vs decode
        per-token) inside one padded dense [N, nkv, T_slot_max, head_dim] stack.
        The RoPE position base and the attention valid-len mask both use this
        real length, NOT the buffer width, so padded positions stay inert.
        """
        if self._k_caches is None:
            raise RuntimeError("No K caches in registry")
        if self._slot_lens is None:
            # Legacy fallback (registry populated before _slot_lens existed):
            # uniform T_slot, no padding.
            t_slot = self._k_caches.size(-2)
            return torch.full(
                indices.shape, t_slot, dtype=torch.long, device=indices.device
            )
        lens = self._slot_lens.to(device=indices.device)
        return lens[indices.long().flatten()].view_as(indices)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "slot_ids": self._slot_ids,
            "routing_keys": (
                self._routing_keys.cpu().tolist()
                if self._routing_keys is not None
                else []
            ),
            "k_caches": (
                self._k_caches.cpu().tolist() if self._k_caches is not None else []
            ),
            "v_caches": (
                self._v_caches.cpu().tolist() if self._v_caches is not None else []
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serializable state."""
        self._slot_ids = state.get("slot_ids", [])
        rk = state.get("routing_keys", [])
        self._routing_keys = torch.tensor(rk) if rk else None
        kc = state.get("k_caches", [])
        self._k_caches = torch.tensor(kc) if kc else None
        vc = state.get("v_caches", [])
        self._v_caches = torch.tensor(vc) if vc else None
        self._id_to_idx = {sid: i for i, sid in enumerate(self._slot_ids)}
        self._slot_ids_tensor = None


class SparseRouter(nn.Module):
    """Routes queries to the top-k most relevant memory slots.

    Computes dot-product scores between the query hidden state and slot routing
    keys, then returns the top-k slot IDs, raw scores, and softmax-normalised
    weights. When the registry exceeds ``lsh_threshold`` slots, an LSH
    (random-projection sign-hash) pre-filter narrows the candidate set before
    the exact top-k — sublinear retrieval as the store scales toward millions
    of slots. Below the threshold the exact matmul+topk path runs unchanged.
    """

    def __init__(
        self,
        hidden_size: int,
        key_dim: int | None = None,
        lsh_threshold: int = 0,
        n_hashes: int = 8,
        bucket_bits: int = 10,
        probe_buckets: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim if key_dim is not None else hidden_size
        self.query_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.lsh_threshold = int(lsh_threshold)
        self.n_hashes = int(n_hashes)
        self.bucket_bits = int(bucket_bits)
        self.probe_buckets = int(probe_buckets)
        # Random projection planes for LSH: n_hashes projections, each
        # bucket_bits wide. Non-persistent: deterministic, no checkpoint impact.
        if self.lsh_threshold > 0:
            proj = torch.randn(self.n_hashes * self.bucket_bits, self.key_dim)
            self.register_buffer("_lsh_proj", proj, persistent=False)

    @torch.no_grad()
    def _bucket_codes(self, keys: torch.Tensor) -> torch.Tensor:
        """Hash keys to per-hash bucket indices via sign of random projections.

        keys: [..., key_dim] -> bucket bits packed per hash. Returns
        [..., n_hashes] long bucket ids in [0, 2**bucket_bits).
        """
        lsh_proj = getattr(self, "_lsh_proj", None)
        assert isinstance(
            lsh_proj, torch.Tensor
        ), "LSH projection planes not initialised"
        proj = keys @ lsh_proj.transpose(-2, -1)  # [..., n_hashes*bits]
        proj = proj.view(*keys.shape[:-1], self.n_hashes, self.bucket_bits)
        bits = (proj > 0).long()
        weights = (
            1 << torch.arange(self.bucket_bits - 1, -1, -1, device=bits.device)
        ).long()
        return (bits * weights).sum(dim=-1)  # [..., n_hashes]

    def route_top_k_lsh(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        slot_ids: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LSH-filtered top-k. query [.., key_dim], keys [N, key_dim].

        Probe the candidate set = union of buckets closest to the query across
        ``n_hashes`` hashes (top ``probe_buckets`` per hash), then exact topk
        over only those candidates. Falls back to exact topk over all N when the
        LSH candidate set would be empty or smaller than top_k.
        """
        N = keys.size(0)
        q_codes = self._bucket_codes(query)  # [.., n_hashes]
        k_codes = self._bucket_codes(keys)  # [N, n_hashes] -> [n_hashes, N]
        # match[q, h, n] = (q's bucket on hash h == key n's bucket on hash h).
        # q_codes: [Q, n_hashes, 1]; k_codes transposed: [1, n_hashes, N].
        match = q_codes.unsqueeze(-1) == k_codes.transpose(-2, -1).unsqueeze(-3)
        cand_mask = match.any(dim=-2)  # [Q, N] bool

        # cand_mask is [Q, N]. Pad per-query candidate lists to a common length;
        # pad with -1 (filtered to -inf afterwards). cand_idx: [Q, max_cand].
        Q = cand_mask.size(0)
        cand_idx_list = [
            cand_mask[i].nonzero(as_tuple=False).squeeze(-1) for i in range(Q)
        ]
        maxc = max((c.numel() for c in cand_idx_list), default=0)
        cand_idx = torch.full((Q, maxc), -1, dtype=torch.long, device=query.device)
        for i, c in enumerate(cand_idx_list):
            if c.numel() > 0:
                cand_idx[i, : c.numel()] = c

        # Gather candidate scores: [Q, max_cand]. -1 idx -> gather index 0,
        # then masked to -inf via pad_mask.
        safe_idx = cand_idx.clamp(min=0)
        cand_keys = keys[safe_idx]  # [Q, max_cand, key_dim]
        cand_scores = torch.einsum("qd,qcd->qc", query, cand_keys)
        pad_mask = cand_idx < 0
        cand_scores = cand_scores.masked_fill(pad_mask, float("-inf"))

        top_k = min(top_k, N)
        tk_weights, tk_local = torch.topk(cand_scores, k=top_k, dim=-1, sorted=False)
        # Map local candidate index -> global slot index -> slot id.
        tk_global = safe_idx.gather(-1, tk_local)
        tk_global = torch.where(
            torch.isfinite(tk_weights), tk_global, torch.zeros_like(tk_global)
        )
        top_k_ids = slot_ids[tk_global]
        weights = F.softmax(tk_weights, dim=-1)
        return top_k_ids, tk_weights, weights

    def load_balance_loss(
        self,
        scores: torch.Tensor,
        top_k_indices: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Shazeer/Switch load-balance loss over the full routing distribution.

        L = alpha * N * sum_s( f_s * P_s ), where f_s is the (detached) fraction
        of queries whose top-k selected slot s, and P_s is the mean full-softmax
        probability of slot s. f_s carries no gradient (detached counter); P_s is
        differentiable through query_proj AND the routing keys (registry), so this
        is what actually trains the router. Minimum = alpha * top_k at uniform use.
        Mirror of moe.py MoESwiGLU aux loss.

        scores: [B, T, N] or [B, N] dot-product logits (pre-topk).
        top_k_indices: [..., top_k] long indices into N (from torch.topk).
        """
        n = scores.size(-1)
        probs = F.softmax(scores, dim=-1)  # full distribution over all N slots
        flat_probs = probs.reshape(-1, n)  # [BT, N]
        mean_prob = flat_probs.mean(dim=0)  # [N]
        flat_idx = top_k_indices.reshape(-1, top_k_indices.size(-1))  # [BT, top_k]
        one_hot = torch.zeros(
            flat_idx.size(0),
            n,
            device=scores.device,
            dtype=mean_prob.dtype,
        )
        one_hot.scatter_(1, flat_idx, 1.0)
        fraction = one_hot.mean(dim=0).detach()  # [N], detached counter
        # Normalize by top_k (NOT n): the registry slot count n is VARIADIC —
        # memory-aware HRM grows the registry every l_cycle (N -> 2N), so the
        # Shazeer `alpha * N * sum(f_s * P_s)` term would scale with n and the
        # loss would double across cycles / grow as slots accumulate, dragging
        # grad_norm up (logged L_msa_lb 8.5 -> 9.6 over steps 25-75). top_k is
        # fixed, so the loss minimum stays alpha*top_k at uniform use regardless
        # of how many slots are registered. The gradient signal (push toward
        # uniform slot use) is unchanged; only the spurious scale drift is gone.
        return alpha * float(top_k_indices.size(-1)) * (fraction * mean_prob).sum()

    @torch.compiler.disable
    def route_top_k(
        self,
        query_hidden: torch.Tensor,
        registry: SlotRegistry,
        top_k: int,
        compute_lb: bool = False,
        lb_alpha: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return top-k slot IDs, scores, weights, and optional load-balance loss.

        Args:
            query_hidden: [B, T, hidden_size] or [B, hidden_size].
            registry: slot registry with pre-registered routing keys.
            top_k: number of slots to select.
            compute_lb: when True (training), also return the load-balance aux
                loss so the router receives a gradient from the LM loss. The
                full-softmax term is differentiable through query_proj and the
                registry routing keys.

        Returns:
            top_k_ids: [B, T, top_k] or [B, top_k] long tensor of slot IDs.
            scores:    same shape as top_k_ids, raw dot-product scores.
            weights:   same shape, softmax-normalised weights.
            lb_loss:   scalar load-balance loss when compute_lb, else None.

        ``@torch.compiler.disable``: the routing keys tensor is [N, key_dim]
        with VARIADIC N — memory-aware HRM grows the registry every l_cycle
        (0 -> N -> 2N), so Dynamo would specialize a guard on N and hit
        recompile_limit(8) -> eager fallback (the recompile storm logged as
        `_fetch_kv_from_slots` / route matmul size mismatch). The matmul over a
        variable-N key set recompiles regardless, so eager here is strictly
        cheaper than the storm. ``query_proj`` (the only trained weight) still
        gets its gradient — disable stops graph capture, not autograd.
        """
        if len(registry) == 0:
            raise RuntimeError("Cannot route: registry is empty")

        query = self.query_proj(query_hidden)  # [B, T, key_dim] or [B, key_dim]
        keys = registry.keys_tensor(device=str(query.device))  # [N, key_dim]

        # Move keys to query device/dtype if needed
        if keys.device != query.device or keys.dtype != query.dtype:
            keys = keys.to(device=query.device, dtype=query.dtype)

        slot_ids = registry.slot_ids_tensor(device=str(query.device))

        # Cosine routing: L2-normalize query and keys before the dot product so
        # scores lie in [-1, 1] and are invariant to the hidden-state magnitude.
        # The raw dot product `query @ keys.T` makes the load-balance loss scale
        # with ||h||: as hidden drifts/low-rank-concentrates during training the
        # routing keys correlate, softmax peaks onto a few slots, and L_msa_lb
        # climbs monotonically (5 -> 54 over ~500 steps in the rtx3070 run).
        # Normalising keeps the router's load-balance loss at its floor
        # (alpha * top_k) regardless of magnitude drift — verified offline
        # (raw: 60-71, cosine: 5.0). eps matches the bfloat16 floor.
        query = F.normalize(query, dim=-1, eps=1e-8)
        keys = F.normalize(keys, dim=-1, eps=1e-8)

        # LSH path: only when enabled AND registry is large enough to benefit.
        # (Exact matmul+topk is faster and exact for small N — the ANN overhead
        # pays off only once the candidate set is much smaller than N.)
        if (
            self.lsh_threshold > 0
            and len(registry) >= self.lsh_threshold
            and query.shape[-1] == self.key_dim
        ):
            orig_shape = query.shape
            q_flat = query.reshape(-1, self.key_dim)
            top_k_ids, top_k_weights, weights = self.route_top_k_lsh(
                q_flat, keys, slot_ids, top_k
            )
            top_k_ids = top_k_ids.reshape(*orig_shape[:-1], top_k)
            top_k_weights = top_k_weights.reshape(*orig_shape[:-1], top_k)
            weights = weights.reshape(*orig_shape[:-1], top_k)
            # LSH path has no full scores; compute lb from an exact pass if asked.
            lb_loss = None
            if compute_lb:
                full_scores = torch.matmul(
                    query.reshape(-1, self.key_dim), keys.transpose(-2, -1)
                )
                flat_tk = torch.topk(full_scores, k=top_k, dim=-1, sorted=False).indices
                lb_loss = self.load_balance_loss(
                    full_scores.reshape(*query.shape[:-1], -1),
                    flat_tk.reshape(*query.shape[:-1], top_k),
                    lb_alpha,
                )
            return top_k_ids, top_k_weights, weights, lb_loss

        # Dot-product scores: [B, T, N] or [B, N]
        scores = torch.matmul(query, keys.transpose(-2, -1))

        top_k = min(top_k, scores.size(-1))
        top_k_weights, top_k_indices = torch.topk(scores, k=top_k, dim=-1, sorted=False)
        weights = F.softmax(top_k_weights, dim=-1)

        # Map indices back to actual slot IDs (cached tensor)
        top_k_ids = slot_ids[top_k_indices]

        lb_loss = None
        if compute_lb:
            lb_loss = self.load_balance_loss(scores, top_k_indices, lb_alpha)

        return top_k_ids, top_k_weights, weights, lb_loss


class DocumentWiseRoPE(nn.Module):
    """RoPE with per-slot document offsets.

    Each slot maintains its own position offset so that tokens from different
    memory domains do not share the same rotation angles.
    """

    _cos: torch.Tensor
    _sin: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        # Precompute cos/sin as non-persistent buffers. The prior Python dict
        # cache (_get_cache) stored tensors that CUDAGraphs captured and then
        # overwrote on replay, causing "accessing tensor output of CUDAGraphs
        # that has been overwritten by a subsequent run" under torch.compile.
        # Buffers are part of the module's state and are properly managed by
        # CUDAGraphs — no stale references.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
        t = torch.arange(end=max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("_cos", freqs.cos(), persistent=False)
        self.register_buffer("_sin", freqs.sin(), persistent=False)

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
        seq_len = self.max_seq_len
        # Cast to input dtype — creates a fresh tensor when dtypes differ, or
        # returns the buffer itself when already matching. Either way, the
        # buffer is not a CUDAGraph output, so it cannot be stale.
        cos = self._cos.to(x.dtype)
        sin = self._sin.to(x.dtype)

        # slot_offsets must be [B, T]
        if slot_offsets.dim() == 3:
            slot_offsets = slot_offsets[..., 0]
        if slot_offsets.dim() != 2:
            raise ValueError(f"slot_offsets must be [B, T], got {slot_offsets.shape}")

        positions = slot_offsets.unsqueeze(1)  # [B, 1, T]
        positions = positions + torch.arange(end=T, device=x.device).view(1, 1, T)
        pos_indices = positions.long().clamp(0, seq_len - 1)
        pos_indices = pos_indices.expand(B, H, T)  # [B, H, T]

        cos_idx = cos[pos_indices]  # [B, H, T, D//2]
        sin_idx = sin[pos_indices]  # [B, H, T, D//2]

        x1, x2 = x[..., 0::2], x[..., 1::2]
        rx1 = x1 * cos_idx - x2 * sin_idx
        rx2 = x1 * sin_idx + x2 * cos_idx
        return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class HostKvCache:
    """Append-only K/V cache for memory slots.

    Contract **CacheAppendOnly**: tokens are appended via ``append()``, never
    overwritten in-place.  Pre-allocates a buffer to avoid O(n²) memory copies.
    """

    def __init__(
        self,
        k_cache: torch.Tensor | MemorySlot,
        v_cache: torch.Tensor | None = None,
        max_len: int = 4096,
    ) -> None:
        self.max_len = max_len
        self._slot: MemorySlot | None = None
        if isinstance(k_cache, MemorySlot):
            self._slot = k_cache
            k_cache = self._slot.k_cache
            v_cache = self._slot.v_cache
        assert v_cache is not None
        nkv, _, head_dim = k_cache.shape
        self._k = torch.empty(
            nkv,
            max_len,
            head_dim,
            dtype=k_cache.dtype,
            device=k_cache.device,
        )
        self._v = torch.empty(
            nkv,
            max_len,
            head_dim,
            dtype=v_cache.dtype,
            device=v_cache.device,
        )
        self._len = k_cache.size(-2)
        self._k[..., : self._len, :] = k_cache
        self._v[..., : self._len, :] = v_cache

    @property
    def k(self) -> torch.Tensor:
        return self._k[..., : self._len, :]

    @property
    def v(self) -> torch.Tensor:
        return self._v[..., : self._len, :]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append new K/V tokens to the slot cache.

        Args:
            k_new: [..., new_len, head_dim] key tokens.
            v_new: [..., new_len, head_dim] value tokens.
        """
        new_len = k_new.size(-2)
        if self._len + new_len > self.max_len:
            raise RuntimeError(
                f"cache overflow: {self._len + new_len} > {self.max_len}"
            )
        self._k[..., self._len : self._len + new_len, :] = k_new
        self._v[..., self._len : self._len + new_len, :] = v_new
        self._len += new_len
        if self._slot is not None:
            self._slot.k_cache = self.k
            self._slot.v_cache = self.v


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
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_query_heads
        )
        assert (
            num_query_heads % num_kv_heads == 0
        ), "query heads must be divisible by kv heads"

        def _make(i: int, o: int) -> nn.Module:
            if use_binary_factorized:
                return BinaryFactorizedLinear(i, o, binary_factorized_rank)
            return nn.Linear(i, o, bias=False)

        self.q_proj = _make(hidden_size, num_query_heads * self.head_dim)
        if not use_binary_factorized:
            self.kv_proj = nn.Linear(
                hidden_size, 2 * num_kv_heads * self.head_dim, bias=False
            )
        else:
            self.k_proj = _make(hidden_size, num_kv_heads * self.head_dim)
            self.v_proj = _make(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = _make(num_query_heads * self.head_dim, hidden_size)

        self.rope = DocumentWiseRoPE(self.head_dim, max_seq_len, rope_theta)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if hasattr(self, "kv_proj"):
            kv = self.kv_proj.weight
            k, v = kv.split(self.num_kv_heads * self.head_dim, dim=0)
            state[f"{prefix}k_proj.weight"] = k
            state[f"{prefix}v_proj.weight"] = v
            state.pop(f"{prefix}kv_proj.weight", None)
        return state

    @torch.compiler.disable
    def _fetch_kv_from_slots(
        self,
        slot_ids: torch.Tensor,
        registry: SlotRegistry,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch K, V, and offsets for the selected slots.

        Deduplicates unique slot IDs to avoid O(B*T*top_k) redundant copies.

        ``@torch.compiler.disable``: torch.unique returns a VARIADIC-size tensor
        (unique slot count depends on the registry contents), and memory-aware
        HRM grows the registry every l_cycle, so the unique-count changes
        between forwards (0 -> N -> 2N across cycles). Dynamo specializes a
        guard on that size and hits recompile_limit(8) -> eager fallback (wasted
        compile, 68% gpu_util vs 90%). Disabling makes Dynamo treat this as an
        opaque Python side-effect (the SlotRegistry methods it calls are already
        disabled); the heavy tensor ops (matmul/softmax attention) stay in the
        compiled ``forward``.
        """
        B = slot_ids.size(0)
        top_k = slot_ids.size(-1)
        is_3d = slot_ids.dim() == 3
        T = slot_ids.size(1) if is_3d else 1

        device = slot_ids.device
        flat_ids = slot_ids.flatten().long()
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)

        # Fetch unique caches once using vectorised indices
        unique_indices = registry.get_indices(unique_ids)
        unique_k, unique_v = registry.get_kv(unique_indices.to(device))
        if unique_k.device != device:
            unique_k = unique_k.to(device)
            unique_v = unique_v.to(device)
        unique_offsets = registry.get_offsets(unique_indices.to(device))

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

        q = (
            self.q_proj(hidden_states)
            .view(B, T, self.num_query_heads, self.head_dim)
            .transpose(1, 2)
        )
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
        scale = self.head_dim**0.5

        if is_3d:
            # GQA: broadcast KV heads instead of repeat_interleave (zero-copy)
            # k_all: [B, T, top_k, nkv, T_slot, head_dim] -> [B, nq, T, top_k, T_slot, head_dim]
            kk = (
                k_all.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1, -1, -1)
                .reshape(B, T, top_k, self.num_query_heads, -1, self.head_dim)
                .permute(0, 3, 1, 2, 4, 5)
            )
            # v_all: [B, T, top_k, nkv, T_slot, head_dim] -> [B, nq, T, top_k, T_slot, head_dim]
            vk = (
                v_all.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1, -1, -1)
                .reshape(B, T, top_k, self.num_query_heads, -1, self.head_dim)
                .permute(0, 3, 1, 2, 4, 5)
            )

            qk = q.unsqueeze(3)  # [B, nq, T, 1, head_dim]
            qk_mat = qk.reshape(B * self.num_query_heads * T, 1, self.head_dim)
            kk_mat = kk.reshape(
                B * self.num_query_heads * T, top_k * kk.size(4), self.head_dim
            )
            scores = torch.matmul(qk_mat, kk_mat.transpose(-2, -1)) / scale
            scores = scores.view(B, self.num_query_heads, T, top_k, kk.size(4))
            # Valid-len mask: padded K positions (>= real slot length) get -inf
            # so mixed-T_slot generation (prefill chunk_size vs decode per-token)
            # does not bleed attention mass onto zero-padded tail rows. Training
            # has uniform T_slot == real_len, so the mask is all-True (no-op).
            T_slot = kk.size(4)
            if T_slot > 1:
                # offsets: [B, T, top_k] -> broadcast against scores [B, nq, T, top_k, T_slot]
                pos = torch.arange(T_slot, device=scores.device)
                valid = pos[None, None, None, None, :] < offsets[:, None, :, :, None]
                scores = scores.masked_fill(~valid, float("-inf"))
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1).to(q.dtype)
            if nars_weights is not None:
                w = nars_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, top_k, 1]
                attn = attn * w
            attn_mat = attn.reshape(B * self.num_query_heads * T, 1, top_k * kk.size(4))
            vk_mat = vk.reshape(
                B * self.num_query_heads * T, top_k * vk.size(4), self.head_dim
            )
            out_k = torch.matmul(attn_mat, vk_mat)  # [B*nq*T, 1, head_dim]
            out_k = out_k.view(B, self.num_query_heads, T, self.head_dim)
        else:
            # GQA: broadcast KV heads instead of repeat_interleave
            # k_all: [B, top_k, nkv, T_slot, head_dim] -> [B, nq, top_k, T_slot, head_dim]
            kk = (
                k_all.unsqueeze(2)
                .expand(-1, -1, rep, -1, -1, -1)
                .reshape(B, top_k, self.num_query_heads, -1, self.head_dim)
                .permute(0, 2, 1, 3, 4)
            )
            vk = (
                v_all.unsqueeze(2)
                .expand(-1, -1, rep, -1, -1, -1)
                .reshape(B, top_k, self.num_query_heads, -1, self.head_dim)
                .permute(0, 2, 1, 3, 4)
            )

            qk = q.unsqueeze(2)  # [B, nq, 1, T, head_dim]
            qk_mat = qk.reshape(B * self.num_query_heads, T, self.head_dim)
            kk_mat = kk.reshape(
                B * self.num_query_heads, top_k * kk.size(3), self.head_dim
            )
            scores = torch.matmul(qk_mat, kk_mat.transpose(-2, -1)) / scale
            scores = scores.view(B, self.num_query_heads, T, top_k, kk.size(3))
            scores = scores.permute(0, 1, 3, 2, 4)  # [B, nq, top_k, T, T_slot]
            # Valid-len mask (mirror of is_3d branch): pad positions >= real
            # slot length get -inf. No-op when all slots share T_slot == lens.
            T_slot = kk.size(3)
            if T_slot > 1:
                pos = torch.arange(T_slot, device=scores.device)
                valid = pos[None, None, None, None, :] < offsets[:, None, :, None, None]
                scores = scores.masked_fill(~valid, float("-inf"))
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1).to(q.dtype)
            if nars_weights is not None:
                w = (
                    nars_weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # [B, 1, top_k, 1, 1]
                attn = attn * w
            attn_mat = attn.permute(0, 1, 3, 2, 4).reshape(
                B * self.num_query_heads * T, 1, top_k * kk.size(3)
            )
            vk_mat = (
                vk.reshape(B * self.num_query_heads, top_k * vk.size(3), self.head_dim)
                .unsqueeze(1)
                .expand(-1, T, -1, -1)
                .reshape(
                    B * self.num_query_heads * T, top_k * vk.size(3), self.head_dim
                )
            )
            out_k = torch.matmul(attn_mat, vk_mat)  # [B*nq*T, 1, head_dim]
            out_k = out_k.squeeze(1)  # [B*nq*T, head_dim]
            out_k = out_k.view(B, self.num_query_heads, T, self.head_dim)

        out = out_k.transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(out)
        return out


class HDIMSlotRouter(nn.Module):
    """HDIM-aware slot router that derives routing keys from Clifford invariants.

    Each slot corresponds to a separate HDIM domain; the routing key is the
    scalar (grade-0) invariant extracted via the inner product of the domain
    rotor with the hidden state.
    """

    def __init__(
        self, hidden_size: int, blade_count: int = BLADE_COUNT, key_dim: int = 64
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.blade_count = blade_count
        self.key_dim = key_dim
        self.hidden_to_mv = nn.Linear(hidden_size, blade_count, bias=False)
        self.key_proj = nn.Linear(blade_count, key_dim, bias=False)

    def routing_key(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute scalar Clifford invariant from hidden states.

        Args:
            hidden_states: [B, T, hidden_size] or [hidden_size].

        Returns:
            [B, T, key_dim] or [key_dim] tensor of invariant values.
        """
        mv = self.hidden_to_mv(hidden_states)
        # Project multivector to key_dim
        return self.key_proj(mv)

    def batch_create_slots(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_id_base: int = 0,
        domain_id: int = 0,
        chunk_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create slot tensors for all positions in a batch.

        Args:
            hidden_states: [B, T, hidden_size].
            k_cache: [B, nkv, T, head_dim].
            v_cache: [B, nkv, T, head_dim].
            slot_id_base: starting slot ID.
            domain_id: domain ID for all slots.
            chunk_size: group ``chunk_size`` adjacent tokens into ONE slot by
                mean-pooling their K/V and routing key. ``chunk_size=1`` (default)
                reproduces the legacy per-token slots (T_slot=1). ``chunk_size>1``
                yields compressed slots (T_slot=chunk_size): each slot summarizes
                that many source tokens, so the registry holds ``B*T/chunk_size``
                slots and represents ``chunk_size``x more source context at the
                same max_slots budget. A trailing partial chunk (< chunk_size
                tokens) is dropped to keep fixed T_slot.

        Returns:
            (slot_ids, routing_keys, k_caches, v_caches) where:
            - slot_ids: [N] tensor (N = B*T//chunk_size)
            - routing_keys: [N, key_dim]
            - k_caches: [N, nkv, chunk_size, head_dim]
            - v_caches: [N, nkv, chunk_size, head_dim]
        """
        B, T, _ = hidden_states.shape
        # chunk_size<=1 -> per-token slots. Also fall back to per-token when
        # T < chunk_size (single-token decode step): n_full_chunks would be 0
        # and the chunked reshape below raises on a 0-element tensor. Pooling
        # the available tokens as per-token slots keeps generation working.
        if chunk_size <= 1 or T < chunk_size:
            inv = self.routing_key(hidden_states)  # [B, T, key_dim]
            inv_flat = inv.reshape(B * T, -1)
            total = B * T
            k_t = k_cache.transpose(1, 2)  # [B, T, nkv, head_dim]
            v_t = v_cache.transpose(1, 2)
            k_flat = k_t.reshape(total, k_cache.size(1), k_cache.size(-1)).unsqueeze(2)
            v_flat = v_t.reshape(total, v_cache.size(1), v_cache.size(-1)).unsqueeze(2)
            slot_ids = torch.arange(
                slot_id_base,
                slot_id_base + total,
                dtype=torch.long,
                device=hidden_states.device,
            )
            return slot_ids, inv_flat, k_flat.detach(), v_flat.detach()

        # Chunked (compressed) path: pool every `chunk_size` tokens into a slot.
        n_full_chunks = T // chunk_size  # trailing partial chunk dropped
        T_use = n_full_chunks * chunk_size
        inv = self.routing_key(hidden_states[:, :T_use])  # [B, T_use, key_dim]
        k_use = k_cache[:, :, :T_use]  # [B, nkv, T_use, head_dim]
        v_use = v_cache[:, :, :T_use]
        nkv, head_dim = k_cache.size(1), k_cache.size(-1)

        # Reshape to chunks and mean-pool: [B, n_chunks, chunk_size, ...] -> mean
        # routing key: [B, T_use, key_dim] -> [B, n_chunks, chunk_size, key_dim]
        inv_chunked = inv.view(B, n_full_chunks, chunk_size, -1).mean(dim=2)
        inv_flat = inv_chunked.reshape(B * n_full_chunks, -1)

        # K/V: [B, nkv, T_use, head_dim] -> [B, nkv, n_chunks, chunk_size, head_dim]
        k_chunked = k_use.view(B, nkv, n_full_chunks, chunk_size, head_dim)
        v_chunked = v_use.view(B, nkv, n_full_chunks, chunk_size, head_dim)
        # -> [B, n_chunks, nkv, chunk_size, head_dim] -> [B*n_chunks, nkv, chunk_size, head_dim]
        k_flat = k_chunked.permute(0, 2, 1, 3, 4).reshape(
            B * n_full_chunks, nkv, chunk_size, head_dim
        )
        v_flat = v_chunked.permute(0, 2, 1, 3, 4).reshape(
            B * n_full_chunks, nkv, chunk_size, head_dim
        )

        total = B * n_full_chunks
        slot_ids = torch.arange(
            slot_id_base,
            slot_id_base + total,
            dtype=torch.long,
            device=hidden_states.device,
        )
        return slot_ids, inv_flat, k_flat.detach(), v_flat.detach()

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


class MSAMemory:
    """Functional bridge binding MSA components for memory-aware HRM.

    Memory-aware HRM makes the slot registry part of the reasoning loop instead
    of a post-reasoning block: each l_cycle READs memory (retrieval -> sparse
    attention over the registry accumulated so far) and WRITEs memory (registers
    the current hidden state's K/V as slots). This holds no parameters — it
    borrows the model's existing ``msa`` / ``msa_router`` / ``hdim_slot_router``
    nn.Modules so the checkpoint state_dict is unchanged and the routers stay
    shared (one set of trained routing weights, read and written identically).

    Contract:
      - ``registry`` is owned by the caller (HRMCore). It persists across the
        l_cycles within one forward and is cleared at forward start, so cycle
        k>0 reads what cycle k-1 wrote — the intra-forward memory link.
      - ``read`` is a no-op (returns zero) when the registry is empty, so the
        first cycle (no prior write) is just reasoning + write.
      - ``write`` always registers the current hidden as slots; the registry's
        ``max_slots`` cap bounds growth.
      - ``slot_id_base`` is threaded by the caller so successive writes get
        disjoint slot ids (the registry dedups by id, but disjoint ids avoid
        the in-place overwrite path and keep the eviction order meaningful).

    Returns from ``read``: (msa_out, lb_loss, slot_ids, scores) where lb_loss is
    None unless ``compute_lb`` and the registry is non-empty.
    """

    def __init__(
        self,
        msa: MSAAttention,
        msa_router: SparseRouter,
        hdim_slot_router: HDIMSlotRouter,
        cfg: Any,
    ):
        self.msa = msa
        self.msa_router = msa_router
        self.hdim_slot_router = hdim_slot_router
        self.top_k = int(getattr(cfg, "msa_top_k", 5))
        self.chunk_size = int(getattr(cfg, "msa_chunk_size", 1))
        self.aux_loss = bool(getattr(cfg, "msa_aux_loss", True))
        self.lb_alpha = float(getattr(cfg, "msa_lb_alpha", 1.0))
        self.use_nars = bool(getattr(cfg, "use_nars", False))
        self.nars_msa: Any = None  # set by the model when use_nars=true

    def read(
        self,
        hidden_states: torch.Tensor,
        registry: SlotRegistry,
        training_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Read memory: route + sparse-attend over the current registry.

        Returns (msa_out [B,T,H], lb_loss | None, slot_ids | None, scores | None).
        When the registry is empty returns (zeros, None, None, None) so the first
        cycle has no memory to read and HRM proceeds to reasoning + write.
        """
        h = hidden_states
        b, t, _ = h.shape
        if len(registry) == 0:
            return torch.zeros_like(h), None, None, None

        if self.use_nars and self.nars_msa is not None:
            with torch.no_grad():
                query_hidden = h.mean(dim=(0, 1))  # [H] pooled for NARS routing
                # NARS path returns [B=1] ids/scores; expand to [B, T, top_k].
                top_k_ids, top_values = self.nars_msa.route_top_k_with_nars(
                    registry, query_hidden, self.top_k
                )
                slot_ids = top_k_ids.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                scores = top_values.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                nars_weights = self.nars_msa.compute_attention_weights(slot_ids)
                lb_loss = None
        else:
            slot_ids, _raw_scores, weights, lb = self.msa_router.route_top_k(
                h,
                registry,
                self.top_k,
                compute_lb=self.aux_loss and training_mode,
                lb_alpha=self.lb_alpha,
            )
            scores = weights
            nars_weights = None
            lb_loss = lb

        msa_out = self.msa(h, slot_ids, registry, nars_weights=nars_weights)
        return msa_out, lb_loss, slot_ids, scores

    def write(
        self,
        hidden_states: torch.Tensor,
        registry: SlotRegistry,
        slot_id_base: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write memory: project the hidden to K/V and register fresh slots.

        Returns (slot_ids [N], routing_keys [N, key_dim]) so the caller can
        thread the next ``slot_id_base`` (base + N) for disjoint ids.
        """
        h = hidden_states
        b, t, _ = h.shape
        nkv = self.msa.num_kv_heads
        head_dim = self.msa.head_dim

        if hasattr(self.msa, "kv_proj"):
            kv = (
                self.msa.kv_proj(h).view(b, t, 2 * nkv, head_dim).transpose(1, 2)
            )
            k, v = kv.split(nkv, dim=1)
        else:
            k = self.msa.k_proj(h).view(b, t, nkv, head_dim).transpose(1, 2)
            v = self.msa.v_proj(h).view(b, t, nkv, head_dim).transpose(1, 2)

        slot_ids, routing_keys, k_caches, v_caches = (
            self.hdim_slot_router.batch_create_slots(
                hidden_states=h,
                k_cache=k,
                v_cache=v,
                slot_id_base=slot_id_base,
                domain_id=0,
                chunk_size=self.chunk_size,
            )
        )
        registry.batch_register(slot_ids, routing_keys, k_caches, v_caches)
        return slot_ids, routing_keys
