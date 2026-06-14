"""NARS adapters for HAGI components.

Bridges OpenNARS-style truth revision, budget, and bag mechanisms to the
HAGI training loop (HRM controller), HDIM domain transfer, and MSA slot
routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hagi.nars.bag import Bag
from hagi.nars.budget import BudgetValue, budget_decay
from hagi.nars.truth import TruthValue, truth_revision

DomainId = int

# Singleton defaults to avoid repeated dataclass construction overhead
_DEFAULT_TV_00 = TruthValue(0.5, 0.0)


@dataclass(frozen=True, slots=True)
class HrmControlPolicy:
    h_cycles: int
    l_cycles: int
    convergence_eps: float
    bp_steps: int


class NarsHrmController:
    """Observes training loss / gradient norms and resolves HRM control policies.

    Control concepts (``loss_low``, ``grad_stable``) accumulate truth values via
    revision.  A policy bag of capacity 10 stores candidate policies ranked by
    the average truth strength of the current control state.
    """

    def __init__(self, policy_capacity: int = 10, max_observations: int = 1000) -> None:
        self.observed_losses: list[float] = []
        self.observed_grad_norms: list[float] = []
        self.control_budgets: dict[str, BudgetValue] = {}
        self.control_truths: dict[str, TruthValue] = {}
        self._policy_capacity = policy_capacity
        self._policy_bag: Bag[HrmControlPolicy] = Bag()
        self._policy_counter: int = 0
        self._max_obs = max_observations

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_train_step(self, loss: float, grad_norm: float) -> None:
        """Record a training step and revise control concept truths."""
        self.observed_losses.append(loss)
        self.observed_grad_norms.append(grad_norm)
        if len(self.observed_losses) > self._max_obs:
            self.observed_losses = self.observed_losses[-self._max_obs :]
            self.observed_grad_norms = self.observed_grad_norms[-self._max_obs :]

        # Loss concept: lower loss → higher frequency
        loss_freq = 1.0 / (1.0 + loss)
        loss_truth = TruthValue(loss_freq, 0.9)
        self.control_truths["loss_low"] = truth_revision(
            self.control_truths.get("loss_low", _DEFAULT_TV_00),
            loss_truth,
        )

        # Gradient concept: lower norm → higher frequency
        grad_freq = 1.0 / (1.0 + grad_norm)
        grad_truth = TruthValue(grad_freq, 0.9)
        self.control_truths["grad_stable"] = truth_revision(
            self.control_truths.get("grad_stable", _DEFAULT_TV_00),
            grad_truth,
        )

        # Decay all existing control budgets
        for name in list(self.control_budgets.keys()):
            self.control_budgets[name] = budget_decay(self.control_budgets[name], 0.95)

        self.control_budgets.setdefault("loss_low", BudgetValue(0.5, 0.5, 0.5))
        self.control_budgets.setdefault("grad_stable", BudgetValue(0.5, 0.5, 0.5))

        # Generate a fresh policy and inject into the bag
        policy = self._build_policy()
        self._inject_policy(policy)

    # ------------------------------------------------------------------
    # Policy resolution
    # ------------------------------------------------------------------

    def _build_policy(self) -> HrmControlPolicy:
        """Derive a control policy from the current control truths."""
        loss_tv = self.control_truths.get("loss_low", _DEFAULT_TV_00)
        grad_tv = self.control_truths.get("grad_stable", _DEFAULT_TV_00)

        h_cycles = max(1, min(5, int(2 + (1.0 - loss_tv.frequency) * 3)))
        l_cycles = max(1, min(5, int(2 + (1.0 - grad_tv.frequency) * 3)))
        convergence_eps = 1e-5 if loss_tv.frequency < 0.5 else 1e-7
        bp_steps = max(1, min(5, int(1 + grad_tv.frequency * 4)))

        return HrmControlPolicy(h_cycles, l_cycles, convergence_eps, bp_steps)

    def _inject_policy(self, policy: HrmControlPolicy) -> None:
        """Insert a policy into the bag, keeping capacity."""
        if not self.control_truths:
            priority = 0.5
        else:
            avg_freq = sum(tv.frequency for tv in self.control_truths.values()) / len(
                self.control_truths
            )
            avg_conf = sum(tv.confidence for tv in self.control_truths.values()) / len(
                self.control_truths
            )
            priority = (avg_freq + avg_conf) / 2.0

        while len(self._policy_bag) >= self._policy_capacity:
            self._policy_bag.take()

        self._policy_bag.put(policy, priority)

    def resolve_policy(self) -> HrmControlPolicy:
        """Return the highest-priority policy from the bag.

        Falls back to a freshly built policy if the bag is empty.
        """
        if not self._policy_bag:
            return self._build_policy()

        best_key = min(
            self._policy_bag.items,
            key=lambda name: (
                -self._policy_bag._priorities[name],
                self._policy_bag._sequence[name],
                name,
            ),
        )
        return self._policy_bag.items[best_key]

    # ------------------------------------------------------------------
    # Policy application
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "observed_losses": self.observed_losses,
            "observed_grad_norms": self.observed_grad_norms,
            "control_budgets": {
                k: {
                    "priority": v.priority,
                    "durability": v.durability,
                    "quality": v.quality,
                }
                for k, v in self.control_budgets.items()
            },
            "control_truths": {
                k: {"frequency": v.frequency, "confidence": v.confidence}
                for k, v in self.control_truths.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serializable state."""
        self.observed_losses = state.get("observed_losses", [])
        self.observed_grad_norms = state.get("observed_grad_norms", [])
        self.control_budgets = {
            k: BudgetValue(v["priority"], v["durability"], v["quality"])
            for k, v in state.get("control_budgets", {}).items()
        }
        self.control_truths = {
            k: TruthValue(v["frequency"], v["confidence"])
            for k, v in state.get("control_truths", {}).items()
        }

    def apply_policy(self, policy: HrmControlPolicy, hrm_config: Any) -> None:
        """Apply a resolved policy to an HRM config (object or dict).

        Only adapts runtime-safe scalar knobs (convergence_eps, bp_steps). The
        HRM cycle counts (h_cycles/l_cycles) are deliberately NOT mutated here:
        h_cycles drives conditional module construction (h_transition exists
        only when h_cycles>1 at init), so overriding it post-init either crashes
        the forward pass or changes the model's math per-step and destabilises
        training. Cycles stay at their config values; NARS still adapts via the
        scalar knobs and via compute_gating.
        """
        if isinstance(hrm_config, dict):
            hrm_config["convergence_eps"] = policy.convergence_eps
            hrm_config["bp_steps"] = policy.bp_steps
        else:
            for attr, value in (
                ("convergence_eps", policy.convergence_eps),
                ("bp_steps", policy.bp_steps),
            ):
                if hasattr(hrm_config, attr):
                    setattr(hrm_config, attr, value)

    def compute_gating(
        self, z_H: torch.Tensor, z_L: torch.Tensor
    ) -> tuple[float, float]:
        """Return truth-weighted gating coefficients for z_H and z_L.

        Default gate 1.0 — HRM always contributes fully. NARS observation
        can still dampen if training is highly unstable, but minimum is 0.5.
        """
        loss_tv = self.control_truths.get("loss_low", _DEFAULT_TV_00)
        grad_tv = self.control_truths.get("grad_stable", _DEFAULT_TV_00)
        # Combined truth strength: high = stable training -> full gate
        h_gate = max(1.0, float(loss_tv.frequency * grad_tv.frequency))
        l_gate = max(1.0, float(grad_tv.frequency))
        return h_gate, l_gate


class NarsHdimReasoner:
    """NARS-based reasoner for HDIM domain transfer.

    Maintains beliefs about which target domains are effective for each source
    domain and updates them via truth revision.
    """

    def __init__(self) -> None:
        self.domain_concepts: dict[DomainId, TruthValue] = {}
        self.transfer_beliefs: dict[tuple[DomainId, DomainId], TruthValue] = {}

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "domain_concepts": {
                k: {"frequency": v.frequency, "confidence": v.confidence}
                for k, v in self.domain_concepts.items()
            },
            "transfer_beliefs": {
                f"{k[0]},{k[1]}": {"frequency": v.frequency, "confidence": v.confidence}
                for k, v in self.transfer_beliefs.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serializable state."""
        self.domain_concepts = {
            int(k): TruthValue(v["frequency"], v["confidence"])
            for k, v in state.get("domain_concepts", {}).items()
        }
        self.transfer_beliefs = {}
        for k_str, v in state.get("transfer_beliefs", {}).items():
            parts = k_str.split(",")
            if len(parts) == 2:
                key = (int(parts[0]), int(parts[1]))
                self.transfer_beliefs[key] = TruthValue(v["frequency"], v["confidence"])


class NarsMsaReasoner:
    """NARS-based reasoner for MSA slot routing.

    Blends raw dot-product scores with NARS truth frequencies and recency
    weights to select top-k memory slots.
    """

    def __init__(self) -> None:
        self.slot_beliefs: dict[int, TruthValue] = {}
        self.slot_budgets: dict[int, BudgetValue] = {}
        self.recency_weights: dict[int, float] = {}

    def route_top_k_with_nars(
        self,
        registry: Any,
        query: Any,
        top_k: int,
        nars_weight: float = 0.6,
        recency_weight: float = 0.3,
        dot_weight: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend NARS belief, recency, and dot-product to route a query.

        Returns:
            (top_k_ids, blended_scores) — both tensors on the query device.
        """
        if not hasattr(registry, "keys_tensor"):
            raise TypeError("registry must have a keys_tensor() method")
        if not hasattr(registry, "slot_ids"):
            raise TypeError("registry must have a slot_ids() method")

        query_t = (
            query
            if isinstance(query, torch.Tensor)
            else torch.tensor(query, dtype=torch.float32)
        )
        if query_t.dim() > 1:
            query_t = query_t.view(-1)
        device = str(query_t.device) if query_t.device.type != "cpu" else None
        keys = registry.keys_tensor(device=device)  # [N, key_dim]
        slot_ids = registry.slot_ids()  # [N]

        if query_t.shape[-1] != keys.shape[-1]:
            raise ValueError(
                f"Query dim {query_t.shape[-1]} != key dim {keys.shape[-1]}"
            )

        if keys.device != query_t.device or keys.dtype != query_t.dtype:
            keys = keys.to(device=query_t.device, dtype=query_t.dtype)

        # Normalised dot-product scores
        dot_scores = torch.matmul(keys, query_t)  # [N]
        dot_min = dot_scores.min()
        dot_max = dot_scores.max()
        dot_scores_norm = (dot_scores - dot_min) / (dot_max - dot_min + 1e-8)

        # Vectorized: build tensors for frequency, recency, and dot scores
        N = len(slot_ids)
        freq_list = [
            self.slot_beliefs.get(sid, _DEFAULT_TV_00).frequency for sid in slot_ids
        ]
        recency_list = [self.recency_weights.get(sid, 0.0) for sid in slot_ids]
        freq_tensor = torch.as_tensor(freq_list, dtype=torch.float32).to(query_t.device)
        recency_tensor = torch.as_tensor(recency_list, dtype=torch.float32).to(
            query_t.device
        )
        blended_tensor = (
            nars_weight * freq_tensor
            + recency_weight * recency_tensor
            + dot_weight * dot_scores_norm[:N]
        )
        top_k = min(top_k, N)
        top_values, top_indices = torch.topk(blended_tensor, k=top_k)
        top_k_ids = torch.index_select(
            torch.as_tensor(slot_ids, dtype=torch.long, device=query_t.device),
            0,
            top_indices,
        )

        return top_k_ids, top_values

    def compute_attention_weights(self, slot_ids: torch.Tensor) -> torch.Tensor | None:
        """Return truth-weighted attention weights for slot IDs.

        Args:
            slot_ids: [B, T, top_k] or [B, top_k] long tensor.

        Returns:
            Tensor of same shape with truth frequencies, or None if no beliefs.
        """
        if not self.slot_beliefs:
            return None
        ids_list = slot_ids.flatten().tolist()
        weights = [
            self.slot_beliefs.get(sid, _DEFAULT_TV_00).frequency for sid in ids_list
        ]
        return torch.as_tensor(
            weights, dtype=torch.float32, device=slot_ids.device
        ).view_as(slot_ids)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "slot_beliefs": {
                k: {"frequency": v.frequency, "confidence": v.confidence}
                for k, v in self.slot_beliefs.items()
            },
            "slot_budgets": {
                k: {
                    "priority": v.priority,
                    "durability": v.durability,
                    "quality": v.quality,
                }
                for k, v in self.slot_budgets.items()
            },
            "recency_weights": self.recency_weights,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from serializable state."""
        self.slot_beliefs = {
            int(k): TruthValue(v["frequency"], v["confidence"])
            for k, v in state.get("slot_beliefs", {}).items()
        }
        self.slot_budgets = {
            int(k): BudgetValue(v["priority"], v["durability"], v["quality"])
            for k, v in state.get("slot_budgets", {}).items()
        }
        self.recency_weights = {
            int(k): float(v) for k, v in state.get("recency_weights", {}).items()
        }
