# NARS -- Non-Axiomatic Reasoning System

HAGI includes optional NARS (Non-Axiomatic Reasoning System) controllers that bridge OpenNARS-style truth revision, budget allocation, and bag-based concept selection to the HAGI training loop.

**NARS is disabled by default** (`use_nars: false`). The integration is experimental.

---

## Overview

NARS provides a framework for adaptive control of reasoning systems under bounded resources. HAGI uses it to dynamically adjust:

| Component | NARS Controller | What it adjusts |
|-----------|----------------|-----------------|
| HRM | `NarsHrmController` | H_cycles, L_cycles, convergence_eps, bp_steps |
| HDIM | `NarsHdimReasoner` | Domain rotor selection, transfer gating |
| MSA | `NarsMsaReasoner` | Slot routing priority, top_k adaptation |

---

## Core Concepts

### Truth Values

A `TruthValue` is a `(frequency, confidence)` tuple:

- **Frequency** (f): proportion of positive evidence, in [0, 1]
- **Confidence** (c): total amount of evidence, in [0, 1]

Truth revision combines multiple truth values into a consensus:

```python
def truth_revision(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    w1 = f1 * c1 * (1 - c1) + f2 * c2 * (1 - c2)
    w2 = c1 * (1 - c1) + c2 * (1 - c2)
    f_new = w1 / w2 if w2 > 0 else 0.5
    c_new = w2 / (w2 + 1) if w2 > 0 else 0.0
    return TruthValue(f_new, c_new)
```

Source: `src/hagi/nars/truth.py`

### Budget Values

A `BudgetValue` is a `(priority, durability, quality)` tuple:

- **Priority**: immediate relevance, decays over time
- **Durability**: resistance to decay
- **Quality**: long-term usefulness

Budget decay reduces priority each cycle:

```python
def budget_decay(bv: BudgetValue, dt: float = 1.0) -> BudgetValue:
    return BudgetValue(
        priority=bv.priority * (1 - dt / (bv.durability + 1)),
        durability=bv.durability,
        quality=bv.quality,
    )
```

Source: `src/hagi/nars/budget.py`

### Bag

A `Bag[T]` is a priority-queue with probabilistic selection. Items are selected proportional to their priority, not deterministically. This provides exploration vs. exploitation in concept selection.

Source: `src/hagi/nars/bag.py`

---

## HRM Controller

`NarsHrmController` observes training signals and resolves HRM control policies.

### Observation

The controller observes:
- Training loss (windowed average)
- Gradient norms (windowed average)

### Control Concepts

Control concepts accumulate truth values via revision:

| Concept | Condition | Truth Value |
|---------|-----------|-------------|
| `loss_low` | loss < threshold | (1.0, confidence) |
| `grad_stable` | grad_norm < threshold | (1.0, confidence) |

### Policy Bag

A bag of `HrmControlPolicy` candidates ranked by the average truth strength of the current control state:

```python
@dataclass(frozen=True, slots=True)
class HrmControlPolicy:
    h_cycles: int
    l_cycles: int
    convergence_eps: float
    bp_steps: int
```

When `loss_low` and `grad_stable` are both confident, the controller selects policies with fewer cycles (efficiency). When they're not, it selects policies with more cycles (depth).

Source: `src/hagi/nars/adapters.py`

---

## HDIM Reasoner

`NarsHdimReasoner` adapts domain rotor selection and transfer gating based on the truth values of domain-invariant quality.

### Domain Quality

Each domain (reasoning iteration) accumulates a truth value for "how well the invariant was extracted." High-quality domains are preferred for transfer; low-quality domains are skipped.

### Rotor Selection

Instead of the LCG-based deterministic rotor schedule, NARS selects rotors probabilistically based on their accumulated truth values. Rotors that produce better invariants (measured by L_iso) get higher priority.

---

## MSA Reasoner

`NarsMsaReasoner` adapts slot routing priority and top_k based on memory utility.

### Slot Utility

Each MSA slot accumulates a budget value. Slots that are frequently selected by the router and contribute to attention output get higher priority. Slots that are never selected decay.

### Adaptive Top_k

NARS can dynamically adjust top_k per query: queries with high-confidence routing get fewer slots (the router knows where to look), while low-confidence queries get more slots (broader search).

---

## Integration with Training Loop

When `use_nars: true`:

1. Before each step, the controllers observe the previous step's metrics.
2. The HRM controller resolves a control policy (H_cycles, L_cycles, etc.).
3. The HDIM reasoner adjusts rotor selection.
4. The MSA reasoner adjusts routing.
5. After the step, truth values and budgets are revised with the new observations.

The controllers are designed to be **non-blocking**: they run on CPU and do not add GPU synchronization points.

---

## Configuration

```yaml
model:
  use_nars: false  # Enable NARS controllers (experimental)
```

When disabled, NARS modules are not created and standard routing is used:
- HRM: fixed H_cycles and L_cycles from config
- HDIM: LCG-based deterministic rotor schedule
- MSA: fixed top_k from config

---

## Current Status

The NARS integration is **experimental and disabled by default**. The controllers are implemented and functional, but their impact on training quality has not been systematically evaluated. Enabling NARS adds CPU overhead for truth revision and bag management, but no GPU overhead.

Future work:
- Systematic ablation: NARS vs fixed scheduling
- Truth value calibration: verify that control concepts track training quality
- Budget decay tuning: ensure slots/policies don't starve or saturate
