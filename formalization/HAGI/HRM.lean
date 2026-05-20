import HAGI.CoreTypes

/-!
Formalization of the HRM layer.

HRM stores two states: high-level z_H and low-level z_L.
The Lean specification fixes shape preservation, PrefixLM mask legality, and partitioning
for packed sequence training.
-/

namespace HAGI
namespace HRM

open CoreTypes

/-- HRM recurrence state. -/
structure HRMState where
  z_H : TensorSpec
  z_L : TensorSpec
  sameShape : SameShape z_H z_L

/-- HRM cycle configuration. -/
structure HRMConfig where
  hCycles : Nat
  lCycles : Nat
  bpSteps : Nat
  bpWarmupRatio : Float
  hiddenSize : Nat
  numHeads : Nat
  maxSequenceLength : Nat
  validHeads : HeadDivisible hiddenSize numHeads
  positiveHCycles : hCycles > 0
  positiveLCycles : lCycles > 0
  maxSequenceLength_pos : maxSequenceLength > 0
  bpWarmupRatio_le_one : bpWarmupRatio ≤ 1.0
  bpWarmupRatio_nonneg : bpWarmupRatio ≥ 0.0

/-- Low-level transition. In the specification, the transition is abstract but preserves shape. -/
structure LTransition where
  apply : HRMState → HRMState
  preserves_z_H : ∀ s, SameShape (apply s).z_H s.z_H
  preserves_z_L : ∀ s, SameShape (apply s).z_L s.z_L

/-- High-level transition. -/
structure HTransition where
  apply : HRMState → HRMState
  preserves_z_H : ∀ s, SameShape (apply s).z_H s.z_H
  preserves_z_L : ∀ s, SameShape (apply s).z_L s.z_L

/-- One full HRM step: L cycle, then H cycle. -/
def step (l : LTransition) (h : HTransition) (s : HRMState) : HRMState :=
  h.apply (l.apply s)

/-- The full step preserves the shape of z_H. -/
theorem step_preserves_z_H (l : LTransition) (h : HTransition) (s : HRMState) :
    SameShape (step l h s).z_H s.z_H := by
  unfold step SameShape
  rw [h.preserves_z_H (l.apply s), l.preserves_z_H s]

/-- The full step preserves the shape of z_L. -/
theorem step_preserves_z_L (l : LTransition) (h : HTransition) (s : HRMState) :
    SameShape (step l h s).z_L s.z_L := by
  unfold step SameShape
  rw [h.preserves_z_L (l.apply s), l.preserves_z_L s]

/-- PrefixLM position type: prefix can see prefix bidirectionally, suffix is causal. -/
inductive Segment where
  | prefix
  | suffix
  deriving DecidableEq, Repr

/-- Packed sequence position. -/
structure PackedIndex where
  sequenceId : Nat
  position : Nat
  segment : Segment
  deriving Repr

/-- PrefixLM attention legality. -/
def PrefixLMLegal (q k : PackedIndex) : Prop :=
  q.sequenceId = k.sequenceId ∧
    match q.segment, k.segment with
    | Segment.prefix, Segment.prefix => True
    | Segment.prefix, Segment.suffix => False
    | Segment.suffix, Segment.prefix => True
    | Segment.suffix, Segment.suffix => k.position ≤ q.position

/-- A prefix token cannot attend to a suffix token. -/
theorem prefix_cannot_attend_suffix {q k : PackedIndex}
    (hq : q.segment = Segment.prefix) (hk : k.segment = Segment.suffix) :
    ¬ PrefixLMLegal q k := by
  intro h
  unfold PrefixLMLegal at h
  cases h with
  | intro _ hmask =>
    rw [hq, hk] at hmask
    exact hmask

/-- A suffix token can attend to a prefix token from the same sequence. -/
theorem suffix_can_attend_prefix_same_sequence {q k : PackedIndex}
    (hs : q.sequenceId = k.sequenceId)
    (hq : q.segment = Segment.suffix) (hk : k.segment = Segment.prefix) :
    PrefixLMLegal q k := by
  unfold PrefixLMLegal
  constructor
  · exact hs
  · rw [hq, hk]
    trivial

/-- Packed range defines the half-open interval [start, stop). -/
structure PackedRange (cfg : HRMConfig) where
  sequenceId : Nat
  start : Nat
  stop : Nat
  valid : start ≤ stop
  valid_in_sequence : stop ≤ cfg.maxSequenceLength
  deriving Repr

/-- Non-overlap of two ranges. -/
def Disjoint {cfg : HRMConfig} (a b : PackedRange cfg) : Prop :=
  a.stop ≤ b.start ∨ b.stop ≤ a.start

/-- Number of tokens represented by packed ranges. -/
def packedTokenCount {cfg : HRMConfig} (ranges : List (PackedRange cfg)) : Nat :=
  ranges.foldl (fun acc r => acc + (r.stop - r.start)) 0

/-- Partition invariant: ranges do not overlap and total packed tokens fit the sequence bound. -/
def PackedPartition {cfg : HRMConfig} (ranges : List (PackedRange cfg)) : Prop :=
  (∀ a ∈ ranges, ∀ b ∈ ranges, a.sequenceId ≠ b.sequenceId → Disjoint a b) ∧
    packedTokenCount ranges ≤ cfg.maxSequenceLength

/-- The empty set of packed ranges is valid. -/
theorem empty_packed_partition {cfg : HRMConfig} : PackedPartition ([] : List (PackedRange cfg)) := by
  constructor
  · intro a ha
    cases ha
  · simp [packedTokenCount]

/-- Packed partition token count respects max sequence length. -/
theorem packed_partition_total_tokens_le_max {cfg : HRMConfig} {ranges : List (PackedRange cfg)}
    (h : PackedPartition ranges) : packedTokenCount ranges ≤ cfg.maxSequenceLength :=
  h.right

/-- Cycle depth is monotone when one step is added. -/
def DepthMonotone (before after : CycleId) : Prop :=
  before.value ≤ after.value

/-- CycleId increment is always monotone. -/
theorem cycle_increment_monotone (c : CycleId) : DepthMonotone c ⟨c.value + 1⟩ := by
  unfold DepthMonotone
  exact Nat.le_succ c.value

end HRM
end HAGI
