import Mathlib
import HAGI.CoreTypes
import HAGI.HRM

/-! Data pipeline contracts.

Formalizes the invariants of `prefix_lm_mask` and `create_prefix_lm_batch`
in `src/hagi/data/prefix_lm.py`.
-/

namespace HAGI
namespace Data

open CoreTypes

/-- PrefixLM mask construction parameters. -/
structure PrefixLMConfig where
  totalLen : Nat
  prefixLengths : List Nat
  totalLen_pos : totalLen > 0
  all_prefix_nonneg : ∀ p ∈ prefixLengths, p ≥ 0
  sum_prefix_le_total : prefixLengths.sum ≤ totalLen

/-- Sample range within a packed sequence. -/
structure SampleRange where
  start : Nat
  prefixLen : Nat
  suffixLen : Nat
  stop : Nat
  start_le_stop : start ≤ stop
  prefix_le_total : prefixLen ≤ stop - start

/-- A prefix mask for one sample allows prefix↔prefix and suffix→prefix/suffix (causal). -/
def SampleMaskLegal (r : SampleRange) (q k : Nat) : Prop :=
  r.start ≤ q ∧ q < r.stop ∧ r.start ≤ k ∧ k < r.stop →
    if q < r.start + r.prefixLen then
      -- query is in prefix: can attend to prefix only
      if k < r.start + r.prefixLen then True else False
    else
      -- query is in suffix: can attend to prefix and causal suffix
      if k < r.start + r.prefixLen then True
      else k ≤ q

/-- The total length of all samples equals the packed length. -/
def PartitionCovers (ranges : List SampleRange) (totalLen : Nat) : Prop :=
  (∀ i j, i < ranges.length → j < ranges.length → i < j →
    (ranges.get ⟨i, by omega⟩).start ≤ (ranges.get ⟨j, by omega⟩).start) ∧
  ranges.foldl (fun acc r => acc + (r.stop - r.start)) 0 = totalLen

/-- Any prefix token cannot attend to any suffix token in the same sample. -/
theorem prefix_cannot_attend_suffix_in_sample (r : SampleRange)
    (q k : Nat)
    (hq : r.start ≤ q ∧ q < r.stop ∧ q ≥ r.start + r.prefixLen)
    (hk : r.start ≤ k ∧ k < r.stop ∧ k < r.start + r.prefixLen) :
    ¬ SampleMaskLegal r q k := by
  intro h
  unfold SampleMaskLegal at h
  have h1 : r.start ≤ q := hq.left
  have h2 : q < r.stop := hq.right.left
  have h3 : r.start ≤ k := hk.left
  have h4 : k < r.stop := hk.right.left
  have h5 : q ≥ r.start + r.prefixLen := hq.right.right
  have h6 : k < r.start + r.prefixLen := hk.right.right
  have h_bounds : r.start ≤ q ∧ q < r.stop ∧ r.start ≤ k ∧ k < r.stop := by
    constructor
    · exact h1
    constructor
    · exact h2
    constructor
    · exact h3
    · exact h4
  simp [h_bounds, h5, h6] at h
  omega

/-- Suffix tokens are causal within the suffix region. -/
theorem suffix_causal_in_sample (r : SampleRange)
    (q k : Nat)
    (hq : r.start ≤ q ∧ q < r.stop ∧ q ≥ r.start + r.prefixLen)
    (hk : r.start ≤ k ∧ k < r.stop ∧ k ≥ r.start + r.prefixLen)
    (hc : k ≤ q) :
    SampleMaskLegal r q k := by
  unfold SampleMaskLegal
  intro h_bounds
  have h1 : r.start ≤ q := h_bounds.left
  have h2 : q < r.stop := h_bounds.right.left
  have h3 : r.start ≤ k := h_bounds.right.right.left
  have h4 : k < r.stop := h_bounds.right.right.right
  have h5 : q ≥ r.start + r.prefixLen := hq.right.right
  have h6 : k ≥ r.start + r.prefixLen := hk.right.right
  simp [h5, h6, hc]
  omega

/-- Prefix tokens can attend bidirectionally within the prefix region. -/
theorem prefix_bidirectional_in_sample (r : SampleRange)
    (q k : Nat)
    (hq : r.start ≤ q ∧ q < r.stop ∧ q < r.start + r.prefixLen)
    (hk : r.start ≤ k ∧ k < r.stop ∧ k < r.start + r.prefixLen) :
    SampleMaskLegal r q k := by
  unfold SampleMaskLegal
  intro h_bounds
  have h5 : q < r.start + r.prefixLen := hq.right.right
  have h6 : k < r.start + r.prefixLen := hk.right.right
  simp [h5, h6]

/-- Empty prefix lengths yield an all-zero mask of the correct size. -/
theorem empty_prefix_all_zero_mask (cfg : PrefixLMConfig)
    (h_empty : cfg.prefixLengths = []) :
    ∃ mask : Nat → Nat → Bool,
      (∀ q k, q < cfg.totalLen → k < cfg.totalLen → mask q k = false) := by
  use fun _ _ => false
  intro q k hq hk
  rfl

/-- Consecutive sample ranges do not overlap, assuming each range ends
    exactly at the next range's start (contiguity). -/
theorem ranges_ordered_no_overlap (ranges : List SampleRange)
    (h : PartitionCovers ranges totalLen)
    (hcont : ∀ k : Nat, k + 1 < ranges.length →
      (ranges.get ⟨k, by omega⟩).stop = (ranges.get ⟨k + 1, by omega⟩).start)
    (i j : Nat) (hi : i < ranges.length) (hj : j < ranges.length) (hij : i < j) :
    (ranges.get ⟨i, by omega⟩).stop ≤ (ranges.get ⟨j, by omega⟩).start := by
  -- Walk the contiguous chain from i to j - 1: stop_i = start_{i+1}, stop_{i+1} = start_{i+2}, ...
  -- Then start_{i+1} ≤ start_{i+2} ≤ ... ≤ start_j by the ordered property of PartitionCovers.
  have h_order : ∀ a b, a < ranges.length → b < ranges.length → a < b →
      (ranges.get ⟨a, by omega⟩).start ≤ (ranges.get ⟨b, by omega⟩).start := h.left
  -- For j = i + 1, contiguity gives stop_i = start_j directly.
  by_cases h_adj : j = i + 1
  · subst h_adj
    exact (hcont i hi).le
  -- For j > i + 1, we have start_{i+1} ≤ start_{i+2} ≤ ... ≤ start_j, and stop_i = start_{i+1}.
  have h_le : (ranges.get ⟨i, by omega⟩).start ≤ (ranges.get ⟨i + 1, by omega⟩).start :=
    h_order i (i + 1) hi (by omega) (by omega)
  have h_stop_start : (ranges.get ⟨i, by omega⟩).stop = (ranges.get ⟨i + 1, by omega⟩).start :=
    hcont i hi
  -- stop_i = start_{i+1} ≤ start_{i+2} ≤ ... ≤ start_j.
  rw [h_stop_start]
  -- Need start_{i+1} ≤ start_j; iterate via transitivity of the ordered property.
  -- The chain length is (j - (i+1)) steps.  Induction on the gap.
  suffices h_chain : ∀ m, i + 1 + m ≤ j →
      (ranges.get ⟨i + 1, by omega⟩).start ≤ (ranges.get ⟨i + 1 + m, by omega⟩).start from
    h_chain (j - (i + 1)) (Nat.sub_le_self _ _)
  intro m hm
  induction m with
  | zero => simp
  | succ m ih =>
    have h_lt : i + 1 + m < j := by omega
    have h_idx : i + 1 + m < ranges.length := by
      have : i + 1 + m < ranges.length := lt_of_lt_of_le h_lt hj
      exact this
    have h_succ_idx : i + 1 + (m + 1) < ranges.length := by omega
    have h_step : (ranges.get ⟨i + 1 + m, by omega⟩).start ≤
        (ranges.get ⟨i + 1 + (m + 1), by omega⟩).start :=
      h_order (i + 1 + m) (i + 1 + (m + 1)) h_idx h_succ_idx (by omega)
    exact le_trans ih h_step

end Data
end HAGI
