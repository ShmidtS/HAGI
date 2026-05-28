import Mathlib

namespace HAGI
namespace NARS

/-- NARS truth value with frequency and confidence in the unit interval. -/
structure TruthValue where
  frequency : ℝ
  confidence : ℝ

/-- Truth values are valid when both components are bounded by `[0, 1]`. -/
def valid_truth (tv : TruthValue) : Prop :=
  0 ≤ tv.frequency ∧ tv.frequency ≤ 1 ∧ 0 ≤ tv.confidence ∧ tv.confidence ≤ 1

/-- Confidence-weighted truth revision. -/
noncomputable def revision (tv1 tv2 : TruthValue) : TruthValue :=
  let w1 := tv1.confidence
  let w2 := tv2.confidence
  if w1 + w2 = 0 then
    { frequency := 0.5, confidence := 0.0 }
  else
    { frequency := (tv1.frequency * w1 + tv2.frequency * w2) / (w1 + w2)
      confidence := (w1 + w2) / (1 + w1 + w2) }

/-- Confidence-weighted revision keeps truth components in the unit interval. -/
theorem nars_truth_revision_bounded (tv1 tv2 : TruthValue) :
    valid_truth tv1 → valid_truth tv2 → valid_truth (revision tv1 tv2) := by
  intro h1 h2
  rcases h1 with ⟨hf1_nonneg, hf1_le, hw1_nonneg, hw1_le⟩
  rcases h2 with ⟨hf2_nonneg, hf2_le, hw2_nonneg, hw2_le⟩
  unfold revision valid_truth
  by_cases hzero : tv1.confidence + tv2.confidence = 0
  · simp [hzero]
    norm_num
  · simp [hzero]
    have hsum_nonneg : 0 ≤ tv1.confidence + tv2.confidence := by positivity
    have hsum_pos : 0 < tv1.confidence + tv2.confidence :=
      lt_of_le_of_ne hsum_nonneg (Ne.symm hzero)
    have hden_pos : 0 < 1 + tv1.confidence + tv2.confidence := by linarith
    have hnum_nonneg : 0 ≤ tv1.frequency * tv1.confidence + tv2.frequency * tv2.confidence := by
      positivity
    have hnum_le : tv1.frequency * tv1.confidence + tv2.frequency * tv2.confidence ≤
        tv1.confidence + tv2.confidence := by
      have hmul1 : tv1.frequency * tv1.confidence ≤ 1 * tv1.confidence :=
        mul_le_mul_of_nonneg_right hf1_le hw1_nonneg
      have hmul2 : tv2.frequency * tv2.confidence ≤ 1 * tv2.confidence :=
        mul_le_mul_of_nonneg_right hf2_le hw2_nonneg
      nlinarith
    constructor
    · exact div_nonneg hnum_nonneg (le_of_lt hsum_pos)
    constructor
    · calc
        (tv1.frequency * tv1.confidence + tv2.frequency * tv2.confidence) /
            (tv1.confidence + tv2.confidence)
            ≤ (tv1.confidence + tv2.confidence) / (tv1.confidence + tv2.confidence) := by
              exact div_le_div_of_nonneg_right hnum_le (le_of_lt hsum_pos)
        _ = 1 := by exact div_self (ne_of_gt hsum_pos)
    constructor
    · exact div_nonneg hsum_nonneg (le_of_lt hden_pos)
    · calc
        (tv1.confidence + tv2.confidence) / (1 + tv1.confidence + tv2.confidence)
            ≤ (1 + tv1.confidence + tv2.confidence) /
                (1 + tv1.confidence + tv2.confidence) := by
              apply div_le_div_of_nonneg_right
              · linarith
              · exact le_of_lt hden_pos
        _ = 1 := by exact div_self (ne_of_gt hden_pos)

/-- Two zero-confidence inputs revise to the explicit neutral fallback truth value. -/
theorem nars_truth_revision_neutral (tv1 tv2 : TruthValue) :
    (tv1.confidence = 0 ∧ tv2.confidence = 0) →
      (revision tv1 tv2).frequency = 0.5 ∧ (revision tv1 tv2).confidence = 0.0 := by
  intro h
  rcases h with ⟨h1, h2⟩
  unfold revision
  simp [h1, h2]

private theorem confidence_revision_ge_half_left {c1 c2 : ℝ}
    (hc1_nonneg : 0 ≤ c1) (hc1_le : c1 ≤ 1) (hc2_nonneg : 0 ≤ c2) :
    c1 / 2 ≤ (c1 + c2) / (1 + c1 + c2) := by
  have hs_nonneg : 0 ≤ c1 + c2 := by linarith
  have hden_pos : 0 < 1 + c1 + c2 := by linarith
  have htwo_pos : (0 : ℝ) < 2 := by norm_num
  have hmul_le : c1 * (c1 + c2) ≤ c1 + c2 := by
    have := mul_le_mul_of_nonneg_right hc1_le hs_nonneg
    nlinarith
  have hmain : c1 * (1 + c1 + c2) ≤ 2 * (c1 + c2) := by
    nlinarith
  field_simp [ne_of_gt htwo_pos, ne_of_gt hden_pos]
  nlinarith

/-- Revised confidence is at least half the larger source confidence. -/
theorem nars_truth_revision_confidence_increase (tv1 tv2 : TruthValue) :
    valid_truth tv1 → valid_truth tv2 →
      (revision tv1 tv2).confidence ≥ max tv1.confidence tv2.confidence / 2 := by
  intro h1 h2
  rcases h1 with ⟨_, _, hw1_nonneg, hw1_le⟩
  rcases h2 with ⟨_, _, hw2_nonneg, hw2_le⟩
  unfold revision
  by_cases hzero : tv1.confidence + tv2.confidence = 0
  · have hw1_zero : tv1.confidence = 0 := by linarith
    have hw2_zero : tv2.confidence = 0 := by linarith
    simp [hw1_zero, hw2_zero]
    norm_num
  · simp [hzero]
    have hleft : tv1.confidence / 2 ≤
        (tv1.confidence + tv2.confidence) / (1 + tv1.confidence + tv2.confidence) :=
      confidence_revision_ge_half_left hw1_nonneg hw1_le hw2_nonneg
    have hright : tv2.confidence / 2 ≤
        (tv1.confidence + tv2.confidence) / (1 + tv1.confidence + tv2.confidence) := by
      have hright' : tv2.confidence / 2 ≤
          (tv2.confidence + tv1.confidence) / (1 + tv2.confidence + tv1.confidence) :=
        confidence_revision_ge_half_left hw2_nonneg hw2_le hw1_nonneg
      convert hright' using 1
      ring
    by_cases hle : tv1.confidence ≤ tv2.confidence
    · rw [max_eq_right hle]
      exact hright
    · have hle' : tv2.confidence ≤ tv1.confidence := le_of_not_ge hle
      rw [max_eq_left hle']
      exact hleft

/-- NARS budget value with priority, durability, and quality. -/
structure BudgetValue where
  priority : ℝ
  durability : ℝ
  quality : ℝ

/-- Budget values are valid when all components are bounded by `[0, 1]`. -/
def valid_budget (b : BudgetValue) : Prop :=
  0 ≤ b.priority ∧ b.priority ≤ 1 ∧
  0 ≤ b.durability ∧ b.durability ≤ 1 ∧
  0 ≤ b.quality ∧ b.quality ≤ 1

/-- Budget decay scales priority and durability while preserving quality. -/
def decay (b : BudgetValue) (factor : ℝ) : BudgetValue :=
  { priority := b.priority * factor
    durability := b.durability * factor
    quality := b.quality }

/-- Decay by a factor in `[0, 1]` cannot increase priority or durability. -/
theorem nars_budget_decay_monotone (b : BudgetValue) (factor : ℝ) :
    valid_budget b → 0 ≤ factor → factor ≤ 1 →
      (decay b factor).priority ≤ b.priority ∧
        (decay b factor).durability ≤ b.durability := by
  intro hb hfactor_nonneg hfactor_le
  rcases hb with ⟨hp_nonneg, _, hd_nonneg, _, _, _⟩
  unfold decay
  constructor
  · have := mul_le_mul_of_nonneg_left hfactor_le hp_nonneg
    nlinarith
  · have := mul_le_mul_of_nonneg_left hfactor_le hd_nonneg
    nlinarith

/-- Budget merge keeps the componentwise maximum budget. -/
def merge (b1 b2 : BudgetValue) : BudgetValue :=
  { priority := max b1.priority b2.priority
    durability := max b1.durability b2.durability
    quality := max b1.quality b2.quality }

/-- Componentwise max merge is monotone in every budget component. -/
theorem nars_budget_merge_monotone (b1 b2 : BudgetValue) :
    valid_budget b1 → valid_budget b2 →
      (merge b1 b2).priority ≥ b1.priority ∧
      (merge b1 b2).priority ≥ b2.priority ∧
      (merge b1 b2).durability ≥ b1.durability ∧
      (merge b1 b2).durability ≥ b2.durability ∧
      (merge b1 b2).quality ≥ b1.quality ∧
      (merge b1 b2).quality ≥ b2.quality := by
  intro _ _
  unfold merge
  constructor
  · exact le_max_left _ _
  constructor
  · exact le_max_right _ _
  constructor
  · exact le_max_left _ _
  constructor
  · exact le_max_right _ _
  constructor
  · exact le_max_left _ _
  · exact le_max_right _ _

/-- Simplified priority bag represented by `(item, priority)` pairs. -/
def Bag (α : Type) := List (α × ℝ)

/-- A bag is bounded when its list length fits the configured capacity. -/
def bounded {α : Type} (bag : Bag α) (capacity : ℕ) : Prop :=
  bag.length ≤ capacity

/-- Insert an item, sort by descending priority, and truncate to capacity. -/
noncomputable def insert_with_overflow {α : Type} (bag : Bag α) (item : α)
    (priority : ℝ) (capacity : ℕ) : Bag α :=
  (((item, priority) :: bag).mergeSort (fun a b => a.2 ≥ b.2)).take capacity

/-- Insertion with overflow preserves the configured bag capacity. -/
theorem nars_bag_insert_preserves_capacity {α : Type} (bag : Bag α) (item : α)
    (priority : ℝ) (capacity : ℕ) :
    0 ≤ priority → priority ≤ 1 → bounded bag capacity →
      bounded (insert_with_overflow bag item priority capacity) capacity := by
  intro _ _ _
  unfold bounded insert_with_overflow
  exact List.length_take_le _ _

end NARS
end HAGI
