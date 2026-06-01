import HAGI.CoreTypes
import HAGI.HDIM

/-!
Grade-Decomposed Recurrence (GDR) contract.

HAGI's per-iteration recurrence splits the hidden state into Clifford grades
(scalar / vector / bivector / trivector / residual) with distinct update
dynamics, then mixes the vector grade via the Cl(3,0,0) geometric product
and projects the result back into scalar and bivector channels.

This file mirrors the Python implementation in `src/hagi/model/gdr.py`.
-/

namespace HAGI
namespace GDR

open CoreTypes
open HDIM

/-- Blade count for Cl(3,0,0). Must equal `2^3`. -/
def BLADE_COUNT : Nat := 8

theorem blade_count_eq : BLADE_COUNT = 2 ^ 3 := rfl

/-- Grade-Decomposed Recurrence configuration. -/
structure GradeConfig where
  scalar : Nat
  vector : Nat
  bivector : Nat
  trivector : Nat
  residual : Nat
  scalarMomentum : Float
  vectorMomentum : Float
  scalar_pos : scalar > 0
  vector_pos : vector > 0
  bivector_pos : bivector > 0
  trivector_pos : trivector > 0
  residual_pos : residual > 0
  vector_div_blades : vector % BLADE_COUNT = 0
  scalarMom_le_one : scalarMomentum ≤ 1.0
  scalarMom_nonneg : scalarMomentum ≥ 0.0
  vectorMom_le_one : vectorMomentum ≤ 1.0
  vectorMom_nonneg : vectorMomentum ≥ 0.0

/-- Total hidden size (sum of all grade widths). -/
def GradeConfig.hiddenSize (cfg : GradeConfig) : Nat :=
  cfg.scalar + cfg.vector + cfg.bivector + cfg.trivector + cfg.residual

/-- Context width: all grades except the residual channel. -/
def GradeConfig.contextSize (cfg : GradeConfig) : Nat :=
  cfg.scalar + cfg.vector + cfg.bivector + cfg.trivector

/-- Boundary indices into the hidden state. Six entries for five slices. -/
def GradeConfig.bounds (cfg : GradeConfig) : List Nat :=
  let s := cfg.scalar
  let v := cfg.vector
  let b := cfg.bivector
  let t := cfg.trivector
  let r := cfg.residual
  [0, s, s + v, s + v + b, s + v + b + t, s + v + b + t + r]

/-- `bounds` always has exactly 6 entries. -/
theorem bounds_length (cfg : GradeConfig) : cfg.bounds.length = 6 := by
  unfold GradeConfig.bounds
  simp

/-- Hidden size equals the last boundary. -/
theorem hiddenSize_eq_last_bound (cfg : GradeConfig) :
    cfg.hiddenSize = cfg.bounds[cfg.bounds.length - 1]! := by
  unfold GradeConfig.hiddenSize GradeConfig.bounds
  simp

/-- Context size equals the second-to-last boundary. -/
theorem contextSize_eq_fifth_bound (cfg : GradeConfig) :
    cfg.contextSize = cfg.bounds[4]! := by
  unfold GradeConfig.contextSize GradeConfig.bounds
  simp

/-- Number of structural multivector heads packed into the vector grade. -/
def GradeConfig.numMultivectors (cfg : GradeConfig) : Nat :=
  cfg.vector / BLADE_COUNT

/-- Number of structural heads is positive. -/
theorem numMultivectors_pos (cfg : GradeConfig) : cfg.numMultivectors > 0 := by
  unfold GradeConfig.numMultivectors
  have h1 : cfg.vector > 0 := cfg.vector_pos
  exact Nat.div_pos h1 (by decide)

/-- Number of multivectors times blade count equals the vector grade width. -/
theorem numMultivectors_times_blades (cfg : GradeConfig) :
    cfg.numMultivectors * BLADE_COUNT = cfg.vector := by
  unfold GradeConfig.numMultivectors
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero cfg.vector_div_blades)

/-- Boundaries form a strictly increasing sequence of length 6. -/
theorem bounds_strictlyIncreasing (cfg : GradeConfig) :
    ∀ i : Nat, i + 1 < cfg.bounds.length →
      cfg.bounds[i]! < cfg.bounds[i + 1]! := by
  intro i hi
  unfold GradeConfig.bounds at hi ⊢
  simp at hi
  rcases hi with h1 | h2 | h3 | h4 | h5
  · -- i = 0: 0 < scalar
    simp
    exact cfg.scalar_pos
  · -- i = 1: scalar < scalar + vector
    simp
    linarith [cfg.scalar_pos, cfg.vector_pos : cfg.vector > 0]
  · -- i = 2: scalar + vector < scalar + vector + bivector
    simp
    linarith [cfg.bivector_pos : cfg.bivector > 0]
  · -- i = 3: ... + bivector < ... + bivector + trivector
    simp
    linarith [cfg.trivector_pos : cfg.trivector > 0]
  · -- i = 4: ... + trivector < ... + trivector + residual
    simp
    linarith [cfg.residual_pos : cfg.residual > 0]

/-- Sum of consecutive boundary differences equals the slice widths.

`scalar = bounds[1] - bounds[0]`, `vector = bounds[2] - bounds[1]`, etc. -/
theorem bounds_diff_scalar (cfg : GradeConfig) :
    cfg.bounds[1]! - cfg.bounds[0]! = cfg.scalar := by
  unfold GradeConfig.bounds
  simp

theorem bounds_diff_vector (cfg : GradeConfig) :
    cfg.bounds[2]! - cfg.bounds[1]! = cfg.vector := by
  unfold GradeConfig.bounds
  simp
  linarith

theorem bounds_diff_bivector (cfg : GradeConfig) :
    cfg.bounds[3]! - cfg.bounds[2]! = cfg.bivector := by
  unfold GradeConfig.bounds
  simp
  linarith

theorem bounds_diff_trivector (cfg : GradeConfig) :
    cfg.bounds[4]! - cfg.bounds[3]! = cfg.trivector := by
  unfold GradeConfig.bounds
  simp
  linarith

theorem bounds_diff_residual (cfg : GradeConfig) :
    cfg.bounds[5]! - cfg.bounds[4]! = cfg.residual := by
  unfold GradeConfig.bounds
  simp
  linarith

/-- The Cl(3,0,0) signature used by GDR. -/
def gdrSignature : CliffordSignature := { p := 3, q := 0, r := 0 }

/-- Number of blades in the GDR Clifford algebra. -/
theorem gdr_blade_count : gdrSignature.dim = 3 := rfl

/-- The blade count from the signature matches the structural constant. -/
theorem gdr_signature_blades : 2 ^ gdrSignature.dim = BLADE_COUNT := by
  unfold gdrSignature BLADE_COUNT
  rfl

/-- Vector grade width equals `n_mv * 2^dim(Cl(3,0,0))`. -/
theorem vector_equals_nmv_times_blades (cfg : GradeConfig) :
    cfg.vector = cfg.numMultivectors * (2 ^ gdrSignature.dim) := by
  rw [gdr_signature_blades]
  exact (numMultivectors_times_blades cfg).symm

/-- Geometric product table on Cl(3,0,0) blades. Anchors the GDR
`geometric_interaction` function: for blade ids `a, b`, the table returns
a `(sign, result_blade_id)` pair. -/
def gdrGeometricProduct : BasisBlade gdrSignature.dim →
                         BasisBlade gdrSignature.dim →
                         (Float × BasisBlade gdrSignature.dim) :=
  geometricProductTable gdrSignature

/-- GDR's `geometric_interaction` reduces a vector of length
`n_mv * BLADE_COUNT` into two signals of width `scalar` and `bivector`. -/
structure GDROperation (cfg : GradeConfig) where
  apply : TensorSpec → TensorSpec
  preservesShape : ∀ t, SameShape (apply t) t
  preservesDim : ∀ t, t.shape.dims[0]! = cfg.hiddenSize →
    (apply t).shape.dims[0]! = cfg.hiddenSize

/-- Split a hidden state into five per-grade tensors. -/
structure GradeSplit (cfg : GradeConfig) where
  scalar : TensorSpec
  vector : TensorSpec
  bivector : TensorSpec
  trivector : TensorSpec
  residual : TensorSpec
  scalarDim : scalar.shape.dims[0]! = cfg.scalar
  vectorDim : vector.shape.dims[0]! = cfg.vector
  bivectorDim : bivector.shape.dims[0]! = cfg.bivector
  trivectorDim : trivector.shape.dims[0]! = cfg.trivector
  residualDim : residual.shape.dims[0]! = cfg.residual

/-- Widths across the five slices sum to the hidden size. -/
theorem GradeSplit.total_width {cfg : GradeConfig} (s : GradeSplit cfg) :
    s.scalar.shape.dims[0]! + s.vector.shape.dims[0]! +
      s.bivector.shape.dims[0]! + s.trivector.shape.dims[0]! +
        s.residual.shape.dims[0]! = cfg.hiddenSize := by
  unfold GradeConfig.hiddenSize
  rw [s.scalarDim, s.vectorDim, s.bivectorDim, s.trivectorDim, s.residualDim]
  rfl

end GDR
end HAGI
