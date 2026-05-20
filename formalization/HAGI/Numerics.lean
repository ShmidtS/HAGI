import Mathlib

namespace HAGI

/-- Approximate equality for mathematical real values within epsilon. -/
def ApproxEq (ε : Real) (x y : Real) : Prop :=
  |x - y| < ε

/-- Scalar operations needed for approximate numeric reasoning. -/
class ApproxScalar (α : Type) where
  zero : α
  one  : α
  add  : α → α → α
  mul  : α → α → α

instance : ApproxScalar Real where
  zero := 0
  one  := 1
  add  := (· + ·)
  mul  := (· * ·)

/-- Real addition commutes exactly, hence within any positive epsilon. -/
theorem approx_add_comm (ε : Real) (hε : 0 < ε) (a b : Real)
    : ApproxEq ε (a + b) (b + a) := by
  rw [ApproxEq, add_comm]
  simpa using hε

/-- Real multiplication associates exactly, hence within any positive epsilon. -/
theorem approx_mul_assoc (ε : Real) (hε : 0 < ε) (a b c : Real)
    : ApproxEq ε ((a * b) * c) (a * (b * c)) := by
  rw [ApproxEq, mul_assoc]
  simpa using hε

/-- Float bridge predicate for runtime claims against the Real model. -/
def FloatApprox (ε : Real) (observed expected : Real) : Prop :=
  ApproxEq ε observed expected

end HAGI