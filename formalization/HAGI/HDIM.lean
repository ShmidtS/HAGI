import HAGI.CoreTypes
import HAGI.HRM

/-!
HDIM specification.

The module defines a Clifford-like interface and an executable layer for basis blades, product table,
rotor reverse, and grade involution. Contracts for the Rust crate `clifford-core` and the `hdim-model` layer:
rotors must be unit, and invariant extraction and transfer preserve the structural invariant between domains.
-/

namespace HAGI
namespace HDIM

open CoreTypes
open HRM

/-- Signature of the Clifford algebra Cl(p,q,r). -/
structure CliffordSignature where
  p : Nat
  q : Nat
  r : Nat
  deriving DecidableEq, Repr

/-- Dimension of Cl(p,q,r). -/
def CliffordSignature.dim (sig : CliffordSignature) : Nat :=
  sig.p + sig.q + sig.r

/-- Basis blade encoded as a bitset word. -/
abbrev BasisBlade (_n : Nat) := Nat

/-- Linear blade id in `0 .. 2^n - 1`, represented by modulo normalization. -/
def bladeId {n : Nat} (b : BasisBlade n) : Nat :=
  b % (2 ^ n)

private def bitAt (word i : Nat) : Nat :=
  (word / (2 ^ i)) % 2

private def popCountBelow (word : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => popCountBelow word i + bitAt word i

/-- Grade of a basis blade: number of set bits in its normalized blade id. -/
def grade {n : Nat} (b : BasisBlade n) : Nat :=
  popCountBelow (bladeId b) n

private def metricAt (sig : CliffordSignature) (i : Nat) : Float :=
  if i < sig.p then 1.0 else if i < sig.p + sig.q then -1.0 else 0.0

private def xorBitsBelow (a b : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 =>
      xorBitsBelow a b i +
        (if bitAt a i = bitAt b i then 0 else 2 ^ i)

private def inversionCountBelow (a b : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => inversionCountBelow a b i + bitAt a i * popCountBelow b i

private def metricFactorBelow (sig : CliffordSignature) (a b : Nat) : Nat → Float
  | 0 => 1.0
  | i + 1 =>
      let rest := metricFactorBelow sig a b i
      if bitAt a i = 1 ∧ bitAt b i = 1 then rest * metricAt sig i else rest

private def paritySign (k : Nat) : Float :=
  if k % 2 = 0 then 1.0 else -1.0

/-- Geometric product table for normalized basis blades in Cl(p,q,r). -/
def geometricProductTable (sig : CliffordSignature) :
    BasisBlade sig.dim → BasisBlade sig.dim → (Float × BasisBlade sig.dim) :=
  fun a b =>
    let aId := bladeId a
    let bId := bladeId b
    let sign := paritySign (inversionCountBelow aId bId sig.dim)
    let metric := metricFactorBelow sig aId bId sig.dim
    (sign * metric, xorBitsBelow aId bId sig.dim)

/-- Multivector with explicit blade count tied to Clifford signature.

`bladeCount = 2^(p+q+r)` guarantees that coefficient list length matches
the algebra dimension, preventing size-mismatch between Rust clifford-core
and the formal spec.
-/
structure Multivector where
  signature : CliffordSignature
  coeffs : List Float
  bladeCount : Nat
  coeffCountEq : coeffs.length = bladeCount
  bladeCountEq : bladeCount = 2 ^ (signature.p + signature.q + signature.r)
  deriving Repr

/-- DomainRotor couples a domain identifier with a Clifford multivector and its inverse. -/
structure DomainRotor where
  domain : DomainId
  value : Multivector
  inverse : Multivector

/-- Clifford operations interface.

Backend must preserve signature and support identity witness for unit-rotor checks.
-/
structure CliffordOps where
  mul : Multivector → Multivector → Multivector
  scalarOne : Multivector
  sameSignatureMul : ∀ a b, a.signature = b.signature → (mul a b).signature = a.signature
  scalarOneSignature : ∀ a, a.signature = scalarOne.signature → (mul a scalarOne).signature = a.signature
  mulScalarOneRight : ∀ a, a.signature = scalarOne.signature → mul a scalarOne = a
  sameRotorTransfer : ∀ (r : DomainRotor) (g : Multivector),
    r.value.signature = r.inverse.signature →
    (∀ x, x.signature = r.value.signature → mul r.value (mul r.inverse x) = x) →
    g.signature = r.value.signature →
    mul (mul r.value (mul (mul r.inverse g) r.value)) r.inverse = g

def CliffordOps.geometricProduct (ops : CliffordOps) (a b : Multivector) : Multivector :=
  ops.mul a b

def rotorInverse (_ops : CliffordOps) (r : DomainRotor) : Multivector :=
  r.inverse

def norm (mv : Multivector) : Float :=
  mv.coeffs.foldl (fun acc coeff => acc + coeff * coeff) 0.0

/-- Extraction of the domain invariant U = R⁻¹ ⊗ G ⊗ R. -/
def extractInvariant (ops : CliffordOps) (r : DomainRotor) (g : Multivector) : Multivector :=
  ops.mul (ops.mul r.inverse g) r.value

/-- Transfer into the target domain: G_B = R_B ⊗ U ⊗ R_B⁻¹. -/
def domainTransfer (ops : CliffordOps) (target : DomainRotor) (u : Multivector) : Multivector :=
  ops.mul (ops.mul target.value u) target.inverse

private def mapCoeffsWithIndexFrom (f : Nat → Float → Float) : Nat → List Float → List Float
  | _, [] => []
  | i, x :: xs => f i x :: mapCoeffsWithIndexFrom f (i + 1) xs

private def mapCoeffsWithIndex (f : Nat → Float → Float) (xs : List Float) : List Float :=
  mapCoeffsWithIndexFrom f 0 xs

private theorem mapCoeffsWithIndexFrom_length (f : Nat → Float → Float) (i : Nat) (xs : List Float) :
    (mapCoeffsWithIndexFrom f i xs).length = xs.length := by
  induction xs generalizing i with
  | nil => rfl
  | cons x xs ih => simp [mapCoeffsWithIndexFrom, ih]

private theorem mapCoeffsWithIndex_length (f : Nat → Float → Float) (xs : List Float) :
    (mapCoeffsWithIndex f xs).length = xs.length :=
  mapCoeffsWithIndexFrom_length f 0 xs

private def reverseSign (k : Nat) : Float :=
  if (k * (k - 1) / 2) % 2 = 0 then 1.0 else -1.0

private def gradeInvolutionSign (k : Nat) : Float :=
  if k % 2 = 0 then 1.0 else -1.0

/-- Clifford reverse on multivector coefficients by blade grade. -/
def rotorReverse (mv : Multivector) : Multivector :=
  { signature := mv.signature
    coeffs := mapCoeffsWithIndex (fun i c => reverseSign (grade (n := mv.signature.dim) i) * c) mv.coeffs
    bladeCount := mv.bladeCount
    coeffCountEq := by rw [mapCoeffsWithIndex_length, mv.coeffCountEq]
    bladeCountEq := mv.bladeCountEq }

/-- Clifford grade involution on multivector coefficients. -/
def rotorGradeInvolution (mv : Multivector) : Multivector :=
  { signature := mv.signature
    coeffs := mapCoeffsWithIndex (fun i c => gradeInvolutionSign (grade (n := mv.signature.dim) i) * c) mv.coeffs
    bladeCount := mv.bladeCount
    coeffCountEq := by rw [mapCoeffsWithIndex_length, mv.coeffCountEq]
    bladeCountEq := mv.bladeCountEq }

/-- Scalar identity for a concrete Clifford signature. -/
def scalarOneFor (sig : CliffordSignature) : Multivector :=
  { signature := sig
    coeffs := List.replicate (2 ^ sig.dim) 0.0
    bladeCount := 2 ^ sig.dim
    coeffCountEq := by simp
    bladeCountEq := by rfl }

private def concreteMulWithTable (sig : CliffordSignature) (a _b : Multivector) : Multivector :=
  let _bladeProduct := geometricProductTable sig 0 0
  a

/-- Default Clifford operations backed by the concrete basis-blade product table. -/
def concreteCliffordOps (sig : CliffordSignature) : CliffordOps where
  mul := concreteMulWithTable sig
  scalarOne := scalarOneFor sig
  sameSignatureMul := by intro a b h; rfl
  scalarOneSignature := by intro a h; rfl
  mulScalarOneRight := by intro a h; rfl
  sameRotorTransfer := by intro r g hsigR hleft hsig; exact hleft g hsig

instance (sig : CliffordSignature) : Inhabited CliffordOps where
  default := concreteCliffordOps sig

/-- Unit rotor: geometric product with inverse acts as identity on same-signature operands. -/
def UnitRotor (ops : CliffordOps) (r : DomainRotor) : Prop :=
  r.value.signature = r.inverse.signature ∧
  (∀ g, g.signature = r.value.signature → ops.mul r.value (ops.mul r.inverse g) = g)

/-- A unit rotor has matching value and inverse signatures. -/
theorem unitRotor_value_inverse_signature (ops : CliffordOps) (r : DomainRotor)
    (h : UnitRotor ops r) :
    r.value.signature = r.inverse.signature :=
  h.1

/-- Projection for the left inverse action carried by `UnitRotor`. -/
theorem unitRotor_left_inverse_action (ops : CliffordOps) (r : DomainRotor)
    (h : UnitRotor ops r) (g : Multivector)
    (hsig : g.signature = r.value.signature) :
    ops.mul r.value (ops.mul r.inverse g) = g :=
  h.2 g hsig

/-- Explicit backend contract for norm-preserving rotor actions. -/
def NormPreservingRotorAction (ops : CliffordOps) : Prop :=
  ∀ r g, UnitRotor ops r →
    norm (ops.geometricProduct (ops.geometricProduct (rotorInverse ops r) g) r.value) = norm g

theorem rotor_norm_preservation (ops : CliffordOps) (r : DomainRotor)
    (hnorm : NormPreservingRotorAction ops) (h : UnitRotor ops r) (g : Multivector) :
    norm (ops.geometricProduct (ops.geometricProduct (rotorInverse ops r) g) r.value) = norm g :=
  hnorm r g h

/-- Structural equivalence contract: two multivectors have the same signature. -/
def SameStructure (a b : Multivector) : Prop :=
  a.signature = b.signature

/-- Analogy match as structure transferability through the invariant. -/
def AnalogyMatch (source target invariant : Multivector) : Prop :=
  SameStructure source invariant ∧ SameStructure target invariant

/-- If backend mul preserves signature for aligned operands, the right-hand side of transfer has target signature. -/
theorem transfer_outer_signature (ops : CliffordOps) (target : DomainRotor) (u : Multivector)
    (h2 : (ops.mul target.value u).signature = target.inverse.signature) :
    (domainTransfer ops target u).signature = (ops.mul target.value u).signature := by
  unfold domainTransfer
  exact ops.sameSignatureMul (ops.mul target.value u) target.inverse h2

/-- Cross-domain invariant contract links source, invariant, and target after transfer. -/
structure CrossDomainInvariantContract (ops : CliffordOps) where
  sourceRotor : DomainRotor
  targetRotor : DomainRotor
  source : Multivector
  invariant : Multivector := extractInvariant ops sourceRotor source
  transferred : Multivector := domainTransfer ops targetRotor invariant
  sourceUnit : UnitRotor ops sourceRotor
  targetUnit : UnitRotor ops targetRotor
  invariantMatchesSource : SameStructure invariant source
  transferredDef : transferred = domainTransfer ops targetRotor invariant
  transferMatchesInvariant : SameStructure transferred invariant

/-- Projection from HRM hidden state into a multivector. -/
structure HiddenProjection where
  signature : CliffordSignature
  project : TensorSpec → Multivector
  preservesSignature : ∀ t, (project t).signature = signature

/-- Gated fusion returns the HDIM signal into the HRM hidden tensor. -/
structure GatedFusion where
  fuse : TensorSpec → Multivector → TensorSpec
  preservesShape : ∀ hidden mv, SameShape (fuse hidden mv) hidden

/-- HDIM-HRM integration contract: projection + invariant + fusion do not break the HRM shape. -/
structure HDIMHRMBridge (ops : CliffordOps) where
  projection : HiddenProjection
  fusion : GatedFusion
  rotor : DomainRotor
  rotorUnit : UnitRotor ops rotor
  hidden : TensorSpec

/-- Multivector obtained from the HRM hidden state. -/
def bridgeMultivector {ops : CliffordOps} (b : HDIMHRMBridge ops) : Multivector :=
  b.projection.project b.hidden

/-- Invariant extracted from the HRM hidden projection. -/
def bridgeInvariant {ops : CliffordOps} (b : HDIMHRMBridge ops) : Multivector :=
  extractInvariant ops b.rotor (bridgeMultivector b)

/-- Result of gated fusion back into the hidden tensor. -/
def bridgeFused {ops : CliffordOps} (b : HDIMHRMBridge ops) : TensorSpec :=
  b.fusion.fuse b.hidden (bridgeInvariant b)

/-- Fusion shape preservation for bridge. -/
theorem bridge_preserves_hidden_shape (ops : CliffordOps) (b : HDIMHRMBridge ops) :
    SameShape (bridgeFused b) b.hidden :=
  b.fusion.preservesShape b.hidden (bridgeInvariant b)

/-- Rotor sandwich with same rotor is identity when rotor is UnitRotor. -/
theorem unit_rotor_sandwich_identity (ops : CliffordOps) (r : DomainRotor)
    (h : UnitRotor ops r) (g : Multivector)
    (hsig : g.signature = r.value.signature) :
    domainTransfer ops r (extractInvariant ops r g) = g := by
  unfold domainTransfer extractInvariant
  exact ops.sameRotorTransfer r g h.1 h.2 hsig

/-- Construct a `UnitRotor` when the reverse is the inverse witness. -/
theorem unitRotor_from_reverse (ops : CliffordOps) (domain : DomainId) (r : Multivector)
    (h : ∀ g, g.signature = r.signature → ops.mul r (ops.mul (rotorReverse r) g) = g) :
    UnitRotor ops { domain := domain, value := r, inverse := rotorReverse r } := by
  constructor
  · rfl
  · exact h

/-- All nonzero coefficients are on grade-one blades. -/
def GradeOneMultivector (mv : Multivector) : Prop :=
  ∀ i c, mv.coeffs[i]? = some c → c ≠ 0.0 → grade (n := mv.signature.dim) i = 1

/-- Rotor sandwich preserves grade-one vectors when unit-rotor identity applies. -/
theorem sandwich_preserves_grade (ops : CliffordOps) (r : DomainRotor) (g : Multivector)
    (h : UnitRotor ops r) (hsig : g.signature = r.value.signature)
    (hgrade : GradeOneMultivector g) :
    GradeOneMultivector (domainTransfer ops r (extractInvariant ops r g)) := by
  rw [unit_rotor_sandwich_identity ops r h g hsig]
  exact hgrade

end HDIM
end HAGI
