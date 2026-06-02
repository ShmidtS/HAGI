import HAGI.CoreTypes
import HAGI.HRM
import HAGI.HDIM
import HAGI.TensorRuntime

/-! System invariants of the HAGI pipeline.

This module links HRM recurrence, HDIM transfer, and tensor runtime dispatch.
The statements are intentionally kept at the contract level that the Rust implementation must
maintain in the forward-only CPU reference path and future CUDA-oxide backend kernels.
-/

namespace HAGI
namespace Invariants

open CoreTypes
open HRM
open HDIM
open TensorRuntime

/-- Model forward pass as an abstract spec-preserving transformer. -/
structure ForwardPass where
  run : Tensor → Tensor
  preservesSpec : ∀ t, (run t).spec = t.spec

/-- End-to-end forward pass preserves tensor shapes. -/
theorem forward_pass_preserves_tensor_shape (f : ForwardPass) (t : Tensor) :
    (f.run t).spec.shape = t.spec.shape := by
  rw [f.preservesSpec t]

/-- End-to-end forward pass preserves dtype. -/
theorem forward_pass_preserves_dtype (f : ForwardPass) (t : Tensor) :
    (f.run t).spec.dtype = t.spec.dtype := by
  rw [f.preservesSpec t]

/-- Signature preservation modulo target domain, i.e., `SameStructure`. -/
def TransferSameStructureModuloTarget (ops : CliffordOps) (target : DomainRotor) (u : Multivector) : Prop :=
  SameStructure (domainTransfer ops target u) u

/-- Signature preservation modulo target domain, i.e., `SameStructure`. -/
theorem contract_transfer_same_structure (ops : CliffordOps)
    (c : CrossDomainInvariantContract ops) :
    TransferSameStructureModuloTarget ops c.targetRotor c.invariant := by
  unfold TransferSameStructureModuloTarget
  rw [← c.transferredDef]
  exact c.transferMatchesInvariant

/-- This theorem exposes the identity law required for same-rotor transfer. -/
theorem same_rotor_transfer_identity (ops : CliffordOps) (r : DomainRotor) (g : Multivector)
    (h : UnitRotor ops r) (hsig : g.signature = r.value.signature) :
    domainTransfer ops r (extractInvariant ops r g) = g :=
  HDIM.unit_rotor_sandwich_identity ops r h g hsig

/-- HRM recurrence depth monotonicity for one increment. -/
theorem hrm_recurrence_depth_monotone (c : CycleId) :
    DepthMonotone c ⟨c.value + 1⟩ :=
  cycle_increment_monotone c

end Invariants
end HAGI
