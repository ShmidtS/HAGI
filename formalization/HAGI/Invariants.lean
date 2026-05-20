import HAGI.CoreTypes
import HAGI.HRM
import HAGI.HDIM
import HAGI.TensorRuntime
import HAGI.MSA

/-!
System invariants of the HAGI pipeline.

This module links HRM recurrence, HDIM transfer, tensor runtime dispatch, and MSA cache.
The statements are intentionally kept at the contract level that the Rust implementation must
maintain in the forward-only CPU reference path and future CUDA-oxide backend kernels.
-/

namespace HAGI
namespace Invariants

open CoreTypes
open HRM
open HDIM
open TensorRuntime
open MSA

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

/-- This theorem exposes the identity law already required by `UnitRotor`; it does not construct the rotor. -/
theorem same_rotor_transfer_identity (ops : CliffordOps) (r : DomainRotor) (g : Multivector)
    (h : UnitRotor ops r) (hsig : g.signature = r.value.signature) :
    domainTransfer ops r (extractInvariant ops r g) = g := by
  unfold domainTransfer extractInvariant
  exact h.right g hsig

/-- HRM recurrence depth monotonicity for one increment. -/
theorem hrm_recurrence_depth_monotone (c : CycleId) :
    DepthMonotone c ⟨c.value + 1⟩ :=
  cycle_increment_monotone c

/-- Memory cache append-only invariant for adding a slot. -/
theorem memory_cache_append_only {cfg : MSAConfig} (cache : KVCache cfg) (slot : MemorySlot cfg)
    (hcap : cache.entries + 1 ≤ cfg.slotCount) :
    CacheAppendOnly cache (appendSlot cache slot hcap) :=
  append_slot_append_only cache slot hcap

/-- Minimal pipeline state used by current invariants. -/
structure PipelineState (cfg : MSAConfig) where
  tensor : Tensor
  hrm : HRMState
  cache : KVCache cfg

/-- Pipeline step contract: tensor spec, HRM shapes, and cache monotonicity. -/
structure PipelineStep (cfg : MSAConfig) where
  run : PipelineState cfg → PipelineState cfg
  preservesTensorSpec : ∀ s, (run s).tensor.spec = s.tensor.spec
  preservesHRMHigh : ∀ s, SameShape (run s).hrm.z_H s.hrm.z_H
  preservesHRMLow : ∀ s, SameShape (run s).hrm.z_L s.hrm.z_L
  cacheAppendOnly : ∀ s, CacheAppendOnly s.cache (run s).cache

/-- Pipeline step preserves tensor shape. -/
theorem pipeline_step_preserves_tensor_shape {cfg : MSAConfig} (p : PipelineStep cfg) (s : PipelineState cfg) :
    (p.run s).tensor.spec.shape = s.tensor.spec.shape := by
  rw [p.preservesTensorSpec s]

/-- Pipeline step preserves HRM z_H shape. -/
theorem pipeline_step_preserves_hrm_high {cfg : MSAConfig} (p : PipelineStep cfg) (s : PipelineState cfg) :
    SameShape (p.run s).hrm.z_H s.hrm.z_H :=
  p.preservesHRMHigh s

/-- Pipeline step preserves HRM z_L shape. -/
theorem pipeline_step_preserves_hrm_low {cfg : MSAConfig} (p : PipelineStep cfg) (s : PipelineState cfg) :
    SameShape (p.run s).hrm.z_L s.hrm.z_L :=
  p.preservesHRMLow s

/-- Pipeline step does not remove memory cache. -/
theorem pipeline_step_cache_append_only {cfg : MSAConfig} (p : PipelineStep cfg) (s : PipelineState cfg) :
    CacheAppendOnly s.cache (p.run s).cache :=
  p.cacheAppendOnly s

end Invariants
end HAGI
