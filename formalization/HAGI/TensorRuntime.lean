import HAGI.CoreTypes
import HAGI.Numerics

/-!
Tensor Runtime boundary.

Lean fixes the safety contract between the CPU reference backend, future CUDA-oxide kernels,
and upper models: backend dispatch does not change shape, dtype, or layout alignment.
-/

namespace HAGI
namespace TensorRuntime

open CoreTypes

/-- Backend runtime. -/
inductive Backend where
  | cpuReference
  | cudaOxide
  deriving DecidableEq, Repr

/-- Tensor value at the specification level: data is abstract, but the spec is strict. -/
structure Tensor where
  spec : TensorSpec
  backend : Backend

/-- Backend operation must preserve the tensor spec. -/
structure BackendOp where
  name : String
  run : Tensor → Tensor
  preservesSpec : ∀ t, (run t).spec = t.spec

/-- Dispatch operation on a concrete backend. -/
def dispatch (op : BackendOp) (target : Backend) (t : Tensor) : Tensor :=
  { spec := (op.run t).spec, backend := target }

/-- Dispatch is shape-safe because op preserves the spec. -/
theorem dispatch_preserves_shape (op : BackendOp) (target : Backend) (t : Tensor) :
    (dispatch op target t).spec.shape = t.spec.shape := by
  unfold dispatch
  rw [op.preservesSpec t]

/-- Dispatch is dtype-safe. -/
theorem dispatch_preserves_dtype (op : BackendOp) (target : Backend) (t : Tensor) :
    (dispatch op target t).spec.dtype = t.spec.dtype := by
  unfold dispatch
  rw [op.preservesSpec t]

/-- Runtime tensor is aligned if its layout is aligned. -/
def TensorAligned (t : Tensor) : Prop :=
  t.spec.layout.Aligned

/-- Memory layout alignment is preserved by a spec-preserving operation. -/
theorem backend_op_preserves_alignment (op : BackendOp) (t : Tensor)
    (h : TensorAligned t) : TensorAligned (op.run t) := by
  unfold TensorAligned
  rw [op.preservesSpec t]
  exact h

/-- Binary operation with two input tensors and one output. -/
structure BinaryOp where
  name : String
  run : Tensor → Tensor → Tensor
  requiresSameShape : Prop
  preservesLeftSpec : ∀ a b, SameShape a.spec b.spec → (run a b).spec = a.spec

/-- Shape safety for a binary op with aligned input shapes. -/
theorem binary_op_shape_safe (op : BinaryOp) (a b : Tensor)
    (h : SameShape a.spec b.spec) : SameShape (op.run a b).spec a.spec := by
  unfold SameShape
  exact congrArg TensorSpec.shape (op.preservesLeftSpec a b h)

/-- Backend boundary allows CPU and CUDA, but does not change the mathematical spec. -/
def BackendDispatchSafe (op : BackendOp) : Prop :=
  ∀ target t, (dispatch op target t).spec = t.spec

/-- Backend numeric outputs are valid only when bounded by the Real-model error contract. -/
def BackendRealErrorBounded (ε reference actual : Real) : Prop :=
  FloatApprox ε actual reference

/-- `BackendRealErrorBounded` is definitionally equal to `FloatApprox`. -/
theorem backend_real_error_bounded_intro (ε reference actual : Real)
    (h : FloatApprox ε actual reference) : BackendRealErrorBounded ε reference actual :=
  h

/-- Any BackendOp with preservesSpec is safe for dispatch. -/
theorem backend_dispatch_safe (op : BackendOp) : BackendDispatchSafe op := by
  intro target t
  unfold dispatch
  exact op.preservesSpec t

end TensorRuntime
end HAGI
