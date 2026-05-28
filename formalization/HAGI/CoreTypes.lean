import Mathlib

namespace HAGI
namespace CoreTypes

/-! Core HAGI types.

Minimal contract shared by the Rust runtime and Lean layer:
tensor shape, dtype, domain/cycle identifiers, and verifiable layout properties.
Invariants:
- Shape: all axes > 0, list is nonempty.
- Layout: strides match rank, all > 0, alignment > 0.
- AxisIndex: length matches shape rank, each coordinate is in bounds.

The AxisIndex ↔ FlatIndex bijection is proved constructively.
-/

/-- Supported numeric formats. -/
inductive DType where
  | f32
  | f64
  | i32
  | u8
  deriving DecidableEq, Repr

/-- Domain identifier for HDIM/MSA transfer. -/
structure DomainId where
  value : Nat
  deriving DecidableEq, Repr

/-- HRM recurrence step identifier. -/
structure CycleId where
  value : Nat
  deriving DecidableEq, Repr

/-- Shape stores axis sizes and a proof that all axes are nonzero. -/
structure Shape where
  dims : List Nat
  nonEmptyDims : dims ≠ []
  positiveDims : ∀ d ∈ dims, d > 0

/-- Shape rank. -/
def Shape.rank (s : Shape) : Nat :=
  s.dims.length

/-- Number of elements in the shape. The empty list is not used because of nonEmptyDims. -/
def Shape.numel (s : Shape) : Nat :=
  s.dims.foldl (· * ·) 1

/-- Runtime buffer length tied to tensor shape numel. -/
def BufferLength (shape : Shape) : Nat :=
  shape.numel

/-- Layout describes strides and alignment that the runtime must respect. -/
structure Layout where
  shape : Shape
  strides : List Nat
  offset : Nat
  alignment : Nat
  rankMatches : strides.length = shape.rank
  positiveAlignment : alignment > 0
  positiveStrides : ∀ st ∈ strides, st > 0

/-- Offset alignment relative to alignment. -/
def Layout.Aligned (l : Layout) : Prop :=
  l.offset % l.alignment = 0

/-- TensorSpec — shape + dtype + layout. -/
structure TensorSpec where
  shape : Shape
  dtype : DType
  layout : Layout
  layoutShapeMatches : layout.shape = shape

/-- Multi-head attention safety: hidden_size is divisible by the number of heads. -/
def HeadDivisible (hiddenSize numHeads : Nat) : Prop :=
  numHeads > 0 ∧ hiddenSize % numHeads = 0

/-- Head size is defined only under HeadDivisible. -/
def headDim (hiddenSize numHeads : Nat) (_ : HeadDivisible hiddenSize numHeads) : Nat :=
  hiddenSize / numHeads

/-- Shape preservation as equality of shapes before and after an operation. -/
def SameShape (a b : TensorSpec) : Prop :=
  a.shape = b.shape

/-- Multi-axis tensor index with one Nat per shape dimension. -/
def AxisIndex (shape : List Nat) : Type :=
  { idx : List Nat // idx.length = shape.length }

/-- Axis index bounds against a tensor shape. -/
def inBounds (idx : List Nat) (shape : List Nat) : Prop :=
  idx.length = shape.length ∧
    ∀ (i : Nat) (hIdx : i < idx.length) (hShape : i < shape.length),
      idx.get ⟨i, hIdx⟩ < shape.get ⟨i, hShape⟩

/-- Flattened tensor index represented by its runtime offset. -/
abbrev FlatIndex (_shape : List Nat) : Type :=
  Nat

/-- Row-major offset computation. -/
def index_to_offset_list : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: ds, i :: is => i * ds.prod + index_to_offset_list ds is

def index_to_offset (shape : List Nat) (idx : AxisIndex shape) : Nat :=
  index_to_offset_list shape idx.val

private theorem inBounds_tail {i d : Nat} {is ds : List Nat}
    (h : inBounds (i :: is) (d :: ds)) : inBounds is ds := by
  constructor
  · exact Nat.succ.inj h.1
  · intro n hIdx hShape
    simpa using h.2 (n + 1) (Nat.succ_lt_succ hIdx) (Nat.succ_lt_succ hShape)

private theorem index_to_offset_list_lt_prod (shape idx : List Nat)
    (h : inBounds idx shape) : index_to_offset_list shape idx < shape.prod := by
  induction shape generalizing idx with
  | nil =>
      cases idx with
      | nil => simp [index_to_offset_list]
      | cons i is => cases h.1
  | cons d ds ih =>
      cases idx with
      | nil => cases h.1
      | cons i is =>
          have hi : i < d := by simpa using h.2 0 (by simp) (by simp)
          have htail : inBounds is ds := inBounds_tail h
          have hrec : index_to_offset_list ds is < ds.prod := ih is htail
          have hds_pos : 0 < ds.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) hrec
          calc
            index_to_offset_list (d :: ds) (i :: is)
                = i * ds.prod + index_to_offset_list ds is := rfl
            _ < (i + 1) * ds.prod := by nlinarith
            _ ≤ d * ds.prod := Nat.mul_le_mul_right ds.prod (Nat.succ_le_of_lt hi)
            _ = (d :: ds).prod := by simp

/-- Product positivity from all nonzero dimensions. -/
theorem numel_positive_of_nonzero_dims (shape : List Nat)
    (h : shape.all (· > 0)) : shape.prod > 0 := by
  induction shape with
  | nil => simp
  | cons d ds ih =>
      simp only [List.all_cons, Bool.and_eq_true] at h
      have hd : d > 0 := of_decide_eq_true h.1
      have hds : ds.all (· > 0) := h.2
      exact Nat.mul_pos hd (ih hds)

/-- Inverse mapping from a flat offset to a row-major axis index. -/
def offset_to_index_list : List Nat → Nat → List Nat
  | [], _ => []
  | d :: ds, offset =>
      let stride := ds.prod
      (offset / stride) % d :: offset_to_index_list ds (offset % stride)

theorem offset_to_index_list_length (shape : List Nat) (offset : Nat) :
    (offset_to_index_list shape offset).length = shape.length := by
  induction shape generalizing offset with
  | nil => rfl
  | cons d ds ih => simp [offset_to_index_list, ih]

def offset_to_index (shape : List Nat) (offset : FlatIndex shape) : AxisIndex shape :=
  ⟨offset_to_index_list shape offset, offset_to_index_list_length shape offset⟩

private theorem index_offset_inverse_list (shape : List Nat) (offset : Nat)
    (h : offset < shape.prod) :
    index_to_offset_list shape (offset_to_index_list shape offset) = offset := by
  induction shape generalizing offset with
  | nil =>
      simp [index_to_offset_list] at h ⊢
      omega
  | cons d ds ih =>
      have hpos : 0 < d * ds.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) (by simpa using h)
      have hstride : 0 < ds.prod := Nat.pos_of_mul_pos_left hpos
      have hq_lt_d : offset / ds.prod < d := by
        rw [Nat.div_lt_iff_lt_mul hstride]
        simpa [Nat.mul_comm] using h
      have htail : offset % ds.prod < ds.prod := Nat.mod_lt offset hstride
      calc
        index_to_offset_list (d :: ds) (offset_to_index_list (d :: ds) offset)
            = ((offset / ds.prod) % d) * ds.prod +
                index_to_offset_list ds (offset_to_index_list ds (offset % ds.prod)) := rfl
        _ = (offset / ds.prod) * ds.prod + offset % ds.prod := by
              rw [Nat.mod_eq_of_lt hq_lt_d, ih (offset % ds.prod) htail]
        _ = offset := Nat.div_add_mod' offset ds.prod

private theorem offset_index_inverse_list (shape idx : List Nat)
    (h : inBounds idx shape) :
    offset_to_index_list shape (index_to_offset_list shape idx) = idx := by
  induction shape generalizing idx with
  | nil =>
      cases idx with
      | nil => simp [offset_to_index_list]
      | cons i is => cases h.1
  | cons d ds ih =>
      cases idx with
      | nil => cases h.1
      | cons i is =>
          have hi : i < d := by simpa using h.2 0 (by simp) (by simp)
          have htail : inBounds is ds := inBounds_tail h
          have hrec : index_to_offset_list ds is < ds.prod := index_to_offset_list_lt_prod ds is htail
          have hstride : 0 < ds.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) hrec
          have hmod : (i * ds.prod + index_to_offset_list ds is) % ds.prod =
              index_to_offset_list ds is := by
            rw [Nat.mul_add_mod_self_right]
            exact Nat.mod_eq_of_lt hrec
          have hdiv : (i * ds.prod + index_to_offset_list ds is) / ds.prod = i := by
            rw [Nat.add_comm, Nat.mul_comm i ds.prod, Nat.add_mul_div_left _ _ hstride]
            rw [Nat.div_eq_of_lt hrec, zero_add]
          have hhead : ((i * ds.prod + index_to_offset_list ds is) / ds.prod) % d = i := by
            rw [hdiv, Nat.mod_eq_of_lt hi]
          rw [index_to_offset_list, offset_to_index_list, hhead, hmod, ih is htail]

/-- Computed row-major offset is bounded by total element count. -/
theorem index_to_offset_lt_numel (shape : List Nat) (idx : AxisIndex shape)
    (h : inBounds idx.val shape) :
    index_to_offset shape idx < shape.prod := by
  exact index_to_offset_list_lt_prod shape idx.val h

/-- Row-major flat offset recovered after conversion to an axis index. -/
theorem index_offset_inverse (shape : List Nat) (offset : FlatIndex shape)
    (h : offset < shape.prod) :
    index_to_offset shape (offset_to_index shape offset) = offset := by
  exact index_offset_inverse_list shape offset h

/-- Axis index recovered after conversion to a row-major flat offset. -/
theorem offset_index_inverse (shape : List Nat) (idx : AxisIndex shape)
    (h : inBounds idx.val shape) :
    offset_to_index shape (index_to_offset shape idx) = idx := by
  apply Subtype.ext
  exact offset_index_inverse_list shape idx.val h

/-- Layout consistency: strides rank equals shape rank. -/
theorem layout_rank_safe (l : Layout) : l.strides.length = l.shape.rank :=
  l.rankMatches

/-- The proof transports shape equality into SameShape. -/
theorem same_shape_refl (t : TensorSpec) : SameShape t t :=
  rfl

/-- If a layout belongs to TensorSpec, its shape matches the tensor shape. -/
theorem tensor_layout_shape_safe (t : TensorSpec) : t.layout.shape = t.shape :=
  t.layoutShapeMatches

/-- Divisibility of hidden_size makes num_heads nonzero by contract. -/
theorem head_divisible_num_heads_positive {hiddenSize numHeads : Nat}
    (h : HeadDivisible hiddenSize numHeads) : numHeads > 0 :=
  h.left

instance : ToString DomainId where
  toString d := toString d.value

instance : ToString CycleId where
  toString c := toString c.value

end CoreTypes
end HAGI