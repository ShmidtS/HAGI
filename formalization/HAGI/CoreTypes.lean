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

/-- Multi-axis tensor index with one bounded Nat per shape dimension. -/
def AxisIndex (shape : List Nat) : Type :=
  { idx : List Nat // idx.length = shape.length ∧ ∀ i (h : i < idx.length), idx[i] < shape[i] }

/-- Flattened tensor index bounded by total element count. -/
def FlatIndex (shape : List Nat) : Type :=
  Fin shape.prod

/-- Row-major offset computation. -/
def index_to_offset (shape : List Nat) (idx : AxisIndex shape) : Nat :=
  match shape, idx with
  | [], _ => 0
  | _d :: ds, ⟨[], hlen, _⟩ =>
      -- definitional: an empty index cannot have successor shape length
      by simp at hlen
  | d :: ds, ⟨i :: is, hlen, hbound⟩ =>
      -- definitional: successor list length equality injects to tail length equality
      have htail_len : is.length = ds.length := by simpa using Nat.succ.inj hlen
      have htail_bound : ∀ j (hj : j < is.length), is[j] < ds[j] := by
        intro j hj
        -- definitional: tail index position j corresponds to full index position j + 1
        have hb := hbound (j + 1) (by simpa using Nat.succ_lt_succ hj)
        -- definitional: vector indexing through matching cons tails
        simpa using hb
      i * ds.prod + index_to_offset ds ⟨is, htail_len, htail_bound⟩

/-- Product positivity from all nonzero dimensions. -/
theorem numel_positive_of_nonzero_dims (shape : List Nat)
    (h : shape.all (· > 0)) : shape.prod > 0 := by
  induction shape with
  | nil =>
      -- definitional: empty list product is one and `all` is true
      simp
  | cons d ds ih =>
      -- definitional: list product and `all` unfold over cons
      simp at h ⊢
      exact Nat.mul_pos h.1 (ih h.2)

private theorem all_pos_of_mem {shape : List Nat}
    (h : ∀ i (hi : i < shape.length), 0 < shape[i]) : shape.all (· > 0) := by
  induction shape with
  | nil =>
      -- definitional: `all` on an empty list is true
      simp
  | cons d ds ih =>
      -- definitional: `all` on cons splits into head and tail predicates
      simp
      constructor
      · exact h 0 (by
          -- definitional: zero is a valid index into a cons list
          simp)
      · apply ih
        intro i hi
        -- definitional: tail index i corresponds to cons index i + 1
        exact h (i + 1) (by simpa using Nat.succ_lt_succ hi)

private theorem axis_shape_prod_pos (shape : List Nat) (idx : AxisIndex shape) : shape.prod > 0 := by
  apply numel_positive_of_nonzero_dims
  apply all_pos_of_mem
  intro i hi
  have hidx_len : idx.val.length = shape.length := idx.property.1
  -- definitional: AxisIndex length equality transports a shape index bound to idx.val
  have hi_idx : i < idx.val.length := by simpa [hidx_len] using hi
  exact Nat.lt_of_le_of_lt (Nat.zero_le _) (idx.property.2 i hi_idx)

/-- Inverse mapping from a flat offset to a row-major axis index. -/
def offset_to_index (shape : List Nat) (offset : FlatIndex shape) : AxisIndex shape :=
  match shape with
  | [] =>
      -- definitional: empty shape has empty index and vacuous bounds
      ⟨[], by simp, by simp⟩
  | d :: ds =>
      have hprod : (d :: ds).prod > 0 := Nat.pos_of_ne_zero (Fin.pos_iff_nonempty.mp ⟨offset⟩)
      have hd : d > 0 := Nat.pos_of_mul_pos_left hprod
      have hds : ds.prod > 0 := Nat.pos_of_mul_pos_right hprod
      let q := offset.val / ds.prod
      let r := offset.val % ds.prod
      have hq : q < d := by
        apply Nat.div_lt_of_lt_mul
        -- definitional: cons product is d * ds.prod, modulo associativity/commutativity
        simpa [q, List.prod, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using offset.isLt
      have hr : r < ds.prod := by
        exact Nat.mod_lt _ hds
      let tail := offset_to_index ds ⟨r, hr⟩
      ⟨q :: tail.val,
        -- definitional: cons index length is successor of tail length
        by simp [tail.property.1], by
        intro i hi
        cases i with
        | zero =>
            -- definitional: head lookup of q :: tail.val is q
            simpa [q]
        | succ j =>
            -- definitional: successor bound on cons list gives tail bound
            have hj : j < tail.val.length := by simpa using hi
            have hb := tail.property.2 j hj
            -- definitional: successor lookup through cons equals tail lookup
            simpa [tail.property.1] using hb⟩

/-- Computed row-major offset is bounded by total element count. -/
theorem index_to_offset_lt_numel (shape : List Nat) (idx : AxisIndex shape)
    : index_to_offset shape idx < shape.prod := by
  induction shape generalizing idx with
  | nil =>
      -- definitional: empty shape offset is zero
      simp [index_to_offset]
  | cons d ds ih =>
      rcases idx with ⟨idxs, hlen, hbound⟩
      cases idxs with
      | nil =>
          -- definitional: empty index cannot have successor shape length
          simp at hlen
      | cons i is =>
          -- definitional: successor list length equality injects to tail length equality
          have htail_len : is.length = ds.length := by simpa using Nat.succ.inj hlen
          have htail_bound : ∀ j (hj : j < is.length), is[j] < ds[j] := by
            intro j hj
            -- definitional: tail index position j corresponds to full index position j + 1
            have hb := hbound (j + 1) (by simpa using Nat.succ_lt_succ hj)
            -- definitional: vector indexing through matching cons tails
            simpa using hb
          have hi : i < d := by
            -- definitional: head index position is zero in a cons list
            have hb := hbound 0 (by simp)
            -- definitional: head lookup of i :: is and d :: ds
            simpa using hb
          let tail : AxisIndex ds := ⟨is, htail_len, htail_bound⟩
          have hrest : index_to_offset ds tail < ds.prod := ih tail
          have hds : 0 < ds.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) hrest
          -- definitional: row-major offset unfolds over cons shape and cons index
          simp [index_to_offset, tail, List.prod]
          nlinarith

/-- Row-major flat offset recovered after conversion to an axis index. -/
theorem index_offset_inverse (shape : List Nat) (offset : FlatIndex shape) :
    index_to_offset shape (offset_to_index shape offset) = offset.val := by
  induction shape generalizing offset with
  | nil =>
      exact Fin.elim0 offset
  | cons d ds ih =>
      have hprod : (d :: ds).prod > 0 := Nat.pos_of_ne_zero (Fin.pos_iff_nonempty.mp ⟨offset⟩)
      have hds : ds.prod > 0 := Nat.pos_of_mul_pos_right hprod
      let r := offset.val % ds.prod
      have hr : r < ds.prod := Nat.mod_lt _ hds
      have ih_tail := ih ⟨r, hr⟩
      -- definitional: offset/index conversion unfolds to quotient-remainder identity
      simp [offset_to_index, index_to_offset, r, ih_tail, Nat.div_add_mod]

/-- Axis index recovered after conversion to a row-major flat offset. -/
theorem offset_index_inverse (shape : List Nat) (idx : AxisIndex shape) :
    offset_to_index shape ⟨index_to_offset shape idx, index_to_offset_lt_numel shape idx⟩ = idx := by
  induction shape generalizing idx with
  | nil =>
      cases idx with
      | mk xs h =>
          -- definitional: empty shape permits only empty index
          cases xs <;> simp [offset_to_index]
  | cons d ds ih =>
      rcases idx with ⟨idxs, hlen, hbound⟩
      cases idxs with
      | nil =>
          -- definitional: empty index cannot have successor shape length
          simp at hlen
      | cons i is =>
          -- definitional: successor list length equality injects to tail length equality
          have htail_len : is.length = ds.length := by simpa using Nat.succ.inj hlen
          have htail_bound : ∀ j (hj : j < is.length), is[j] < ds[j] := by
            intro j hj
            -- definitional: tail index position j corresponds to full index position j + 1
            have hb := hbound (j + 1) (by simpa using Nat.succ_lt_succ hj)
            -- definitional: vector indexing through matching cons tails
            simpa using hb
          let tail : AxisIndex ds := ⟨is, htail_len, htail_bound⟩
          have htail_off : index_to_offset ds tail < ds.prod := index_to_offset_lt_numel ds tail
          have hds : 0 < ds.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) htail_off
          have hi : i < d := by
            -- definitional: head index position is zero in a cons list
            have hb := hbound 0 (by simp)
            -- definitional: head lookup of i :: is and d :: ds
            simpa using hb
          have hdiv : (i * ds.prod + index_to_offset ds tail) / ds.prod = i := by
            rw [Nat.add_comm]
            exact Nat.add_mul_div_right _ _ hds
          have hmod : (i * ds.prod + index_to_offset ds tail) % ds.prod = index_to_offset ds tail := by
            rw [Nat.add_comm]
            exact Nat.add_mul_mod_self_right _ _
          have htail_eq := ih tail
          apply Subtype.ext
          -- definitional: offset_to_index/index_to_offset unfold to matching head and tail
          simp [offset_to_index, index_to_offset, tail, hdiv, hmod, htail_eq]

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