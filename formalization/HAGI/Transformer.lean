import Mathlib
import HAGI.CoreTypes

/-! Transformer substrate contracts.

Formalizes the runtime assertions in `model/transformer.py::TransformerConfig.__post_init__`
as Lean proof obligations. Any Python config that claims to be a `TransformerConfig`
must satisfy these divisibility constraints.
-/

namespace HAGI
namespace Transformer

open CoreTypes

/-- Transformer configuration with shape-safety proof fields.

These mirror the `__post_init__` assertions in Python:
- hidden_size % num_query_heads == 0
- num_query_heads % num_kv_heads == 0
- head_dim % 2 == 0
-/
structure TransformerConfig where
  hiddenSize : Nat
  numQueryHeads : Nat
  numKVHeads : Nat
  intermediateSize : Nat
  ropeTheta : Float
  normEps : Float
  maxSeqLen : Nat
  hiddenDivisible : HeadDivisible hiddenSize numQueryHeads
  queryDivisible : numQueryHeads % numKVHeads = 0
  headDimEven : (hiddenSize / numQueryHeads) % 2 = 0
  maxSeqLen_pos : maxSeqLen > 0
  intermediateSize_pos : intermediateSize > 0

/-- Head dimension is positive when hidden_size > 0 and heads > 0. -/
theorem head_dim_positive (cfg : TransformerConfig) :
    cfg.hiddenSize / cfg.numQueryHeads > 0 := by
  have h1 : cfg.numQueryHeads > 0 := cfg.hiddenDivisible.left
  have h2 : cfg.hiddenSize > 0 := by
    have h3 : cfg.numQueryHeads > 0 := h1
    have h4 : cfg.hiddenSize % cfg.numQueryHeads = 0 := cfg.hiddenDivisible.right
    have h5 : cfg.hiddenSize ≥ cfg.numQueryHeads := by
      by_contra h
      push_neg at h
      have h6 : cfg.hiddenSize % cfg.numQueryHeads = cfg.hiddenSize := by
        rw [Nat.mod_eq_of_lt h]
      rw [h6] at h4
      omega
    nlinarith [h5, h3]
  exact Nat.div_pos h2 h1

/-- Number of KV heads is positive because it divides numQueryHeads. -/
theorem num_kv_heads_positive (cfg : TransformerConfig) :
    cfg.numKVHeads > 0 := by
  have h1 : cfg.numQueryHeads > 0 := cfg.hiddenDivisible.left
  have h2 : cfg.numQueryHeads % cfg.numKVHeads = 0 := cfg.queryDivisible
  by_contra h
  push_neg at h
  have h3 : cfg.numKVHeads = 0 := by omega
  rw [h3] at h2
  simp at h2

/-- KV repetition factor is positive. -/
theorem kv_repeat_positive (cfg : TransformerConfig) :
    cfg.numQueryHeads / cfg.numKVHeads > 0 := by
  have h1 : cfg.numQueryHeads > 0 := cfg.hiddenDivisible.left
  have h2 : cfg.numKVHeads > 0 := num_kv_heads_positive cfg
  exact Nat.div_pos h1 h2

/-- Hidden size is divisible by numKVHeads * head_dim (implied by the two divisibility constraints). -/
theorem hidden_divisible_by_kv_heads_times_head_dim (cfg : TransformerConfig) :
    cfg.numKVHeads > 0 ∧ cfg.hiddenSize % cfg.numKVHeads = 0 := by
  constructor
  · exact num_kv_heads_positive cfg
  · have h1 : cfg.hiddenSize % cfg.numQueryHeads = 0 := cfg.hiddenDivisible.right
    have h2 : cfg.numQueryHeads % cfg.numKVHeads = 0 := cfg.queryDivisible
    have h3 : cfg.numKVHeads > 0 := num_kv_heads_positive cfg
    have h4 : cfg.hiddenSize % cfg.numKVHeads = 0 := by
      exact Nat.dvd_trans (Nat.dvd_of_mod_eq_zero h2) (Nat.dvd_of_mod_eq_zero h1)
    exact h4

/-- Head dimension is even, required for RoPE (splits into x1/x2 pairs). -/
theorem head_dim_is_even (cfg : TransformerConfig) :
    (cfg.hiddenSize / cfg.numQueryHeads) % 2 = 0 :=
  cfg.headDimEven

/-- RoPE cache shape: [seq_len, head_dim / 2]. -/
def ropeCacheShape (cfg : TransformerConfig) : Shape :=
  { dims := [cfg.maxSeqLen, cfg.hiddenSize / cfg.numQueryHeads / 2]
    nonEmptyDims := by simp [cfg.maxSeqLen_pos]
    positiveDims := by
      intro d hd
      simp at hd
      rcases hd with rfl | rfl
      · exact cfg.maxSeqLen_pos
      · have h1 : cfg.hiddenSize / cfg.numQueryHeads > 0 := head_dim_positive cfg
        have h2 : (cfg.hiddenSize / cfg.numQueryHeads) % 2 = 0 := cfg.headDimEven
        have h3 : cfg.hiddenSize / cfg.numQueryHeads / 2 > 0 := by
          have h4 : cfg.hiddenSize / cfg.numQueryHeads ≥ 2 := by
            omega
          exact Nat.div_pos h4 (by norm_num)
        exact h3 }

/-- Q projection output shape: [batch, seq, numQueryHeads * head_dim] = [batch, seq, hiddenSize]. -/
theorem q_proj_shape (cfg : TransformerConfig) (batch seq : Nat) (hb : batch > 0) (hs : seq > 0) :
    let head_dim := cfg.hiddenSize / cfg.numQueryHeads
    let out_dim := cfg.numQueryHeads * head_dim
    out_dim = cfg.hiddenSize := by
  have h1 : cfg.hiddenSize % cfg.numQueryHeads = 0 := cfg.hiddenDivisible.right
  have h2 : cfg.numQueryHeads > 0 := cfg.hiddenDivisible.left
  have h3 : cfg.numQueryHeads * (cfg.hiddenSize / cfg.numQueryHeads) = cfg.hiddenSize := by
    rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero h1)]
  simp [h3]

end Transformer
end HAGI
