import Mathlib
import HAGI.CoreTypes

/-! Loss function input contracts.

Formalizes the shape requirements for `cross_entropy_loss` and `composite_loss`
in `src/hagi/losses.py`.
-/

namespace HAGI
namespace Losses

open CoreTypes

/-- Cross-entropy input contract.

Python contract (`cross_entropy_loss`):
- logits last dimension = vocab_size (number of classes)
- targets shape matches logits with last dimension removed
- targets are class indices in [0, vocab_size)
-/
structure CrossEntropyInput where
  batchSize : Nat
  seqLen : Nat
  vocabSize : Nat
  logits : Shape
  targets : Shape
  batchSeq_pos : batchSize > 0
  seqLen_pos : seqLen > 0
  vocabSize_pos : vocabSize > 0
  logits_is_2d : logits.dims = [batchSize * seqLen, vocabSize]
  targets_is_1d : targets.dims = [batchSize * seqLen]
  dims_match : logits.dims.dropLast = targets.dims

/-- Logits shape is well-formed for 2D cross-entropy. -/
theorem logits_flattened_shape (input : CrossEntropyInput) :
    input.logits.dims = [input.batchSize * input.seqLen, input.vocabSize] :=
  input.logits_is_2d

/-- Targets shape matches flattened batch*sequence. -/
theorem targets_flattened_shape (input : CrossEntropyInput) :
    input.targets.dims = [input.batchSize * input.seqLen] :=
  input.targets_is_1d

/-- The flattened batch*sequence is positive. -/
theorem batch_seq_positive (input : CrossEntropyInput) :
    input.batchSize * input.seqLen > 0 := by
  exact Nat.mul_pos input.batchSeq_pos input.seqLen_pos

/-- Loss input is total when all components are present. -/
structure CompositeLossInput extends CrossEntropyInput where
  hasAuxiliary : Bool
  hasIsomorphic : Bool
  auxiliaryShape : Option Shape
  isomorphicShape : Option Shape
  auxiliary_matches_logits : hasAuxiliary → auxiliaryShape = some logits
  isomorphic_matches_logits : hasIsomorphic → isomorphicShape = some logits

/-- When composite loss is computed with all components, each has the same shape as logits. -/
theorem composite_components_same_shape (input : CompositeLossInput)
    (h_aux : input.hasAuxiliary) (h_iso : input.hasIsomorphic) :
    input.auxiliaryShape = some input.logits ∧ input.isomorphicShape = some input.logits := by
  constructor
  · exact input.auxiliary_matches_logits h_aux
  · exact input.isomorphic_matches_logits h_iso

/-- Total loss is well-defined when weights are non-negative. -/
structure LossWeights where
  wCE : Float
  wAux : Float
  wIso : Float
  wCE_nonneg : 0.0 ≤ wCE
  wAux_nonneg : 0.0 ≤ wAux
  wIso_nonneg : 0.0 ≤ wIso

/-- Total loss formula as weighted sum. -/
noncomputable def totalLoss (l_ce l_aux l_iso : Float) (w : LossWeights) : Float :=
  w.wCE * l_ce + w.wAux * l_aux + w.wIso * l_iso

/-- Total loss is non-negative when all components and weights are non-negative. -/
theorem total_loss_nonneg (l_ce l_aux l_iso : Float)
    (hl_ce : 0.0 ≤ l_ce) (hl_aux : 0.0 ≤ l_aux) (hl_iso : 0.0 ≤ l_iso)
    (w : LossWeights) :
    0.0 ≤ totalLoss l_ce l_aux l_iso w := by
  unfold totalLoss
  have h1 : 0.0 ≤ w.wCE * l_ce := mul_nonneg w.wCE_nonneg hl_ce
  have h2 : 0.0 ≤ w.wAux * l_aux := mul_nonneg w.wAux_nonneg hl_aux
  have h3 : 0.0 ≤ w.wIso * l_iso := mul_nonneg w.wIso_nonneg hl_iso
  linarith

/-- MSE loss input contract (for auxiliary and isomorphic losses). -/
structure MSEInput where
  predShape : Shape
  targetShape : Shape
  shapes_equal : predShape = targetShape

/-- L2 regularization input (fallback when auxiliary labels are missing). -/
structure L2RegularizationInput where
  featuresShape : Shape
  batchSeq_pos : featuresShape.numel > 0

end Losses
end HAGI
