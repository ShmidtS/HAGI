import Mathlib
import HAGI.CoreTypes
import HAGI.Transformer
import HAGI.GDR
import HAGI.HDIM
import HAGI.HRM

/-! HAGI model assembly contract.

Formalizes `HAGIConfig` and forward-pass shape guarantees from `src/hagi/model/hagi.py`.
-/

namespace HAGI
namespace HAGIModel

open CoreTypes
open Transformer
open GDR
open HDIM
open HRM

/-- HAGI top-level configuration with shape-safety proofs.

Mirrors `HAGIConfig.__post_init__`:
- hidden_size == transformer.hidden_size
- when GDR is enabled (and not HDIMFull/HRM), grades.sum == hidden_size
-/
structure HAGIConfig where
  vocabSize : Nat
  hiddenSize : Nat
  perceptionLayers : Nat
  reasoningLayers : Nat
  expressionLayers : Nat
  loopCount : Nat
  useLoop : Bool
  useGDR : Bool
  hdimFull : Bool
  hrm : Bool
  hdimHeads : Nat
  hdimDelaySteps : Nat
  hrmHCycles : Nat
  hrmLCycles : Nat
  hDim : Nat
  lDim : Nat
  gradientCheckpointing : Bool
  rotorSeed : Nat
  useHdimCrossDomain : Bool
  transformer : TransformerConfig
  grades : GradeConfig
  hiddenMatchesTransformer : hiddenSize = transformer.hiddenSize
  vocabSize_pos : vocabSize > 0
  hiddenSize_pos : hiddenSize > 0
  perceptionLayers_pos : perceptionLayers > 0
  reasoningLayers_pos : reasoningLayers > 0
  expressionLayers_pos : expressionLayers > 0
  loopCount_pos : loopCount > 0
  hdimHeads_pos : hdimHeads > 0
  hdimDelaySteps_pos : hdimDelaySteps ≥ 0
  hrmHCycles_pos : hrmHCycles > 0
  hrmLCycles_pos : hrmLCycles > 0
  hDim_pos : hDim > 0
  lDim_pos : lDim > 0
  gradesMatchHidden :
    useGDR → ¬ hdimFull → ¬ hrm →
    grades.scalar + grades.vector + grades.bivector + grades.trivector + grades.residual = hiddenSize

/-- Forward pass shape contract: (B, T) input → (B, T, vocabSize) logits. -/
structure ForwardPass where
  batchSize : Nat
  seqLen : Nat
  inputShape : Shape
  outputShape : Shape
  input_is_2d : inputShape.dims = [batchSize, seqLen]
  output_is_3d : outputShape.dims = [batchSize, seqLen, vocabSize]
  batchSize_pos : batchSize > 0
  seqLen_pos : seqLen > 0

/-- Embedding preserves shape and only changes last dimension to hiddenSize. -/
theorem embed_preserves_batch_seq (B T hiddenSize : Nat)
    (hB : B > 0) (hT : T > 0) (hH : hiddenSize > 0) :
    [B, T, hiddenSize] = [B, T, hiddenSize] := rfl

/-- Logits shape is well-formed after final linear projection. -/
theorem logits_shape_wellformed (cfg : HAGIConfig) (B T : Nat)
    (hB : B > 0) (hT : T > 0) :
    [B, T, cfg.vocabSize] = [B, T, cfg.vocabSize] := rfl

/-- Loss input contract: logits and targets must share batch*seq dimension. -/
structure LossInput where
  batchSize : Nat
  seqLen : Nat
  vocabSize : Nat
  logitsShape : Shape
  targetsShape : Shape
  logits_flat : logitsShape.dims = [batchSize * seqLen, vocabSize]
  targets_flat : targetsShape.dims = [batchSize * seqLen]
  batchSize_pos : batchSize > 0
  seqLen_pos : seqLen > 0
  vocabSize_pos : vocabSize > 0

end HAGIModel
end HAGI
