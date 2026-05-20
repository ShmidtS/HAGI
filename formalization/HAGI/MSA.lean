import HAGI.CoreTypes
import HAGI.HDIM

/-!
MSA sparse memory adapter.

MSA adds long-context memory as a set of slots that can act as HDIM domains.
The specification fixes routing safety, document-wise RoPE domain separation, and append-only K/V cache.
-/

namespace HAGI
namespace MSA

open CoreTypes
open HDIM

/-- MSA slot and routing bounds. -/
structure MSAConfig where
  slotCount : Nat
  slotCount_pos : slotCount > 0
  topK : Nat
  topK_le_slotCount : topK ≤ slotCount

/-- Memory slot corresponds to a document/memory chunk and an HDIM domain. -/
structure MemorySlot (cfg : MSAConfig) where
  slotId : Fin cfg.slotCount
  domain : DomainId
  keyShape : Shape
  valueShape : Shape
  sameKVShape : keyShape = valueShape

/-- Routing decision selects a bounded slot and weight. -/
structure RoutingDecision (cfg : MSAConfig) where
  slotId : Fin cfg.slotCount
  weight : Float
  weightNonNeg : 0.0 ≤ weight
  weightLeOne : weight ≤ 1.0
  deriving Repr

/-- Sparse route is valid if all selected slotIds exist. -/
def RouteWithinSlots {cfg : MSAConfig} (slots : List (MemorySlot cfg))
    (route : List (RoutingDecision cfg)) : Prop :=
  ∀ r ∈ route, ∃ s ∈ slots, s.slotId = r.slotId

/-- An empty route is always valid. -/
theorem empty_route_within_slots {cfg : MSAConfig} (slots : List (MemorySlot cfg)) :
    RouteWithinSlots slots [] := by
  intro r hr
  cases hr

/-- RoPE scope: document-wise for memory docs, global for the active query. -/
inductive RoPEScope where
  | document (docId : Nat)
  | global
  deriving DecidableEq, Repr

/-- Position encoding links the position with the RoPE scope. -/
structure RoPEPosition where
  scope : RoPEScope
  position : Nat
  deriving Repr

/-- Document-wise RoPE correctness: positions from different documents are not mixed. -/
def SameRoPEDocument (a b : RoPEPosition) : Prop :=
  match a.scope, b.scope with
  | RoPEScope.document da, RoPEScope.document db => da = db
  | RoPEScope.global, RoPEScope.global => True
  | _, _ => False

/-- Two positions in the same document scope are validly comparable. -/
theorem document_rope_same_doc (doc posA posB : Nat) :
    SameRoPEDocument ⟨RoPEScope.document doc, posA⟩ ⟨RoPEScope.document doc, posB⟩ := by
  unfold SameRoPEDocument
  rfl

/-- Global RoPE is comparable with itself. -/
theorem global_rope_same_scope (a b : Nat) :
    SameRoPEDocument ⟨RoPEScope.global, a⟩ ⟨RoPEScope.global, b⟩ := by
  unfold SameRoPEDocument
  trivial

/-- K/V cache stores a bounded number of entries and a list of slots. -/
structure KVCache (cfg : MSAConfig) where
  slots : List (MemorySlot cfg)
  entries : Nat
  kvCacheCapacity : entries ≤ cfg.slotCount

/-- Append-only invariant: after the operation, entries do not decrease and old slots remain a prefix. -/
def CacheAppendOnly {cfg : MSAConfig} (before after : KVCache cfg) : Prop :=
  before.entries ≤ after.entries ∧ before.slots <+: after.slots

/-- Adding one slot is append-only when a capacity proof is available. -/
def appendSlot {cfg : MSAConfig} (cache : KVCache cfg) (slot : MemorySlot cfg)
    (hcap : cache.entries + 1 ≤ cfg.slotCount) : KVCache cfg :=
  { slots := cache.slots ++ [slot], entries := cache.entries + 1, kvCacheCapacity := hcap }

/-- appendSlot does not decrease the cache and preserves old slots as a prefix. -/
theorem append_slot_append_only {cfg : MSAConfig} (cache : KVCache cfg) (slot : MemorySlot cfg)
    (hcap : cache.entries + 1 ≤ cfg.slotCount) :
    CacheAppendOnly cache (appendSlot cache slot hcap) := by
  unfold CacheAppendOnly appendSlot
  constructor
  · exact Nat.le_succ cache.entries
  · exact List.prefix_append cache.slots [slot]

/-- Capacity invariant survives appendSlot. -/
theorem append_slot_capacity {cfg : MSAConfig} (cache : KVCache cfg) (slot : MemorySlot cfg)
    (hcap : cache.entries + 1 ≤ cfg.slotCount) :
    (appendSlot cache slot hcap).entries ≤ cfg.slotCount :=
  hcap

/-- Memory slot as an additional HDIM domain. -/
def slotDomain {cfg : MSAConfig} (slot : MemorySlot cfg) : DomainId :=
  slot.domain

/-- MSA adapter defines slots and a proof of routing correctness. -/
structure MSAAdapter (cfg : MSAConfig) where
  slots : List (MemorySlot cfg)
  route : TensorSpec → List (RoutingDecision cfg)
  routeSafe : ∀ q, RouteWithinSlots slots (route q)
  routeLengthBound : ∀ q, (route q).length ≤ cfg.topK
  routeUnique : ∀ q, (route q).Pairwise (fun a b => a.slotId ≠ b.slotId)

/-- Routing adapter does not select unknown slots. -/
theorem adapter_route_safe {cfg : MSAConfig} (a : MSAAdapter cfg) (q : TensorSpec) :
    RouteWithinSlots a.slots (a.route q) :=
  a.routeSafe q

/-- All slot ids in a route are distinct by adapter invariant. -/
theorem route_slot_unique {cfg : MSAConfig} (a : MSAAdapter cfg) (q : TensorSpec) :
    (a.route q).Pairwise (fun x y => x.slotId ≠ y.slotId) :=
  a.routeUnique q

end MSA
end HAGI
