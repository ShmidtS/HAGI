import HAGI

open HAGI

def sampleShape : List Nat := [2, 3]

def sampleIndex : CoreTypes.AxisIndex sampleShape := ⟨[1, 2], by decide⟩

#check CoreTypes.index_to_offset_lt_numel sampleShape sampleIndex
#check HDIM.unit_rotor_sandwich_identity
#check MSA.empty_route_within_slots

#eval "CoreTypes.index_to_offset_lt_numel: type-checks"
#eval "HDIM.unit_rotor_sandwich_identity: type-checks"
#eval "MSA.empty_route_within_slots: type-checks"

def main : IO Unit := do
  IO.println "HAGI formalization imports loaded"
  IO.println "Lean traceability contracts: OK"
