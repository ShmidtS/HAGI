import Lake
open Lake DSL

package hagi_formalization where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.30.0"

lean_lib HAGI where
  roots := #[`HAGI]

@[test_driver]
lean_exe TestRunner where
  root := `TestRunner
