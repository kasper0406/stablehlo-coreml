/-
  CI audit target for the kernel-checked pilot.

  The audit is intentionally kept in a small source file so the workflow can
  fail closed on `sorryAx` or an unexpected project axiom instead of treating
  `lake build` as an independent certificate checker.
-/

import StableHLOCoreML.RemoveNoopSliceUpdate
import Lean

open Lean

set_option warningAsError true

run_cmd do
  let allowed : List Name := [`propext, `Classical.choice, `Quot.sound]
  let theoremNames : List Name := [
    `StableHLOCoreML.RemoveNoopSliceUpdate.ruleMatches_sound,
    `StableHLOCoreML.RemoveNoopSliceUpdate.ruleMatches_slice_update_eq_update,
    `StableHLOCoreML.RemoveNoopSliceUpdate.ruleMatches_context_preserves,
    `StableHLOCoreML.RemoveNoopSliceUpdate.checkedContextClosure_sound
  ]
  for theoremName in theoremNames do
    match (← Lean.getEnv).find? theoremName with
    | none => throwError m!"axiom audit target is missing: {theoremName}"
    | some (.thmInfo _) => pure ()
    | some _ => throwError m!"axiom audit target is not a theorem: {theoremName}"
    let axioms ← Lean.collectAxioms theoremName
    let unexpected := axioms.toList.filter (fun name => !allowed.contains name)
    if !unexpected.isEmpty then
      throwError m!"unexpected axioms for {theoremName}: {unexpected}"
    logInfo m!"audited {theoremName}; axioms: {axioms.toList}"
