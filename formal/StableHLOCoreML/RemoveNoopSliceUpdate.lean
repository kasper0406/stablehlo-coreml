/-
  Kernel-checked semantics for the generated remove_noop_slice_update rule.

  `sliceSelected` and `routedCoordinates` model the non-squeezed plain-slice
  fragment used by this rule.  They are not a general model of MIL slicing:
  the `toNat` totalization used here does not claim to model negative bounds,
  non-unit or negative strides, or masked and squeezed slices outside the
  accepted domain.  The generated `ruleMatches` predicate and
  `ruleMatches_sound` establish that the accepted fragment has begin zero,
  stride one, no begin or squeeze mask, equal dimensions, and effective full
  end coverage.  Correspondence between this model and the actual graph
  adapter/type checker remains a trusted interface boundary.
  The theorem consumes the generated predicate; it does not define acceptance
  by branching inside the semantic function.  Tensor values are functions on
  a subtype of valid destination coordinates, so an arbitrary pure context
  cannot observe out-of-bounds function values.
-/

import Std.Tactic
import StableHLOCoreML.Generated.RemoveNoopSliceUpdateRule

namespace StableHLOCoreML
namespace RemoveNoopSliceUpdate

open Generated.RemoveNoopSliceUpdateRule

abbrev Valuation := String → Nat

def dimValue (valuation : Valuation) : Dim → Nat
  | .fixed value => value
  | .symbol id => valuation id

def dimensions (axes : List Axis) (valuation : Valuation) : List Nat :=
  axes.map (fun axis => dimValue valuation axis.dim)

def updateDimensions (axes : List Axis) (valuation : Valuation) : List Nat :=
  axes.map (fun axis => dimValue valuation axis.updateDim)

def outputDimensions (axes : List Axis) (valuation : Valuation) : List Nat :=
  axes.map (fun axis => dimValue valuation axis.outputDim)

def nonnegative (value : Int) : Nat := value.toNat

def sliceBegin (axis : Axis) : Nat := nonnegative axis.begin

def sliceEnd (axis : Axis) (dim : Nat) : Nat :=
  if axis.endMask then dim else nonnegative axis.stop

def sliceStride (axis : Axis) : Nat := nonnegative axis.stride

def sliceSelected (axis : Axis) (dim index : Nat) : Bool :=
  index < dim &&
    sliceBegin axis ≤ index &&
    index < sliceEnd axis dim &&
    (index - sliceBegin axis) % sliceStride axis == 0

def updateIndex (axis : Axis) (index : Nat) : Nat :=
  (index - sliceBegin axis) / sliceStride axis

def validCoordinates : List Nat → List Nat → Prop
  | [], [] => True
  | index :: indices, dim :: dims => index < dim ∧ validCoordinates indices dims
  | _, _ => False

def Coordinate (shape : List Nat) := {coordinates : List Nat // validCoordinates coordinates shape}

abbrev Tensor (shape : List Nat) (α : Type) := Coordinate shape → α

def allSelected (axes : List Axis) (valuation : Valuation) : List Nat → Bool
  | [] => axes.isEmpty
  | index :: indices =>
      match axes with
      | [] => false
      | axis :: rest =>
          sliceSelected axis (dimValue valuation axis.dim) index &&
            allSelected rest valuation indices

def routedCoordinates : List Axis → List Nat → List Nat
  | [], _ => []
  | axis :: rest, index :: indices => updateIndex axis index :: routedCoordinates rest indices
  | _ :: _, [] => []

def sliceUpdate (axes : List Axis) (valuation : Valuation)
    (buffer update : List Nat → α) : Tensor (outputDimensions axes valuation) α :=
  fun coordinate =>
    if allSelected axes valuation coordinate.1 then
      update (routedCoordinates axes coordinate.1)
    else buffer coordinate.1

def updateTensor (axes : List Axis) (valuation : Valuation)
    (update : List Nat → α) : Tensor (outputDimensions axes valuation) α :=
  fun coordinate => update coordinate.1

theorem axisMatches_full_coverage
    (axis : Axis) (valuation : Valuation)
    (accepted : axisMatches axis = true) (index : Nat)
    (indexBound : index < dimValue valuation axis.dim) :
    sliceSelected axis (dimValue valuation axis.dim) index = true ∧
      updateIndex axis index = index := by
  simp [axisMatches] at accepted
  rcases accepted with
    ⟨⟨⟨⟨⟨⟨dimEq, outputEq⟩, beginEq⟩, strideEq⟩, beginMask⟩, squeezeMask⟩, endGuard⟩
  have endFull : sliceEnd axis (dimValue valuation axis.dim) =
      dimValue valuation axis.dim := by
    unfold sliceEnd
    by_cases endMask : axis.endMask
    · simp [endMask]
    · simp only [endMask, Bool.false_eq_true, false_or] at endGuard
      cases dimension : axis.dim with
      | symbol id => simp [dimEqInt, dimension] at endGuard
      | fixed value =>
          by_cases stopNegative : axis.stop < 0
          · simp [dimEqInt, dimension, stopNegative] at endGuard
          · have stopNonnegative : 0 ≤ axis.stop := Int.not_lt.mp stopNegative
            simp [endMask, dimEqInt, dimension, stopNegative, nonnegative] at endGuard ⊢
            exact endGuard.symm
  constructor
  · simp [sliceSelected, sliceBegin, sliceStride, nonnegative,
      beginEq, strideEq, endFull, indexBound, Nat.mod_one]
  · simp [updateIndex, sliceBegin, sliceStride, nonnegative,
      beginEq, strideEq, Nat.div_one]

theorem ruleMatches_axis_full_coverage
    (input : Match) (valuation : Valuation) (accepted : ruleMatches input = true)
    (axis : Axis) (axisIn : axis ∈ input.axes) (index : Nat)
    (indexBound : index < dimValue valuation axis.dim) :
    sliceSelected axis (dimValue valuation axis.dim) index = true ∧
      updateIndex axis index = index := by
  have allAccepted : input.axes.all axisMatches = true := by
    have parsed := by simpa [ruleMatches, Bool.and_eq_true] using accepted
    exact List.all_eq_true.mpr parsed.2
  have axisAccepted : axisMatches axis = true := List.all_eq_true.mp allAccepted axis axisIn
  exact axisMatches_full_coverage axis valuation axisAccepted index indexBound

theorem ruleMatches_axis_dimensions
    (input : Match) (_valuation : Valuation) (accepted : ruleMatches input = true)
    (axis : Axis) (axisIn : axis ∈ input.axes) :
    axis.dim = axis.updateDim ∧ axis.dim = axis.outputDim := by
  have allAccepted : input.axes.all axisMatches = true := by
    have parsed := by simpa [ruleMatches, Bool.and_eq_true] using accepted
    exact List.all_eq_true.mpr parsed.2
  have axisAccepted : axisMatches axis = true := List.all_eq_true.mp allAccepted axis axisIn
  simp [axisMatches] at axisAccepted
  rcases axisAccepted with
    ⟨⟨⟨⟨⟨⟨dimEq, outputEq⟩, beginEq⟩, strideEq⟩, beginMask⟩, squeezeMask⟩, endGuard⟩
  exact ⟨dimEq, outputEq⟩

theorem ruleMatches_axis_domain
    (input : Match) (_valuation : Valuation) (accepted : ruleMatches input = true)
    (axis : Axis) (axisIn : axis ∈ input.axes) :
    axis.begin = 0 ∧ axis.stride = 1 ∧
      axis.beginMask = false ∧ axis.squeezeMask = false := by
  have allAccepted : input.axes.all axisMatches = true := by
    have parsed := by simpa [ruleMatches, Bool.and_eq_true] using accepted
    exact List.all_eq_true.mpr parsed.2
  have axisAccepted : axisMatches axis = true := List.all_eq_true.mp allAccepted axis axisIn
  simp [axisMatches] at axisAccepted
  rcases axisAccepted with
    ⟨⟨⟨⟨⟨⟨dimEq, outputEq⟩, beginEq⟩, strideEq⟩, beginMask⟩, squeezeMask⟩, endGuard⟩
  exact ⟨beginEq, strideEq, beginMask, squeezeMask⟩

theorem ruleMatches_dtype_equal
    (input : Match) (accepted : ruleMatches input = true) :
    input.xDtype = input.updateDtype ∧ input.xDtype = input.outputDtype := by
  have parsed := by simpa [ruleMatches, Bool.and_eq_true] using accepted
  exact ⟨parsed.1.1, parsed.1.2⟩

theorem axisMatches_dimensions
    (axis : Axis) (accepted : axisMatches axis = true) :
    axis.dim = axis.updateDim ∧ axis.dim = axis.outputDim := by
  simp [axisMatches] at accepted
  rcases accepted with
    ⟨⟨⟨⟨⟨⟨dimEq, outputEq⟩, beginEq⟩, strideEq⟩, beginMask⟩, squeezeMask⟩, endGuard⟩
  exact ⟨dimEq, outputEq⟩

theorem map_eq_of_forall
    (f g : Axis → β) (axes : List Axis)
    (same : ∀ axis, axis ∈ axes → f axis = g axis) :
    axes.map f = axes.map g := by
  induction axes with
  | nil => rfl
  | cons axis axes inductionHypothesis =>
      simp only [List.map_cons]
      have axisSame : f axis = g axis := same axis (by simp)
      have tailSame : ∀ item, item ∈ axes → f item = g item := by
        intro item itemIn
        exact same item (by simp [itemIn])
      simp [axisSame, inductionHypothesis tailSame]

theorem ruleMatches_dimensions_equal
    (input : Match) (valuation : Valuation) (accepted : ruleMatches input = true) :
    dimensions input.axes valuation = updateDimensions input.axes valuation ∧
      dimensions input.axes valuation = outputDimensions input.axes valuation := by
  constructor
  · apply map_eq_of_forall
    intro axis axisIn
    exact congrArg (dimValue valuation)
      (ruleMatches_axis_dimensions input valuation accepted axis axisIn).1
  · apply map_eq_of_forall
    intro axis axisIn
    exact congrArg (dimValue valuation)
      (ruleMatches_axis_dimensions input valuation accepted axis axisIn).2

theorem ruleMatches_sound
    (input : Match) (valuation : Valuation) (accepted : ruleMatches input = true) :
    input.xDtype = input.updateDtype ∧ input.xDtype = input.outputDtype ∧
      dimensions input.axes valuation = updateDimensions input.axes valuation ∧
      dimensions input.axes valuation = outputDimensions input.axes valuation ∧
      (∀ axis, axis ∈ input.axes →
        axis.begin = 0 ∧ axis.stride = 1 ∧
          axis.beginMask = false ∧ axis.squeezeMask = false) := by
  have dtype := ruleMatches_dtype_equal input accepted
  have dims := ruleMatches_dimensions_equal input valuation accepted
  have domain := fun axis axisIn =>
    ruleMatches_axis_domain input valuation accepted axis axisIn
  exact ⟨dtype.1, dtype.2, dims.1, dims.2, domain⟩

theorem allSelected_of_axisMatches (valuation : Valuation) :
    ∀ (axes : List Axis) (coordinates : List Nat),
      (∀ axis, axis ∈ axes → axisMatches axis = true) →
      validCoordinates coordinates (axes.map (fun axis => dimValue valuation axis.outputDim)) →
      allSelected axes valuation coordinates = true := by
  intro axes
  induction axes with
  | nil =>
      intro coordinates axisAccepted valid
      cases coordinates with
      | nil => simp [allSelected]
      | cons index indices =>
          simp [validCoordinates, List.map] at valid
  | cons axis axes inductionHypothesis =>
      intro coordinates axisAccepted valid
      cases coordinates with
      | nil =>
          simp [validCoordinates, List.map] at valid
      | cons index indices =>
          simp [validCoordinates] at valid
          have axisProof := axisAccepted axis (by simp)
          have axisDims := axisMatches_dimensions axis axisProof
          have indexBound : index < dimValue valuation axis.dim := by
            simpa [axisDims.2] using valid.1
          have axisFacts := axisMatches_full_coverage axis valuation axisProof index indexBound
          have tailValid : validCoordinates indices
              (axes.map (fun item => dimValue valuation item.outputDim)) := by
            simpa using valid.2
          have tailFacts := inductionHypothesis indices
            (fun item itemIn => axisAccepted item (List.mem_cons_of_mem axis itemIn))
            tailValid
          simp [allSelected, axisFacts.1, tailFacts]

theorem routedCoordinates_of_axisMatches (valuation : Valuation) :
    ∀ (axes : List Axis) (coordinates : List Nat),
      (∀ axis, axis ∈ axes → axisMatches axis = true) →
      validCoordinates coordinates (axes.map (fun axis => dimValue valuation axis.outputDim)) →
      routedCoordinates axes coordinates = coordinates := by
  intro axes
  induction axes with
  | nil =>
      intro coordinates axisAccepted valid
      cases coordinates with
      | nil => rfl
      | cons index indices =>
          simp [validCoordinates, List.map] at valid
  | cons axis axes inductionHypothesis =>
      intro coordinates axisAccepted valid
      cases coordinates with
      | nil =>
          simp [validCoordinates, List.map] at valid
      | cons index indices =>
          simp [validCoordinates] at valid
          have axisProof := axisAccepted axis (by simp)
          have axisDims := axisMatches_dimensions axis axisProof
          have indexBound : index < dimValue valuation axis.dim := by
            simpa [axisDims.2] using valid.1
          have axisFacts := axisMatches_full_coverage axis valuation axisProof index indexBound
          have tailValid : validCoordinates indices
              (axes.map (fun item => dimValue valuation item.outputDim)) := by
            simpa using valid.2
          have tailFacts := inductionHypothesis indices
            (fun item itemIn => axisAccepted item (List.mem_cons_of_mem axis itemIn))
            tailValid
          simp [routedCoordinates, axisFacts.2, tailFacts]

theorem ruleMatches_slice_update_pointwise
    (input : Match) (valuation : Valuation) (accepted : ruleMatches input = true)
    (buffer update : List Nat → α) (coordinate : Coordinate (outputDimensions input.axes valuation)) :
    sliceUpdate input.axes valuation buffer update coordinate =
      update coordinate.1 := by
  have valid : validCoordinates coordinate.1 (outputDimensions input.axes valuation) := coordinate.2
  have axisAccepted : ∀ axis, axis ∈ input.axes → axisMatches axis = true := by
    intro axis axisIn
    have allAccepted : input.axes.all axisMatches = true := by
      have parsed := by simpa [ruleMatches, Bool.and_eq_true] using accepted
      exact List.all_eq_true.mpr parsed.2
    exact List.all_eq_true.mp allAccepted axis axisIn
  have selected := allSelected_of_axisMatches valuation input.axes coordinate.1 axisAccepted
    (by simpa [outputDimensions] using valid)
  have routed := routedCoordinates_of_axisMatches valuation input.axes coordinate.1 axisAccepted
    (by simpa [outputDimensions] using valid)
  simp [sliceUpdate, selected, routed]

theorem ruleMatches_slice_update_eq_update
    (input : Match) (valuation : Valuation) (accepted : ruleMatches input = true)
    (buffer update : List Nat → α) :
    sliceUpdate input.axes valuation buffer update = updateTensor input.axes valuation update := by
  funext coordinate
  exact ruleMatches_slice_update_pointwise input valuation accepted buffer update coordinate

theorem ruleMatches_context_preserves
    (input : Match) (valuation : Valuation) (accepted : ruleMatches input = true)
    (buffer update : List Nat → α)
    (context : Tensor (outputDimensions input.axes valuation) α → β) :
    context (sliceUpdate input.axes valuation buffer update) =
      context (updateTensor input.axes valuation update) := by
  exact congrArg context
    (ruleMatches_slice_update_eq_update input valuation accepted buffer update)

inductive RewriteClosure (step : γ → γ → Prop) : γ → γ → Prop where
  | refl (value : γ) : RewriteClosure step value value
  | tail {before middle after : γ} :
      step before middle → RewriteClosure step middle after → RewriteClosure step before after

theorem closure_preserves
    (step : γ → γ → Prop)
    (stepSound : ∀ before after, step before after → before = after)
    {before after : γ} (closed : RewriteClosure step before after) :
    before = after := by
  induction closed with
  | refl value => rfl
  | tail stepProof closure inductionHypothesis =>
      exact (stepSound _ _ stepProof).trans inductionHypothesis

theorem pure_context_preserves
    (context : Tensor shape α → β) {before after : Tensor shape α}
    (same : before = after) : context before = context after := by
  exact congrArg context same

inductive CheckedContextStep (α : Type) (β : Type) : β → β → Prop where
  | replace (input : Match) (valuation : Valuation)
      (accepted : ruleMatches input = true)
      (buffer update : List Nat → α)
      (context : Tensor (outputDimensions input.axes valuation) α → β) :
      CheckedContextStep α β
        (context (sliceUpdate input.axes valuation buffer update))
        (context (updateTensor input.axes valuation update))

theorem checkedContextStep_sound {before after : β}
    (step : CheckedContextStep α β before after) : before = after := by
  cases step with
  | replace input valuation accepted buffer update context =>
      exact ruleMatches_context_preserves input valuation accepted buffer update context

theorem checkedContextClosure_sound {before after : β}
    (closed : RewriteClosure (CheckedContextStep α β) before after) : before = after := by
  exact closure_preserves (CheckedContextStep α β)
    (fun before after step => checkedContextStep_sound step) closed

end RemoveNoopSliceUpdate
end StableHLOCoreML
