import importlib

import coremltools as ct
import numpy as np
import sympy as sm
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

# Importing the package registers the passes with coremltools' PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401

pass_module = importlib.import_module("stablehlo_coreml.passes.remove_noop_slice_update")


class TestRemoveNoopSliceUpdate:

    def test_generated_rule_is_the_only_mutation_gate(self, monkeypatch):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(update):
            return mb.slice_update(
                x=np.zeros((2, 3), dtype=np.float32),
                update=update,
                begin=[0, 0],
                end=[2, 3],
            )

        monkeypatch.setattr(pass_module.rule, "matches", lambda candidate: False)
        PASS_REGISTRY["common::remove_noop_slice_update"](prog)
        assert get_op_types_in_program(prog) == ["slice_update"]

    def test_failed_replacement_removes_temporary_name_bridge(self, monkeypatch):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(update):
            return mb.slice_update(
                x=np.zeros((2, 3), dtype=np.float32),
                update=update,
                begin=[0, 0],
                end=[2, 3],
                name="written",
            )

        block = prog.functions["main"]
        monkeypatch.setattr(type(block), "try_replace_uses_of_var_after_op", lambda *args, **kwargs: False)
        PASS_REGISTRY["common::remove_noop_slice_update"](prog)
        assert get_op_types_in_program(prog) == ["slice_update"]
        prog.validate(check_essential_scope=True)

    def test_adapter_rejects_composite_symbolic_dimensions(self):
        symbol = sm.Symbol("batch", positive=True, integer=True)
        assert pass_module._SymbolInterner().dim(2 * symbol) is None

    def test_adapter_interns_symbols_by_structural_identity(self):
        positive = sm.Symbol("batch", positive=True)
        integer = sm.Symbol("batch", integer=True)
        assert str(positive) == str(integer)
        assert positive != integer

        interner = pass_module._SymbolInterner()
        assert interner.dim(positive) == pass_module.rule.Dim("symbol", "s0")
        assert interner.dim(positive) == pass_module.rule.Dim("symbol", "s0")
        assert interner.dim(integer) == pass_module.rule.Dim("symbol", "s1")

    def test_is_removed(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20))
            # Because this function ends up being a complete no-op, we need to ensure the naming of inputs and outputs
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=buffer.shape, name="x")
            return x
        self.__test_program(prog, should_remove=True)

    def test_removed_when_update_is_a_function_input(self):
        """The converter never names the `slice_update` after the function input.

        Replacing the (function output) `slice_update` result by the `update`
        var makes coremltools carry the output name over to it, which it
        refuses to do for a function input -- it raises
        `ValueError: It is not allowed to modify function inputs name.`
        and aborts the conversion. The `slice_update` still has to go.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20), dtype=np.float32)
            return mb.slice_update(
                x=buffer, update=x, begin=[0, 0], end=buffer.shape, name="slice_update_0"
            )

        assert get_op_types_in_program(prog) == ["slice_update"]

        apply_pass_and_basic_check(prog, "common::remove_noop_slice_update")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["identity"]
        # The model output keeps its name; the function input keeps its own.
        assert [output.name for output in prog.functions["main"].outputs] == ["slice_update_0"]
        assert list(prog.functions["main"].inputs) == ["x"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32")
        )

    def test_removed_outright_when_result_is_not_a_function_output(self):
        """No `identity` is needed when nothing has to take over an output name."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20), dtype=np.float32)
            updated = mb.slice_update(
                x=buffer, update=x, begin=[0, 0], end=buffer.shape, name="slice_update_0"
            )
            return mb.mul(x=updated, y=np.float32(2.0))

        assert get_op_types_in_program(prog) == ["slice_update", "mul"]

        apply_pass_and_basic_check(prog, "common::remove_noop_slice_update")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["mul"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32")
        )

    def test_not_removed_if_non_zero_begin_shape(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((11, 20))
            x = mb.slice_update(x=buffer, update=x, begin=[1, 0], end=buffer.shape)
            return x
        self.__test_program(prog, should_remove=False)

    def test_not_removed_if_end_not_matching(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((11, 20))
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=[10, 20])
            return x
        self.__test_program(prog, should_remove=False)

    def test_not_removed_if_strided(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((20, 20))
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=buffer.shape, stride=[2, 1])
            return x
        self.__test_program(prog, should_remove=False)

    def test_not_removed_if_begin_mask(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20))
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=buffer.shape, begin_mask=[True, False])
            return x
        self.__test_program(prog, should_remove=False)

    def test_removed_if_end_mask(self):
        """`end_mask[i]` means "to the end of axis i", so the axis is covered."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20))
            # `end[0]` is neglected because of the mask
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=[0, 20], end_mask=[True, False], name="x")
            return x
        self.__test_program(prog, should_remove=True)

    def test_not_removed_if_squeeze_mask(self):
        """A squeezed axis is a pure index, not a write of the whole axis."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(20,))])
        def prog(x):
            buffer = np.zeros((10, 20))
            updated = mb.slice_update(
                x=buffer, update=x, begin=[0, 0], end=[10, 20], squeeze_mask=[True, False]
            )
            return mb.mul(x=updated, y=np.float32(2.0))

        assert get_op_types_in_program(prog) == ["slice_update", "mul"]
        apply_pass_and_basic_check(prog, "common::remove_noop_slice_update")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["slice_update", "mul"]

    def test_removed_for_symbolic_shape_covered_by_end_mask(self):
        """A symbolic dimension can only be covered by `end_mask`."""
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 20)), mb.TensorSpec(shape=(batch, 20))])
        def prog(buffer, update):
            updated = mb.slice_update(
                x=buffer, update=update, begin=[0, 0], end=[0, 20], end_mask=[True, False]
            )
            return mb.mul(x=updated, y=np.float32(2.0))

        assert get_op_types_in_program(prog) == ["slice_update", "mul"]
        apply_pass_and_basic_check(prog, "common::remove_noop_slice_update")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["mul"]

    def test_not_removed_for_symbolic_shape_with_constant_end(self):
        """`end` is a constant, so it can never provably cover a symbolic axis."""
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 20)), mb.TensorSpec(shape=(batch, 20))])
        def prog(buffer, update):
            updated = mb.slice_update(x=buffer, update=update, begin=[0, 0], end=[10, 20])
            return mb.mul(x=updated, y=np.float32(2.0))

        assert get_op_types_in_program(prog) == ["slice_update", "mul"]
        apply_pass_and_basic_check(prog, "common::remove_noop_slice_update")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["slice_update", "mul"]

    def __test_program(self, prog, should_remove: bool):
        assert get_op_types_in_program(prog) == ["slice_update"]

        apply_pass_and_basic_check(
            prog, "common::remove_noop_slice_update"
        )
        _, _, _ = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        if should_remove:
            assert get_op_types_in_program(prog) == []
        else:
            assert get_op_types_in_program(prog) == ["slice_update"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32")
        )
