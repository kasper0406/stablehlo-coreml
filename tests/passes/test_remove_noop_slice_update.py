import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

# Importing the package registers the passes with coremltools' PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401


class TestRemoveNoopSliceUpdate:

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

    def test_not_removed_if_end_mask(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20))
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=buffer.shape, end_mask=[True, False])
            return x
        self.__test_program(prog, should_remove=False)

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
