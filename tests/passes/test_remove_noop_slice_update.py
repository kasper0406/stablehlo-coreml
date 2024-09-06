from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)
import coremltools as ct

import numpy as np

from stablehlo_coreml import register_optimizations

register_optimizations()


class TestRemoveNoopSliceUpdate:

    def test_is_removed(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            buffer = np.zeros((10, 20))
            # Because this function ends up being a complete no-op, we need to ensure the naming of inputs and outputs
            x = mb.slice_update(x=buffer, update=x, begin=[0, 0], end=buffer.shape, name="x")
            return x
        self.__test_program(prog, should_remove=True)

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
