import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb

from stablehlo_coreml.utils import (
    _resolve_slice_spec,
    index_by_slices,
    update_tensor_by_slice,
)


class TestResolveSliceSpec:

    def test_non_trivial_slice(self):
        """
        A slice [start:end:stride] selects ceil((end - start) / stride) elements.
        Previously the shape was computed as `end - start // stride` due to a
        missing parenthesis, which only worked for start=0, stride=1.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            spec = _resolve_slice_spec(x, [slice(2, 9, 3), slice(5, 20, 1)])
            # [2:9:3] selects indices 2, 5, 8 -> 3 elements (ceil(7 / 3))
            assert spec.start_indices == [2, 5]
            assert spec.end_indices == [9, 20]
            assert spec.strides == [3, 1]
            assert spec.shape == [3, 15]
            return x

    def test_slice_with_defaults(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 6))])
        def prog(x):
            spec = _resolve_slice_spec(x, [slice(None), slice(1, None, 2)])
            assert spec.start_indices == [0, 1]
            assert spec.end_indices == [4, 6]
            assert spec.strides == [1, 2]
            # [1:6:2] selects indices 1, 3, 5 -> 3 elements
            assert spec.shape == [4, 3]
            return x

    def test_ellipsis_and_integer_index(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4, 5))])
        def prog(x):
            spec = _resolve_slice_spec(x, [Ellipsis, 2])
            assert spec.start_indices == [0, 0, 2]
            assert spec.end_indices == [3, 4, 3]
            assert spec.strides == [1, 1, 1]
            assert spec.shape == [3, 4, 1]
            return x

    def test_index_by_slices_output_shape(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            result = index_by_slices(x, [slice(2, 9, 3), slice(5, 20, 1)])
            assert tuple(result.shape) == (3, 15)
            return result

    def test_update_tensor_by_slice_with_non_trivial_slice(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 15))], opset_version=ct.target.iOS18)
        def prog(x):
            buffer = np.zeros((10, 20), dtype=np.float32)
            result = update_tensor_by_slice(buffer, [slice(2, 9, 3), slice(5, 20, 1)], x)
            assert tuple(result.shape) == (10, 20)
            return result
