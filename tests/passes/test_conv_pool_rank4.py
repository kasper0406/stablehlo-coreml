import coremltools as ct
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import assert_model_is_valid, get_op_types_in_program

# Importing the package registers the passes with coremltools' PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401
from tests.passes.helpers import apply_pass, count_ops, predict

PASS_NAME = "common::conv_pool_rank4"

# A 2x2 averaging kernel, so that a rank-4 conv is available as a chain end.
KERNEL = np.full((1, 1, 2, 2), 0.25, dtype=np.float32)


def _apply(prog):
    """Run the pass (plus DCE) and return the program as it was before."""
    return apply_pass(prog, PASS_NAME, skip_output_shape_check=True)


def _count_reshapes(prog, recurse: bool = True) -> int:
    return count_ops(prog, "reshape", recurse=recurse)


def _assert_same_prediction(prog, prev_prog, **inputs):
    """The rewritten program computes exactly what the original one did."""
    np.testing.assert_array_equal(predict(prog, **inputs), predict(prev_prog, **inputs))


class TestShareTheLift:
    """Hoisting a rank lift above the slices that feed it."""

    def test_two_pooled_slices_share_one_lift(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            top = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            bottom = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])
            pooled_top = mb.max_pool(
                x=mb.reshape(x=top, shape=[1, 1, 7, 5]),
                kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid",
            )
            pooled_bottom = mb.max_pool(
                x=mb.reshape(x=bottom, shape=[1, 1, 7, 5]),
                kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid",
            )
            return mb.add(x=pooled_top, y=pooled_bottom)

        assert _count_reshapes(prog) == 2
        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == [
            "reshape", "slice_by_index", "slice_by_index", "max_pool", "max_pool", "add"
        ]
        # The lift now sits on the source, and the slices carry the extra axes.
        lift = prog.functions["main"].find_ops(op_type="reshape")[0]
        assert lift.outputs[0].shape == (1, 1, 8, 6)
        assert lift.inputs["x"] is prog.functions["main"].inputs["x"]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(8, 6).astype(np.float32))

    def test_slice_by_size_group(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(6, 5))])
        def prog(x):
            left = mb.slice_by_size(x=x, begin=[0, 0], size=[5, 4])
            right = mb.slice_by_size(x=x, begin=[1, 1], size=[5, 4])
            a = mb.conv(x=mb.reshape(x=left, shape=[1, 1, 5, 4]), weight=KERNEL, pad_type="valid")
            b = mb.conv(x=mb.reshape(x=right, shape=[1, 1, 5, 4]), weight=KERNEL, pad_type="valid")
            return mb.add(x=a, y=b)

        prev_prog = _apply(prog)
        assert _count_reshapes(prog) == 1
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(6, 5).astype(np.float32))

    def test_four_slices_leave_one_reshape(self):
        """The shape the sphere model emits: four shifted windows per pyramid level."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(9, 7))])
        def prog(x):
            pooled = []
            for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
                window = mb.slice_by_index(x=x, begin=[row, col], end=[row + 8, col + 6])
                pooled.append(mb.max_pool(
                    x=mb.reshape(x=window, shape=[1, 1, 8, 6]),
                    kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid",
                ))
            return mb.add(x=mb.add(x=pooled[0], y=pooled[1]), y=mb.add(x=pooled[2], y=pooled[3]))

        assert _count_reshapes(prog) == 4
        prev_prog = _apply(prog)
        assert _count_reshapes(prog) == 1
        assert count_ops(prog, "slice_by_index") == 4
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(9, 7).astype(np.float32))

    def test_strided_slice_keeps_its_stride_and_masks(self):
        """`begin_mask[0]` and `end_mask[1]` below override `begin`/`end`, and must
        keep doing so once two more axes are prepended."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            even_rows = mb.slice_by_index(
                x=x, begin=[5, 0], end=[8, 3], stride=[2, 1],
                begin_mask=[True, False], end_mask=[False, True],
            )
            odd_rows = mb.slice_by_index(x=x, begin=[1, 0], end=[8, 6], stride=[2, 1])
            return mb.add(
                x=mb.reshape(x=even_rows, shape=[1, 1, 4, 6]),
                y=mb.reshape(x=odd_rows, shape=[1, 1, 4, 6]),
            )

        prev_prog = _apply(prog)
        assert _count_reshapes(prog) == 1
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(8, 6).astype(np.float32))

    def test_not_rewritten_for_a_lone_slice(self):
        """One slice would trade its reshape for a reshape of the (larger) source."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            window = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            return mb.max_pool(
                x=mb.reshape(x=window, shape=[1, 1, 7, 5]),
                kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid",
            )

        _apply(prog)
        assert get_op_types_in_program(prog) == ["slice_by_index", "reshape", "max_pool"]

    def test_not_rewritten_when_the_slices_are_smaller_than_the_source(self):
        """Two tiny windows of a big buffer: lifting the source copies more, not less."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(64, 64))])
        def prog(x):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[4, 4])
            b = mb.slice_by_index(x=x, begin=[4, 4], end=[8, 8])
            return mb.add(
                x=mb.reshape(x=a, shape=[1, 1, 4, 4]),
                y=mb.reshape(x=b, shape=[1, 1, 4, 4]),
            )

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_when_a_slice_has_a_second_consumer(self):
        """The rank-2 slice is needed anyway, so hoisting only adds an op."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            b = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])
            lifted_a = mb.reshape(x=a, shape=[1, 1, 7, 5])
            lifted_b = mb.reshape(x=b, shape=[1, 1, 7, 5])
            return mb.add(x=lifted_a, y=lifted_b), mb.abs(x=a), mb.abs(x=b)

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_a_slice_with_a_second_consumer_drops_out_of_the_group(self):
        """The other three still share a lift; the shared one keeps its own reshape."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(9, 7))])
        def prog(x):
            windows = [
                mb.slice_by_index(x=x, begin=[row, col], end=[row + 8, col + 6])
                for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
            ]
            lifted = [mb.reshape(x=window, shape=[1, 1, 8, 6]) for window in windows]
            total = lifted[0]
            for other in lifted[1:]:
                total = mb.add(x=total, y=other)
            return total, mb.abs(x=windows[3])

        prev_prog = _apply(prog)
        # Three reshapes replaced by one, plus the one the shared slice keeps.
        assert _count_reshapes(prog) == 2
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(9, 7).astype(np.float32))

    def test_not_rewritten_for_symbolic_dims(self):
        rows = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(rows, 6))])
        def prog(x):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[4, 5])
            b = mb.slice_by_index(x=x, begin=[1, 1], end=[5, 6])
            return mb.add(
                x=mb.reshape(x=a, shape=[1, 1, 4, 5]),
                y=mb.reshape(x=b, shape=[1, 1, 4, 5]),
            )

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_when_the_reshape_is_not_a_pure_rank_lift(self):
        """(7, 5) -> (1, 5, 7) reorders elements; it is not a rank-only reshape."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            b = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])
            return mb.add(
                x=mb.reshape(x=a, shape=[1, 5, 7]),
                y=mb.reshape(x=b, shape=[1, 5, 7]),
            )

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_when_the_lift_is_not_leading(self):
        """A trailing 1 does not commute with a slice the way a leading one does."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            b = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])
            return mb.add(
                x=mb.reshape(x=a, shape=[7, 5, 1]),
                y=mb.reshape(x=b, shape=[7, 5, 1]),
            )

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_for_mismatched_lifts(self):
        """Two slices lifted by different amounts cannot share one source lift."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(8, 6))])
        def prog(x):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            b = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])
            return mb.reshape(x=a, shape=[1, 1, 7, 5]), mb.reshape(x=b, shape=[1, 7, 5])

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_for_a_constant_source(self):
        """Lifting a constant would materialise a second copy of it as a weight."""
        source = np.random.rand(8, 6).astype(np.float32)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 7, 5))])
        def prog(x):
            a = mb.slice_by_index(x=source, begin=[0, 0], end=[7, 5])
            b = mb.slice_by_index(x=source, begin=[1, 1], end=[8, 6])
            lifted = mb.add(
                x=mb.reshape(x=a, shape=[1, 1, 7, 5]),
                y=mb.reshape(x=b, shape=[1, 1, 7, 5]),
            )
            return mb.add(x=lifted, y=x)

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_rewritten_inside_a_nested_block(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(8, 6)), mb.TensorSpec(shape=(1,), dtype=types.bool)
        ])
        def prog(x, pred):
            def true_fn():
                a = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
                b = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])
                return mb.add(
                    x=mb.reshape(x=a, shape=[1, 1, 7, 5]),
                    y=mb.reshape(x=b, shape=[1, 1, 7, 5]),
                )

            def false_fn():
                lone = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
                return mb.reshape(x=lone, shape=[1, 1, 7, 5])

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        assert _count_reshapes(prog) == 3
        _apply(prog)
        # The two slices of the taken branch share a lift; the lone one in the
        # other branch is not a group and keeps its reshape.
        assert _count_reshapes(prog) == 2
        assert_model_is_valid(
            prog,
            {"x": (8, 6), "pred": (1,)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_not_rewritten_when_the_slice_is_outside_the_block(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(8, 6)), mb.TensorSpec(shape=(1,), dtype=types.bool)
        ])
        def prog(x, pred):
            a = mb.slice_by_index(x=x, begin=[0, 0], end=[7, 5])
            b = mb.slice_by_index(x=x, begin=[1, 1], end=[8, 6])

            def true_fn():
                return mb.add(
                    x=mb.reshape(x=a, shape=[1, 1, 7, 5]),
                    y=mb.reshape(x=b, shape=[1, 1, 7, 5]),
                )

            def false_fn():
                return mb.reshape(x=a, shape=[1, 1, 7, 5])

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        assert _count_reshapes(prog) == 3


class TestCloseTheRoundTrip:
    """Collapsing a rank round trip made across shape-preserving ops."""

    def test_reverse_between_two_convolutions(self):
        """conv -> reshape -> reverse -> reshape -> conv, the sphere model's Sobel pair."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 6, 4))])
        def prog(x):
            first = mb.conv(x=x, weight=KERNEL, pad_type="same")
            flat = mb.reshape(x=first, shape=[6, 4])
            flipped = mb.reverse(x=flat, axes=[0, 1])
            lifted = mb.reshape(x=flipped, shape=[1, 1, 6, 4])
            return mb.conv(x=lifted, weight=KERNEL, pad_type="same")

        assert _count_reshapes(prog) == 2
        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == ["conv", "reverse", "conv"]
        # The reverse now runs on the rank-4 layout, with its axes shifted by 2.
        reverse = prog.functions["main"].find_ops(op_type="reverse")[0]
        assert list(reverse.inputs["axes"].val) == [2, 3]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(1, 1, 6, 4).astype(np.float32))

    def test_round_trip_between_unequal_layouts_leaves_one_reshape(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            flipped = mb.reverse(x=flat, axes=[1])
            trailing = mb.reshape(x=flipped, shape=[5, 4, 1])
            return mb.add(x=trailing, y=np.float32(1.0))

        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == ["reshape", "reverse", "add"]
        reverse = prog.functions["main"].find_ops(op_type="reverse")[0]
        assert list(reverse.inputs["axes"].val) == [1]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(1, 1, 5, 4).astype(np.float32))

    def test_unary_chain_is_rebuilt_at_the_outer_rank(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            chained = mb.sqrt(x=mb.abs(x=mb.exp(x=flat)))
            lifted = mb.reshape(x=chained, shape=[1, 1, 5, 4])
            return mb.add(x=lifted, y=np.float32(1.0))

        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == ["exp", "abs", "sqrt", "add"]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(1, 1, 5, 4).astype(np.float32))

    def test_exact_inverse_pair_disappears(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            return mb.add(x=mb.reshape(x=flat, shape=[1, 1, 5, 4]), y=np.float32(1.0))

        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == ["add"]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(1, 1, 5, 4).astype(np.float32))

    def test_reversing_only_a_size_one_axis_drops_the_reverse(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4, 1])
            flipped = mb.reverse(x=flat, axes=[2])
            return mb.add(x=mb.reshape(x=flipped, shape=[1, 5, 4]), y=np.float32(1.0))

        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == ["add"]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(1, 5, 4).astype(np.float32))

    def test_stacked_reshapes_collapse_in_one_run(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=mb.reshape(x=x, shape=[1, 5, 4]), shape=[5, 4])
            return mb.add(x=mb.reshape(x=flat, shape=[5, 4, 1]), y=np.float32(1.0))

        prev_prog = _apply(prog)
        assert get_op_types_in_program(prog) == ["reshape", "add"]
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(1, 1, 5, 4).astype(np.float32))

    def test_not_rewritten_when_the_chain_op_has_a_second_consumer(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            flipped = mb.reverse(x=flat, axes=[0])
            return mb.reshape(x=flipped, shape=[1, 1, 5, 4]), mb.abs(x=flipped)

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_across_a_binary_op(self):
        """`mul`'s second operand would need a rank change of its own."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4)), mb.TensorSpec(shape=(5, 4))])
        def prog(x, y):
            flat = mb.reshape(x=x, shape=[5, 4])
            return mb.reshape(x=mb.mul(x=flat, y=y), shape=[1, 1, 5, 4])

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_across_a_shape_changing_op(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            reduced = mb.reduce_sum(x=flat, axes=[1], keep_dims=True)
            return mb.reshape(x=reduced, shape=[1, 1, 5, 1])

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_when_the_reshape_is_a_block_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            return mb.reshape(x=mb.abs(x=flat), shape=[1, 1, 5, 4])

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_for_symbolic_dims(self):
        rows = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, rows, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[-1, 4])
            return mb.add(x=mb.reshape(x=mb.abs(x=flat), shape=[1, 1, -1, 4]), y=np.float32(1.0))

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_not_rewritten_when_the_reshape_permutes_elements(self):
        """(5, 4) -> (4, 5) has a different core, so the two are not the same layout."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 5, 4))])
        def prog(x):
            flat = mb.reshape(x=x, shape=[5, 4])
            return mb.add(x=mb.reshape(x=mb.abs(x=flat), shape=[1, 1, 4, 5]), y=np.float32(1.0))

        _apply(prog)
        assert _count_reshapes(prog) == 2

    def test_rewritten_inside_a_nested_block(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 1, 5, 4)), mb.TensorSpec(shape=(1,), dtype=types.bool)
        ])
        def prog(x, pred):
            def true_fn():
                flat = mb.reshape(x=x, shape=[5, 4])
                lifted = mb.reshape(x=mb.abs(x=flat), shape=[1, 1, 5, 4])
                return mb.add(x=lifted, y=np.float32(1.0))

            def false_fn():
                return mb.identity(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        assert _count_reshapes(prog) == 2
        _apply(prog)
        assert _count_reshapes(prog) == 0
        assert_model_is_valid(
            prog,
            {"x": (1, 1, 5, 4), "pred": (1,)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )


class TestConvPoolRank4Invariants:
    @pytest.mark.parametrize("build", ["lift", "round_trip"])
    def test_is_idempotent(self, build):
        if build == "lift":
            @mb.program(input_specs=[mb.TensorSpec(shape=(9, 7))])
            def prog(x):
                windows = [
                    mb.slice_by_index(x=x, begin=[row, col], end=[row + 8, col + 6])
                    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
                ]
                lifted = [mb.reshape(x=window, shape=[1, 1, 8, 6]) for window in windows]
                total = lifted[0]
                for other in lifted[1:]:
                    total = mb.add(x=total, y=other)
                return total
        else:
            @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 6, 4))])
            def prog(x):
                first = mb.conv(x=x, weight=KERNEL, pad_type="same")
                flat = mb.reshape(x=first, shape=[6, 4])
                lifted = mb.reshape(x=mb.reverse(x=flat, axes=[0, 1]), shape=[1, 1, 6, 4])
                return mb.conv(x=lifted, weight=KERNEL, pad_type="same")

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first

    def test_pyramid_level_keeps_its_numerics(self):
        """A whole level of the sphere model: four windows, pooled, then combined."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(9, 7))])
        def prog(x):
            pooled = []
            for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
                window = mb.slice_by_index(x=x, begin=[row, col], end=[row + 8, col + 6])
                pooled.append(mb.max_pool(
                    x=mb.reshape(x=window, shape=[1, 1, 8, 6]),
                    kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid",
                ))
            combined = mb.mul(
                x=mb.maximum(x=mb.add(x=pooled[0], y=pooled[1]), y=np.float32(0.0)),
                y=mb.maximum(x=mb.add(x=pooled[2], y=pooled[3]), y=np.float32(0.0)),
            )
            flat = mb.reshape(x=combined, shape=[4, 3])
            return mb.reverse(x=flat, axes=[0])

        assert _count_reshapes(prog) == 5
        prev_prog = _apply(prog)
        assert _count_reshapes(prog) == 2
        _assert_same_prediction(prog, prev_prog, x=np.random.rand(9, 7).astype(np.float32))
