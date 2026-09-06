import coremltools as ct
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import get_op_types_in_program

# Importing the package registers the passes with coremltools' PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401
from tests.passes.helpers import apply_pass, count_ops, ops_of_type, predict

PASS_NAME = "common::remove_nonnegative_index_guard"

# The indices the test programs gather with. The negative one is what the clamp
# in `_clamped` (and the guard the pass removes) is there for.
INDICES = np.array([-3, 0, 2, 9], dtype=np.int32)
DATA = np.arange(18, dtype=np.float32).reshape(6, 3)


def _guard(var, size):
    """The wrapping guard coremltools' `guard_negative_gather_indices` emits."""
    nptype = types.nptype_from_builtin(var.dtype)
    cond = mb.greater_equal(x=var, y=nptype(0))
    plus = mb.add(x=var, y=nptype(size))
    return mb.select(cond=cond, a=var, b=plus)


def _clamped(var, size):
    """The clamp the converter's `op_gather` puts on start indices."""
    return mb.minimum(x=mb.maximum(x=var, y=np.int32(0)), y=np.int32(size - 1))


class TestRemoveNonnegativeIndexGuard:

    def test_guard_on_clamped_indices_is_removed(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            return mb.gather(x=data, indices=_guard(_clamped(indices, 6), 6), axis=0)

        assert get_op_types_in_program(prog) == [
            "maximum", "minimum", "greater_equal", "add", "select", "gather",
        ]

        apply_pass(prog, PASS_NAME)

        assert get_op_types_in_program(prog) == ["maximum", "minimum", "gather"]
        # The gather now indexes with the clamp itself.
        gather = ops_of_type(prog, "gather")[0]
        assert gather.indices is ops_of_type(prog, "minimum")[0].outputs[0]

    def test_the_result_is_unchanged(self):
        """Same predictions with and without the guard, negative indices included."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            return mb.gather(x=data, indices=_guard(_clamped(indices, 6), 6), axis=0)

        before = apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 0
        np.testing.assert_array_equal(
            predict(prog, data=DATA, indices=INDICES),
            predict(before, data=DATA, indices=INDICES),
        )

    def test_guard_on_gather_nd_of_non_zero_indices_is_removed(self):
        """`non_zero` indices are non-negative by construction, dynamic shape and all.

        This is the shape the exported models hit most often: one guarded index
        vector of unknown length feeding a `gather_nd`.
        """
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6,)),
            mb.TensorSpec(shape=(6,), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(data, mask):
            return mb.gather_nd(x=data, indices=_guard(mb.non_zero(x=mask), 6))

        assert count_ops(prog, "select") == 1

        apply_pass(prog, PASS_NAME)

        assert get_op_types_in_program(prog) == ["non_zero", "gather_nd"]

    def test_guard_on_indices_gathered_from_a_clamped_tensor_is_removed(self):
        """Non-negativity carries through the ops that only move elements around."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4, 1), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            picked = mb.gather(x=_clamped(indices, 6), indices=np.int32([0]), axis=1)
            return mb.gather(x=data, indices=_guard(mb.squeeze(x=picked, axes=[1]), 6), axis=0)

        assert count_ops(prog, "select") == 1

        apply_pass(prog, PASS_NAME)

        assert get_op_types_in_program(prog) == ["maximum", "minimum", "gather", "squeeze", "gather"]

    def test_one_guard_feeding_several_gathers_is_removed_once(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            guarded = _guard(_clamped(indices, 6), 6)
            return (
                mb.gather(x=data, indices=guarded, axis=0),
                mb.gather(x=data, indices=guarded, axis=1),
            )

        apply_pass(prog, PASS_NAME)

        assert get_op_types_in_program(prog) == ["maximum", "minimum", "gather", "gather"]

    def test_guard_is_kept_when_the_indices_can_be_negative(self):
        """Without the clamp the wrap is what makes a negative index work."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            return mb.gather(x=data, indices=_guard(indices, 6), axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 1
        np.testing.assert_array_equal(
            predict(prog, data=DATA, indices=np.array([-3, 0, 2, 5], dtype=np.int32)),
            DATA[[3, 0, 2, 5]],
        )

    def test_guard_is_kept_when_only_the_lower_clamp_is_missing(self):
        """`minimum(indices, 5)` bounds the indices from above, not from below."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            capped = mb.minimum(x=indices, y=np.int32(5))
            return mb.gather(x=data, indices=_guard(capped, 6), axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 1

    def test_guard_is_kept_when_it_guards_another_value(self):
        """The `select` has to pass through the very var its `cond` tests."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            clamped = _clamped(indices, 6)
            cond = mb.greater_equal(x=indices, y=np.int32(0))
            guarded = mb.select(cond=cond, a=clamped, b=mb.add(x=clamped, y=np.int32(6)))
            return mb.gather(x=data, indices=guarded, axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 1

    def test_guard_is_kept_behind_a_narrowing_cast(self):
        """A clamped index too large for int16 comes back out of the cast negative."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            narrowed = mb.cast(x=mb.maximum(x=indices, y=np.int32(0)), dtype="int16")
            return mb.gather(x=data, indices=_guard(mb.cast(x=narrowed, dtype="int32"), 6), axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 1

    def test_guard_is_removed_behind_an_unsigned_cast(self):
        """What `add_int16_cast` leaves on gather indices: uint16 cannot be negative."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            unsigned = mb.cast(x=indices, dtype="uint16")
            return mb.gather(x=data, indices=_guard(mb.cast(x=unsigned, dtype="int32"), 6), axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 0

    def test_guard_is_kept_when_the_guarded_value_does_not_span_the_result(self):
        """A symbolic dimension the guarded value does not carry: skip.

        `b` is what gives the `select` its length here, so passing the guarded
        value through would gather 1 row instead of `s`. The dimension is
        symbolic, so nothing can prove the two are the same.
        """
        s = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(s,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, dynamic):
            clamped = _clamped(mb.reduce_max(x=dynamic, axes=[0], keep_dims=True), 6)
            cond = mb.greater_equal(x=clamped, y=np.int32(0))
            guarded = mb.select(cond=cond, a=clamped, b=mb.add(x=clamped, y=dynamic))
            return mb.gather(x=data, indices=guarded, axis=0)

        # The `select` broadcasts its (1,) operand up to the symbolic length.
        assert ops_of_type(prog, "select")[0].outputs[0].shape != (1,)

        apply_pass(prog, PASS_NAME, skip_output_shape_check=True)

        assert count_ops(prog, "select") == 1

    def test_guard_inside_a_nested_block_is_removed(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
            mb.TensorSpec(shape=(1,), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices, flag):
            def gathered():
                return mb.gather(x=data, indices=_guard(_clamped(indices, 6), 6), axis=0)

            def zeros():
                return mb.fill(shape=(4, 3), value=0.0)

            return mb.cond(pred=mb.squeeze(x=flag), _true_fn=gathered, _false_fn=zeros)

        assert count_ops(prog, "select", recurse=True) == 1

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select", recurse=True) == 0
        assert count_ops(prog, "gather", recurse=True) == 1

    @pytest.mark.parametrize("guarded_is_output", [True, False])
    def test_guard_is_kept_when_its_result_is_a_function_output(self, guarded_is_output):
        """Replacing a block output means renaming it, which is not this pass' job."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            guarded = _guard(_clamped(indices, 6), 6)
            gather = mb.gather(x=data, indices=guarded, axis=0)
            return (guarded, gather) if guarded_is_output else (gather,)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == (1 if guarded_is_output else 0)
