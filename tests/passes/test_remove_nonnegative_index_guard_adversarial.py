"""Adversarial cases for `remove_nonnegative_index_guard`: nested consumers,
loops, stacked guards, constant and symbolic indices, and the full pipeline."""

import time

import coremltools as ct
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import get_op_types_in_program

# Importing the package registers the passes with coremltools' PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401
from stablehlo_coreml.passes.utils import build_pass_pipeline
from tests.passes.helpers import apply_pass, count_ops, ops_of_type, predict

PASS_NAME = "common::remove_nonnegative_index_guard"

INDICES = np.array([-3, 0, 2, 9], dtype=np.int32)
DATA = np.arange(18, dtype=np.float32).reshape(6, 3)


def _guard(var, size):
    nptype = types.nptype_from_builtin(var.dtype)
    cond = mb.greater_equal(x=var, y=nptype(0))
    plus = mb.add(x=var, y=nptype(size))
    return mb.select(cond=cond, a=var, b=plus)


def _clamped(var, size):
    return mb.minimum(x=mb.maximum(x=var, y=np.int32(0)), y=np.int32(size - 1))


class TestAdversarial:

    def test_stacked_guards_are_both_removed(self):
        """A guard on a guard: removing the inner one must not confuse the outer."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            return mb.gather(x=data, indices=_guard(_guard(_clamped(indices, 6), 6), 6), axis=0)

        before = apply_pass(prog, PASS_NAME)

        assert get_op_types_in_program(prog) == ["maximum", "minimum", "gather"]
        np.testing.assert_array_equal(
            predict(prog, data=DATA, indices=INDICES),
            predict(before, data=DATA, indices=INDICES),
        )

    def test_guard_consumed_only_inside_a_later_nested_block(self):
        """The select lives in the outer block, its only use is inside a `cond` body."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
            mb.TensorSpec(shape=(1,)),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices, flag):
            guarded = _guard(_clamped(indices, 6), 6)
            pred = mb.squeeze(x=mb.cast(x=flag, dtype="bool"))

            def gathered():
                return mb.gather(x=data, indices=guarded, axis=0)

            def zeros():
                return mb.fill(shape=(4, 3), value=0.0)

            return mb.cond(pred=pred, _true_fn=gathered, _false_fn=zeros)

        before = apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select", recurse=True) == 0
        gather = ops_of_type(prog, "gather", recurse=True)[0]
        assert gather.indices is ops_of_type(prog, "minimum")[0].outputs[0]
        flag = np.array([1.0], dtype=np.float32)
        np.testing.assert_array_equal(
            predict(prog, data=DATA, indices=INDICES, flag=flag),
            predict(before, data=DATA, indices=INDICES, flag=flag),
        )

    def test_while_loop_keeps_the_guard_on_the_loop_variable(self):
        """Inside a loop body: the clamped outer indices lose their guard, the
        loop-carried counter (a block input, unprovable) keeps it."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            clamped = _clamped(indices, 6)

            def cond_fn(i, acc):
                return mb.less(x=i, y=np.int32(3))

            def body_fn(i, acc):
                rows = mb.gather(x=data, indices=_guard(clamped, 6), axis=0)
                row = mb.gather(x=data, indices=_guard(mb.sub(x=i, y=np.int32(1)), 6), axis=0)
                return mb.add(x=i, y=np.int32(1)), mb.add(x=acc, y=mb.add(x=rows, y=row))

            _, out = mb.while_loop(
                _cond=cond_fn, _body=body_fn,
                loop_vars=(np.int32(0), np.zeros((4, 3), dtype=np.float32)),
            )
            return out

        assert count_ops(prog, "select", recurse=True) == 2

        before = apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select", recurse=True) == 1
        np.testing.assert_array_equal(
            predict(prog, data=DATA, indices=INDICES),
            predict(before, data=DATA, indices=INDICES),
        )

    @pytest.mark.parametrize("values, removed", [([0, 2, 5, 1], True), ([0, -2, 5, 1], False)])
    def test_constant_indices(self, values, removed):
        @mb.program(input_specs=[mb.TensorSpec(shape=(6, 3))], opset_version=ct.target.iOS18)
        def prog(data):
            idx = mb.const(val=np.array(values, dtype=np.int32))
            return mb.gather(x=data, indices=_guard(idx, 6), axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == (0 if removed else 1)

    def test_guard_is_kept_when_b_has_a_different_symbol(self):
        """Two distinct symbolic lengths broadcast to a fresh symbol: not provably `a`."""
        s, t = get_new_symbol(), get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(s,), dtype=types.int32),
            mb.TensorSpec(shape=(t,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, a, b):
            clamped = _clamped(a, 6)
            cond = mb.greater_equal(x=clamped, y=np.int32(0))
            guarded = mb.select(cond=cond, a=clamped, b=_clamped(b, 6))
            return mb.gather(x=data, indices=guarded, axis=0)

        apply_pass(prog, PASS_NAME, skip_output_shape_check=True)

        assert count_ops(prog, "select") == 1

    def test_guard_is_kept_when_b_is_concretely_wider(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, a, b):
            clamped = _clamped(a, 6)
            cond = mb.greater_equal(x=clamped, y=np.int32(0))
            guarded = mb.select(cond=cond, a=clamped, b=b)
            return mb.gather(x=data, indices=guarded, axis=0)

        apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 1

    def test_data_movement_chain_with_negative_stride(self):
        """transpose/slice_by_index(stride=-1)/reshape/tile/concat only re-address."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(2, 2), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            c = _clamped(indices, 6)
            c = mb.transpose(x=c, perm=[1, 0])
            c = mb.slice_by_index(x=c, begin=[1, 1], end=[-3, -3], stride=[-1, -1])
            c = mb.reshape(x=c, shape=[-1])
            c = mb.concat(values=[c, mb.tile(x=c, reps=[2])], axis=0)
            return mb.gather(x=data, indices=_guard(c, 6), axis=0)

        before = apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 0
        idx = np.array([[-3, 9], [2, -1]], dtype=np.int32)
        np.testing.assert_array_equal(
            predict(prog, data=DATA, indices=idx),
            predict(before, data=DATA, indices=idx),
        )

    def test_pass_is_idempotent(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(4,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            return mb.gather(x=data, indices=_guard(_clamped(indices, 6), 6), axis=0)

        apply_pass(prog, PASS_NAME)
        first = get_op_types_in_program(prog)
        apply_pass(prog, PASS_NAME)
        assert get_op_types_in_program(prog) == first

    def test_float_nan_through_maximum(self):
        """`maximum(NaN, 0)` is where a float `select(x >= 0, x, ...)` could still
        diverge from `x`; Core ML resolves it like fmax (to 0), so no divergence."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4,))], opset_version=ct.target.iOS18)
        def prog(x):
            m = mb.maximum(x=x, y=np.float32(0.0))
            cond = mb.greater_equal(x=m, y=np.float32(0.0))
            sel = mb.select(cond=cond, a=m, b=mb.fill(shape=(4,), value=-1.0))
            return mb.mul(x=sel, y=np.float32(2.0))

        before = apply_pass(prog, PASS_NAME)

        assert count_ops(prog, "select") == 0
        x = np.array([np.nan, -2.0, 3.0, -np.inf], dtype=np.float32)
        np.testing.assert_array_equal(predict(prog, x=x), predict(before, x=x))

    @pytest.mark.parametrize("precision, selects_left", [
        # fp32: `add_int16_cast` is off, both guards go.
        (ct.precision.FLOAT32, 0),
        # fp16: `add_int16_cast` runs *before* the guard and narrows every index
        # vector to int16 first, so the guarded value is `cast(int32)(cast(int16)(..))`
        # and the pass (rightly) cannot prove it; nothing is removed.
        (ct.precision.FLOAT16, 2),
    ])
    def test_full_pipeline_matches_default_pipeline(self, precision, selects_left):
        """End to end through `build_pass_pipeline()`, next to coremltools' own
        `add_int16_cast`/`cast_optimization`, with negative and oversized indices.

        `ct.convert` mutates a milinternal program in place, so each conversion
        gets a fresh one.
        """
        def make_prog():
            @mb.program(input_specs=[
                mb.TensorSpec(shape=(6, 3)),
                mb.TensorSpec(shape=(4,), dtype=types.int32),
                mb.TensorSpec(shape=(6,)),
            ], opset_version=ct.target.iOS18)
            def prog(data, indices, mask):
                rows = mb.gather(x=data, indices=_clamped(indices, 6), axis=0)
                picked = mb.gather_nd(x=data, indices=mb.non_zero(x=mb.cast(x=mask, dtype="bool")))
                return mb.add(x=mb.reduce_sum(x=rows, axes=[0]), y=mb.reduce_sum(x=picked, axes=[0]))
            return prog

        def convert(pipeline):
            return ct.convert(
                make_prog(), source="milinternal", minimum_deployment_target=ct.target.iOS18,
                compute_units=ct.ComputeUnit.CPU_ONLY, compute_precision=precision,
                pass_pipeline=pipeline,
            )

        ours = convert(build_pass_pipeline())
        theirs = convert(ct.PassPipeline.DEFAULT)
        inputs = {"data": DATA, "indices": INDICES, "mask": np.array([1, 0, 1, 1, 0, 0], dtype=np.float32)}
        a = np.array(next(iter(ours.predict(inputs).values())))
        b = np.array(next(iter(theirs.predict(inputs).values())))
        np.testing.assert_allclose(a, b, rtol=1e-3)

        assert count_ops(theirs._mil_program, "select", recurse=True) == 2
        assert count_ops(ours._mil_program, "select", recurse=True) == selects_left

    def test_a_fanned_out_proof_graph_is_not_walked_once_per_path(self):
        """`concat(values=[v] * 8)` names the same var eight times, so a stack of
        them has `8 ** levels` distinct paths back to the clamp underneath.

        The proof walks a DAG, so it is linear in the number of ops once its
        answers are memoized -- and exponential without, which at these numbers
        (2M paths, and `MAX_PROOF_DEPTH` allows far worse) takes tens of seconds.
        """
        k, levels = 8, 7

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(6, 3)),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
        ], opset_version=ct.target.iOS18)
        def prog(data, indices):
            fanned = mb.maximum(x=indices, y=np.int32(0))
            for _ in range(levels):
                fanned = mb.concat(values=[fanned] * k, axis=0)
            return mb.gather(x=data, indices=_guard(fanned, 6), axis=0)

        assert count_ops(prog, "concat") == levels
        assert count_ops(prog, "select") == 1

        start = time.perf_counter()
        apply_pass(prog, PASS_NAME)
        elapsed = time.perf_counter() - start

        assert count_ops(prog, "select") == 0
        assert elapsed < 2.0, f"the pass took {elapsed:.1f}s on {k ** levels} paths"
