import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

from stablehlo_coreml.passes.fuse_reduce_keep_dims import _REDUCE_OPS
from tests.utils import get_model_instruction_types, run_and_compare

PASS_NAME = "common::fuse_reduce_keep_dims"
DCE_PASS_NAME = "common::dead_code_elimination"


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME)
    # The pass leaves the matched ops behind; DCE is what removes them.
    apply_pass_and_basic_check(prog, DCE_PASS_NAME)


def _ops_of_type(prog, op_type):
    return [op for op in prog.functions["main"].operations if op.op_type == op_type]


class TestFuseReduceKeepDims:
    """Unit tests on hand-built MIL programs."""

    @pytest.mark.parametrize("reduce_op", sorted(_REDUCE_OPS))
    def test_every_supported_reduce_op_is_fused(self, reduce_op):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = getattr(mb, reduce_op)(x=x, axes=[2], keep_dims=False)
            return mb.reshape(x=reduced, shape=[2, 8, 1])

        assert get_op_types_in_program(prog) == [reduce_op, "reshape"]
        _apply(prog)
        assert get_op_types_in_program(prog) == [reduce_op]

        fused = _ops_of_type(prog, reduce_op)
        assert len(fused) == 1
        assert fused[0].keep_dims.val
        assert prog.functions["main"].outputs[0].shape == (2, 8, 1)
        assert_model_is_valid(
            prog, {"x": (2, 8, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_multiple_axes_are_fused(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[0, 2], keep_dims=False)
            return mb.reshape(x=reduced, shape=[1, 8, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum"]
        assert prog.functions["main"].outputs[0].shape == (1, 8, 1)

    def test_negative_axes_are_normalized(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_mean(x=x, axes=[-1], keep_dims=False)
            return mb.reshape(x=reduced, shape=[2, 8, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_mean"]
        assert list(_ops_of_type(prog, "reduce_mean")[0].axes.val) == [2]

    def test_reduce_over_all_axes_is_fused(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[0, 1], keep_dims=False)
            return mb.reshape(x=reduced, shape=[1, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum"]
        assert prog.functions["main"].outputs[0].shape == (1, 1)

    def test_symbolic_dims_are_fused(self):
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=False)
            return mb.reshape(x=reduced, shape=[-1, 8, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum"]
        assert prog.functions["main"].outputs[0].shape == (batch, 8, 1)

    def test_keep_dims_already_true_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=True)
            return mb.reshape(x=reduced, shape=[2, 8, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "reshape"]

    def test_reshape_to_a_different_shape_is_untouched(self):
        """The reshape merges dimensions instead of re-inserting the reduced one."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=False)
            return mb.reshape(x=reduced, shape=[16, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "reshape"]

    def test_size_one_axis_in_the_wrong_place_is_untouched(self):
        """`(2, 8, 16) -> reduce axis 1 -> (2, 16)` reshaped to `(2, 16, 1)`, not `(2, 1, 16)`."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[1], keep_dims=False)
            return mb.reshape(x=reduced, shape=[2, 16, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "reshape"]

    def test_arg_reductions_are_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_argmax(x=x, axis=2, keep_dims=False)
            return mb.reshape(x=reduced, shape=[2, 8, 1])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_argmax", "reshape"]

    def test_reduce_with_extra_consumers_is_kept(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=False)
            keep_dims = mb.reshape(x=reduced, shape=[2, 8, 1])
            squeezed = mb.mul(x=reduced, y=2.0)
            return keep_dims, squeezed

        _apply(prog)
        ops = get_op_types_in_program(prog)
        # The reshape is gone, but the keep_dims=False reduce stays for `mul`.
        assert ops.count("reshape") == 0
        assert ops.count("reduce_sum") == 2
        main = prog.functions["main"]
        assert main.outputs[0].shape == (2, 8, 1)
        assert main.outputs[1].shape == (2, 8)

    def test_two_reshapes_on_the_same_reduce(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=False)
            return mb.reshape(x=reduced, shape=[2, 8, 1]), mb.reshape(x=reduced, shape=[2, 8, 1])

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("reshape") == 0
        assert ops.count("reduce_sum") == 2

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=False)
                return mb.reshape(x=reduced, shape=[2, 8, 1])

            def false_fn():
                return mb.reduce_max(x=x, axes=[2], keep_dims=True)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        assert "reshape" in get_op_types_in_program(prog, recurse=True)
        _apply(prog)
        assert "reshape" not in get_op_types_in_program(prog, recurse=True)

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8, 16))])
        def prog(x):
            reduced = mb.reduce_sum(x=x, axes=[2], keep_dims=False)
            return mb.reshape(x=reduced, shape=[2, 8, 1])

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestFuseReduceKeepDimsEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_rms_norm_becomes_reduce_mean(self):
        def rms_norm(x):
            return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)

        cml_model = run_and_compare(rms_norm, [jax.ShapeDtypeStruct((2, 8, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        # Removing the keep-dims reshape lets coremltools' own `fuse_reduce_mean`
        # collapse `reduce_sum -> mul(1/N)` into a single `reduce_mean`.
        assert ops.count("reduce_mean") == 1
        assert "reduce_sum" not in ops
        assert "reshape" not in ops

    def test_mean_and_variance_become_reduce_means(self):
        def layer_norm(x):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            return (x - mean) * jax.lax.rsqrt(var + 1e-6)

        cml_model = run_and_compare(layer_norm, [jax.ShapeDtypeStruct((2, 8, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert ops.count("reduce_mean") == 2
        assert "reduce_sum" not in ops

    def test_keepdims_max_keeps_a_single_reduce(self):
        def f(x):
            return x - jnp.max(x, axis=1, keepdims=True)

        cml_model = run_and_compare(f, [jax.ShapeDtypeStruct((4, 6, 8), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert ops.count("reduce_max") == 1
        assert "reshape" not in ops

    def test_symbolic_batch_dim(self):
        symbolic_shape = jax.export.symbolic_shape("(b, 8, 16)")

        def f(x):
            return jnp.sum(x, axis=-1, keepdims=True)

        from tests.utils import run_and_compare_symbolic  # noqa: PLC0415

        cml_model = run_and_compare_symbolic(
            f,
            [jax.ShapeDtypeStruct(symbolic_shape, jnp.float32)],
            [(np.zeros((3, 8, 16), dtype=np.float32),), (np.ones((5, 8, 16), dtype=np.float32),)],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("reduce_sum") == 1
        assert "reshape" not in ops
