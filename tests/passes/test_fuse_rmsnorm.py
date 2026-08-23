import math

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

from tests.utils import get_model_instruction_types, run_and_compare

PASS_NAME = "common::fuse_rmsnorm"
DCE_PASS_NAME = "common::dead_code_elimination"
# `mb.rsqrt`'s own epsilon, folded into the one the `add` contributes.
RSQRT_EPSILON = 1e-12
EPSILON = 1e-6


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME)
    # The pass removes the chain itself, but leaves its constants behind.
    apply_pass_and_basic_check(prog, DCE_PASS_NAME)


def _rmsnorm(x, scale=None, axes=(-1,), keep_dims=True, epsilon=EPSILON):
    """Build `x * rsqrt(mean(x*x, axes) + epsilon) * scale` in the current block."""
    variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=list(axes), keep_dims=keep_dims)
    normalized = mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=variance, y=epsilon)))
    if scale is None:
        return normalized
    return mb.mul(x=normalized, y=scale)


def _ops_of_type(prog, op_type):
    return [op for op in prog.functions["main"].operations if op.op_type == op_type]


def _expected_epsilon(d, epsilon=EPSILON):
    """`l2_norm` sums the squares where the chain averaged them, so eps' = d * eps."""
    return d * (epsilon + RSQRT_EPSILON)


class TestFuseRmsNorm:
    """Unit tests on hand-built MIL programs."""

    def test_chain_with_a_constant_scale_is_fused(self):
        scale = np.linspace(0.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, scale)

        assert get_op_types_in_program(prog) == ["mul", "reduce_mean", "add", "rsqrt", "mul", "mul"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]

        l2_norm = _ops_of_type(prog, "l2_norm")[0]
        assert np.isclose(l2_norm.epsilon.val, _expected_epsilon(16))
        # The `sqrt(d)` of the identity is folded into the scale constant.
        factor = _ops_of_type(prog, "mul")[0].y.val
        np.testing.assert_allclose(factor, math.sqrt(16) * scale, rtol=1e-6)
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_chain_without_a_scale_is_fused(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]
        assert np.isclose(_ops_of_type(prog, "mul")[0].y.val, math.sqrt(16))

    def test_batch_dimensions_are_fused(self):
        """`l2_norm` treats everything before the last three dims as batch."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 2, 1, 1, 16))])
        def prog(x):
            return _rmsnorm(x)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]

    def test_scalar_scale_is_absorbed(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, np.float32(2.0))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]
        assert np.isclose(_ops_of_type(prog, "mul")[0].y.val, 2.0 * math.sqrt(16))

    @pytest.mark.parametrize("shape", [(1, 4, 8, 16), (1, 8, 16), (4, 16)])
    def test_off_canonical_shape_is_left_alone(self, shape):
        """`l2_norm` reduces the last three dims, so anything but `(..., 1, 1, d)`
        would need a reshape around the op -- more ops than the rewrite saves."""
        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            return _rmsnorm(x)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert "l2_norm" not in ops
        assert ops.count("reduce_mean") == 1
        assert ops.count("rsqrt") == 1

    def test_reduction_over_another_axis_is_left_alone(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, axes=(0,))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_reduction_over_several_axes_is_left_alone(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, axes=(1, 2))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_reduce_sum_is_left_alone(self):
        """Only the mean form is the identity below; a plain sum is not."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_sum(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_squares_of_different_vars_are_left_alone(self):
        """`mul(x, y)` is not `x**2`, so the reduction is not the RMS of `x`."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16)), mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x, y):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=y), axes=[-1], keep_dims=True)
            return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_normalizing_another_var_is_left_alone(self):
        """The statistics and the value being scaled must be the same tensor."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16)), mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x, y):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            return mb.mul(x=y, y=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_non_uniform_epsilon_is_left_alone(self):
        """A per-element shift is not an epsilon and does not fold into `l2_norm`."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            shift = np.linspace(1e-6, 1e-3, 16, dtype=np.float32).reshape(1, 1, 16)
            return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=variance, y=shift)))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_shared_variance_is_left_alone(self):
        """The chain may not be rewritten while another consumer observes it."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            normalized = mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)))
            return normalized, variance

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_extra_consumer_of_the_input_is_left_alone(self):
        """The normalized value must be read by the chain and nothing else.

        Conservative: the rewrite would in fact be safe here (it never removes
        the op producing the input), but requiring the chain to own its input
        keeps the match to the shape RMSNorm actually has.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            scaled = mb.mul(x=x, y=2.0)
            return _rmsnorm(scaled), mb.add(x=scaled, y=1.0)

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_scale_varying_over_a_batch_axis_is_not_absorbed(self):
        """Such a scale cannot become part of the single folded factor, but the
        chain around it still fuses."""
        scale = np.linspace(0.5, 1.5, 32, dtype=np.float32).reshape(2, 1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, scale)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul", "mul"]
        assert np.isclose(_ops_of_type(prog, "mul")[0].y.val, math.sqrt(16))

    def test_keep_dims_false_is_left_alone(self):
        """Without `keep_dims` the `mul` broadcasts differently; not this pattern."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, keep_dims=False)

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                return _rmsnorm(x)

            def false_fn():
                # Not `identity(x)`: another reader of `x` blocks the rewrite
                # (see `test_extra_consumer_of_the_input_is_left_alone`).
                return mb.fill(shape=[1, 1, 16], value=1.0)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        ops = get_op_types_in_program(prog, recurse=True)
        assert "l2_norm" in ops
        assert "rsqrt" not in ops

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


def _jax_rmsnorm(x, scale=None, epsilon=EPSILON):
    """RMSNorm as a decoder writes it: fp32 statistics over an fp16 activation."""
    x32 = x.astype(jnp.float32)
    normalized = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + epsilon)
    if scale is not None:
        normalized = normalized * scale.astype(jnp.float32)
    return normalized.astype(x.dtype)


class TestFuseRmsNormEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_rmsnorm_fuses_to_l2_norm(self):
        """`(1, 1, D)` needs no reshape: for that shape `l2_norm`'s last-three-dims
        reduction is exactly the last one."""
        scale = jnp.asarray(np.full((16,), 0.5, np.float16))

        cml_model = run_and_compare(
            lambda x: _jax_rmsnorm(x, scale),
            [jax.ShapeDtypeStruct((1, 1, 16), jnp.float16)],
            atol=1e-02,
            rtol=1e-02,
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("l2_norm") == 1
        for op_type in ("reduce_mean", "reduce_sum", "rsqrt", "reshape"):
            assert op_type not in ops, f"unfused RMSNorm leftover: {op_type}"
        # cast(fp32) -> l2_norm -> mul(sqrt(d) * scale) -> cast(fp16), nothing else.
        assert [op for op in ops if op != "const"] == ["cast", "l2_norm", "mul", "cast"]

    def test_epsilon_is_scaled_by_the_normalized_size(self):
        """`l2_norm` sums the squares where the chain averaged them."""
        cml_model = run_and_compare(
            _jax_rmsnorm, [jax.ShapeDtypeStruct((1, 1, 64), jnp.float32)]
        )
        mil_program = cml_model._mil_program
        l2_norm = next(
            op for op in mil_program.functions["main"].operations if op.op_type == "l2_norm"
        )
        assert np.isclose(l2_norm.epsilon.val, 64 * EPSILON, rtol=1e-5)

    @pytest.mark.parametrize("shape", [(1, 4, 8, 16), (1, 8, 16)])
    def test_off_canonical_shape_is_left_alone(self, shape):
        cml_model = run_and_compare(_jax_rmsnorm, [jax.ShapeDtypeStruct(shape, jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert "l2_norm" not in ops
        assert ops.count("reduce_mean") == 1
        assert ops.count("rsqrt") == 1

    def test_adjacent_rmsnorms_are_both_fused(self):
        scale = jnp.asarray(np.full((16,), 0.5, np.float16))

        cml_model = run_and_compare(
            lambda x: _jax_rmsnorm(_jax_rmsnorm(x, scale), scale),
            [jax.ShapeDtypeStruct((1, 1, 16), jnp.float16)],
            atol=1e-02,
            rtol=1e-02,
        )
        assert get_model_instruction_types(cml_model).count("l2_norm") == 2
