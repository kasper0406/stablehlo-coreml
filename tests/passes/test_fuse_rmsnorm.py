import math

import coremltools as ct
import equinox as eqx
import equinox.internal as eqxi
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
from flax import nnx

from tests.utils import get_model_instruction_types, run_and_compare

PASS_NAME = "common::fuse_rmsnorm"
DCE_PASS_NAME = "common::dead_code_elimination"
# `mb.rsqrt`'s own epsilon, folded into the one the `add` contributes.
RSQRT_EPSILON = 1e-12
EPSILON = 1e-6
# The normalized size used by the end-to-end library tests.
D = 16


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

    def test_extra_consumer_of_the_input_is_fused(self):
        """The chain need not own its input: the rewrite never removes the op
        producing it, and no removed op other than the square and the normalize
        mul reads it."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            scaled = mb.mul(x=x, y=2.0)
            return _rmsnorm(scaled), mb.add(x=scaled, y=1.0)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("l2_norm") == 1
        assert "rsqrt" not in ops
        # The extra reader still sees the unchanged input.
        assert "add" in ops
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_residual_path_reading_the_input_is_fused(self):
        """`h = x + f(rmsnorm(x))` is the shape every transformer block has. In an
        fp32 graph no cast insulates `x` from the residual add, so the norm's input
        has a third reader at every site."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return mb.add(x=x, y=_rmsnorm(x))

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops == ["l2_norm", "mul", "add"]
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

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

    def test_keep_dims_false_is_fused(self):
        """A squeezed `(1, 1)` statistic left-pads to the keep-dims `(1, 1, 1)`,
        so it broadcasts exactly like the keep-dims one."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            return _rmsnorm(x, keep_dims=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]
        assert np.isclose(_ops_of_type(prog, "mul")[0].y.val, math.sqrt(16))
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_scale_applied_to_the_rsqrt_is_fused(self):
        """flax's `_normalize` computes `mul = rsqrt(var + eps); mul *= scale`,
        so the scale lands *before* the normalize mul."""
        scale = np.linspace(0.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            inv = mb.mul(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), y=scale)
            return mb.mul(x=x, y=inv)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]
        factor = _ops_of_type(prog, "mul")[0].y.val
        np.testing.assert_allclose(factor, math.sqrt(16) * scale, rtol=1e-6)
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_reshaped_statistic_is_fused(self):
        """equinox's `jnp.mean(x**2)` reduces to a scalar, so JAX broadcasts the
        statistic back with a `reshape` between the rsqrt and the normalize mul."""
        weight = np.linspace(0.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=False)
            inv = mb.reshape(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), shape=[1, 1, 1])
            return mb.mul(x=weight, y=mb.mul(x=inv, y=x))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]
        factor = _ops_of_type(prog, "mul")[0].y.val
        np.testing.assert_allclose(factor, math.sqrt(16) * weight, rtol=1e-6)
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_scales_on_both_sides_are_folded_into_one_constant(self):
        pre = np.linspace(0.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 16)
        post = np.linspace(2.0, 3.0, 16, dtype=np.float32).reshape(1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            inv = mb.mul(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), y=pre)
            return mb.mul(x=mb.mul(x=x, y=inv), y=post)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["l2_norm", "mul"]
        factor = _ops_of_type(prog, "mul")[0].y.val
        np.testing.assert_allclose(factor, math.sqrt(16) * pre * post, rtol=1e-6)
        assert_model_is_valid(
            prog, {"x": (1, 1, 16)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_reshape_that_is_not_a_broadcast_partner_is_left_alone(self):
        """A reshape that permutes a non-unit batch axis makes the statistic
        broadcast against a different axis than the one that was reduced."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            inv = mb.reshape(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), shape=[1, 2, 1, 1])
            return mb.mul(x=x, y=inv)

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_pre_scale_varying_over_a_batch_axis_is_left_alone(self):
        """Unlike a trailing scale, one folded onto the rsqrt cannot be left
        behind -- the whole chain is skipped instead."""
        scale = np.linspace(0.5, 1.5, 32, dtype=np.float32).reshape(2, 1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            inv = mb.mul(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), y=scale)
            return mb.mul(x=x, y=inv)

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_two_reshapes_on_the_walk_are_left_alone(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=False)
            inv = mb.reshape(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), shape=[1, 1])
            return mb.mul(x=x, y=mb.reshape(x=inv, shape=[1, 1, 1]))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_two_constant_muls_on_the_walk_are_left_alone(self):
        first = np.linspace(0.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 16)
        second = np.linspace(2.0, 3.0, 16, dtype=np.float32).reshape(1, 1, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16))])
        def prog(x):
            variance = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
            inv = mb.mul(x=mb.rsqrt(x=mb.add(x=variance, y=EPSILON)), y=first)
            return mb.mul(x=x, y=mb.mul(x=inv, y=second))

        _apply(prog)
        assert "l2_norm" not in get_op_types_in_program(prog)

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 16)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                return _rmsnorm(x)

            def false_fn():
                # Also a reader of `x`, from a sibling block -- which no longer
                # blocks the rewrite (see
                # `test_extra_consumer_of_the_input_is_fused`).
                return mb.identity(x=x)

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



def _model_ops_of_type(cml_model, op_type):
    return [
        op
        for func in cml_model._mil_program.functions.values()
        for op in func.operations
        if op.op_type == op_type
    ]


def _assert_norm_is_fused(cml_model):
    """The whole statistics chain is gone, replaced by a single `l2_norm`."""
    ops = get_model_instruction_types(cml_model)
    assert ops.count("l2_norm") == 1
    for op_type in ("reduce_mean", "reduce_sum", "rsqrt", "reshape"):
        assert op_type not in ops, f"unfused RMSNorm leftover: {op_type}"


def _assert_norm_is_not_fused(cml_model):
    ops = get_model_instruction_types(cml_model)
    assert "l2_norm" not in ops
    assert ops.count("reduce_mean") == 1
    assert ops.count("rsqrt") == 1


def _flax_rmsnorm(use_scale=True):
    """A flax RMSNorm with a *non-unit* scale.

    With the default all-ones scale coremltools' `noop_elimination` deletes the
    scale `mul` before this pass runs, so the reassociation flax actually emits
    (`mul = rsqrt(var + eps); mul *= scale; y = x * mul`) would never be exercised.
    """
    return nnx.RMSNorm(
        num_features=D,
        use_scale=use_scale,
        scale_init=nnx.initializers.normal(),
        rngs=nnx.Rngs(0),
    )


def _equinox_rmsnorm(shape, use_bias=False):
    """An equinox RMSNorm with a non-unit weight, vmapped up to ``shape``.

    `eqx.nn.RMSNorm` normalizes a single vector, and its `jnp.mean(x**2)` takes
    no axis, so it reduces to a scalar and JAX broadcasts the statistic back
    with a `reshape` placed after the `rsqrt`.
    """
    model = eqx.nn.RMSNorm(shape=(D,), use_bias=use_bias)
    weight = jnp.asarray(np.linspace(0.5, 1.5, D, dtype=np.float32))
    model = eqx.tree_at(lambda m: m.weight, model, weight)

    fn = eqxi.finalise_fn(eqx.nn.inference_mode(model))
    for _ in range(len(shape) - 1):
        fn = jax.vmap(fn)
    return fn


class TestFuseRmsNormLibraryLayers:
    """End-to-end tests driving the actual flax / equinox RMSNorm layers.

    Each library reassociates the tail of the chain its own way, which is what
    the forward walk from the `rsqrt` exists to absorb.
    """

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_flax_nnx_rmsnorm_is_fused(self, dtype):
        cml_model = run_and_compare(
            nnx.jit(_flax_rmsnorm()),
            [jax.ShapeDtypeStruct((1, 1, D), dtype)],
            atol=1e-02,
            rtol=1e-02,
        )
        _assert_norm_is_fused(cml_model)
        # The scale flax folded onto the rsqrt is part of the single constant.
        factor = _model_ops_of_type(cml_model, "mul")[0].y.val
        assert factor.shape == (1, 1, D)

    def test_flax_nnx_rmsnorm_without_a_scale_is_fused(self):
        cml_model = run_and_compare(
            nnx.jit(_flax_rmsnorm(use_scale=False)),
            [jax.ShapeDtypeStruct((1, 1, D), jnp.float32)],
        )
        _assert_norm_is_fused(cml_model)
        assert np.isclose(_model_ops_of_type(cml_model, "mul")[0].y.val, math.sqrt(D))

    def test_equinox_rmsnorm_is_fused(self):
        cml_model = run_and_compare(
            _equinox_rmsnorm((1, 1, D)),
            [jax.ShapeDtypeStruct((1, 1, D), jnp.float32)],
        )
        _assert_norm_is_fused(cml_model)
        factor = _model_ops_of_type(cml_model, "mul")[0].y.val
        assert factor.shape == (1, 1, D)

    def test_equinox_rmsnorm_with_a_bias_is_fused(self):
        """The bias `add` is not absorbed -- it simply stays after the fused norm."""
        cml_model = run_and_compare(
            _equinox_rmsnorm((1, 1, D), use_bias=True),
            [jax.ShapeDtypeStruct((1, 1, D), jnp.float32)],
        )
        _assert_norm_is_fused(cml_model)
        assert get_model_instruction_types(cml_model).count("add") == 1

    def test_flax_nnx_rmsnorm_off_canonical_shape_is_left_alone(self):
        cml_model = run_and_compare(
            nnx.jit(_flax_rmsnorm()),
            [jax.ShapeDtypeStruct((1, 4, 8, D), jnp.float32)],
        )
        _assert_norm_is_not_fused(cml_model)

    def test_equinox_rmsnorm_off_canonical_shape_is_left_alone(self):
        cml_model = run_and_compare(
            _equinox_rmsnorm((1, 8, D)),
            [jax.ShapeDtypeStruct((1, 8, D), jnp.float32)],
        )
        _assert_norm_is_not_fused(cml_model)

    def test_flax_rmsnorm_on_a_residual_path_is_fused(self):
        """The shape a transformer block actually has. In an fp32 graph there is
        no cast between the block input and the residual `add`, so the norm's
        input has a reader outside the chain at every site."""
        norm = _flax_rmsnorm()
        linear = nnx.Linear(in_features=D, out_features=D, rngs=nnx.Rngs(1))

        cml_model = run_and_compare(
            nnx.jit(lambda x: x + linear(norm(x))),
            [jax.ShapeDtypeStruct((1, 1, D), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("l2_norm") == 1
        for op_type in ("reduce_mean", "reduce_sum", "rsqrt"):
            assert op_type not in ops, f"unfused RMSNorm leftover: {op_type}"
