import coremltools as ct
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
)
from flax import nnx

from tests.passes.helpers import apply_pass
from tests.utils import (
    get_model_instruction_types,
    run_and_compare,
    run_and_compare_specific_input,
)

PASS_NAME = "common::replace_decomposed_softmax"

NEG_INF = np.float32(-np.inf)


def _apply(prog):
    apply_pass(prog, PASS_NAME, skip_output_shape_check=True)


def _decomposed_softmax(x, axis, shape, *, clamps=1, keep_dims=False, tile=False):
    """Build the op chain JAX emits for ``softmax(x, axis)``."""
    keep_dims_shape = list(shape)
    keep_dims_shape[axis] = 1

    maximum = mb.reduce_max(x=x, axes=[axis], keep_dims=keep_dims)
    for _ in range(clamps):
        maximum = mb.maximum(x=NEG_INF, y=maximum)
    if not keep_dims:
        maximum = mb.reshape(x=maximum, shape=keep_dims_shape)
    if tile:
        reps = [1] * len(shape)
        reps[axis] = shape[axis]
        maximum = mb.tile(x=maximum, reps=reps)

    shifted = mb.sub(x=x, y=maximum)
    exponentiated = mb.exp(x=shifted)

    total = mb.reduce_sum(x=exponentiated, axes=[axis], keep_dims=keep_dims)
    if not keep_dims:
        total = mb.reshape(x=total, shape=keep_dims_shape)
    if tile:
        reps = [1] * len(shape)
        reps[axis] = shape[axis]
        total = mb.tile(x=total, reps=reps)

    return mb.real_div(x=exponentiated, y=total)


def _masked_softmax(x, mask, axis, shape, *, fill=NEG_INF, out_fill=0.0, sum_fill=0.0):
    """Build the op chain JAX emits for ``softmax(x, axis, where=mask)``.

    JAX materialises the mask once per use, so each ``select`` gets its own
    ``cond`` var; here they share one, which is the shape the converter produces
    for an unbroadcast mask.
    """
    keep_dims_shape = list(shape)
    keep_dims_shape[axis] = 1

    safe = mb.select(cond=mask, a=x, b=np.full(shape, fill, dtype=np.float32))
    maximum = mb.reduce_max(x=safe, axes=[axis], keep_dims=False)
    maximum = mb.maximum(x=NEG_INF, y=maximum)
    maximum = mb.reshape(x=maximum, shape=keep_dims_shape)

    exponentiated = mb.exp(x=mb.sub(x=safe, y=maximum))
    masked = mb.select(cond=mask, a=exponentiated, b=np.full(shape, sum_fill, dtype=np.float32))
    total = mb.reduce_sum(x=masked, axes=[axis], keep_dims=True)
    quotient = mb.real_div(x=exponentiated, y=total)
    return mb.select(cond=mask, a=quotient, b=np.full(shape, out_fill, dtype=np.float32))


class TestReplaceDecomposedSoftmax:
    """Unit tests on hand-built MIL programs."""

    def test_replaces_the_jax_chain(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            return _decomposed_softmax(x, 1, (2, 4))

        assert get_op_types_in_program(prog) == [
            "reduce_max", "maximum", "reshape", "sub", "exp", "reduce_sum", "reshape", "real_div",
        ]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]
        softmax_op = prog.functions["main"].operations[-1]
        assert softmax_op.axis.val == 1
        assert_model_is_valid(
            prog, {"x": (2, 4)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    @pytest.mark.parametrize("clamps", [0, 1, 2])
    def test_handles_zero_to_two_neg_inf_clamps(self, clamps):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            return _decomposed_softmax(x, 1, (2, 4), clamps=clamps)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]

    def test_handles_keep_dims_reductions(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 4))])
        def prog(x):
            return _decomposed_softmax(x, 2, (2, 3, 4), keep_dims=True)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]

    def test_handles_broadcast_tiles(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            return _decomposed_softmax(x, 1, (2, 4), tile=True)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]

    def test_handles_negative_axis(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            maximum = mb.reduce_max(x=x, axes=[-1], keep_dims=True)
            shifted = mb.sub(x=x, y=maximum)
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_sum(x=exponentiated, axes=[-1], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]
        assert prog.functions["main"].operations[-1].axis.val == 1

    def test_handles_the_fp16_cast_pair(self):
        """JAX accumulates the fp16 sum in fp32, adding a cast on each side."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4), dtype=types.fp16)])
        def prog(x):
            maximum = mb.reduce_max(x=x, axes=[1], keep_dims=True)
            shifted = mb.sub(x=x, y=maximum)
            exponentiated = mb.exp(x=shifted)
            promoted = mb.cast(x=exponentiated, dtype="fp32")
            total = mb.reduce_sum(x=promoted, axes=[1], keep_dims=True)
            demoted = mb.cast(x=total, dtype="fp16")
            return mb.real_div(x=exponentiated, y=demoted)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]

    def test_handles_the_where_mask(self):
        """`jax.nn.softmax(where=mask)` puts a `select` inside the sum."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            return _masked_softmax(x, mask, 1, (2, 4))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["select", "softmax", "select"]
        softmax_op = next(
            op for op in prog.functions["main"].operations if op.op_type == "softmax"
        )
        assert softmax_op.axis.val == 1
        # The softmax runs on the -inf-filled input, not on the raw one.
        assert softmax_op.x.op.op_type == "select"

    def test_handles_the_where_mask_with_the_fp16_cast_pair(self):
        """In fp16 the cast to fp32 sits between the `exp` and the mask `select`."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4), dtype=types.fp16),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            safe = mb.select(cond=mask, a=x, b=np.full((2, 4), -np.inf, dtype=np.float16))
            maximum = mb.reduce_max(x=safe, axes=[1], keep_dims=True)
            exponentiated = mb.exp(x=mb.sub(x=safe, y=maximum))
            promoted = mb.cast(x=exponentiated, dtype="fp32")
            masked = mb.select(cond=mask, a=promoted, b=np.zeros((2, 4), dtype=np.float32))
            total = mb.reduce_sum(x=masked, axes=[1], keep_dims=True)
            demoted = mb.cast(x=total, dtype="fp16")
            quotient = mb.real_div(x=exponentiated, y=demoted)
            return mb.select(cond=mask, a=quotient, b=np.zeros((2, 4), dtype=np.float16))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["select", "softmax", "select"]

    def test_handles_a_where_mask_broadcast_by_separate_tiles(self):
        """The converter tiles the mask up to the operand shape once per use."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 3, 4)),
            mb.TensorSpec(shape=(2, 1, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            fill = np.full((2, 3, 4), -np.inf, dtype=np.float32)
            zeros = np.zeros((2, 3, 4), dtype=np.float32)
            safe = mb.select(cond=mb.tile(x=mask, reps=[1, 3, 1]), a=x, b=fill)
            maximum = mb.reduce_max(x=safe, axes=[2], keep_dims=True)
            exponentiated = mb.exp(x=mb.sub(x=safe, y=maximum))
            masked = mb.select(cond=mb.tile(x=mask, reps=[1, 3, 1]), a=exponentiated, b=zeros)
            total = mb.reduce_sum(x=masked, axes=[2], keep_dims=True)
            quotient = mb.real_div(x=exponentiated, y=total)
            return mb.select(cond=mb.tile(x=mask, reps=[1, 3, 1]), a=quotient, b=zeros)

        _apply(prog)
        assert "softmax" in get_op_types_in_program(prog)
        assert "real_div" not in get_op_types_in_program(prog)

    def test_keeps_a_masked_sum_when_the_fill_is_finite(self):
        """Only `-inf` guarantees the masked lanes of `exp` are exactly zero.

        With a finite fill (``jnp.finfo.min``, as flax spells it) the masked
        terms are tiny but non-zero, so dropping the `select` would change the
        denominator.
        """
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            return _masked_softmax(x, mask, 1, (2, 4), fill=np.finfo(np.float32).min)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_keeps_a_masked_sum_without_the_outer_select(self):
        """The outer `select` is what turns an entirely masked row into zeros."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            safe = mb.select(cond=mask, a=x, b=np.full((2, 4), -np.inf, dtype=np.float32))
            maximum = mb.reduce_max(x=safe, axes=[1], keep_dims=True)
            exponentiated = mb.exp(x=mb.sub(x=safe, y=maximum))
            masked = mb.select(cond=mask, a=exponentiated, b=np.zeros((2, 4), dtype=np.float32))
            total = mb.reduce_sum(x=masked, axes=[1], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_keeps_a_masked_sum_for_a_different_mask(self):
        """All three masks must come from one source."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask, other):
            safe = mb.select(cond=other, a=x, b=np.full((2, 4), -np.inf, dtype=np.float32))
            maximum = mb.reduce_max(x=safe, axes=[1], keep_dims=True)
            exponentiated = mb.exp(x=mb.sub(x=safe, y=maximum))
            masked = mb.select(cond=mask, a=exponentiated, b=np.zeros((2, 4), dtype=np.float32))
            total = mb.reduce_sum(x=masked, axes=[1], keep_dims=True)
            quotient = mb.real_div(x=exponentiated, y=total)
            return mb.select(cond=mask, a=quotient, b=np.zeros((2, 4), dtype=np.float32))

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_keeps_a_masked_sum_when_the_summand_fill_is_not_zero(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            return _masked_softmax(x, mask, 1, (2, 4), sum_fill=1.0)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_keeps_a_masked_sum_when_the_output_fill_is_not_zero(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(2, 4), dtype=types.bool),
        ])
        def prog(x, mask):
            return _masked_softmax(x, mask, 1, (2, 4), out_fill=1.0)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_handles_symbolic_batch_dim(self):
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 4))])
        def prog(x):
            maximum = mb.reduce_max(x=x, axes=[1], keep_dims=True)
            shifted = mb.sub(x=x, y=maximum)
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_sum(x=exponentiated, axes=[1], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["softmax"]

    def test_keeps_the_shift_when_it_varies_along_the_axis(self):
        """Softmax is only invariant under a shift that is constant along the axis."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(x, bias):
            shifted = mb.sub(x=x, y=bias)
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_sum(x=exponentiated, axes=[1], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["sub", "softmax"]

    def test_keeps_a_statically_nonfinite_shift(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            shifted = mb.sub(x=x, y=np.float32(np.inf))
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_sum(x=exponentiated, axes=[1], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["sub", "softmax"]

    def test_uses_the_reduce_sum_axis(self):
        """The softmax axis is the one the sum reduces over, not the max's."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            maximum = mb.reduce_max(x=x, axes=[1], keep_dims=True)
            shifted = mb.sub(x=x, y=maximum)
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_sum(x=exponentiated, axes=[0], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        # The shift is not constant along axis 0, so it has to stay.
        assert get_op_types_in_program(prog) == ["reduce_max", "sub", "softmax"]
        assert prog.functions["main"].operations[-1].axis.val == 0

    def test_not_replaced_for_a_multi_axis_sum(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 4))])
        def prog(x):
            exponentiated = mb.exp(x=x)
            total = mb.reduce_sum(x=exponentiated, axes=[1, 2], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_not_replaced_when_the_sum_is_reshaped_incompatibly(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4))])
        def prog(x):
            exponentiated = mb.exp(x=x)
            total = mb.reduce_sum(x=exponentiated, axes=[1], keep_dims=False)
            # Broadcast along the wrong axis: this is not a softmax.
            reshaped = mb.reshape(x=total, shape=[1, 4])
            return mb.real_div(x=exponentiated, y=reshaped)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_not_replaced_when_denominator_is_not_a_sum(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            maximum = mb.reduce_max(x=x, axes=[1], keep_dims=True)
            shifted = mb.sub(x=x, y=maximum)
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_max(x=exponentiated, axes=[1], keep_dims=True)
            return mb.real_div(x=exponentiated, y=total)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_not_replaced_when_exp_is_used_elsewhere(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            maximum = mb.reduce_max(x=x, axes=[1], keep_dims=True)
            shifted = mb.sub(x=x, y=maximum)
            exponentiated = mb.exp(x=shifted)
            total = mb.reduce_sum(x=exponentiated, axes=[1], keep_dims=True)
            softmax = mb.real_div(x=exponentiated, y=total)
            return mb.add(x=softmax, y=exponentiated)

        _apply(prog)
        assert "softmax" not in get_op_types_in_program(prog)

    def test_replaced_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                return _decomposed_softmax(x, 1, (2, 4))

            def false_fn():
                return mb.identity(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        assert "softmax" not in get_op_types_in_program(prog, recurse=True)
        _apply(prog)
        assert "softmax" in get_op_types_in_program(prog, recurse=True)
        assert "real_div" not in get_op_types_in_program(prog, recurse=True)

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            return _decomposed_softmax(x, 1, (2, 4))

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestReplaceDecomposedSoftmaxEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_jax_softmax_last_axis(self):
        cml_model = run_and_compare(
            lambda x: jax.nn.softmax(x, axis=-1),
            [jax.ShapeDtypeStruct((3, 16), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        for op_type in ("reduce_max", "reduce_sum", "exp", "real_div"):
            assert op_type not in ops

    def test_jax_softmax_middle_axis(self):
        cml_model = run_and_compare(
            lambda x: jax.nn.softmax(x, axis=1),
            [jax.ShapeDtypeStruct((2, 5, 4), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        assert "real_div" not in ops

    def test_jax_softmax_fp16(self):
        """The fp16 lowering wraps `reduce_sum` in a cast pair; it must still fuse.

        CoreML cannot take fp16 model inputs here, so the check is done on the
        MIL program produced by the very same pipeline the other tests use.
        """
        from jax._src.interpreters import mlir as jax_mlir  # noqa: PLC0415
        from jax._src.lib.mlir import ir  # noqa: PLC0415

        from stablehlo_coreml.converter import convert  # noqa: PLC0415
        from tests.utils import _convert_mil_to_coreml, jax_export  # noqa: PLC0415

        exported = jax_export(
            jax.jit(lambda x: jax.nn.softmax(x, axis=-1)),
            [jax.ShapeDtypeStruct((3, 16), jnp.float16)],
        )
        context = jax_mlir.make_ir_context()
        hlo_module = ir.Module.parse(exported.mlir_module(), context=context)
        mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)

        cml_model = _convert_mil_to_coreml(mil_program)
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        assert "real_div" not in ops
        assert "reduce_sum" not in ops

    def test_flax_softmax(self):
        cml_model = run_and_compare(nnx.softmax, [jax.ShapeDtypeStruct((3, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        assert "real_div" not in ops

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_jax_softmax_with_a_where_mask(self, dtype):
        """`where=` masks the summand and the quotient with a `select` each.

        The mask deliberately blanks out one row entirely: JAX answers that row
        with zeros (the outer `select` swallows the NaN), and the fused model
        has to do the same.
        """
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 16), dtype)
        mask = jax.random.bernoulli(jax.random.PRNGKey(1), 0.6, (3, 16)).at[1].set(False)
        precision_loss = jnp.finfo(dtype).eps / jnp.finfo(jnp.float32).eps

        cml_model = run_and_compare_specific_input(
            lambda a, m: jax.nn.softmax(a, axis=-1, where=m),
            (x, mask),
            atol=1e-04 * precision_loss,
            rtol=1e-05 * precision_loss,
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        for op_type in ("reduce_max", "reduce_sum", "exp", "real_div"):
            assert op_type not in ops

    def test_jax_softmax_with_a_broadcast_where_mask(self):
        """A mask of a lower rank is tiled up to the operand shape once per use."""
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8, 8), jnp.float32)
        mask = jax.random.bernoulli(jax.random.PRNGKey(1), 0.7, (2, 1, 1, 8))

        cml_model = run_and_compare_specific_input(
            lambda a, m: jax.nn.softmax(a, axis=-1, where=m), (x, mask)
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        assert "real_div" not in ops

    def test_jax_softmax_with_a_where_mask_on_a_middle_axis(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4), jnp.float32)
        mask = jax.random.bernoulli(jax.random.PRNGKey(1), 0.7, (2, 5, 4))

        cml_model = run_and_compare_specific_input(
            lambda a, m: jax.nn.softmax(a, axis=1, where=m), (x, mask)
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        assert "real_div" not in ops

    def test_masked_softmax_as_flax_and_equinox_write_it(self):
        """Both libraries fill the masked logits themselves, then call `softmax`.

        `flax.nnx.dot_product_attention` and `equinox.nn.MultiheadAttention`
        both use `jnp.where(mask, logits, finfo.min)`, so no `select` ends up
        inside the sum and the plain decomposition is what reaches MIL.
        """
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8, 8), jnp.float32)
        mask = jax.random.bernoulli(jax.random.PRNGKey(1), 0.7, (2, 1, 8, 8))

        cml_model = run_and_compare_specific_input(
            lambda a, m: jax.nn.softmax(
                jnp.where(m, a, jnp.finfo(jnp.float32).min), axis=-1
            ),
            (x, mask),
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("softmax") == 1
        assert "real_div" not in ops

    def test_flax_multi_head_attention(self):
        """The softmax inside `nnx.MultiHeadAttention` must be recognised.

        `fuse_attention_to_sdpa` then absorbs it, so what is asserted is that
        nothing of the decomposition survives.
        """
        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=8, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
        )
        layer.eval()
        causal = nnx.make_causal_mask(jnp.ones((2, 6)))

        cml_model = run_and_compare(
            nnx.jit(lambda x: layer(x, mask=causal)),
            [jax.ShapeDtypeStruct((2, 6, 8), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        for op_type in ("reduce_max", "reduce_sum", "exp", "real_div"):
            assert op_type not in ops

    def test_equinox_multihead_attention(self):
        layer = eqx.nn.MultiheadAttention(
            num_heads=2, query_size=8, key=jax.random.PRNGKey(0)
        )
        layer = eqx.nn.inference_mode(layer)
        mask = jnp.tril(jnp.ones((6, 6), dtype=bool))

        cml_model = run_and_compare(
            eqxi.finalise_fn(lambda x: layer(x, x, x, mask=mask)),
            [jax.ShapeDtypeStruct((6, 8), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        for op_type in ("reduce_max", "reduce_sum", "exp", "real_div"):
            assert op_type not in ops

    def test_log_softmax_is_left_alone(self):
        """MIL has no `log_softmax`, and `log(softmax(x))` is not the same op."""
        cml_model = run_and_compare(
            jax.nn.log_softmax, [jax.ShapeDtypeStruct((3, 16), jnp.float32)]
        )
        ops = get_model_instruction_types(cml_model)
        assert "softmax" not in ops
        assert ops.count("log") == 1

    @pytest.mark.parametrize("axis", [None, (1, 2)])
    def test_multi_axis_softmax_is_left_alone(self, axis):
        """`softmax` reduces over a single axis, so a multi-axis sum cannot fuse."""
        cml_model = run_and_compare(
            lambda x: jax.nn.softmax(x, axis=axis),
            [jax.ShapeDtypeStruct((2, 3, 4), jnp.float32)],
        )
        assert "softmax" not in get_model_instruction_types(cml_model)
