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

from stablehlo_coreml import register_optimizations
from tests.utils import get_model_instruction_types, run_and_compare

register_optimizations()

PASS_NAME = "common::replace_decomposed_softmax"

NEG_INF = np.float32(-np.inf)


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME, skip_output_shape_check=True)


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
