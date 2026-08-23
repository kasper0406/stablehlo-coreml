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

PASS_NAME = "common::fuse_logit_softcap"
DCE_PASS_NAME = "common::dead_code_elimination"


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME)
    # The pass leaves the matched ops behind; DCE is what removes them.
    apply_pass_and_basic_check(prog, DCE_PASS_NAME)


def _scaled_tanh_ops(prog):
    return [op for op in prog.functions["main"].operations if op.op_type == "scaled_tanh"]


class TestFuseLogitSoftcap:
    """Unit tests on hand-built MIL programs."""

    def test_div_then_mul(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.real_div(x=x, y=30.0)
            return mb.mul(x=mb.tanh(x=scaled), y=30.0)

        assert get_op_types_in_program(prog) == ["real_div", "tanh", "mul"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_tanh"]

        fused = _scaled_tanh_ops(prog)[0]
        assert np.isclose(fused.alpha.val, 30.0)
        assert np.isclose(fused.beta.val, 1.0 / 30.0)
        assert_model_is_valid(
            prog, {"x": (2, 8)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_mul_then_mul(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.mul(x=x, y=1.0 / 30.0)
            return mb.mul(x=mb.tanh(x=scaled), y=30.0)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_tanh"]

    def test_reversed_operand_orders(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.mul(x=1.0 / 30.0, y=x)
            return mb.mul(x=30.0, y=mb.tanh(x=scaled))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_tanh"]

    def test_uniform_tensor_constants(self):
        """The cap may be a (broadcast) constant tensor as long as it is uniform."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            cap = np.full((1,), 30.0, dtype=np.float32)
            scaled = mb.real_div(x=x, y=cap)
            return mb.mul(x=mb.tanh(x=scaled), y=cap)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_tanh"]

    @pytest.mark.parametrize("cap", [30.0, 50.0])
    def test_fp16_reciprocal_rounding_is_still_fused(self, cap):
        """``1 / cap`` is rounded to fp16 long before MIL sees it.

        ``fp16(1/30) * 30 == 0.99976``, which is 2.4e-4 away from unity -- more
        than the fp32 tolerance allows, but well inside one fp16 ulp.
        """
        beta = np.float16(1.0) / np.float16(cap)
        assert abs(float(beta) * cap - 1.0) > 1e-4

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8), dtype=types.fp16)])
        def prog(x):
            scaled = mb.mul(x=x, y=beta)
            return mb.mul(x=mb.tanh(x=scaled), y=np.float16(cap))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_tanh"]

        fused = _scaled_tanh_ops(prog)[0]
        assert fused.alpha.val.dtype == np.float16
        assert np.isclose(fused.alpha.val, cap)
        assert fused.beta.val == beta

    def test_fp32_keeps_the_tight_tolerance(self):
        """The widened tolerance is fp16-only; fp32 constants stay tightly checked."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float32(1.0 / 30.0))
            # 5e-4 off unity: representable in fp32, so not a rounding artifact.
            return mb.mul(x=mb.tanh(x=scaled), y=np.float32(30.0 * 1.0005))

        _apply(prog)
        assert "scaled_tanh" not in get_op_types_in_program(prog)

    def test_not_fused_in_fp16_when_the_product_is_far_from_one(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8), dtype=types.fp16)])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float16(1.0 / 30.0))
            return mb.mul(x=mb.tanh(x=scaled), y=np.float16(20.0))

        _apply(prog)
        assert "scaled_tanh" not in get_op_types_in_program(prog)

    def test_not_fused_when_product_is_not_one(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.real_div(x=x, y=30.0)
            return mb.mul(x=mb.tanh(x=scaled), y=20.0)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["real_div", "tanh", "mul"]

    def test_not_fused_without_outer_scale(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8)), mb.TensorSpec(shape=(2, 8))])
        def prog(x, y):
            return mb.mul(x=mb.tanh(x=x), y=y)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tanh", "mul"]

    def test_not_fused_when_tanh_has_multiple_consumers(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.real_div(x=x, y=30.0)
            capped = mb.tanh(x=scaled)
            return mb.mul(x=capped, y=30.0), mb.add(x=capped, y=1.0)

        _apply(prog)
        assert "scaled_tanh" not in get_op_types_in_program(prog)

    def test_not_fused_for_non_uniform_constant(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2,))])
        def prog(x):
            cap = np.array([30.0, 20.0], dtype=np.float32)
            scaled = mb.real_div(x=x, y=cap)
            return mb.mul(x=mb.tanh(x=scaled), y=cap)

        _apply(prog)
        assert "scaled_tanh" not in get_op_types_in_program(prog)

    def test_inner_scale_shared_keeps_beta_one(self):
        """When the scaled value escapes, only `a * tanh(v)` may be fused (a == 1)."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.real_div(x=x, y=30.0)
            return mb.mul(x=mb.tanh(x=scaled), y=30.0), scaled

        _apply(prog)
        # beta would have to be 1 (the div cannot be peeled), and 30 * 1 != 1.
        assert "scaled_tanh" not in get_op_types_in_program(prog)

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                scaled = mb.real_div(x=x, y=30.0)
                return mb.mul(x=mb.tanh(x=scaled), y=30.0)

            def false_fn():
                return mb.identity(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        ops = get_op_types_in_program(prog, recurse=True)
        assert "scaled_tanh" in ops
        assert "tanh" not in ops

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8))])
        def prog(x):
            scaled = mb.real_div(x=x, y=30.0)
            return mb.mul(x=mb.tanh(x=scaled), y=30.0)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestFuseLogitSoftcapEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_softcap(self):
        def f(x):
            return jnp.tanh(x / 30.0) * 30.0

        cml_model = run_and_compare(f, [jax.ShapeDtypeStruct((4, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert ops.count("scaled_tanh") == 1
        assert "tanh" not in ops

    def test_softcap_with_multiplication(self):
        def f(x):
            return jnp.tanh(x * 0.02) * 50.0

        cml_model = run_and_compare(f, [jax.ShapeDtypeStruct((4, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert ops.count("scaled_tanh") == 1
        assert "tanh" not in ops

    def test_unbalanced_scaling_is_not_fused(self):
        def f(x):
            return jnp.tanh(x / 30.0) * 20.0

        cml_model = run_and_compare(f, [jax.ShapeDtypeStruct((4, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert "scaled_tanh" not in ops
        assert ops.count("tanh") == 1

    @pytest.mark.parametrize("cap", [30.0, 50.0])
    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param(lambda x, cap: cap * jnp.tanh(x / cap), id="cap_times_tanh_of_div"),
            pytest.param(lambda x, cap: jnp.tanh(x / cap) * cap, id="tanh_of_div_times_cap"),
            pytest.param(lambda x, cap: cap * jnp.tanh(x * (1.0 / cap)), id="cap_times_tanh_of_mul"),
            pytest.param(
                lambda x, cap: jnp.asarray(cap, x.dtype) * jnp.tanh(x / jnp.asarray(cap, x.dtype)),
                id="cap_as_traced_array",
            ),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_gemma_style_softcap_spellings(self, spelling, cap, dtype):
        """Every way Gemma-2 writes ``cap * tanh(logits / cap)`` must fuse.

        The fp16 cases matter: JAX folds ``1 / cap`` in the operand dtype, so
        ``alpha * beta`` only reaches unity to within an fp16 ulp.
        """
        precision_loss = jnp.finfo(dtype).eps / jnp.finfo(jnp.float32).eps
        cml_model = run_and_compare(
            lambda x: spelling(x, cap),
            [jax.ShapeDtypeStruct((4, 16), dtype)],
            atol=1e-04 * precision_loss,
            rtol=1e-05 * precision_loss,
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("scaled_tanh") == 1
        assert "tanh" not in ops

    def test_attention_logit_softcap(self):
        """Gemma-2 softcaps the attention logits, right before the softmax."""
        def f(query, key):
            logits = jnp.einsum("qd,kd->qk", query, key)
            logits = 50.0 * jnp.tanh(logits / 50.0)
            return jax.nn.softmax(logits, axis=-1)

        cml_model = run_and_compare(
            f,
            [
                jax.ShapeDtypeStruct((6, 4), jnp.float32),
                jax.ShapeDtypeStruct((6, 4), jnp.float32),
            ],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("scaled_tanh") == 1
        assert ops.count("softmax") == 1
        assert "tanh" not in ops
