import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

from stablehlo_coreml import register_optimizations
from tests.utils import get_model_instruction_types, run_and_compare

register_optimizations()

PASS_NAME = "common::fuse_logit_softcap"


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME)


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
