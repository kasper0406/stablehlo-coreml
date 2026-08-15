import math

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
from tests.utils import get_model_instruction_types, run_and_compare, run_and_compare_jit_lowering

register_optimizations()

PASS_NAME = "common::fuse_gelu_erfc"
INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME)


def _erfc_gelu(x, negate_first=True, factor=INV_SQRT2, half=0.5, one=1.0):
    """Build `half * x * (one - erf(-factor * x))` in the current MIL block."""
    half_x = mb.mul(x=x, y=half)
    if negate_first:
        arg = mb.mul(x=mb.sub(x=0.0, y=x), y=factor)
    else:
        arg = mb.sub(x=0.0, y=mb.mul(x=x, y=factor))
    return mb.mul(x=half_x, y=mb.sub(x=one, y=mb.erf(x=arg)))


class TestFuseGeluErfc:
    """Unit tests on hand-built MIL programs."""

    def test_negate_then_scale(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(x)

        assert get_op_types_in_program(prog) == ["mul", "sub", "mul", "erf", "sub", "mul"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]
        assert_model_is_valid(
            prog, {"x": (4, 8)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_scale_then_negate(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(x, negate_first=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_division_form(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            half_x = mb.mul(x=0.5, y=x)
            arg = mb.real_div(x=mb.sub(x=0.0, y=x), y=math.sqrt(2.0))
            return mb.mul(x=mb.sub(x=1.0, y=mb.erf(x=arg)), y=half_x)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_single_negative_multiplier(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            half_x = mb.mul(x=x, y=0.5)
            arg = mb.mul(x=x, y=-INV_SQRT2)
            return mb.mul(x=half_x, y=mb.sub(x=1.0, y=mb.erf(x=arg)))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_uniform_tensor_constants(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(
                x,
                factor=np.full((1,), INV_SQRT2, dtype=np.float32),
                half=np.full((1,), 0.5, dtype=np.float32),
                one=np.full((1,), 1.0, dtype=np.float32),
            )

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_not_fused_for_the_wrong_factor(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(x, factor=0.5)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_a_nearby_factor(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(x, factor=0.7062)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_without_the_negation(self):
        """`0.5 * x * (1 - erf(x/sqrt(2)))` is not GELU (it is x - gelu(x))."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            half_x = mb.mul(x=x, y=0.5)
            arg = mb.mul(x=x, y=INV_SQRT2)
            return mb.mul(x=half_x, y=mb.sub(x=1.0, y=mb.erf(x=arg)))

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_the_wrong_half(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(x, half=0.25)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_a_different_input(self):
        """The `0.5 *` and the `erf` must be applied to the same var."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(4, 8))])
        def prog(x, y):
            half_x = mb.mul(x=x, y=0.5)
            arg = mb.mul(x=mb.sub(x=0.0, y=y), y=INV_SQRT2)
            return mb.mul(x=half_x, y=mb.sub(x=1.0, y=mb.erf(x=arg)))

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_when_erf_is_shared(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            half_x = mb.mul(x=x, y=0.5)
            arg = mb.mul(x=mb.sub(x=0.0, y=x), y=INV_SQRT2)
            erf = mb.erf(x=arg)
            return mb.mul(x=half_x, y=mb.sub(x=1.0, y=erf)), erf

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                return _erfc_gelu(x)

            def false_fn():
                return mb.identity(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        ops = get_op_types_in_program(prog, recurse=True)
        assert "gelu" in ops
        assert "erf" not in ops

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _erfc_gelu(x)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestFuseGeluErfcEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_exact_gelu_via_jit_lowering(self):
        """The `jax.jit(...).lower()` path keeps `chlo.erfc`, which becomes `1 - erf`."""
        inputs = (jax.random.normal(jax.random.PRNGKey(0), (4, 16), dtype=jnp.float32),)
        cml_model = run_and_compare_jit_lowering(
            lambda x: jax.nn.gelu(x, approximate=False), inputs
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("gelu") == 1
        assert "erf" not in ops

    def test_exact_gelu_via_jax_export(self):
        """`jax.export` expands `chlo.erfc` itself, so only the numerics are checked.

        (jax.export legalizes CHLO before we ever see the module and, unlike
        `chlo.erf`, `chlo.erfc` has no composite representation there.)
        """
        cml_model = run_and_compare(
            lambda x: jax.nn.gelu(x, approximate=False),
            [jax.ShapeDtypeStruct((4, 16), jnp.float32)],
        )
        assert "gelu" not in get_model_instruction_types(cml_model)

    def test_tanh_approximated_gelu_still_converts(self):
        """The tanh approximation converts correctly, but is not fused.

        coremltools' `fuse_gelu_tanh_approximation` insists on a literal
        `pow(x, 3)`, while JAX emits `x * x * x`. This pass deliberately only
        covers the `erfc` form, so the approximation stays decomposed.
        """
        cml_model = run_and_compare(
            lambda x: jax.nn.gelu(x, approximate=True),
            [jax.ShapeDtypeStruct((4, 16), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("tanh") == 1
