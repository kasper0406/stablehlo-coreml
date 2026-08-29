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

from tests.utils import (
    get_model_instruction_types,
    run_and_compare,
    run_and_compare_jit_lowering,
    run_and_compare_specific_input,
)

PASS_NAME = "common::fuse_gelu_tanh"
DCE_PASS_NAME = "common::dead_code_elimination"

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
CUBE_COEFFICIENT = 0.044715


def _apply(prog):
    apply_pass_and_basic_check(prog, PASS_NAME)
    # The pass leaves the matched ops behind; DCE is what removes them.
    apply_pass_and_basic_check(prog, DCE_PASS_NAME)


def _tanh_gelu(
    x,
    *,
    outer=SQRT_2_OVER_PI,
    cube=CUBE_COEFFICIENT,
    half=0.5,
    one=1.0,
    pow_cube=False,
    half_on_x=False,
):
    """Build `half * x * (one + tanh(outer * (x + cube * x**3)))`."""
    if pow_cube:
        cubed = mb.pow(x=x, y=3.0)
    else:
        cubed = mb.mul(x=mb.mul(x=x, y=x), y=x)
    inner = mb.add(x=x, y=mb.mul(x=cube, y=cubed))
    cdf = mb.add(x=one, y=mb.tanh(x=mb.mul(x=outer, y=inner)))
    if half_on_x:
        return mb.mul(x=mb.mul(x=x, y=half), y=cdf)
    return mb.mul(x=x, y=mb.mul(x=half, y=cdf))


def _gelu_ops(prog):
    return [op for op in prog.functions["main"].operations if op.op_type == "gelu"]


class TestFuseGeluTanh:
    """Unit tests on hand-built MIL programs."""

    def test_replaces_the_jax_chain(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x)

        assert get_op_types_in_program(prog) == [
            "mul", "mul", "mul", "add", "mul", "tanh", "add", "mul", "mul",
        ]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]
        assert _gelu_ops(prog)[0].mode.val == "TANH_APPROXIMATION"
        assert_model_is_valid(
            prog, {"x": (4, 8)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_replaces_the_pow_spelling(self):
        """coremltools' own pass only covers this one; ours has to cover both."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x, pow_cube=True)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_replaces_the_half_folded_into_x(self):
        """`(0.5 * x) * (1 + tanh(...))` is the same product, associated differently."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x, half_on_x=True)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_replaces_a_distributed_tanh_argument(self):
        """`k*x + k*a*x**3` instead of the factored `k*(x + a*x**3)`."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            cubed = mb.mul(x=mb.mul(x=x, y=x), y=x)
            inner = mb.add(
                x=mb.mul(x=x, y=SQRT_2_OVER_PI),
                y=mb.mul(x=cubed, y=SQRT_2_OVER_PI * CUBE_COEFFICIENT),
            )
            cdf = mb.add(x=mb.tanh(x=inner), y=1.0)
            return mb.mul(x=mb.mul(x=cdf, y=0.5), y=x)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_replaces_the_x_squared_times_x_spelling(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            cubed = mb.mul(x=x, y=mb.mul(x=x, y=x))
            inner = mb.add(x=mb.mul(x=CUBE_COEFFICIENT, y=cubed), y=x)
            cdf = mb.add(x=1.0, y=mb.tanh(x=mb.mul(x=SQRT_2_OVER_PI, y=inner)))
            return mb.mul(x=mb.mul(x=0.5, y=cdf), y=x)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_uniform_tensor_constants(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(
                x,
                outer=np.full((1,), SQRT_2_OVER_PI, dtype=np.float32),
                cube=np.full((1,), CUBE_COEFFICIENT, dtype=np.float32),
                half=np.full((1,), 0.5, dtype=np.float32),
                one=np.full((1,), 1.0, dtype=np.float32),
            )

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_fp16_rounded_coefficients_are_still_fused(self):
        """`0.044715` and `sqrt(2/pi)` do not survive the trip through fp16.

        The matcher multiplies the two together, so the error compounds to about
        two fp16 ulps -- well outside an fp32-calibrated tolerance.
        """
        cube = np.float16(CUBE_COEFFICIENT)
        outer = np.float16(SQRT_2_OVER_PI)
        cubic_term = float(outer) * float(cube)
        assert abs(cubic_term / (SQRT_2_OVER_PI * CUBE_COEFFICIENT) - 1.0) > 1e-4

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8), dtype=types.fp16)])
        def prog(x):
            return _tanh_gelu(x, outer=outer, cube=cube, half=np.float16(0.5), one=np.float16(1.0))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["gelu"]

    def test_not_fused_for_the_wrong_cube_coefficient(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x, cube=0.05)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_the_wrong_outer_coefficient(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x, outer=0.8)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_the_wrong_half(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x, half=0.25)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_the_wrong_offset(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x, one=2.0)

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_a_square_instead_of_a_cube(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            squared = mb.mul(x=x, y=x)
            inner = mb.add(x=x, y=mb.mul(x=CUBE_COEFFICIENT, y=squared))
            cdf = mb.add(x=1.0, y=mb.tanh(x=mb.mul(x=SQRT_2_OVER_PI, y=inner)))
            return mb.mul(x=x, y=mb.mul(x=0.5, y=cdf))

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_for_a_different_input(self):
        """The cube, the linear term and the outer factor must share one var."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(4, 8))])
        def prog(x, y):
            cubed = mb.mul(x=mb.mul(x=y, y=y), y=y)
            inner = mb.add(x=x, y=mb.mul(x=CUBE_COEFFICIENT, y=cubed))
            cdf = mb.add(x=1.0, y=mb.tanh(x=mb.mul(x=SQRT_2_OVER_PI, y=inner)))
            return mb.mul(x=x, y=mb.mul(x=0.5, y=cdf))

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_when_the_tanh_is_shared(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            cubed = mb.mul(x=mb.mul(x=x, y=x), y=x)
            inner = mb.add(x=x, y=mb.mul(x=CUBE_COEFFICIENT, y=cubed))
            tanh = mb.tanh(x=mb.mul(x=SQRT_2_OVER_PI, y=inner))
            cdf = mb.add(x=1.0, y=tanh)
            return mb.mul(x=x, y=mb.mul(x=0.5, y=cdf)), tanh

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_not_fused_when_the_square_is_shared(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            squared = mb.mul(x=x, y=x)
            cubed = mb.mul(x=squared, y=x)
            inner = mb.add(x=x, y=mb.mul(x=CUBE_COEFFICIENT, y=cubed))
            cdf = mb.add(x=1.0, y=mb.tanh(x=mb.mul(x=SQRT_2_OVER_PI, y=inner)))
            return mb.mul(x=x, y=mb.mul(x=0.5, y=cdf)), squared

        _apply(prog)
        assert "gelu" not in get_op_types_in_program(prog)

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                return _tanh_gelu(x)

            def false_fn():
                return mb.identity(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        ops = get_op_types_in_program(prog, recurse=True)
        assert "gelu" in ops
        assert "tanh" not in ops

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            return _tanh_gelu(x)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestFuseGeluTanhEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_jax_gelu_default_is_the_tanh_approximation(self):
        """`approximate=True` is the default, so this is the common spelling."""
        cml_model = run_and_compare(
            jax.nn.gelu, [jax.ShapeDtypeStruct((4, 16), jnp.float32)]
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("gelu") == 1
        assert "tanh" not in ops

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_tanh_gelu_via_jit_lowering(self, dtype):
        inputs = (jax.random.normal(jax.random.PRNGKey(0), (4, 16), dtype=dtype),)
        precision_loss = jnp.finfo(dtype).eps / jnp.finfo(jnp.float32).eps
        cml_model = run_and_compare_jit_lowering(
            lambda x: jax.nn.gelu(x, approximate=True),
            inputs,
            atol=1e-04 * precision_loss,
            rtol=1e-05 * precision_loss,
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("gelu") == 1
        assert "tanh" not in ops

    def test_tanh_gelu_saturates_the_same_way(self):
        """The fused op has to agree where the approximation flattens out."""
        big = jnp.linspace(-30.0, 30.0, 64, dtype=jnp.float32).reshape(4, 16)
        cml_model = run_and_compare_specific_input(jax.nn.gelu, (big,))
        assert get_model_instruction_types(cml_model).count("gelu") == 1

    def test_flax_gelu(self):
        cml_model = run_and_compare(nnx.gelu, [jax.ShapeDtypeStruct((4, 16), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert ops.count("gelu") == 1
        assert "tanh" not in ops

    def test_equinox_mlp_with_a_gelu_activation(self):
        model = eqx.nn.MLP(
            in_size=8, out_size=4, width_size=16, depth=2,
            activation=jax.nn.gelu, key=jax.random.PRNGKey(0),
        )
        cml_model = run_and_compare(
            eqxi.finalise_fn(eqx.nn.inference_mode(model)),
            [jax.ShapeDtypeStruct((8,), jnp.float32)],
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("gelu") == 2
        assert "tanh" not in ops

    def test_the_exact_gelu_still_fuses_to_the_exact_mode(self):
        """The two GELU passes must not steal each other's pattern."""
        inputs = (jax.random.normal(jax.random.PRNGKey(0), (4, 16), jnp.float32),)
        cml_model = run_and_compare_jit_lowering(
            lambda x: jax.nn.gelu(x, approximate=False), inputs
        )
        gelu_ops = [
            op
            for func in cml_model._mil_program.functions.values()
            for op in func.operations
            if op.op_type == "gelu"
        ]
        assert len(gelu_ops) == 1
        assert gelu_ops[0].mode.val == "EXACT"
