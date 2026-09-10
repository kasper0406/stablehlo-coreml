"""Exact structural proof subset for ``fuse_logit_softcap``.

The subset follows coremltools' executable operand order exactly:
``alpha * tanh(x * beta)``.  Equality is conditional on ``scaled_tanh`` using
the same typed multiply rounding and tanh kernel as the three separate ops.
This is deliberately not a proof of a native backend's IEEE-754 behavior.
"""

import copy

import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

# Importing the package registers the production pass in PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401
from tests.formal.proof import InequivalentMILPrograms, UnsupportedMILGraph, prove_mil_programs_equivalent


def _apply_pass(prog):
    before = copy.deepcopy(prog)
    before.validate(check_essential_scope=True)
    PASS_REGISTRY["common::fuse_logit_softcap"](prog)
    prog.validate(check_essential_scope=True)
    return before, prog


def _numpy_dtype(mil_dtype):
    return np.float16 if mil_dtype is types.fp16 else np.float32


def _raw_scalar(value, dtype):
    return np.asarray(value, dtype=dtype).reshape(()).tobytes()


@pytest.mark.parametrize("dtype", [types.fp16, types.fp32])
@pytest.mark.parametrize(("alpha", "beta"), [(0.5, 2.0), (3.0, 0.5)])
def test_production_multiply_softcap_fusion_preserves_ordered_operation_bits(dtype, alpha, beta):
    numpy_dtype = _numpy_dtype(dtype)
    alpha_value, beta_value = numpy_dtype(alpha), numpy_dtype(beta)

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=dtype)])
    def prog(x):
        # Match scaled_tanh.value_inference exactly: alpha * tanh(x * beta).
        inner = mb.mul(x=x, y=beta_value)
        activated = mb.tanh(x=inner)
        return mb.mul(x=alpha_value, y=activated, name="output")

    before, after = _apply_pass(prog)
    fused = after.functions["main"].outputs[0].op
    assert fused.op_type == "scaled_tanh"
    assert _raw_scalar(fused.alpha.val, numpy_dtype) == _raw_scalar(alpha_value, numpy_dtype)
    assert _raw_scalar(fused.beta.val, numpy_dtype) == _raw_scalar(beta_value, numpy_dtype)
    prove_mil_programs_equivalent(before, after)


@pytest.mark.parametrize(("wrong_alpha", "wrong_beta"), [(2.0, 2.0), (0.5, 3.0)])
def test_checker_finds_altered_scaled_tanh_parameters(wrong_alpha, wrong_beta):
    specs = [mb.TensorSpec(shape=(2, 3))]

    @mb.program(input_specs=specs)
    def before(x):
        inner = mb.mul(x=x, y=np.float32(2.0))
        return mb.mul(x=np.float32(0.5), y=mb.tanh(x=inner), name="output")

    @mb.program(input_specs=specs)
    def wrong_after(x):
        return mb.scaled_tanh(
            x=x,
            alpha=np.float32(wrong_alpha),
            beta=np.float32(wrong_beta),
            name="output",
        )

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_finds_reordered_abstract_multiplications():
    """The bit model intentionally gives multiplication no commutativity axiom."""
    specs = [mb.TensorSpec(shape=(2, 3))]

    @mb.program(input_specs=specs)
    def before(x):
        inner = mb.mul(x=x, y=np.float32(2.0))
        return mb.mul(x=np.float32(0.5), y=mb.tanh(x=inner), name="output")

    @mb.program(input_specs=specs)
    def reordered(x):
        inner = mb.mul(x=np.float32(2.0), y=x)
        return mb.mul(x=mb.tanh(x=inner), y=np.float32(0.5), name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, reordered)


def test_constant_left_inner_multiply_is_outside_current_model_proof_subset():
    """The abstract UF counterexample does not establish a numerical difference.

    Multiplication commutativity may hold under a richer IEEE/native contract;
    this deliberately axiom-free model cannot use it.
    """
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        inner = mb.mul(x=np.float32(2.0), y=x)
        return mb.mul(x=np.float32(0.5), y=mb.tanh(x=inner), name="output")

    before, after = _apply_pass(prog)
    assert after.functions["main"].outputs[0].op.op_type == "scaled_tanh"
    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, after)


def test_missing_inner_multiply_is_outside_current_model_proof_subset():
    """The abstract UF counterexample does not establish a numerical difference.

    Multiplication by one may be an identity under a richer IEEE/native
    contract; this deliberately axiom-free model cannot use that identity.
    """
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        return mb.mul(x=np.float32(3.0), y=mb.tanh(x=x), name="output")

    before, after = _apply_pass(prog)
    fused = after.functions["main"].outputs[0].op
    assert fused.op_type == "scaled_tanh"
    assert fused.beta.val == np.float32(1.0)
    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, after)


def test_checker_fails_closed_for_non_scalar_scaled_tanh_parameter():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def malformed(x):
        return mb.scaled_tanh(
            x=x,
            alpha=np.array([0.5, 0.5], dtype=np.float32),
            beta=np.float32(2.0),
            name="output",
        )

    with pytest.raises(UnsupportedMILGraph, match="must be rank-0 scalars"):
        prove_mil_programs_equivalent(malformed, malformed)


def test_checker_fails_closed_for_nonconstant_scaled_tanh_parameter():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def malformed(x):
        return mb.scaled_tanh(x=x, alpha=np.float32(0.5), beta=np.float32(2.0), name="output")

    malformed.functions["main"].outputs[0].op.alpha._sym_val = None
    with pytest.raises(UnsupportedMILGraph, match="compile-time constants"):
        prove_mil_programs_equivalent(malformed, malformed)
