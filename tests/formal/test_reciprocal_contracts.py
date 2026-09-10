"""Scalar IEEE contracts for the reciprocal spellings used by softcapping.

These tests establish only scalar input-rounding facts.  They do not prove the
softcap matcher, the final ``tanh`` result, or Core ML backend behavior.
"""

import numpy as np
import pytest
import z3
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

import stablehlo_coreml  # noqa: F401  # register production passes
from tests.formal.proof import prove_for_all


@pytest.mark.parametrize(
    ("exponent_bits", "significand_bits", "divisor"),
    [(5, 11, 2.0), (5, 11, 4.0), (8, 24, 2.0), (8, 24, 4.0)],
)
def test_power_of_two_division_matches_multiply_by_exact_reciprocal(
    exponent_bits, significand_bits, divisor
):
    """Prove raw result bits agree for every non-NaN input encoding.

    The unconstrained input bit-vector includes infinities, both signed zeros,
    and subnormals. Only NaN inputs are excluded; the lemma proves both
    operation outputs are non-NaN before comparing their raw bits.
    """
    sort = z3.FPSort(exponent_bits, significand_bits)
    width = exponent_bits + significand_bits
    input_bits = z3.BitVec(f"reciprocal_power_two_{width}_{int(divisor)}", width)
    value = z3.fpToFP(input_bits, sort)
    divided = z3.fpDiv(z3.RNE(), value, z3.FPVal(divisor, sort))
    multiplied = z3.fpMul(z3.RNE(), value, z3.FPVal(1.0 / divisor, sort))

    prove_for_all(
        [z3.Not(z3.fpIsNaN(value))],
        z3.And(
            z3.Not(z3.fpIsNaN(divided)),
            z3.Not(z3.fpIsNaN(multiplied)),
            z3.fpToIEEEBV(divided) == z3.fpToIEEEBV(multiplied),
        ),
        theorem=f"fp{width} division by {int(divisor)} preserves raw result bits",
    )


def _bits(value, dtype) -> int:
    unsigned_dtype = np.uint16 if dtype is np.float16 else np.uint32
    return int(np.asarray(value, dtype=dtype).reshape(()).view(unsigned_dtype))


def _value_from_bits(bits: int, dtype):
    unsigned_dtype = np.uint16 if dtype is np.float16 else np.uint32
    return np.asarray([bits], dtype=unsigned_dtype).view(dtype)[0]


@pytest.mark.parametrize(
    ("numpy_dtype", "x_bits", "divided_bits", "multiplied_bits", "beta_bits"),
    [
        (np.float16, 0x1A25, 0x068E, 0x068D, 0x2844),
        (np.float32, 0x3A83126F, 0x380BCF65, 0x380BCF66, 0x3D088889),
    ],
)
def test_c30_fixed_finite_input_rounding_witness(
    numpy_dtype, x_bits, divided_bits, multiplied_bits, beta_bits
):
    """Record a stable input-stage mismatch for c=30, without claiming output mismatch."""
    x = _value_from_bits(x_bits, numpy_dtype)
    cap = numpy_dtype(30.0)
    beta = numpy_dtype(1.0 / 30.0)
    divided = numpy_dtype(x / cap)
    multiplied = numpy_dtype(x * beta)

    assert _bits(x, numpy_dtype) == x_bits
    assert _bits(beta, numpy_dtype) == beta_bits
    assert _bits(divided, numpy_dtype) == divided_bits
    assert _bits(multiplied, numpy_dtype) == multiplied_bits
    assert divided_bits != multiplied_bits


@pytest.mark.parametrize(
    ("exponent_bits", "significand_bits", "x_bits", "divided_bits", "multiplied_bits", "beta_bits"),
    [
        (5, 11, 0x1A25, 0x068E, 0x068D, 0x2844),
        (8, 24, 0x3A83126F, 0x380BCF65, 0x380BCF66, 0x3D088889),
    ],
)
def test_c30_fixed_witness_is_replayed_by_z3(
    exponent_bits, significand_bits, x_bits, divided_bits, multiplied_bits, beta_bits
):
    """Replay the fixed NumPy witness with IEEE operations in Z3."""
    sort = z3.FPSort(exponent_bits, significand_bits)
    width = exponent_bits + significand_bits
    value = z3.fpToFP(z3.BitVecVal(x_bits, width), sort)
    beta = z3.fpToFP(z3.BitVecVal(beta_bits, width), sort)
    divided = z3.fpDiv(z3.RNE(), value, z3.FPVal(30.0, sort))
    multiplied = z3.fpMul(z3.RNE(), value, beta)

    prove_for_all(
        [],
        z3.And(
            z3.fpToIEEEBV(divided) == z3.BitVecVal(divided_bits, width),
            z3.fpToIEEEBV(multiplied) == z3.BitVecVal(multiplied_bits, width),
            z3.fpToIEEEBV(divided) != z3.fpToIEEEBV(multiplied),
        ),
        theorem=f"fp{width} c30 fixed input-rounding witness",
    )


@pytest.mark.parametrize(
    ("mil_dtype", "numpy_dtype", "beta_bits"),
    [(types.fp16, np.float16, 0x2844), (types.fp32, np.float32, 0x3D088889)],
)
def test_production_softcap_emits_inner_scale_python_reciprocal(mil_dtype, numpy_dtype, beta_bits):
    """Tie the c=30 reciprocal witness to the production matcher output."""
    scalar_type = types.nptype_from_builtin(mil_dtype)
    cap = scalar_type(30.0)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=mil_dtype)])
    def program(x):
        scaled = mb.real_div(x=x, y=cap)
        return mb.mul(x=mb.tanh(x=scaled), y=cap)

    program.validate(check_essential_scope=True)
    PASS_REGISTRY["common::fuse_logit_softcap"](program)
    program.validate(check_essential_scope=True)
    fused = program.functions["main"].outputs[0].op
    assert fused.op_type == "scaled_tanh"

    assert _bits(fused.alpha.val, numpy_dtype) == _bits(cap, numpy_dtype)
    assert _bits(fused.beta.val, numpy_dtype) == beta_bits
