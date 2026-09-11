"""Scalar contracts for the identity operation in ``broadcast_select_operands``.

These lemmas describe the scalar operation inserted by the pass.  They do not
verify the pass matcher, MIL graph mutation, Core ML backend behavior, or
flush-to-zero floating-point modes.
"""

import pytest
import z3

from tests.formal.proof import ProofFailure, prove_for_all


def _fp_sort(exponent_bits: int, significand_bits: int) -> z3.FPSortRef:
    return z3.FPSort(exponent_bits, significand_bits)


def _fp_eq_or_both_nan(left: z3.FPRef, right: z3.FPRef) -> z3.BoolRef:
    """IEEE equality, treating any two NaNs as equivalent for this contract."""
    return z3.Or(z3.fpEQ(left, right), z3.And(z3.fpIsNaN(left), z3.fpIsNaN(right)))


@pytest.mark.parametrize("exponent_bits, significand_bits", [(5, 11), (8, 24)])
def test_float_add_positive_zero_preserves_fp_eq_or_both_nan(exponent_bits, significand_bits):
    sort = _fp_sort(exponent_bits, significand_bits)
    bits = z3.BitVec(f"select_float_{exponent_bits}_bits", exponent_bits + significand_bits)
    value = z3.fpToFP(bits, sort)
    widened = z3.fpAdd(z3.RNE(), value, z3.FPVal(0.0, sort))

    # The unconstrained bit-vector ranges over every IEEE encoding, including
    # both signed zeros, infinities, and NaNs.  fpEQ intentionally ignores the
    # sign of zero; the second branch ignores NaN payload differences.
    prove_for_all(
        [],
        _fp_eq_or_both_nan(value, widened),
        theorem=f"fp{exponent_bits + significand_bits} add +0 preserves fpEQ",
    )


@pytest.mark.parametrize("exponent_bits, significand_bits", [(5, 11), (8, 24)])
def test_float_add_positive_zero_is_not_exact_bit_preserving_for_negative_zero(exponent_bits, significand_bits):
    sort = _fp_sort(exponent_bits, significand_bits)
    width = exponent_bits + significand_bits
    bits = z3.BitVec(f"select_negative_zero_{width}_bits", width)
    value = z3.fpToFP(bits, sort)
    widened = z3.fpAdd(z3.RNE(), value, z3.FPVal(0.0, sort))
    negative_zero_bits = z3.BitVecVal(1 << (width - 1), width)

    # Pin the operand to -0 and exclude NaN before comparing raw IEEE bits.  A
    # raw-bit equality contract is therefore refuted by the +0 result.
    with pytest.raises(ProofFailure, match="counterexample"):
        prove_for_all(
            [bits == negative_zero_bits, z3.Not(z3.fpIsNaN(value))],
            z3.fpToIEEEBV(value) == z3.fpToIEEEBV(widened),
            theorem=f"fp{width} add +0 preserves exact bits for -0",
        )


@pytest.mark.parametrize("exponent_bits, significand_bits", [(5, 11), (8, 24)])
def test_relaxed_float_contract_is_not_contextual_for_reciprocal(exponent_bits, significand_bits):
    sort = _fp_sort(exponent_bits, significand_bits)
    width = exponent_bits + significand_bits
    bits = z3.BitVec(f"select_reciprocal_{width}_bits", width)
    value = z3.fpToFP(bits, sort)
    widened = z3.fpAdd(z3.RNE(), value, z3.FPVal(0.0, sort))
    one = z3.FPVal(1.0, sort)
    reciprocal = z3.fpDiv(z3.RNE(), one, value)
    widened_reciprocal = z3.fpDiv(z3.RNE(), one, widened)
    negative_zero_bits = z3.BitVecVal(1 << (width - 1), width)

    # fpEQ identifies signed zeros, but that relation is not safe to substitute
    # through every context: 1 / -0 is -inf while 1 / +0 is +inf.
    with pytest.raises(ProofFailure, match="counterexample"):
        prove_for_all(
            [bits == negative_zero_bits, z3.Not(z3.fpIsNaN(value))],
            _fp_eq_or_both_nan(reciprocal, widened_reciprocal),
            theorem=f"fp{width} relaxed +0 contract is contextual under reciprocal",
        )


@pytest.mark.parametrize("exponent_bits, significand_bits", [(5, 11), (8, 24)])
def test_float_add_positive_zero_is_bit_exact_off_nan_and_negative_zero(exponent_bits, significand_bits):
    sort = _fp_sort(exponent_bits, significand_bits)
    width = exponent_bits + significand_bits
    bits = z3.BitVec(f"select_exact_domain_{width}_bits", width)
    value = z3.fpToFP(bits, sort)
    widened = z3.fpAdd(z3.RNE(), value, z3.FPVal(0.0, sort))

    # This stronger contract is valid after excluding NaNs and negative zero;
    # positive zero remains in the domain. It is still a scalar IEEE lemma, not
    # a proof of the pass.
    prove_for_all(
        [z3.Not(z3.fpIsNaN(value)), z3.Not(z3.And(z3.fpIsZero(value), z3.fpIsNegative(value)))],
        z3.fpToIEEEBV(value) == z3.fpToIEEEBV(widened),
        theorem=f"fp{width} add +0 preserves exact bits off zero and NaN",
    )


def test_int32_add_zero_preserves_exact_bits():
    value = z3.BitVec("select_int32_bits", 32)
    prove_for_all(
        [],
        value + z3.BitVecVal(0, 32) == value,
        theorem="int32 add zero preserves exact bits",
    )


def test_bool_or_false_preserves_exact_value():
    value = z3.Bool("select_bool_value")
    prove_for_all([], z3.Or(value, z3.BoolVal(False)) == value, theorem="bool or false preserves value")
