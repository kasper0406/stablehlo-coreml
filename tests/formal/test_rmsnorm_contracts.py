"""Bounded RMSNorm counterexample under IEEE fp32 gradual underflow.

The replay uses Z3's IEEE-754 binary32 operations with round-to-nearest-even.
The MIL comparison checks reference value inference for the same fixed tensor;
neither test makes a claim about every native Core ML backend or FTZ mode.
"""

import numpy as np
import z3
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

import stablehlo_coreml  # noqa: F401  # registers the production pass
from tests.formal.proof import prove_for_all


def _prove_fp32_bits(value, bits):
    prove_for_all(
        [],
        z3.fpToIEEEBV(value) == z3.BitVecVal(bits, 32),
        theorem=f"fixed fp32 expression has bits 0x{bits:08x}",
    )


def test_ieee_replay_exposes_mean_underflow_and_rmsnorm_mismatch():
    sort = z3.FPSort(8, 24)
    rne = z3.RNE()

    def fp(value):
        return z3.FPVal(value, sort)

    x = fp(2**-74)
    zero, one = fp(0.0), fp(1.0)

    square = z3.fpMul(rne, x, x)
    total = square
    for _ in range(3):
        total = z3.fpAdd(rne, total, zero)
    mean = z3.fpDiv(rne, total, fp(4.0))
    old = z3.fpMul(rne, x, z3.fpDiv(rne, one, z3.fpSqrt(rne, mean)))
    fused = z3.fpMul(rne, z3.fpDiv(rne, x, z3.fpSqrt(rne, total)), fp(2.0))

    _prove_fp32_bits(square, 0x00000002)  # 2**-148, twice the minimum subnormal.
    _prove_fp32_bits(mean, 0x00000000)
    _prove_fp32_bits(old, 0x7F800000)  # +infinity.
    _prove_fp32_bits(fused, 0x40000000)  # 2.0.


def _rms_pattern(x):
    square = mb.mul(x=x, y=x)
    mean = mb.reduce_mean(x=square, axes=[-1], keep_dims=True)
    shifted = mb.add(x=mean, y=np.float32(0.0))
    inverse = mb.rsqrt(x=shifted, epsilon=np.float32(0.0))
    return mb.mul(x=x, y=inverse, name="result")


def test_mil_value_inference_replays_the_same_counterexample():
    value = np.array([[[2**-74, 0.0, 0.0, 0.0]]], dtype=np.float32)

    @mb.program(input_specs=[])
    def original():
        return _rms_pattern(value)

    @mb.program(input_specs=[])
    def fused_reference():
        normalized = mb.l2_norm(x=value, epsilon=np.float32(0.0))
        return mb.mul(x=normalized, y=np.float32(2.0), name="result")

    values = []
    for program in (original, fused_reference):
        output = program.functions["main"].outputs[0]
        assert types.builtin_to_string(output.dtype) == "fp32"
        assert output.val is not None
        value = np.asarray(output.val, dtype=np.float32)
        assert value.shape == (1, 1, 4)
        values.append(value)
    old_value, new_value = values
    old_bits = old_value.view(np.uint32)
    new_bits = new_value.view(np.uint32)
    assert old_bits[0, 0, 0] == 0x7F800000
    assert new_bits[0, 0, 0] == 0x40000000
    assert np.isposinf(old_value[0, 0, 0])
    assert np.isfinite(new_value[0, 0, 0])


def test_production_rmsnorm_matcher_emits_the_fused_route():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4), dtype=types.fp32)])
    def program(x):
        return _rms_pattern(x)

    program.validate()
    PASS_REGISTRY["common::fuse_rmsnorm"](program)
    program.validate()
    operations = [op for op in program.functions["main"].operations if op.op_type != "const"]
    assert [op.op_type for op in operations] == ["l2_norm", "mul"]
    l2_norm, final_mul = operations
    function = program.functions["main"]
    assert function.outputs[0].op is final_mul
    assert l2_norm.x is function.inputs["x"]
    assert final_mul.x is l2_norm.outputs[0] or final_mul.y is l2_norm.outputs[0]
    assert l2_norm.epsilon.val == np.float32(0.0)
    assert final_mul.y.val == np.float32(2.0)
