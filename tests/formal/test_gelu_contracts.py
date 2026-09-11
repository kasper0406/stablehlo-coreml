"""Concrete counterexamples to treating tolerant GELU matches as exact.

The symbolic fixtures show that the production matchers consume the perturbed
constants and replace the whole pattern with a native ``gelu``.  The separate
constant-only fixtures use MIL value inference for the original pattern and
the native op at ``x=20``; their declared-fp32 outputs differ, so saturation
does not hide the coefficient error in these examples. This is evidence from
MIL's reference value inference, not a claim about every native backend.
"""

import math

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

import stablehlo_coreml  # noqa: F401  # registers the production passes


def _erfc_pattern(x, half):
    half_x = mb.mul(x=x, y=half)
    argument = mb.mul(x=mb.sub(x=0.0, y=x), y=1.0 / math.sqrt(2.0))
    return mb.mul(x=half_x, y=mb.sub(x=1.0, y=mb.erf(x=argument)))


def _tanh_pattern(x, half):
    cubed = mb.mul(x=mb.mul(x=x, y=x), y=x)
    inner = mb.add(x=x, y=mb.mul(x=0.044715, y=cubed))
    cdf = mb.add(x=1.0, y=mb.tanh(x=mb.mul(x=math.sqrt(2.0 / math.pi), y=inner)))
    return mb.mul(x=x, y=mb.mul(x=half, y=cdf))


def _assert_symbolic_pattern_fuses(builder, half, pass_name, mode):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
    def program(x):
        return builder(x, half)

    program.validate()
    PASS_REGISTRY[pass_name](program)
    PASS_REGISTRY["common::dead_code_elimination"](program)
    program.validate()
    gelus = [op for op in program.functions["main"].operations if op.op_type == "gelu"]
    assert len(gelus) == 1
    assert program.functions["main"].outputs[0].op is gelus[0]
    assert gelus[0].mode.val == mode


def _inferred_output(builder, half, mode):
    value = np.array([20.0], dtype=np.float32)

    @mb.program(input_specs=[])
    def original():
        return builder(value, half)

    @mb.program(input_specs=[])
    def native():
        return mb.gelu(x=value, mode=mode, name="result")

    values = []
    for program in (original, native):
        output = program.functions["main"].outputs[0]
        assert types.builtin_to_string(output.dtype) == "fp32"
        assert output.val is not None
        value = np.asarray(output.val, dtype=np.float32)
        assert value.shape == (1,)
        assert np.isfinite(value).all()
        values.append(value)
    original_value, native_value = values
    return original_value, native_value


def test_erfc_match_accepts_noncanonical_half_with_different_mil_value():
    half = np.nextafter(np.float32(0.5), np.float32(np.inf))
    assert half != np.float32(0.5)
    assert abs(float(half) - 0.5) < 1e-4
    _assert_symbolic_pattern_fuses(
        _erfc_pattern,
        half,
        "common::fuse_gelu_erfc",
        "EXACT",
    )

    original, native = _inferred_output(_erfc_pattern, half, "EXACT")
    np.testing.assert_array_equal(original.view(np.uint32), np.array([1101004801], dtype=np.uint32))
    np.testing.assert_array_equal(native.view(np.uint32), np.array([1101004800], dtype=np.uint32))
    assert not np.array_equal(original.view(np.uint32), native.view(np.uint32))


def test_tanh_match_accepts_noncanonical_half_with_different_mil_value():
    half = np.float32(0.50004)
    assert half != np.float32(0.5)
    assert abs(float(half) - 0.5) < 1e-4
    _assert_symbolic_pattern_fuses(
        _tanh_pattern,
        half,
        "common::fuse_gelu_tanh",
        "TANH_APPROXIMATION",
    )

    original, native = _inferred_output(_tanh_pattern, half, "TANH_APPROXIMATION")
    np.testing.assert_array_equal(original.view(np.uint32), np.array([1101005639], dtype=np.uint32))
    np.testing.assert_array_equal(native.view(np.uint32), np.array([1101004800], dtype=np.uint32))
    assert not np.array_equal(original.view(np.uint32), native.view(np.uint32))
