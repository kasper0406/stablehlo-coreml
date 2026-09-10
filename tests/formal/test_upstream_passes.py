"""Bounded smoke checks for reusing coremltools' upstream graph passes.

These fixtures validate one upstream pass translation against the formal
checker; they are intentionally not a claim that every upstream pass is
formally covered.
"""

import copy

import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

from tests.formal.proof import prove_mil_programs_equivalent


@pytest.mark.parametrize("op_name, shape", [("add", (4,)), ("mul", (2, 3))])
def test_noop_elimination_removes_same_shape_reshape_with_proof(op_name, shape):
    @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
    def program(x):
        reshaped = mb.reshape(x=x, shape=list(shape), name="noop")
        other = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
        return getattr(mb, op_name)(x=reshaped, y=other, name="result")

    before = copy.deepcopy(program)
    before.validate()
    program.validate()
    PASS_REGISTRY["common::noop_elimination"](program)
    program.validate()

    assert any(op.op_type == "reshape" for op in before.functions["main"].operations)
    assert all(op.op_type != "reshape" for op in program.functions["main"].operations)
    prove_mil_programs_equivalent(before, program)
