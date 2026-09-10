"""Cross-check the proof interpreter against MIL's constant value inference.

These fixtures keep the route operations concrete, while the proof checker
still has to show equality at an arbitrary output index.  The expected graph
is built from the value inferred by MIL, rather than from a second copy of the
tile or slice-update indexing rules.
"""

import numpy as np
from coremltools.converters.mil.mil import Builder as mb

from tests.formal.proof import prove_mil_programs_equivalent


def _prove_against_mil_value(program):
    output = program.functions["main"].outputs[0]
    assert output.name == "result"
    assert output.val is not None

    inferred_value = np.array(output.val, copy=True)

    @mb.program(input_specs=[])
    def expected():
        return mb.identity(x=inferred_value, name="result")

    expected_output = expected.functions["main"].outputs[0]
    assert expected_output.name == "result"
    prove_mil_programs_equivalent(program, expected)


def test_tile_rank1_routes_nonuniform_values_with_nonunit_repetition():
    @mb.program(input_specs=[])
    def program():
        return mb.tile(
            x=np.array([11, -3, 29], dtype=np.int32),
            reps=[3],
            name="result",
        )

    _prove_against_mil_value(program)


def test_tile_rank2_routes_nonuniform_values_with_nonunit_repetition():
    @mb.program(input_specs=[])
    def program():
        return mb.tile(
            x=np.array([[1.5, -2.0], [7.25, 9.0]], dtype=np.float32),
            reps=[2, 3],
            name="result",
        )

    _prove_against_mil_value(program)


def test_slice_update_routes_partial_stride_two_update():
    @mb.program(input_specs=[])
    def program():
        return mb.slice_update(
            x=np.arange(10, dtype=np.float32).reshape(2, 5),
            update=np.array([[100, 101], [200, 201]], dtype=np.float32),
            begin=[0, 1],
            end=[2, 5],
            stride=[1, 2],
            name="result",
        )

    _prove_against_mil_value(program)


def test_slice_update_routes_masked_end():
    @mb.program(input_specs=[])
    def program():
        return mb.slice_update(
            x=np.arange(8, dtype=np.int32).reshape(2, 4),
            update=np.array([[100, 101, 102], [200, 201, 202]], dtype=np.int32),
            begin=[0, 1],
            end=[2, 0],
            end_mask=[False, True],
            name="result",
        )

    _prove_against_mil_value(program)


def test_identity_preserves_raw_fp32_bits():
    nan = np.array([0x7FC01234], dtype=np.uint32).view(np.float32)[0]
    values = np.array([0.0, -0.0, nan, np.inf, -np.inf], dtype=np.float32)

    @mb.program(input_specs=[])
    def program():
        return mb.identity(x=values, name="result")

    output = program.functions["main"].outputs[0]
    np.testing.assert_array_equal(output.val.view(np.uint32), values.view(np.uint32))
    _prove_against_mil_value(program)
