"""Cross-check shape-routing models against MIL value inference."""

import numpy as np
from coremltools.converters.mil.mil import Builder as mb

from tests.formal.proof import prove_mil_programs_equivalent
from tests.formal.test_semantics import _prove_against_mil_value


def test_reshape_preserves_multidimensional_row_major_order():
    @mb.program(input_specs=[])
    def program():
        return mb.reshape(
            x=np.arange(24, dtype=np.int32).reshape(2, 3, 4),
            shape=[4, 2, 3],
            name="result",
        )

    _prove_against_mil_value(program)


def test_reshape_infers_one_negative_dimension():
    @mb.program(input_specs=[])
    def program():
        return mb.reshape(
            x=np.arange(24, dtype=np.int32).reshape(2, 3, 4),
            shape=[-1, 4],
            name="result",
        )

    _prove_against_mil_value(program)


def test_reshape_copies_zero_dimension_and_infers_middle_dimension():
    @mb.program(input_specs=[])
    def program():
        return mb.reshape(
            x=np.arange(24, dtype=np.int32).reshape(2, 3, 4),
            shape=[0, -1, 2],
            name="result",
        )

    _prove_against_mil_value(program)


def test_expand_dims_accepts_multiple_negative_axes():
    @mb.program(input_specs=[])
    def program():
        return mb.expand_dims(
            x=np.arange(6, dtype=np.int32).reshape(2, 3),
            axes=[-1, -3],
            name="result",
        )

    _prove_against_mil_value(program)


def test_reshape_then_expand_dims_routes_nontrivial_shape_order():
    @mb.program(input_specs=[])
    def program():
        reshaped = mb.reshape(
            x=np.arange(24, dtype=np.int32).reshape(2, 3, 4),
            shape=[4, 2, 3],
        )
        return mb.expand_dims(x=reshaped, axes=[-1], name="result")

    _prove_against_mil_value(program)


def test_scalar_reduction_can_be_restored_to_all_unit_shape():
    values = np.arange(6, dtype=np.int32).reshape(2, 3)

    @mb.program(input_specs=[])
    def program():
        reduced = mb.reduce_sum(x=values, axes=[0, 1], keep_dims=False)
        return mb.reshape(x=reduced, shape=[1, 1], name="result")

    output = program.functions["main"].outputs[0]
    assert output.name == "result"
    assert output.shape == (1, 1)
    np.testing.assert_array_equal(output.val, np.array([[15]], dtype=np.int32))

    # Reductions are modeled as uninterpreted operations in the proof checker;
    # compare the restored scalar route with the independent keep-dims route
    # instead of asking SMT to reproduce reduction arithmetic.
    @mb.program(input_specs=[])
    def expected():
        return mb.reduce_sum(x=values, axes=[0, 1], keep_dims=True, name="result")

    expected_output = expected.functions["main"].outputs[0]
    assert expected_output.name == "result"
    np.testing.assert_array_equal(output.val.view(np.uint32), expected_output.val.view(np.uint32))
    prove_mil_programs_equivalent(program, expected)
