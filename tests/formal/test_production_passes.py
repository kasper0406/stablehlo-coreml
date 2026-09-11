import copy

import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

# Importing the package registers the production passes in PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401
from tests.formal.proof import (
    InequivalentMILPrograms,
    MILGraphContractError,
    UnsupportedMILGraph,
    prove_mil_programs_equivalent,
)


def _apply_registered_pass(prog, pass_name):
    before = copy.deepcopy(prog)
    before.validate(check_essential_scope=True)
    PASS_REGISTRY[pass_name](prog)
    prog.validate(check_essential_scope=True)
    return before, prog


def _op_types(prog):
    return [op.op_type for op in prog.functions["main"].operations if op.op_type != "const"]


@pytest.mark.parametrize("dtype", [types.fp16, types.fp32, types.int32, types.bool])
def test_production_remove_noop_slice_update_is_bit_exact(dtype):
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4), dtype=dtype), mb.TensorSpec(shape=(3, 4), dtype=dtype)])
    def prog(buffer, update):
        return mb.slice_update(
            x=buffer,
            update=update,
            begin=[0, 0],
            end=[0, 4],
            end_mask=[True, False],
            name="updated",
        )

    before, after = _apply_registered_pass(prog, "common::remove_noop_slice_update")
    assert _op_types(before) == ["slice_update"]
    assert _op_types(after) == ["identity"]
    prove_mil_programs_equivalent(before, after)


@pytest.mark.parametrize(
    ("op_name", "dtype"),
    [
        ("add", types.fp16),
        ("sub", types.fp32),
        ("mul", types.int32),
        ("equal", types.int32),
        ("logical_and", types.bool),
    ],
)
def test_production_remove_broadcast_tile_is_bit_exact_for_pointwise_consumers(op_name, dtype):
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4), dtype=dtype), mb.TensorSpec(shape=(3, 1), dtype=dtype)])
    def prog(x, y):
        tiled = mb.tile(x=y, reps=[1, 4])
        return getattr(mb, op_name)(x=x, y=tiled)

    before, after = _apply_registered_pass(prog, "common::remove_broadcast_tiles")
    assert _op_types(before) == ["tile", op_name]
    assert _op_types(after) == [op_name]
    prove_mil_programs_equivalent(before, after)


def test_production_remove_both_broadcast_tiles_is_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 1)), mb.TensorSpec(shape=(1, 4))])
    def prog(x, y):
        tiled_x = mb.tile(x=x, reps=[1, 4])
        tiled_y = mb.tile(x=y, reps=[3, 1])
        return mb.add(x=tiled_x, y=tiled_y)

    before, after = _apply_registered_pass(prog, "common::remove_broadcast_tiles")
    assert _op_types(before) == ["tile", "tile", "add"]
    assert _op_types(after) == ["add"]
    prove_mil_programs_equivalent(before, after)


def test_production_remove_tile_with_lower_rank_other_operand_is_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(4,)), mb.TensorSpec(shape=(3, 1))])
    def prog(x, y):
        return mb.maximum(x=x, y=mb.tile(x=y, reps=[1, 4]))

    before, after = _apply_registered_pass(prog, "common::remove_broadcast_tiles")
    assert _op_types(before) == ["tile", "maximum"]
    assert _op_types(after) == ["maximum"]
    prove_mil_programs_equivalent(before, after)


def test_production_remove_unit_repetition_from_unit_axis_is_bit_exact():
    """The consumer, rather than the tile, supplies both output dimensions."""
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(1, 1))])
    def prog(x, y):
        return mb.add(x=x, y=mb.tile(x=y, reps=[1, 1]))

    before, after = _apply_registered_pass(prog, "common::remove_broadcast_tiles")
    assert _op_types(before) == ["tile", "add"]
    assert _op_types(after) == ["add"]
    prove_mil_programs_equivalent(before, after)


def test_production_keeps_partial_slice_update_and_proof_still_holds():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(2, 4))])
    def prog(buffer, update):
        return mb.slice_update(x=buffer, update=update, begin=[1, 0], end=[3, 4], name="output")

    before, after = _apply_registered_pass(prog, "common::remove_noop_slice_update")
    assert _op_types(before) == ["slice_update"]
    assert _op_types(after) == ["slice_update"]
    prove_mil_programs_equivalent(before, after)


def test_production_keeps_nonbroadcast_tile_and_proof_still_holds():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2,)), mb.TensorSpec(shape=(6,))])
    def prog(x, y):
        return mb.add(x=mb.tile(x=x, reps=[3]), y=y)

    before, after = _apply_registered_pass(prog, "common::remove_broadcast_tiles")
    assert _op_types(before) == ["tile", "add"]
    assert _op_types(after) == ["tile", "add"]
    prove_mil_programs_equivalent(before, after)


def test_production_keeps_tile_used_as_both_operands_and_proof_still_holds():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 1))])
    def prog(x):
        tiled = mb.tile(x=x, reps=[1, 4])
        return mb.mul(x=tiled, y=tiled)

    before, after = _apply_registered_pass(prog, "common::remove_broadcast_tiles")
    assert _op_types(before) == ["tile", "mul"]
    assert _op_types(after) == ["tile", "mul"]
    prove_mil_programs_equivalent(before, after)


def test_checker_finds_a_wrong_slice_update_replacement_with_same_shape_and_dtype():
    specs = [mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(2, 3))]

    @mb.program(input_specs=specs)
    def before(buffer, update):
        return mb.slice_update(x=buffer, update=update, begin=[0, 0], end=[2, 3], name="output")

    @mb.program(input_specs=specs)
    def wrong_after(buffer, update):
        return mb.identity(x=buffer, name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_finds_a_wrong_pointwise_rewrite_with_same_shape_and_dtype():
    specs = [mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(3, 1))]

    @mb.program(input_specs=specs)
    def before(x, y):
        return mb.sub(x=x, y=mb.tile(x=y, reps=[1, 4]), name="output")

    @mb.program(input_specs=specs)
    def wrong_after(x, y):
        return mb.sub(x=y, y=x, name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_rejects_output_shape_changes_before_solver_use():
    specs = [mb.TensorSpec(shape=(3, 1))]

    @mb.program(input_specs=specs)
    def before(x):
        return mb.tile(x=x, reps=[1, 4], name="output")

    @mb.program(input_specs=specs)
    def wrong_after(x):
        return mb.identity(x=x, name="output")

    with pytest.raises(MILGraphContractError, match="changed shape"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_rejects_dtype_changes_before_solver_use():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp16)])
    def before(x):
        return mb.identity(x=x, name="output")

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.int32)])
    def wrong_after(x):
        return mb.identity(x=x, name="output")

    with pytest.raises(MILGraphContractError, match="changed dtype"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_fails_closed_for_unsupported_operations():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        return mb.relu(x=x)

    with pytest.raises(UnsupportedMILGraph, match="unsupported operation 'relu'"):
        prove_mil_programs_equivalent(prog, prog)


def test_checker_fails_closed_for_clipped_slice_indices():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3,)), mb.TensorSpec(shape=(3,))])
    def prog(buffer, update):
        return mb.slice_update(x=buffer, update=update, begin=[0], end=[99])

    with pytest.raises(UnsupportedMILGraph, match="must lie within"):
        prove_mil_programs_equivalent(prog, prog)


def test_checker_fails_closed_for_symbolic_fixture_shapes():
    dynamic = get_new_symbol()

    @mb.program(input_specs=[mb.TensorSpec(shape=(dynamic, 3))])
    def prog(x):
        return mb.identity(x=x)

    with pytest.raises(UnsupportedMILGraph, match="symbolic fixture shape"):
        prove_mil_programs_equivalent(prog, prog)


def test_checker_compares_constant_bits_instead_of_only_shapes():
    @mb.program(input_specs=[])
    def before():
        return mb.identity(x=np.array([0.0, -0.0], dtype=np.float32), name="output")

    @mb.program(input_specs=[])
    def wrong_after():
        return mb.identity(x=np.array([0.0, 0.0], dtype=np.float32), name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)
