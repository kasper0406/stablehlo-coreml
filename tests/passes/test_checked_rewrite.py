"""Mutation-boundary regressions for the full-tensor slice-update rewrite.

These tests deliberately exercise the public pass registry and MIL's validation
boundary.  They are complementary to the symbolic proof fixtures: a proof of
the replacement rule does not by itself test fan-out, block traversal, output
name handling, or a failed graph mutation.
"""

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY

import stablehlo_coreml  # noqa: F401  # registers the production passes

PASS_NAME = "common::remove_noop_slice_update"


def _apply(program):
    """Run the registered pass directly, then validate its public MIL result."""
    PASS_REGISTRY[PASS_NAME](program)
    program.validate(check_essential_scope=True)


def _all_ops(block):
    ops = list(block.operations)
    for op in block.operations:
        for nested in op.blocks:
            ops.extend(_all_ops(nested))
    return ops


def test_full_update_fanout_and_duplicate_uses_are_rewritten():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(2, 3))])
    def program(buffer, update):
        replaced = mb.slice_update(
            x=buffer, update=update, begin=[0, 0], end=[2, 3], name="whole_update"
        )
        # Both uses must be redirected, including the duplicate operand use.
        return mb.add(x=replaced, y=replaced, name="fanout")

    program.validate(check_essential_scope=True)
    _apply(program)

    assert all(op.op_type != "slice_update" for op in _all_ops(program.functions["main"]))
    output_op = program.functions["main"].outputs[0].op
    assert output_op.op_type == "add"
    assert output_op.x is program.functions["main"].inputs["update"]
    assert output_op.y is program.functions["main"].inputs["update"]


def test_nested_cond_rewrites_local_slice_update_and_preserves_capture():
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(2, 3)),
            mb.TensorSpec(shape=(2, 3)),
            mb.TensorSpec(shape=(1,), dtype=types.bool),
        ],
        opset_version=ct.target.iOS18,
    )
    def program(buffer, update, pred):
        def true_fn():
            local = mb.slice_update(
                x=buffer, update=update, begin=[0, 0], end=[2, 3], name="nested_update"
            )
            return mb.mul(x=local, y=np.float32(2.0), name="nested_consumer")

        def false_fn():
            return mb.identity(x=buffer, name="untouched_capture")

        return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

    program.validate(check_essential_scope=True)
    _apply(program)

    cond = program.functions["main"].outputs[0].op
    assert cond.op_type == "cond"
    true_block = cond.blocks[0]
    assert all(op.op_type != "slice_update" for op in _all_ops(true_block))
    assert true_block.outputs[0].op.op_type == "mul"
    assert true_block.outputs[0].op.x is program.functions["main"].inputs["update"]


def test_function_input_and_output_names_survive_bridge_identity():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def program(update):
        buffer = np.zeros((2, 3), dtype=np.float32)
        return mb.slice_update(
            x=buffer, update=update, begin=[0, 0], end=[2, 3], name="public_result"
        )

    _apply(program)

    function = program.functions["main"]
    assert list(function.inputs) == ["update"]
    assert [output.name for output in function.outputs] == ["public_result"]
    assert function.outputs[0].op.op_type == "identity"
    assert function.outputs[0].op.x is function.inputs["update"]


def test_existing_output_alias_gets_a_bridge_without_renaming_first_output():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def program(x):
        update_value = mb.add(x=x, y=np.float32(1.0), name="original_value")
        written = mb.slice_update(
            x=np.zeros((2, 3), dtype=np.float32),
            update=update_value,
            begin=[0, 0],
            end=[2, 3],
            name="written_value",
        )
        return update_value, written

    _apply(program)

    function = program.functions["main"]
    assert [output.name for output in function.outputs] == ["original_value", "written_value"]
    assert function.outputs[0].op.op_type == "add"
    assert function.outputs[1].op.op_type == "identity"
    assert function.outputs[1].op.x is function.outputs[0]


def test_failed_try_replace_keeps_slice_update_and_graph_usable():
    """A non-replaceable constexpr producer must leave the graph untouched."""
    @mb.program(input_specs=[], opset_version=ct.target.iOS18)
    def program():
        # Both operands are constexpr descendants, but they have distinct
        # non-replaceable upstream vars.  Replacing the slice result with only
        # ``update`` would drop the buffer dependency, so MIL must reject it.
        buffer = mb.constexpr_cast(
            source_val=np.zeros((2, 3), dtype=np.float16), output_dtype="fp32", name="buffer_weight"
        )
        update = mb.constexpr_cast(
            source_val=np.ones((2, 3), dtype=np.float16), output_dtype="fp32", name="cast_weight"
        )
        return mb.slice_update(
            x=buffer, update=update, begin=[0, 0], end=[2, 3], name="protected_update"
        )

    before = [op.op_type for op in program.functions["main"].operations]
    _apply(program)
    after = [op.op_type for op in program.functions["main"].operations]

    assert after == before
    assert program.functions["main"].outputs[0].op.op_type == "slice_update"


def test_malformed_constant_vector_is_rejected_without_mutation():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def program(update):
        buffer = np.zeros((2, 3), dtype=np.float32)
        return mb.slice_update(
            x=buffer, update=update, begin=[0, 0], end=[2, 3], name="malformed_update"
        )

    slice_update = program.functions["main"].operations[-1]
    # This simulates a malformed serialized MIL vector.  The matcher must
    # fail closed rather than indexing or guessing a missing axis.
    slice_update.begin._sym_val = np.asarray([0], dtype=np.int32)
    _apply(program)

    assert program.functions["main"].outputs[0].op is slice_update
    assert slice_update.op_type == "slice_update"


def test_non_integral_constant_vector_is_rejected_without_mutation():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def program(update):
        buffer = np.zeros((2, 3), dtype=np.float32)
        return mb.slice_update(
            x=buffer, update=update, begin=[0, 0], end=[2, 3], name="float_vector_update"
        )

    slice_update = program.functions["main"].operations[-1]
    slice_update.end._sym_val = np.asarray([2.5, 3.0], dtype=np.float32)
    _apply(program)

    assert program.functions["main"].outputs[0].op is slice_update
    assert slice_update.op_type == "slice_update"
