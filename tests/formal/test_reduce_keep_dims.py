import copy

import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

# Importing the package registers the production pass in PASS_REGISTRY.
import stablehlo_coreml  # noqa: F401
from tests.formal.proof import (
    InequivalentMILPrograms,
    UnsupportedMILGraph,
    prove_mil_programs_equivalent,
    prove_singleton_insertion_preserves_flat_index,
)

# Kept independent from the production pass' matcher list: adding an operation
# there does not silently add a proof here, or vice versa.
REDUCE_OPS = (
    "reduce_l1_norm",
    "reduce_l2_norm",
    "reduce_log_sum",
    "reduce_log_sum_exp",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    "reduce_sum_square",
)


def _apply_pass(prog):
    before = copy.deepcopy(prog)
    before.validate(check_essential_scope=True)
    PASS_REGISTRY["common::fuse_reduce_keep_dims"](prog)
    prog.validate(check_essential_scope=True)
    return before, prog


def _restored(reduced, restore):
    if restore == "reshape":
        return mb.reshape(x=reduced, shape=[2, 1], name="output")
    return mb.expand_dims(x=reduced, axes=[1], name="output")


def test_singleton_insertion_flat_index_lemma_is_universal():
    prove_singleton_insertion_preserves_flat_index()


@pytest.mark.parametrize("reduce_op", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [types.fp16, types.fp32])
@pytest.mark.parametrize("restore", ["reshape", "expand_dims"])
def test_production_fusion_preserves_ordered_reduction_slice_bits(reduce_op, dtype, restore):
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=dtype)])
    def prog(x):
        reduced = getattr(mb, reduce_op)(x=x, axes=[1], keep_dims=False)
        return _restored(reduced, restore)

    before, after = _apply_pass(prog)
    assert before.functions["main"].outputs[0].op.op_type == restore
    fused = after.functions["main"].outputs[0].op
    assert fused.op_type == reduce_op
    assert bool(fused.keep_dims.val)
    prove_mil_programs_equivalent(before, after)


def test_negative_axis_expand_dims_fusion_is_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 4))])
    def prog(x):
        reduced = mb.reduce_mean(x=x, axes=[-1], keep_dims=False)
        return mb.expand_dims(x=reduced, axes=[-1], name="output")

    before, after = _apply_pass(prog)
    assert tuple(after.functions["main"].outputs[0].op.axes.val) == (2,)
    prove_mil_programs_equivalent(before, after)


def test_multiple_axes_reshape_fusion_is_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 4))])
    def prog(x):
        reduced = mb.reduce_sum(x=x, axes=[0, 2], keep_dims=False)
        return mb.reshape(x=reduced, shape=[1, 3, 1], name="output")

    before, after = _apply_pass(prog)
    prove_mil_programs_equivalent(before, after)


def test_all_axes_expand_from_scalar_fusion_is_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        reduced = mb.reduce_max(x=x, axes=[0, 1], keep_dims=False)
        return mb.expand_dims(x=reduced, axes=[0, 1], name="output")

    before, after = _apply_pass(prog)
    assert before.functions["main"].operations[-1].x.shape == ()
    prove_mil_programs_equivalent(before, after)


def test_absent_axes_reshape_from_scalar_fusion_is_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        reduced = mb.reduce_min(x=x, keep_dims=False)
        return mb.reshape(x=reduced, shape=[1, 1], name="output")

    before, after = _apply_pass(prog)
    assert before.functions["main"].operations[-1].x.shape == ()
    prove_mil_programs_equivalent(before, after)


def test_extra_consumer_keeps_squeezed_output_bit_exact():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        reduced = mb.reduce_sum_square(x=x, axes=[1], keep_dims=False)
        restored = mb.reshape(x=reduced, shape=[2, 1], name="restored")
        squeezed = mb.identity(x=reduced, name="squeezed")
        return restored, squeezed

    before, after = _apply_pass(prog)
    output_ops = [output.op.op_type for output in after.functions["main"].outputs]
    assert output_ops == ["reduce_sum_square", "identity"]
    prove_mil_programs_equivalent(before, after)


def test_production_keeps_singleton_inserted_at_wrong_axis():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        reduced = mb.reduce_sum(x=x, axes=[1], keep_dims=False)
        return mb.reshape(x=reduced, shape=[1, 2], name="output")

    before, after = _apply_pass(prog)
    assert after.functions["main"].outputs[0].op.op_type == "reshape"
    prove_mil_programs_equivalent(before, after)


def test_checker_finds_wrong_axis_with_same_output_shape_and_dtype():
    specs = [mb.TensorSpec(shape=(2, 2))]

    @mb.program(input_specs=specs)
    def before(x):
        reduced = mb.reduce_sum(x=x, axes=[0], keep_dims=False)
        return mb.reshape(x=reduced, shape=[1, 2], name="output")

    @mb.program(input_specs=specs)
    def wrong_after(x):
        reduced = mb.reduce_sum(x=x, axes=[1], keep_dims=False)
        return mb.reshape(x=reduced, shape=[1, 2], name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_finds_wrong_reducer_with_same_output_shape_and_dtype():
    specs = [mb.TensorSpec(shape=(2, 3))]

    @mb.program(input_specs=specs)
    def before(x):
        return mb.reduce_sum(x=x, axes=[1], keep_dims=True, name="output")

    @mb.program(input_specs=specs)
    def wrong_after(x):
        return mb.reduce_mean(x=x, axes=[1], keep_dims=True, name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_finds_wrong_reduction_input_with_same_shape_and_dtype():
    specs = [mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(2, 3))]

    @mb.program(input_specs=specs)
    def before(x, y):
        return mb.reduce_prod(x=x, axes=[1], keep_dims=True, name="output")

    @mb.program(input_specs=specs)
    def wrong_after(x, y):
        return mb.reduce_prod(x=y, axes=[1], keep_dims=True, name="output")

    with pytest.raises(InequivalentMILPrograms, match="counterexample"):
        prove_mil_programs_equivalent(before, wrong_after)


def test_checker_fails_closed_when_reshape_operand_disagrees_with_annotation():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        reduced = mb.reduce_sum(x=x, axes=[1], keep_dims=False)
        return mb.reshape(x=reduced, shape=[2, 1], name="output")

    reshape = prog.functions["main"].outputs[0].op
    reshape.shape.val[:] = [1, 2]
    with pytest.raises(UnsupportedMILGraph, match="shape operand disagrees"):
        prove_mil_programs_equivalent(prog, prog)


def test_checker_fails_closed_for_unmodeled_arg_reduction():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
    def prog(x):
        return mb.reduce_argmax(x=x, axis=1, keep_dims=True, name="output")

    with pytest.raises(UnsupportedMILGraph, match="unsupported operation 'reduce_argmax'"):
        prove_mil_programs_equivalent(prog, prog)
