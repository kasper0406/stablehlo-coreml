"""MIL pass: drop ``slice_update`` ops that overwrite the whole destination tensor.

The converter builds several results by allocating a buffer and writing into it
with ``slice_update`` (``op_dynamic_update_slice``, and the accumulator loops in
``reductions.py`` / ``DotGeneralOp``). When the written slice happens to cover
the entire buffer the write is just a copy, and the update tensor can be used
directly.
"""

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _match_pattern(op):
    if op.op_type == "slice_update":
        if op.x.shape is None or op.update.shape is None:
            return False

        x_rank = len(op.x.shape)

        x_and_update_shape_matches = op.x.shape == op.update.shape

        all_zeros_start_indices_array = np.array([0] * x_rank, dtype=np.int32)
        start_values_all_zero = np.array_equal(op.begin.val, all_zeros_start_indices_array)

        end_values_matches_x_shape = np.array_equal(op.end.val, op.x.shape)

        all_one_strides_array = np.array([1] * x_rank, dtype=np.int32)
        strides_all_one = not op.stride or np.array_equal(op.stride.val, all_one_strides_array)
        no_extra_options = strides_all_one and not op.begin_mask and not op.end_mask

        return x_and_update_shape_matches and start_values_all_zero and end_values_matches_x_shape and no_extra_options

    return False


def _renames_function_input(slice_update_op, new_var) -> bool:
    """True if replacing the op's output by ``new_var`` would rename a function input.

    When the replaced var is an output of the enclosing block, coremltools
    carries the old name over to the replacement so that the model keeps its
    output names (``Block.replace_block_output_var``). For a ``Function`` it
    refuses to do that to an input var and raises ``ValueError: It is not
    allowed to modify function inputs name.`` -- which aborts the whole
    conversion. That happens for e.g. ``lax.dynamic_update_slice(buffer, x, 0)``
    returned as-is, where ``update`` is a function argument.
    """
    block = slice_update_op.enclosing_block
    if not isinstance(block, Function):
        return False
    out_var = slice_update_op.outputs[0]
    if out_var not in block.outputs:
        return False
    return new_var in block.inputs.values() and new_var.name != out_var.name


def _try_to_transform(slice_update_op):
    block = slice_update_op.enclosing_block
    out_var = slice_update_op.outputs[0]

    new_var = slice_update_op.update
    if _renames_function_input(slice_update_op, new_var):
        # The function input cannot be renamed, so route the output through an
        # `identity` that can take over the name instead. The `slice_update` --
        # the op this pass is here to remove -- still goes away.
        new_var = mb.identity(x=new_var, before_op=slice_update_op)

    # Replace occurences of the `slice_update_op` output with `new_var`.
    # `try_...` rather than the unguarded variant: the update may descend from a
    # var coremltools refuses to replace (a `constexpr_*` weight, say), in which
    # case the rewrite is skipped instead of raising.
    if not block.try_replace_uses_of_var_after_op(
        anchor_op=slice_update_op, old_var=out_var, new_var=new_var
    ):
        return False
    slice_update_op.remove_from_block()
    return True


@block_context_manager
def _remove_noop_slice_update(block):
    did_optimize = False
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _remove_noop_slice_update(b)
        if len(op.blocks) > 0:
            continue

        if _match_pattern(op):
            if _try_to_transform(op):
                did_optimize = True
    return did_optimize


@register_pass(namespace="common")
class remove_noop_slice_update(AbstractGraphPass):
    """
    If a slice_update is called on the full tensor with an update of the same shape,
    simply use the update tensor going forward.

    This optimization is very useful for the way the HLO DotGeneralOp is implemented,
    in case the DotGeneralOp reduces to a single matrix multiplication.

    Given:
        %1 = <buffer tensor of shape S>
        %2 = <update tensor of shape S>
        %2 = slice_update(x=%buffer, update=%2, begin=[0] * rank(%1), end=S, stride=[1] * rank(%1))
        %3 = some_op(%2)

    Result:
        %1 = <tensor of shape S>
        %3 = some_op(%1)
        ...
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _remove_noop_slice_update(f)
