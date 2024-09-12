from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

import numpy as np


def _match_pattern(op):
    if op.op_type == "slice_update":
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


def _try_to_transform(slice_update_op):
    # Replace occurences of the `slice_update_op` output with the `slice_update_op.update` variable
    slice_update_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=slice_update_op, old_var=slice_update_op.outputs[0], new_var=slice_update_op.update
    )
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
