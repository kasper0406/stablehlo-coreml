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

from .pattern_utils import const_int_list, dims_equal, shapes_equal


def _mask_values(var, rank) -> list[bool] | None:
    """A ``slice_update`` boolean mask as a list of ``rank`` values.

    An absent mask reads as all-``False``, which is what the op defaults to.
    ``None`` when the mask is present but not a compile-time constant of the
    expected length, so that callers give up on the match.
    """
    if var is None:
        return [False] * rank
    val = getattr(var, "val", None)
    if val is None:
        return None
    values = [bool(v) for v in np.asarray(val).reshape(-1)]
    return values if len(values) == rank else None


def _match_pattern(op):
    if op.op_type != "slice_update":
        return False

    if op.x.shape is None:
        return False

    x_shape = tuple(op.x.shape)
    x_rank = len(x_shape)

    # `begin_mask[i]` neglects `begin[i]`, and `squeeze_mask[i]` turns the axis
    # into a pure index that drops out of the result. Neither is the plain
    # full-tensor write this pass rewrites. A squeeze also lowers the update's
    # rank, so the shape check below rejects it anyway -- the explicit guard
    # keeps that from being an accident.
    begin_mask = _mask_values(op.begin_mask, x_rank)
    squeeze_mask = _mask_values(op.squeeze_mask, x_rank)
    if begin_mask is None or squeeze_mask is None or any(begin_mask) or any(squeeze_mask):
        return False

    # Symbolic dimensions compare structurally: the update has to be shaped by
    # the very same symbols as the buffer it overwrites.
    if not shapes_equal(x_shape, op.update.shape):
        return False

    if const_int_list(op.begin) != [0] * x_rank:
        return False

    if op.stride is not None and const_int_list(op.stride) != [1] * x_rank:
        return False

    end_mask = _mask_values(op.end_mask, x_rank)
    end = const_int_list(op.end)
    if end_mask is None or end is None or len(end) != x_rank:
        return False

    # `end_mask[i] == True` *means* "up to the end of axis i", so it covers the
    # axis whatever its size. Without it the axis has to be concrete for
    # `end[i]` to provably cover it: `end` holds constant ints, and a symbolic
    # dimension is never provably equal to one.
    return all(
        covers_axis or dims_equal(dim, stop)
        for covers_axis, dim, stop in zip(end_mask, x_shape, end)
    )


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
            did_optimize |= _remove_noop_slice_update(b)
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

    A symbolic dimension of S is only ever covered through `end_mask`, since a
    constant `end` is never provably equal to a symbol.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _remove_noop_slice_update(f)
