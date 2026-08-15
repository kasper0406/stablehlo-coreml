"""MIL pass: remove ``tile`` ops that only implement NumPy broadcasting.

StableHLO requires all operands of an elementwise op to have the exact same
shape, so ``broadcast_in_dim`` is lowered to ``reshape (-> reshape) -> tile``.
MIL's elementwise ops broadcast natively, so those tiles are pure overhead --
and if a tile of a constant survives until ``const_elimination`` it is folded
into a full-size constant tensor (this is how exports end up with gigabytes of
constant weights). The pass therefore runs before the first
``const_elimination`` in the pipeline.
"""

import logging

import numpy as np
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import broadcast_shapes, dims_equal, shapes_equal

logger = logging.getLogger(__name__)

# Ops that support implicit NumPy-style broadcasting of their operands.
#
# `select` is deliberately EXCLUDED: E5RT (Apple's CoreML runtime) cannot
# propagate shapes through a `select` with implicit broadcasting when the model
# is loaded as a multifunction .mlpackage. It fails with
#   "Failed to PropagateInputTensorShapes: Validation error during type
#    inference for select: Incompatible Dimension"
# so tiles feeding a `select` must be preserved.
_BROADCAST_OPS = frozenset({
    "add", "sub", "mul", "real_div",
    "maximum", "minimum",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_xor",
    "pow", "floor_div", "mod",
})

# The operand names of the (binary) ops above.
_BINARY_OPERANDS = ("x", "y")


def _is_broadcast_tile(op) -> bool:
    """True if ``op`` is a ``tile`` that only replicates size-1 dimensions.

    A tile of a dimension that is not 1 is *not* a broadcast:
    ``tile([1, 2], reps=[2]) == [1, 2, 1, 2]`` cannot be expressed by implicit
    broadcasting. ``reps`` must be known at compile time.
    """
    if op.op_type != "tile":
        return False

    reps_var = op.inputs.get("reps")
    if reps_var is None or reps_var.val is None:
        return False
    reps = np.asarray(reps_var.val).reshape(-1).tolist()

    x_shape = op.x.shape
    if x_shape is None or len(reps) != len(x_shape):
        return False

    for dim, rep in zip(x_shape, reps):
        if int(rep) == 1:
            continue
        # `dim` may be symbolic; only a literal 1 is safe to broadcast.
        if dims_equal(dim, 1):
            continue
        return False
    return True


def _consumer_output_is_unchanged(consumer, tile_out, tile_in) -> bool:
    """True if replacing ``tile_out`` by ``tile_in`` keeps ``consumer``'s output shape."""
    operand_shapes = []
    for name in _BINARY_OPERANDS:
        operand = consumer.inputs.get(name)
        if operand is None:
            return False
        shape = tile_in.shape if operand is tile_out else operand.shape
        if shape is None:
            return False
        operand_shapes.append(tuple(shape))

    broadcast = broadcast_shapes(*operand_shapes)
    if broadcast is None:
        return False
    return shapes_equal(broadcast, consumer.outputs[0].shape)


def _can_remove(op, block) -> bool:
    if not _is_broadcast_tile(op):
        return False

    tile_out = op.outputs[0]
    if tile_out in block.outputs:
        return False

    consumers = list(tile_out.child_ops)
    if len(consumers) == 0:
        return False

    for consumer in consumers:
        if consumer.op_type not in _BROADCAST_OPS:
            return False
        # The tile output being consumed from inside a nested block (while_loop /
        # cond body) is not safe to rewrite from here.
        if consumer.enclosing_block is not block:
            return False
        if not _consumer_output_is_unchanged(consumer, tile_out, op.x):
            return False
    return True


@block_context_manager
def _remove_broadcast_tiles(block) -> int:
    removed = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            removed += _remove_broadcast_tiles(nested_block)
        if len(op.blocks) > 0:
            continue

        if not _can_remove(op, block):
            continue

        # The replacement changes the shape of the operand, so type inference on
        # the consumers must be skipped (`no_check_var_types`). That is safe: we
        # verified above that every consumer keeps its current output shape.
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=op.x,
            no_check_var_types=True,
        )
        op.remove_from_block()
        removed += 1

    return removed


@register_pass(namespace="common")
class remove_broadcast_tiles(AbstractGraphPass):
    """
    Remove ``tile`` ops that only implement NumPy broadcasting for consumers
    that broadcast natively.

    A tile is removed when all of the following hold:

    1. It is a broadcast: ``reps`` is a compile-time constant and for every
       axis either ``x.shape[i] == 1`` or ``reps[i] == 1``.
    2. Every consumer is an elementwise op with implicit broadcasting support
       (``select`` is excluded on purpose, see the module docstring), and lives
       in the same block as the tile.
    3. No consumer changes its output shape when the tile is bypassed.

    Given:
        %2 = tile(x=%1, reps=[1, 8])   # %1: (4, 1)
        %3 = add(x=%0, y=%2)           # %0: (4, 8)

    Result:
        %3 = add(x=%0, y=%1)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            removed = _remove_broadcast_tiles(f)
            if removed:
                logger.debug("remove_broadcast_tiles: removed %d tile op(s)", removed)
