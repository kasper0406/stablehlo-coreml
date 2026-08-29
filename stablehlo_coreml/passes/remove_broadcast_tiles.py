"""MIL pass: remove ``tile`` ops that only implement NumPy broadcasting.

StableHLO requires all operands of an elementwise op to have the exact same
shape, so ``broadcast_in_dim`` is lowered to ``reshape (-> reshape) -> tile``.
MIL's elementwise ops broadcast natively, so those tiles are pure overhead --
and if a tile of a constant survives until ``const_elimination`` it is folded
into a full-size constant tensor (this is how exports end up with gigabytes of
constant weights). The pass therefore runs before the first
``const_elimination`` in the pipeline.
"""


from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import RewritePass, aligned_dim, const_int_list, dims_equal, is_broadcast_tile

# Ops that support implicit NumPy-style broadcasting of their operands.
#
# `select` is deliberately EXCLUDED: E5RT (Apple's CoreML runtime) cannot
# propagate shapes through a `select` with implicit broadcasting when the model
# is loaded as a multifunction .mlpackage. It fails with
#   "Failed to PropagateInputTensorShapes: Validation error during type
#    inference for select: Incompatible Dimension"
# so tiles feeding a `select` must be preserved. Preserving them only covers the
# tiles that exist, though -- `broadcast_select_operands` runs right after this
# pass and adds the ones JAX never emitted (`jnp.where` broadcasts implicitly).
_BROADCAST_OPS = frozenset({
    "add", "sub", "mul", "real_div",
    "maximum", "minimum",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_xor",
    "pow", "floor_div", "mod",
})

# The operand names of the (binary) ops above.
_BINARY_OPERANDS = ("x", "y")


def _consumer_output_is_unchanged(consumer, tile_op, tile_out) -> bool:
    """True if bypassing ``tile_op`` keeps ``consumer``'s output shape.

    The question is decided per axis rather than by re-broadcasting the whole
    operand shapes. Re-broadcasting cannot answer it under dynamic shapes: MIL
    mints a *fresh* symbol for a dynamic dimension at nearly every op, so the
    two operands of an elementwise op routinely carry different symbols
    (``is4`` vs ``dim_0``) for one and the same runtime value, and a symbolic
    dimension is only ever provably equal to the identical symbol.

    Per axis the reasoning needs no such comparison:

    * ``reps[axis] == 1`` leaves the axis alone -- the tile's output dimension
      *is* its input's, whatever either is called, so the consumer cannot tell
      the two apart.
    * ``reps[axis] > 1`` means the tile replicated a size-1 axis, and bypassing
      it takes the operand back down to 1 there. The consumer's output only
      stays the same if the other operand already carries the full size on that
      axis. A replicated dimension is always ``1 * reps[axis]``, i.e. a literal
      int, so this comparison never involves a symbol on the tile's side.
    """
    reps = const_int_list(tile_op.inputs.get("reps"))
    if reps is None or tile_out.shape is None:
        return False

    out_shape = consumer.outputs[0].shape
    if out_shape is None or len(out_shape) < len(reps):
        return False

    others = []
    for name in _BINARY_OPERANDS:
        operand = consumer.inputs.get(name)
        if operand is None:
            return False
        # An operand that is the tile itself is bypassed too, so it cannot be
        # the one supplying a replicated dimension.
        if operand is not tile_out:
            others.append(operand)

    offset = len(out_shape) - len(reps)
    for axis, rep in enumerate(reps):
        if rep == 1:
            continue
        replicated = tile_out.shape[axis]
        dims = [aligned_dim(other, offset + axis, len(out_shape)) for other in others]
        if not any(dim is not None and dims_equal(dim, replicated) for dim in dims):
            return False
    return True


def _can_remove(op, block) -> bool:
    if not is_broadcast_tile(op):
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
        if not _consumer_output_is_unchanged(consumer, op, tile_out):
            return False
    return True


@register_pass(namespace="common")
class remove_broadcast_tiles(RewritePass):
    """
    Remove ``tile`` ops that only implement NumPy broadcasting for consumers
    that broadcast natively.

    A tile is removed when all of the following hold:

    1. It is a broadcast: ``reps`` is a compile-time constant and for every
       axis either ``x.shape[i] == 1`` or ``reps[i] == 1``.
    2. Every consumer is an elementwise op with implicit broadcasting support
       (``select`` is excluded on purpose, see the module docstring), and lives
       in the same block as the tile.
    3. No consumer changes its output shape when the tile is bypassed: on every
       axis the tile actually replicated (``reps[i] > 1``), the consumer's other
       operand already carries the replicated size. Axes with ``reps[i] == 1``
       need no check -- the tile passes them through unchanged.

    Given:
        %2 = tile(x=%1, reps=[1, 8])   # %1: (4, 1)
        %3 = add(x=%0, y=%2)           # %0: (4, 8)

    Result:
        %3 = add(x=%0, y=%1)
    """

    _REWRITES = "tile op(s)"

    def visit(self, op, block) -> bool:
        if not _can_remove(op, block):
            return False

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
        return True
