"""MIL pass: keep the rank-4 layout across ``conv``/``max_pool`` chains.

``conv`` and the pooling ops are the only ops in MIL that insist on a rank-4
(NCHW) operand. A JAX model that convolves a plain ``(H, W)`` image therefore
comes out of the converter with every convolution wrapped in a pair of
reshapes::

    (H, W) -> reshape -> (1, 1, H, W) -> conv -> (1, 1, H', W') -> reshape -> (H', W')

Those reshapes are *rank-only*: they add and drop size-1 dimensions and leave
the element order alone. Core ML still executes them as full copies -- tens of
microseconds each once the tensor reaches a few megabytes -- and an image
pyramid ends up with hundreds of them.

Two rewrites take them out of the chains. Both decide on shapes alone, and
neither fires unless it removes at least one reshape and copies no more elements
than it saves, so neither can make a graph worse:

**Share the lift.** A convolution's operand is usually a *slice* of a larger
buffer, and the same buffer is sliced several times over (the four shifted 2x2
windows of a stride-2 pyramid level, say). Each slice gets its own reshape.
Slicing commutes with a rank lift, so the lift is hoisted above the slices and
shared: *n* reshapes of the slices become one reshape of their source, and the
slices run at the lifted rank.

**Close the round trip.** Where a chain leaves rank 4 only to re-enter it a few
shape-preserving ops later (``conv -> reshape -> reverse -> reshape -> conv``),
those ops are re-emitted at the outer rank -- ``reverse`` only needs its axes
remapped -- and the two reshapes collapse into one, or into none when they are
exact inverses.

Anything that cannot be proven is skipped: symbolic dimensions, non-constant
slice bounds, and intermediates with a second consumer.
"""

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .pattern_utils import RewritePass, const_int_list, normalize_axis, sole_consumer

# The slice ops a rank lift can be hoisted above.
_SLICE_OPS = frozenset({"slice_by_index", "slice_by_size"})

# Shape-preserving ops that can be re-emitted at any rank with the same non-1
# dimensions. Elementwise unary ops qualify by definition, and their remaining
# operands (``cast``'s dtype, ``clip``'s bounds, ...) are rank-independent
# scalars. ``reverse`` qualifies once its ``axes`` are remapped, which
# :func:`_rebuild` does. Binary ops are deliberately left out: their second
# operand would need a rank change of its own, which is a new op unless it
# happens to broadcast.
_RANK_AGNOSTIC_OPS = frozenset({
    "abs", "acos", "asin", "atan", "atanh", "cast", "ceil", "clip", "cos", "cosh",
    "elu", "erf", "exp", "exp2", "floor", "identity", "inverse", "leaky_relu",
    "log", "logical_not", "relu", "relu6", "reverse", "round", "rsqrt", "sigmoid",
    "sign", "sin", "sinh", "softplus", "sqrt", "square", "tan", "tanh", "threshold",
})

# Upper bound on the number of ops the backward walks step over. The converter
# emits at most a handful; the bound just keeps the search finite.
_MAX_CHAIN = 8


def static_shape(var) -> tuple[int, ...] | None:
    """``var``'s shape as a tuple of ints, or ``None`` if any dimension is symbolic."""
    if var is None:
        return None
    shape = var.shape
    if shape is None:
        return None
    if any(is_symbolic(dim) for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def core_dims(shape) -> tuple[int, ...]:
    """``shape`` without its size-1 dimensions.

    Two shapes with the same core hold the same elements in the same order, so a
    reshape between them is rank-only: a relabelling, not a permutation.
    """
    return tuple(dim for dim in shape if dim != 1)


def num_elements(shape) -> int:
    """How many elements a tensor of ``shape`` holds."""
    return int(np.prod(shape, dtype=np.int64)) if len(shape) > 0 else 1


def rank_only_reshape(op) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """``(in_shape, out_shape)`` if ``op`` is a reshape that only adds/drops 1s."""
    if op.op_type != "reshape":
        return None
    in_shape = static_shape(op.inputs.get("x"))
    out_shape = static_shape(op.outputs[0])
    if in_shape is None or out_shape is None:
        return None
    if core_dims(in_shape) != core_dims(out_shape):
        return None
    return in_shape, out_shape


def map_axes(axes, from_shape, to_shape) -> list[int] | None:
    """Rewrite ``axes`` of ``from_shape`` for the equivalent ``to_shape`` layout.

    The two shapes share a core (see :func:`core_dims`), so the *i*-th non-1
    dimension of one is the *i*-th non-1 dimension of the other. Axes addressing
    a size-1 dimension are dropped: there is nothing to reverse along them, and
    the two layouts need not agree on where their 1s sit. ``None`` when an axis
    is out of range.
    """
    from_core = [i for i, dim in enumerate(from_shape) if dim != 1]
    to_core = [i for i, dim in enumerate(to_shape) if dim != 1]
    mapped = set()
    for axis in axes:
        axis = normalize_axis(axis, len(from_shape))
        if axis is None:
            return None
        if from_shape[axis] == 1:
            continue
        mapped.add(to_core[from_core.index(axis)])
    return sorted(mapped)


def _const_flags(var, length) -> list[bool] | None:
    """A boolean mask operand as a list of ``length`` bools (absent means all-False)."""
    if var is None:
        return [False] * length
    val = getattr(var, "val", None)
    if val is None:
        return None
    flags = np.asarray(val).reshape(-1)
    if flags.size != length:
        return None
    return [bool(flag) for flag in flags]


def _slice_params(op) -> dict | None:
    """The compile-time constant operands of a slice op, or ``None``.

    ``None`` means the slice cannot be re-expressed at a higher rank: some
    operand is not a constant, or a ``squeeze_mask`` bit drops an axis -- which
    would make the output rank differ from ``begin``'s length, and with it the
    correspondence between the two layouts.
    """
    x_shape = static_shape(op.inputs.get("x"))
    out_shape = static_shape(op.outputs[0])
    if x_shape is None or out_shape is None:
        return None
    rank = len(x_shape)

    def indices(name):
        values = const_int_list(op.inputs.get(name))
        return values if values is not None and len(values) == rank else None

    begin = indices("begin")
    if begin is None:
        return None

    if op.op_type == "slice_by_size":
        size = indices("size")
        if size is None:
            return None
        return {"begin": begin, "size": size, "out_shape": out_shape}

    end = indices("end")
    stride = [1] * rank if op.inputs.get("stride") is None else indices("stride")
    if end is None or stride is None:
        return None
    masks = {
        name: _const_flags(op.inputs.get(name), rank)
        for name in ("begin_mask", "end_mask", "squeeze_mask")
    }
    if any(mask is None for mask in masks.values()) or any(masks["squeeze_mask"]):
        return None
    return {"begin": begin, "end": end, "stride": stride, "out_shape": out_shape, **masks}


def _lifted_slice(op, params, lift, source, before_op):
    """Re-emit slice ``op`` against ``source``, which carries ``lift`` extra leading 1s.

    Each prepended axis has extent 1 in ``source``, so ``[0:1:1]`` on it keeps
    the whole (single) index while the original axes keep their bounds.

    ``stride`` and the masks are only passed on when they differ from their
    defaults. Spelling out an all-``False`` mask costs a ``const`` op that
    ``const_deduplication`` does not fold away, which would trade the reshapes
    this pass removes for as many constants.
    """
    ones, zeros, falses = [1] * lift, [0] * lift, [False] * lift
    if op.op_type == "slice_by_size":
        return mb.slice_by_size(
            x=source,
            begin=zeros + params["begin"],
            size=ones + params["size"],
            before_op=before_op,
        )

    optional = {}
    if any(stride != 1 for stride in params["stride"]):
        optional["stride"] = ones + params["stride"]
    for name in ("begin_mask", "end_mask"):
        if any(params[name]):
            optional[name] = falses + params[name]
    return mb.slice_by_index(
        x=source,
        begin=zeros + params["begin"],
        end=ones + params["end"],
        before_op=before_op,
        **optional,
    )


def _leading_lift(op) -> int | None:
    """How many size-1 dimensions ``op`` prepends, if that is all it does."""
    shapes = rank_only_reshape(op)
    if shapes is None:
        return None
    in_shape, out_shape = shapes
    lift = len(out_shape) - len(in_shape)
    if lift <= 0 or out_shape != (1,) * lift + in_shape:
        return None
    return lift


def _first_in_block(block, ops):
    """The op of ``ops`` that comes first in ``block``'s operation order."""
    wanted = {id(op) for op in ops}
    for candidate in block.operations:
        if id(candidate) in wanted:
            return candidate
    return None


def _lift_group(reshape_op, block):
    """All ``slice -> reshape(lift)`` pairs that share ``reshape_op``'s source.

    Returns ``(source, lift, members)``, the members being
    ``(slice_op, params, reshape_op)`` triples, or ``None`` when the pattern does
    not apply. Every sibling is checked on its own; one that cannot be lifted
    simply does not join the group.
    """
    lift = _leading_lift(reshape_op)
    if lift is None:
        return None
    slice_op = reshape_op.inputs["x"].op
    if slice_op is None or slice_op.op_type not in _SLICE_OPS:
        return None
    source = slice_op.inputs["x"]
    source_shape = static_shape(source)
    if source_shape is None:
        return None
    # Reshaping a constant would materialise a second copy of it as a weight,
    # and the slices of a constant are folded away by `const_elimination` anyway.
    if getattr(source, "val", None) is not None:
        return None

    members = []
    for sibling in source.child_ops:
        if sibling.op_type not in _SLICE_OPS or sibling.enclosing_block is not block:
            continue
        # The slice must feed nothing but its lift; otherwise the rank-2
        # spelling has to be computed anyway and hoisting buys nothing.
        consumer = sole_consumer(sibling.outputs[0])
        if consumer is None or consumer.enclosing_block is not block:
            continue
        if _leading_lift(consumer) != lift or consumer.outputs[0] in block.outputs:
            continue
        params = _slice_params(sibling)
        if params is None:
            continue
        members.append((sibling, params, consumer))

    # `reshape_op` not being a member means its own slice has another consumer.
    # Leave the group to be found from a member's reshape, so that a rewrite is
    # only ever reported for an op that it actually removes.
    if not any(member is reshape_op for _, _, member in members):
        return None
    if len(members) < 2:
        return None
    # One reshape of the source replaces one reshape per member: fewer ops, and
    # fewer copied elements as long as the slices together are not smaller than
    # what the source lift has to copy.
    copied = sum(num_elements(params["out_shape"]) for _, params, _ in members)
    if copied < num_elements(source_shape):
        return None
    return source, lift, members


def _peel_reshapes(var, consumer, block):
    """Walk back over a run of single-consumer rank-only reshapes above ``var``.

    ``var`` is the input of the rank-only reshape ``consumer``. Returns the
    var that feeds the outermost such reshape, so that a whole stack collapses
    in one go rather than one reshape per run of the pass.
    """
    for _ in range(_MAX_CHAIN):
        if sole_consumer(var) is not consumer:
            return var
        producer = var.op
        if producer is None or producer.enclosing_block is not block:
            return var
        if rank_only_reshape(producer) is None:
            return var
        var, consumer = producer.inputs["x"], producer
    return var


def _match_round_trip(reshape_op, block):
    """``(source, chain, layout, target)`` for a reshape re-entering a rank it just left.

    Walks back from ``reshape_op`` over shape-preserving, rank-agnostic ops until
    it reaches another rank-only reshape with the same core. ``source`` is what
    feeds that reshape, ``chain`` the ops in between (innermost first), ``layout``
    the shape they run at today and ``target`` the shape they will run at.

    Every intermediate must have a single consumer: the old chain has to become
    dead, or the rebuilt one is pure addition. ``reverse`` axes are remapped here
    as well, so that a chain which cannot be rebuilt is rejected before any op is
    emitted.
    """
    shapes = rank_only_reshape(reshape_op)
    if shapes is None or reshape_op.outputs[0] in block.outputs:
        return None
    layout, target = shapes

    chain = []
    var, consumer = reshape_op.inputs["x"], reshape_op
    for _ in range(_MAX_CHAIN):
        if sole_consumer(var) is not consumer:
            return None
        producer = var.op
        if producer is None or producer.enclosing_block is not block:
            return None
        if rank_only_reshape(producer) is not None:
            source = _peel_reshapes(producer.inputs["x"], producer, block)
            return source, chain, layout, target
        if producer.op_type not in _RANK_AGNOSTIC_OPS:
            return None
        if static_shape(producer.inputs.get("x")) != layout:
            return None
        if producer.op_type == "reverse" and _remapped_axes(producer, layout, target) is None:
            return None
        chain.append(producer)
        var, consumer = producer.inputs["x"], producer
    return None


def _remapped_axes(op, layout, target) -> list[int] | None:
    """``op``'s ``reverse`` axes expressed in the ``target`` layout, or ``None``."""
    axes = op.inputs.get("axes")
    axes = list(range(len(layout))) if axes is None else const_int_list(axes)
    if axes is None:
        return None
    return map_axes(axes, layout, target)


def _rebuild(op, x, layout, target, before_op):
    """Re-emit ``op`` on ``x``, which holds ``layout``'s elements shaped as ``target``."""
    if op.op_type != "reverse":
        return getattr(mb, op.op_type)(**{**dict(op.inputs), "x": x}, before_op=before_op)
    axes = _remapped_axes(op, layout, target)
    # Reversing nothing but size-1 axes is the identity; drop the op entirely.
    if len(axes) == 0:
        return x
    return mb.reverse(x=x, axes=axes, before_op=before_op)


@register_pass(namespace="common")
class conv_pool_rank4(RewritePass):
    """
    Keep the rank-4 layout across ``conv``/``max_pool`` chains, so that the
    rank-only reshapes surrounding them are shared or cancelled.

    Two rewrites are tried, in this order, on every rank-only reshape (one that
    only adds or drops size-1 dimensions):

    1. **Share the lift.** The reshape only prepends 1s, and its input is a slice
       whose sole consumer it is. Every sibling slice of the same source that is
       lifted the same way joins a group; the source is then lifted once and the
       slices run at the higher rank instead. The group needs at least two
       members (so it loses at least one reshape) and its slices must together
       copy no fewer elements than the source lift does.

    2. **Close the round trip.** The ops feeding the reshape are shape-preserving
       and rank-agnostic (elementwise unary ops, and ``reverse`` with remapped
       axes), and the chain starts at another rank-only reshape with the same
       core. The chain is re-emitted at the outer shape, leaving one reshape
       where there were two -- or none, when the two are exact inverses. Every
       intermediate must have a single consumer.

    Everything is decided from static shapes and constant operands; symbolic
    dimensions and computed slice bounds are skipped.

    Given:
        %1 = slice_by_index(x=%0, begin=[0, 0], end=[517, 293])   # %0: (518, 294)
        %2 = reshape(x=%1, shape=[1, 1, 517, 293])
        %3 = max_pool(x=%2, ...)
        %4 = slice_by_index(x=%0, begin=[1, 0], end=[518, 293])
        %5 = reshape(x=%4, shape=[1, 1, 517, 293])
        %6 = max_pool(x=%5, ...)

    Result:
        %l = reshape(x=%0, shape=[1, 1, 518, 294])
        %2 = slice_by_index(x=%l, begin=[0, 0, 0, 0], end=[1, 1, 517, 293])
        %3 = max_pool(x=%2, ...)
        %5 = slice_by_index(x=%l, begin=[0, 0, 1, 0], end=[1, 1, 518, 293])
        %6 = max_pool(x=%5, ...)
    """

    _REWRITES = "rank-only reshape(s)"

    def visit(self, op, block) -> bool:
        return self._share_lift(op, block) or self._close_round_trip(op, block)

    def _share_lift(self, op, block) -> bool:
        group = _lift_group(op, block)
        if group is None:
            return False
        source, lift, members = group

        lifted_source = mb.reshape(
            x=source,
            shape=[1] * lift + list(static_shape(source)),
            before_op=_first_in_block(block, [slice_op for slice_op, _, _ in members]),
        )
        for slice_op, params, reshape_op in members:
            lifted = _lifted_slice(slice_op, params, lift, lifted_source, before_op=slice_op)
            block.replace_uses_of_var_after_op(
                anchor_op=reshape_op, old_var=reshape_op.outputs[0], new_var=lifted
            )
            reshape_op.remove_from_block()
        return True

    def _close_round_trip(self, op, block) -> bool:
        match = _match_round_trip(op, block)
        if match is None:
            return False
        source, chain, layout, target = match

        current = source
        if static_shape(source) != target:
            current = mb.reshape(x=source, shape=list(target), before_op=op)
        # `chain` runs innermost-first; rebuild it in execution order.
        for chain_op in reversed(chain):
            current = _rebuild(chain_op, current, layout, target, before_op=op)
        block.replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=current
        )
        op.remove_from_block()
        return True
