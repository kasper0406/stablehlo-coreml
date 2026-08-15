"""MIL pass: replace JAX's decomposed softmax with the native ``softmax`` op.

``jax.nn.softmax`` has no single HLO instruction; it is lowered to the
numerically stable decomposition, which this converter turns into::

    reduce_max(x, axes=[a]) -> [maximum(., -inf)] -> reshape -> [tile]
      -> sub(x, .) -> exp
      -> reduce_sum(exp, axes=[a]) -> reshape -> [tile]
      -> real_div(exp, .)

For fp16 inputs JAX accumulates the sum in fp32, which adds a ``cast`` before
``reduce_sum`` and another one after it. Depending on the surrounding layout,
JAX may also compute the whole thing in a permuted axis order, so the ``sub``
and the ``reduce_max`` do not always share the very same input tensor.

Rather than pinning down every variant of the "subtract the maximum" part, the
pass matches the part that actually defines softmax::

    real_div(exp(z), broadcast(reduce_sum(exp(z), axis)))  ==  softmax(z, axis)

and then, as a bonus, peels a leading ``sub(x, c)`` off ``z`` whenever ``c`` is
constant along ``axis`` and is not a statically known non-finite value --
softmax is invariant under such a shift, so the whole ``reduce_max``/
``maximum``/``reshape`` chain becomes dead. That covers the ``maximum(-inf,
.)`` clamps and the permuted layouts for free.
"""

import logging

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import (
    const_int_list,
    dims_equal,
    normalize_axis,
    producer_ops,
    remove_dead_ops,
    shapes_equal,
    sole_consumer,
)

logger = logging.getLogger(__name__)

# Shape/dtype-only ops that broadcast the reduction result back over the input.
_BROADCAST_BACK_OPS = frozenset({"reshape", "expand_dims", "squeeze", "tile", "cast", "identity"})


def _match(real_div_op):
    """Match a decomposed softmax ending at ``real_div_op``.

    Returns ``(softmax_input, axis, matched_ops)`` or ``None``.
    """
    if real_div_op.op_type != "real_div":
        return None

    numerator = real_div_op.inputs["x"]
    denominator = real_div_op.inputs["y"]

    exp_op = getattr(numerator, "op", None)
    if exp_op is None or exp_op.op_type != "exp":
        return None
    if numerator.shape is None:
        return None
    full_shape = tuple(numerator.shape)
    rank = len(full_shape)
    if rank == 0:
        return None

    matched = [real_div_op, exp_op]

    # -- denominator: [tile] <- [reshape] <- [cast] <- reduce_sum(exp) -------
    chain = []
    var = denominator
    consumer = real_div_op
    while True:
        op = getattr(var, "op", None)
        if op is None or op.op_type not in _BROADCAST_BACK_OPS:
            break
        if sole_consumer(var) is not consumer:
            return None
        chain.append(op)
        consumer = op
        var = op.inputs["x"]

    reduce_sum_op = getattr(var, "op", None)
    if reduce_sum_op is None or reduce_sum_op.op_type != "reduce_sum":
        return None
    if sole_consumer(var) is not consumer:
        return None

    axes = const_int_list(reduce_sum_op.inputs.get("axes"))
    if axes is None or len(axes) != 1:
        return None
    axis = normalize_axis(axes[0], rank)
    if axis is None:
        return None

    # `reduce_sum` must consume the exp output, possibly through an fp32 cast.
    sum_input = reduce_sum_op.inputs["x"]
    matched += chain
    matched.append(reduce_sum_op)
    if sum_input is not numerator:
        cast_op = getattr(sum_input, "op", None)
        if cast_op is None or cast_op.op_type != "cast":
            return None
        if cast_op.inputs["x"] is not numerator:
            return None
        if sole_consumer(sum_input) is not reduce_sum_op:
            return None
        matched.append(cast_op)

    # The intermediate values must be a plain "broadcast the sum back" chain.
    keep_dims_shape = full_shape[:axis] + (1,) + full_shape[axis + 1:]
    reduced_shape = full_shape[:axis] + full_shape[axis + 1:]
    allowed = (full_shape, keep_dims_shape, reduced_shape)
    for op in chain + [reduce_sum_op]:
        if not any(shapes_equal(op.outputs[0].shape, shape) for shape in allowed):
            return None

    # The exp output must not be observed anywhere else.
    block = real_div_op.enclosing_block
    matched_set = set(matched)
    if numerator in block.outputs:
        return None
    for child in numerator.child_ops:
        if child not in matched_set:
            return None

    # -- optionally peel `sub(x, c)`; softmax is invariant under such a shift --
    softmax_input = exp_op.inputs["x"]
    # `producer_ops` also covers the (now dead) reduce_max chain; ops that are
    # still live are simply not removed.
    candidates = matched + producer_ops(softmax_input)

    sub_op = getattr(softmax_input, "op", None)
    if (
        sub_op is not None
        and sub_op.op_type == "sub"
        and sole_consumer(softmax_input) is exp_op
        and shapes_equal(sub_op.inputs["x"].shape, full_shape)
        and _is_constant_along(sub_op.inputs["y"], axis, rank)
        and not _is_statically_nonfinite(sub_op.inputs["y"])
    ):
        softmax_input = sub_op.inputs["x"]

    return softmax_input, axis, candidates


def _is_constant_along(var, axis: int, rank: int) -> bool:
    """True if ``var`` is constant along ``axis``.

    A ``tile`` that only replicates size-1 dimensions is a broadcast, so it is
    peeled before looking at the shape.
    """
    while True:
        op = getattr(var, "op", None)
        if op is None or op.op_type != "tile":
            break
        reps = const_int_list(op.inputs.get("reps"))
        x_shape = op.inputs["x"].shape
        if reps is None or x_shape is None or len(reps) != len(x_shape):
            break
        if any(rep != 1 and not dims_equal(dim, 1) for dim, rep in zip(x_shape, reps)):
            break
        var = op.inputs["x"]

    if var.shape is None:
        return False
    shape = tuple(var.shape)
    if len(shape) > rank:
        return False
    padded = (1,) * (rank - len(shape)) + shape
    return dims_equal(padded[axis], 1)


def _is_statically_nonfinite(var) -> bool:
    """True if a compile-time shift contains NaN or infinity."""
    val = getattr(var, "val", None)
    if val is None:
        op = getattr(var, "op", None)
        if op is not None and op.op_type == "fill":
            return _is_statically_nonfinite(op.inputs.get("value"))
        return False

    arr = np.asarray(val)
    return np.issubdtype(arr.dtype, np.floating) and not bool(np.all(np.isfinite(arr)))


@block_context_manager
def _replace_decomposed_softmax(block) -> int:
    replaced = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            replaced += _replace_decomposed_softmax(nested_block)
        if len(op.blocks) > 0:
            continue

        match = _match(op)
        if match is None:
            continue
        softmax_input, axis, candidates = match

        softmax_var = mb.softmax(x=softmax_input, axis=axis, before_op=op, name=op.name + "_softmax")
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=softmax_var,
        )
        remove_dead_ops(block, candidates)
        replaced += 1

    return replaced


@register_pass(namespace="common")
class replace_decomposed_softmax(AbstractGraphPass):
    """
    Replace the decomposed softmax that JAX emits with MIL's ``softmax`` op.

    Given:
        %max = reduce_max(x=%x, axes=[a], keep_dims=False)
        %clamped = maximum(x=-inf, y=%max)          # optional
        %kd = reshape(x=%clamped, shape=<keep-dims>)
        %sub = sub(x=%x, y=%kd)
        %exp = exp(x=%sub)
        %sum = reduce_sum(x=%exp, axes=[a], keep_dims=False)
        %sum_kd = reshape(x=%sum, shape=<keep-dims>)
        %out = real_div(x=%exp, y=%sum_kd)

    Result:
        %out = softmax(x=%x, axis=a)

    ``tile`` ops broadcasting the sum back to the input shape and the ``cast``
    pair that JAX adds around ``reduce_sum`` for fp16 inputs are matched as
    well. The subtraction is only peeled off when the subtrahend is constant
    along the softmax axis (which is what makes softmax invariant under it)
    and is not statically known to contain NaN or infinity; otherwise the
    ``sub`` output becomes the softmax input.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            replaced = _replace_decomposed_softmax(f)
            if replaced:
                logger.debug("replace_decomposed_softmax: replaced %d softmax chain(s)", replaced)
