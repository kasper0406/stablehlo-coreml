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

``jax.nn.softmax(x, where=mask)`` additionally wraps the summand in
``select(mask, ., 0)`` and the quotient in ``select(mask, ., 0)``, having
first replaced the masked lanes of ``x`` with ``-inf``. Those masked lanes
exponentiate to exactly 0, so the ``select`` inside the sum is redundant and
the whole thing is ``select(mask, softmax(select(mask, x, -inf)), 0)``.
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
    is_broadcast_tile,
    normalize_axis,
    shapes_equal,
    sole_consumer,
    uniform_scalar_value,
)

logger = logging.getLogger(__name__)

# Shape/dtype-only ops that broadcast the reduction result back over the input.
_BROADCAST_BACK_OPS = frozenset({"reshape", "expand_dims", "squeeze", "tile", "cast", "identity"})


def _match(real_div_op):
    """Match a decomposed softmax ending at ``real_div_op``.

    Returns ``(softmax_input, axis)`` or ``None``.
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

    # `reduce_sum` must consume the exp output, possibly through an fp32 cast
    # and/or the `select(mask, ., 0)` that `jax.nn.softmax(where=...)` puts
    # inside the sum.
    sum_input = reduce_sum_op.inputs["x"]
    matched += chain
    matched.append(reduce_sum_op)
    consumer = reduce_sum_op
    mask = None
    seen_cast = False
    while sum_input is not numerator:
        op = getattr(sum_input, "op", None)
        if op is None or sole_consumer(sum_input) is not consumer:
            return None
        if op.op_type == "cast" and not seen_cast:
            seen_cast = True
            next_input = op.inputs["x"]
        elif op.op_type == "select" and mask is None:
            zero = uniform_scalar_value(op.inputs["b"])
            if zero is None or zero != 0.0:
                return None
            mask = _mask_source(op.inputs["cond"])
            next_input = op.inputs["a"]
        else:
            return None
        matched.append(op)
        consumer = op
        sum_input = next_input

    # The intermediate values must be a plain "broadcast the sum back" chain.
    keep_dims_shape = full_shape[:axis] + (1,) + full_shape[axis + 1:]
    reduced_shape = full_shape[:axis] + full_shape[axis + 1:]
    allowed = (full_shape, keep_dims_shape, reduced_shape)
    for op in chain + [reduce_sum_op]:
        if not any(shapes_equal(op.outputs[0].shape, shape) for shape in allowed):
            return None

    # ... and the denominator itself has to broadcast back along `axis`. A
    # `reduced_shape` denominator only does so when `axis` is the leading one:
    # NumPy aligns shapes to the right, so a rank-1 gap anywhere else lines the
    # sum up against the wrong axes and the quotient is not a softmax.
    if not _is_constant_along(denominator, axis, rank):
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

    if mask is not None and not _is_masked_softmax(real_div_op, softmax_input, mask):
        return None

    return softmax_input, axis


def _mask_source(var):
    """Peel broadcasting ``tile``s off a mask so that equal masks compare equal.

    JAX materialises the same ``where=`` mask once per use, and the converter
    tiles each of those up to the operand shape separately, so the three masks
    of a masked softmax are distinct ``Var``s over one common source.
    """
    while True:
        op = getattr(var, "op", None)
        if op is None or not is_broadcast_tile(op):
            return var
        var = op.inputs["x"]


def _select_over(var, mask):
    """Return the ``select`` op if ``var`` is ``select(mask, ., .)``, else ``None``."""
    op = getattr(var, "op", None)
    if op is None or op.op_type != "select":
        return None
    if _mask_source(op.inputs["cond"]) is not mask:
        return None
    return op


def _is_masked_softmax(real_div_op, softmax_input, mask) -> bool:
    """True if dropping the ``select`` inside the sum leaves the result unchanged.

    ``jax.nn.softmax(x, where=mask)`` is::

        x_safe = select(mask, x, -inf)
        result = select(mask, exp(x_safe - m) / sum(select(mask, exp(...), 0)), 0)

    Summing ``exp`` directly instead of the masked copy is exact as long as the
    masked lanes of ``x_safe`` are ``-inf``: ``exp(-inf - m) == 0`` for any finite
    ``m``, so those terms contribute nothing either way. When a whole row is
    masked ``m`` is ``-inf`` too and both spellings produce NaN, but the outer
    ``select`` then replaces the entire row with zeros -- which is why that
    ``select`` has to be present for the rewrite to be sound.
    """
    fill_select = _select_over(softmax_input, mask)
    if fill_select is None:
        return False
    fill = uniform_scalar_value(fill_select.inputs["b"])
    if fill is None or not (np.isinf(fill) and fill < 0):
        return False

    result = real_div_op.outputs[0]
    out_select = sole_consumer(result)
    if out_select is None or out_select.op_type != "select":
        return False
    if _mask_source(out_select.inputs["cond"]) is not mask:
        return False
    if out_select.inputs["a"] is not result:
        return False
    zero = uniform_scalar_value(out_select.inputs["b"])
    return zero is not None and zero == 0.0


def _is_constant_along(var, axis: int, rank: int) -> bool:
    """True if ``var`` is constant along ``axis``.

    A ``tile`` that only replicates size-1 dimensions is a broadcast, so it is
    peeled before looking at the shape.
    """
    while True:
        op = getattr(var, "op", None)
        if op is None or not is_broadcast_tile(op):
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
        softmax_input, axis = match

        softmax_var = mb.softmax(x=softmax_input, axis=axis, before_op=op, name=op.name + "_softmax")
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=softmax_var,
        )
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
    well, as is the masked form ``jax.nn.softmax(x, where=mask)``:

        %safe = select(cond=%mask, a=%x, b=-inf)
        ...
        %exp = exp(x=sub(%safe, %kd))
        %masked = select(cond=%mask, a=%exp, b=0.0)
        %sum = reduce_sum(x=%masked, axes=[a], keep_dims=True)
        %div = real_div(x=%exp, y=%sum)
        %out = select(cond=%mask, a=%div, b=0.0)     # required

    becomes ``%out = select(cond=%mask, a=softmax(x=%safe, axis=a), b=0.0)``.
    All three masks must come from one source (broadcasting ``tile``s aside),
    the fill of ``%safe`` must be ``-inf`` so that the masked lanes of ``%exp``
    are exactly zero, and the outer ``select`` must be there -- it is what
    makes an entirely masked row come out as zeros rather than NaN.

    The subtraction is only peeled off when the subtrahend is constant
    along the softmax axis (which is what makes softmax invariant under it)
    and is not statically known to contain NaN or infinity; otherwise the
    ``sub`` output becomes the softmax input.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            replaced = _replace_decomposed_softmax(f)
            if replaced:
                logger.debug("replace_decomposed_softmax: replaced %d softmax chain(s)", replaced)
