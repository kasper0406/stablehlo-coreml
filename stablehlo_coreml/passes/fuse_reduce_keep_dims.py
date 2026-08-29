"""MIL pass: fold a keep-dims ``reshape``/``expand_dims`` back into its reduction.

StableHLO's ``reduce`` always drops the reduced dimensions, so JAX's
``jnp.sum(x, axis, keepdims=True)`` lowers to a ``reduce`` followed by a
``broadcast_in_dim`` that re-inserts the size-1 axes. The converter turns that
into::

    %r = reduce_sum(x=%x, axes=[2], keep_dims=False)   # (2, 8)
    %o = reshape(x=%r, shape=[2, 8, 1])                # (2, 8, 1)

which is exactly ``reduce_sum(x=%x, axes=[2], keep_dims=True)``. Rewriting it
into that single op is not just cosmetic: several of coremltools' own fusions
(``fuse_reduce_mean``, ``fuse_layernorm_or_instancenorm``) require the reduction
to be directly followed by the arithmetic that consumes it, and the intervening
``reshape`` blocks them. With this pass in place an RMSNorm
(``x * rsqrt(mean(x**2, -1, keepdims=True) + eps)``) becomes a ``reduce_mean``.

Under symbolic (dynamic) shapes the very same JAX source lowers to
``stablehlo.dynamic_broadcast_in_dim`` instead, and ``op_dynamic_broadcast_in_dim``
re-inserts the axes with ``expand_dims`` rather than ``reshape`` (a ``reshape``
would need the runtime shape as an operand)::

    %r = reduce_sum(x=%x, axes=[1], keep_dims=False)   # %x: (dim_0, 8) -> (dim_0,)
    %o = expand_dims(x=%r, axes=[1])                   # (dim_0, 1)

so both spellings are matched.
"""

import logging

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import shapes_equal

logger = logging.getLogger(__name__)

# The MIL reduction ops whose builder takes an `axes` + `keep_dims` pair.
# The arg-reductions (`reduce_argmax`/`reduce_argmin`) take a single `axis` and
# are deliberately not part of this list.
_REDUCE_OPS = frozenset({
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_l1_norm",
    "reduce_l2_norm",
    "reduce_log_sum",
    "reduce_log_sum_exp",
    "reduce_sum_square",
})


def _reduced_axes(reduce_op) -> list[int] | None:
    """Return the (non-negative, deduplicated) axes of ``reduce_op``, or ``None``.

    ``None`` is returned when the input rank or the axes are not known at
    compile time. An absent `axes` input means "reduce every axis".
    """
    x_shape = reduce_op.x.shape
    if x_shape is None:
        return None
    rank = len(x_shape)

    axes_var = reduce_op.inputs.get("axes")
    if axes_var is None:
        return list(range(rank))
    if axes_var.val is None:
        return None

    axes = {int(axis) % rank for axis in np.asarray(axes_var.val).reshape(-1).tolist()}
    return sorted(axes)


def _keep_dims_shape(x_shape, axes: list[int]) -> tuple:
    shape = list(x_shape)
    for axis in axes:
        shape[axis] = 1
    return tuple(shape)


# The ops that can spell out "re-insert the axes the reduction dropped": a
# `reshape` to the keep-dims shape (static shapes) or an `expand_dims` on the
# reduced axes (dynamic shapes, from `op_dynamic_broadcast_in_dim`). Both are
# order-preserving, so matching the output shape is enough to prove either is
# the keep-dims shape -- see `_match`.
_RESTORE_OPS = frozenset({"reshape", "expand_dims"})


def _match(reshape_op, block):
    """Return ``(reduce_op, axes)`` if ``reshape_op`` only re-inserts the reduced axes."""
    if reshape_op.op_type not in _RESTORE_OPS:
        return None

    reduce_op = reshape_op.x.op
    if reduce_op is None or reduce_op.op_type not in _REDUCE_OPS:
        return None
    if reduce_op.enclosing_block is not block:
        return None

    keep_dims = reduce_op.inputs.get("keep_dims")
    if keep_dims is None or keep_dims.val is None or bool(keep_dims.val):
        # `keep_dims=True` already, or not known at compile time.
        return None

    axes = _reduced_axes(reduce_op)
    if axes is None:
        return None

    # The reshape output must be exactly the input shape with 1s at the reduced
    # axes. Comparing the *output shape* (rather than the `shape` operand) is
    # both symbolic-aware and robust against `-1`/`0` entries in `shape`.
    # Because `keep_dims=False` only deletes those axes, re-inserting them is
    # always an order-preserving reshape, so matching shapes implies identity.
    target = _keep_dims_shape(reduce_op.x.shape, axes)
    if not shapes_equal(target, reshape_op.outputs[0].shape):
        return None

    return reduce_op, axes


@block_context_manager
def _fuse_reduce_keep_dims(block) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            fused += _fuse_reduce_keep_dims(nested_block)
        if len(op.blocks) > 0:
            continue

        match = _match(op, block)
        if match is None:
            continue
        reduce_op, axes = match

        # A fresh reduction with `keep_dims=True` replaces the reshape. The
        # original reduction is left in place; if the reshape was its sole
        # consumer it becomes dead and `dead_code_elimination` picks it up
        # (otherwise the other consumers, which expect the squeezed shape, keep
        # using it).
        keep_dims_var = getattr(mb, reduce_op.op_type)(
            x=reduce_op.x,
            axes=axes,
            keep_dims=True,
            before_op=op,
            name=op.outputs[0].name,
        )
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=keep_dims_var,
        )
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_reduce_keep_dims(AbstractGraphPass):
    """
    Fuse ``reduce_*(keep_dims=False) -> reshape/expand_dims(<keep-dims shape>)``
    into a single ``reduce_*(keep_dims=True)``.

    The rewrite applies when

    1. the reduce is one of the ops whose builder takes ``axes`` + ``keep_dims``
       (the arg-reductions are excluded),
    2. ``axes`` is known at compile time, and
    3. the ``reshape``/``expand_dims`` output shape is exactly the reduce input
       shape with the reduced axes set to 1 (compared symbolically, so dynamic
       dimensions are preserved).

    Neither ``reshape`` nor ``expand_dims`` reorders elements, so an output
    shape equal to the keep-dims shape is enough to prove the op is the identity
    on the reduction's result.

    When the reduce has consumers besides the reshape it is kept for them and
    only the reshape is replaced.

    Given:
        %1 = reduce_sum(x=%0, axes=[2], keep_dims=False)   # %0: (2, 8, 16)
        %2 = reshape(x=%1, shape=[2, 8, 1])

    Result:
        %2 = reduce_sum(x=%0, axes=[2], keep_dims=True)

    Given:
        %1 = reduce_sum(x=%0, axes=[1], keep_dims=False)   # %0: (dim_0, 8)
        %2 = expand_dims(x=%1, axes=[1])

    Result:
        %2 = reduce_sum(x=%0, axes=[1], keep_dims=True)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            fused = _fuse_reduce_keep_dims(f)
            if fused:
                logger.debug("fuse_reduce_keep_dims: fused %d reduce/reshape pair(s)", fused)
