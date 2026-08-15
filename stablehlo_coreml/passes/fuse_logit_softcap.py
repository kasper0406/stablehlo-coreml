"""MIL pass: fuse a "logit softcap" ``tanh`` sandwich into ``scaled_tanh``.

Attention logit softcapping (Gemma-style) is written in JAX as
``jnp.tanh(x / cap) * cap``, which the converter lowers to::

    %1 = real_div(x=%x, y=30.0)     # or mul(x=%x, y=1/30)
    %2 = tanh(x=%1)
    %3 = mul(x=%2, y=30.0)

MIL has a single op for that: ``scaled_tanh(x, alpha, beta) = alpha * tanh(beta * x)``.
Fusing saves two elementwise passes over a (usually large) logits tensor.

Only the exact ``alpha * beta ~= 1`` shape is fused -- i.e. a genuine softcap,
where the composition is the identity for small ``x``. A general
``a * tanh(b * x)`` would also be expressible as ``scaled_tanh``, but scaling by
an arbitrary constant is not necessarily a softcap and is left alone so that the
pass stays a targeted, easily reviewable rewrite.
"""

import logging

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import remove_dead_ops, sole_consumer, uniform_scalar_value

logger = logging.getLogger(__name__)

# Absolute tolerance on `alpha * beta == 1`.
_TOLERANCE = 1e-4


def _scalar_operand(op, other):
    """Return the uniform constant value of the operand of ``op`` that is not ``other``.

    ``None`` when ``other`` is not an operand, when the other operand is not a
    uniform compile-time constant, or when it is not rank-0/1 (a constant with a
    real shape would broadcast the output, which ``scaled_tanh`` cannot do).
    """
    x, y = op.x, op.y
    if x is other:
        const = y
    elif y is other:
        const = x
    else:
        return None

    if const.shape is not None and int(np.prod(const.shape)) != 1:
        return None
    return uniform_scalar_value(const)


def _inner_scale(tanh_op):
    """Return ``(x, beta)`` for ``tanh(x * beta)``, peeling a ``mul``/``real_div`` by a constant."""
    inner = tanh_op.x.op
    if inner is None or inner.enclosing_block is not tanh_op.enclosing_block:
        return tanh_op.x, 1.0
    if sole_consumer(tanh_op.x) is not tanh_op:
        # The scaled value is used elsewhere, so removing the scaling op would
        # not pay off (and it has to stay in the graph anyway).
        return tanh_op.x, 1.0

    if inner.op_type == "mul":
        for operand in (inner.x, inner.y):
            beta = _scalar_operand(inner, operand)
            if beta is not None:
                return operand, beta
        return tanh_op.x, 1.0

    if inner.op_type == "real_div":
        # Only `x / c` is a scaling; `c / x` is not.
        divisor = _scalar_operand(inner, inner.x)
        if divisor is not None and divisor != 0.0:
            return inner.x, 1.0 / divisor

    return tanh_op.x, 1.0


def _match(mul_op, block):
    """Return ``(x, alpha, beta, ops)`` if ``mul_op`` completes a softcap pattern."""
    if mul_op.op_type != "mul":
        return None

    for tanh_var in (mul_op.x, mul_op.y):
        tanh_op = tanh_var.op
        if tanh_op is None or tanh_op.op_type != "tanh":
            continue
        if tanh_op.enclosing_block is not block:
            continue
        # The tanh result must feed nothing but this multiplication.
        if sole_consumer(tanh_var) is not mul_op:
            continue

        alpha = _scalar_operand(mul_op, tanh_var)
        if alpha is None:
            continue

        x, beta = _inner_scale(tanh_op)
        if abs(alpha * beta - 1.0) > _TOLERANCE:
            continue

        ops = [mul_op, tanh_op]
        scale_op = tanh_op.x.op
        if x is not tanh_op.x and scale_op is not None:
            ops.append(scale_op)
        return x, alpha, beta, ops

    return None


@block_context_manager
def _fuse_logit_softcap(block) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            fused += _fuse_logit_softcap(nested_block)
        if len(op.blocks) > 0:
            continue

        match = _match(op, block)
        if match is None:
            continue
        x, alpha, beta, matched_ops = match

        # `alpha`/`beta` are `const T`, with T the element type of `x`.
        dtype = types.nptype_from_builtin(x.dtype)
        capped = mb.scaled_tanh(
            x=x,
            alpha=dtype(alpha),
            beta=dtype(beta),
            before_op=op,
            name=op.outputs[0].name,
        )
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=capped)
        remove_dead_ops(block, matched_ops)
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_logit_softcap(AbstractGraphPass):
    """
    Fuse ``alpha * tanh(x * beta)`` (with ``alpha * beta == 1``) into
    ``scaled_tanh(x, alpha, beta)``.

    Both operand orders of the multiplications are accepted, and the inner
    scaling may be written as ``mul(x, beta)`` or ``real_div(x, 1/beta)``.
    The scaling constants must be uniform, single-element, compile-time
    constants, and the ``tanh`` output must have exactly one consumer.

    Given:
        %1 = real_div(x=%0, y=30.0)
        %2 = tanh(x=%1)
        %3 = mul(x=%2, y=30.0)

    Result:
        %3 = scaled_tanh(x=%0, alpha=30.0, beta=0.0333)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            fused = _fuse_logit_softcap(f)
            if fused:
                logger.debug("fuse_logit_softcap: fused %d softcap(s)", fused)
