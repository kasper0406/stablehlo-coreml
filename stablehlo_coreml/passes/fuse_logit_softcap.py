"""MIL pass: fuse a ``tanh`` sandwiched between two constant scalings into ``scaled_tanh``.

The motivating case is attention logit softcapping (Gemma-style), written in JAX
as ``jnp.tanh(x / cap) * cap``, which the converter lowers to::

    %1 = real_div(x=%x, y=30.0)     # or mul(x=%x, y=1/30)
    %2 = tanh(x=%1)
    %3 = mul(x=%2, y=30.0)

MIL has a single op for that: ``scaled_tanh(x, alpha, beta) = alpha * tanh(beta * x)``.
Fusing saves two elementwise passes over a (usually large) logits tensor.

``scaled_tanh`` computes exactly the matched expression, so any pair of constants
is fused; the softcap shape (``alpha * beta == 1``, where the composition is the
identity for small ``x``) is just the common case, not a requirement.
"""

import logging

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import sole_consumer, uniform_const_operand

logger = logging.getLogger(__name__)


def _scalar_operand(op):
    """Like :func:`uniform_const_operand`, but only for single-element constants.

    Returns ``(const_value, other_var)``, or ``None`` when no operand is a
    uniform compile-time constant or when that constant is not rank-0/1 (a
    constant with a real shape would broadcast the output, which ``scaled_tanh``
    cannot do).
    """
    operand = uniform_const_operand(op)
    if operand is None:
        return None
    value, other = operand
    const = op.y if op.x is other else op.x
    if const.shape is not None and int(np.prod(const.shape)) != 1:
        return None
    return value, other


def _inner_scale(tanh_op):
    """Return ``(x, beta)`` for ``tanh(x * beta)``.

    A ``mul``/``real_div`` by a compile-time constant is peeled off the ``tanh``
    argument; ``beta == 1`` when there is nothing to peel.
    """
    inner = tanh_op.x.op
    if inner is None or inner.enclosing_block is not tanh_op.enclosing_block:
        return tanh_op.x, 1.0
    if sole_consumer(tanh_op.x) is not tanh_op:
        # The scaled value is used elsewhere, so removing the scaling op would
        # not pay off (and it has to stay in the graph anyway).
        return tanh_op.x, 1.0

    if inner.op_type == "mul":
        scalar = _scalar_operand(inner)
        if scalar is not None:
            beta, operand = scalar
            return operand, beta
        return tanh_op.x, 1.0

    if inner.op_type == "real_div":
        # Only `x / c` is a scaling; `c / x` is not.
        scalar = _scalar_operand(inner)
        if scalar is not None and scalar[1] is inner.x and scalar[0] != 0.0:
            return inner.x, 1.0 / scalar[0]

    return tanh_op.x, 1.0


def _match(mul_op, block):
    """Return ``(x, alpha, beta)`` if ``mul_op`` computes ``alpha * tanh(beta * x)``."""
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

        scalar = _scalar_operand(mul_op)
        if scalar is None or scalar[1] is not tanh_var:
            continue
        alpha = scalar[0]

        x, beta = _inner_scale(tanh_op)
        return x, alpha, beta

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
        x, alpha, beta = match

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
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_logit_softcap(AbstractGraphPass):
    """
    Fuse ``alpha * tanh(beta * x)`` into ``scaled_tanh(x, alpha, beta)``.

    The motivating case is Gemma-style logit softcapping, ``cap * tanh(x / cap)``,
    but ``scaled_tanh`` is exactly the matched expression, so no relation between
    ``alpha`` and ``beta`` is required.

    Both operand orders of the multiplications are accepted, and the inner
    scaling may be written as ``mul(x, beta)`` or ``real_div(x, 1/beta)``, or be
    missing entirely (``beta == 1``). The scaling constants must be uniform,
    single-element, compile-time constants, and the ``tanh`` output must have
    exactly one consumer.

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
                logger.debug("fuse_logit_softcap: fused %d scaled tanh(s)", fused)
