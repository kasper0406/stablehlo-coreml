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

from .pattern_utils import sole_consumer, uniform_const_operand

logger = logging.getLogger(__name__)

# Absolute tolerance on `alpha * beta == 1`, for constants that carry full fp32
# precision. It is a lower bound only: `_tolerance` widens it for narrower
# element types, where `1 / cap` cannot be represented anywhere near this well.
_TOLERANCE = 1e-4
# How many ulps of the operand type the product may be away from 1. Both `alpha`
# and `beta` are rounded to that type, and JAX already evaluates `1 / cap` in it,
# so the product drifts by a small multiple of the type's epsilon.
_TOLERANCE_ULPS = 4.0


def _dtype_epsilon(var) -> float:
    """Machine epsilon of ``var``'s element type (``0.0`` if it is not a float)."""
    if var is None:
        return 0.0
    dtype = getattr(var, "dtype", None)
    if dtype is None or not types.is_float(dtype):
        return 0.0
    return float(np.finfo(types.nptype_from_builtin(dtype)).eps)


def _tolerance(*vars_) -> float:
    """Tolerance on ``alpha * beta == 1`` for constants stored in these vars' types."""
    epsilon = max((_dtype_epsilon(var) for var in vars_), default=0.0)
    return max(_TOLERANCE, _TOLERANCE_ULPS * epsilon)


def _scalar_operand(op):
    """Like :func:`uniform_const_operand`, but only for single-element constants.

    Returns ``(const_value, other_var, const_var)``, or ``None`` when no operand
    is a uniform compile-time constant or when that constant is not rank-0/1 (a
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
    return value, other, const


def _inner_scale(tanh_op):
    """Return ``(x, beta, beta_const)`` for ``tanh(x * beta)``.

    A ``mul``/``real_div`` by a compile-time constant is peeled off the ``tanh``
    argument; ``beta_const`` is the ``Var`` holding that constant, or ``None``
    when nothing was peeled (``beta == 1``).
    """
    inner = tanh_op.x.op
    if inner is None or inner.enclosing_block is not tanh_op.enclosing_block:
        return tanh_op.x, 1.0, None
    if sole_consumer(tanh_op.x) is not tanh_op:
        # The scaled value is used elsewhere, so removing the scaling op would
        # not pay off (and it has to stay in the graph anyway).
        return tanh_op.x, 1.0, None

    if inner.op_type == "mul":
        scalar = _scalar_operand(inner)
        if scalar is not None:
            beta, operand, const = scalar
            return operand, beta, const
        return tanh_op.x, 1.0, None

    if inner.op_type == "real_div":
        # Only `x / c` is a scaling; `c / x` is not.
        scalar = _scalar_operand(inner)
        if scalar is not None and scalar[1] is inner.x and scalar[0] != 0.0:
            return inner.x, 1.0 / scalar[0], scalar[2]

    return tanh_op.x, 1.0, None


def _match(mul_op, block):
    """Return ``(x, alpha, beta)`` if ``mul_op`` completes a softcap pattern."""
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
        alpha, _, alpha_const = scalar

        x, beta, beta_const = _inner_scale(tanh_op)
        if abs(alpha * beta - 1.0) > _tolerance(x, alpha_const, beta_const):
            continue

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
    Fuse ``alpha * tanh(x * beta)`` (with ``alpha * beta == 1``) into
    ``scaled_tanh(x, alpha, beta)``.

    Both operand orders of the multiplications are accepted, and the inner
    scaling may be written as ``mul(x, beta)`` or ``real_div(x, 1/beta)``.
    The scaling constants must be uniform, single-element, compile-time
    constants, and the ``tanh`` output must have exactly one consumer.

    ``alpha * beta == 1`` is checked with a tolerance that grows with the
    element type of the operands: for an fp16 model ``1 / cap`` is rounded to
    fp16 before it ever reaches MIL, so the product is only unity to within a
    few fp16 ulps (``30.0 * fp16(1/30) == 0.99976``).

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
