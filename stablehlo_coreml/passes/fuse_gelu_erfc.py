"""MIL pass: fuse the ``erfc`` spelling of the exact GELU into ``gelu(mode="EXACT")``.

``jax.nn.gelu(x, approximate=False)`` is written as
``0.5 * x * erfc(-x / sqrt(2))``. Together with the ``chlo.erfc`` composite
handler (which emits ``1 - erf(...)``) the converter produces::

    %h = mul(x=%x, y=0.5)
    %n = sub(x=0.0, y=%x)
    %s = mul(x=%n, y=0.70710677)     # -x / sqrt(2)
    %e = erf(x=%s)
    %c = sub(x=1.0, y=%e)
    %o = mul(x=%h, y=%c)

which is exactly ``gelu(x, mode="EXACT")``. coremltools' own ``fuse_gelu_exact``
only recognises the algebraically equivalent but structurally different
``0.5 * x * (1 + erf(x / sqrt(2)))`` form, so this pass complements it rather
than duplicating it.
"""

import math

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import (
    RewritePass,
    peel_to_scaled_input,
    shapes_equal,
    sole_consumer,
    uniform_const_operand,
    uniform_scalar_value,
)

# Absolute tolerances on the constants of the pattern.
_HALF_TOLERANCE = 1e-4
_ONE_TOLERANCE = 1e-4
# The erf argument must be -x/sqrt(2); the factor is compared with this tolerance.
_FACTOR = -1.0 / math.sqrt(2.0)
# Wide enough for fp16 rounding, but not for merely GELU-like formulae.
_FACTOR_TOLERANCE = 1e-4


def _other_operand(op, var):
    """The operand of the binary ``op`` that is not ``var`` (``None`` if ``var`` is not one)."""
    if op.x is var:
        return op.y
    if op.y is var:
        return op.x
    return None


def _match_half_x(var, x, block) -> bool:
    """True if ``var`` is ``0.5 * x``."""
    op = var.op
    if op is None or op.op_type != "mul" or op.enclosing_block is not block:
        return False
    scaled = uniform_const_operand(op)
    if scaled is None or scaled[1] is not x:
        return False
    return abs(scaled[0] - 0.5) < _HALF_TOLERANCE


def _match(mul_op, block):
    """Return ``x`` if ``mul_op`` computes ``0.5 * x * (1 - erf(-x/sqrt(2)))``."""
    if mul_op.op_type != "mul":
        return None

    for complement in (mul_op.x, mul_op.y):
        half_var = _other_operand(mul_op, complement)
        if half_var is None or half_var is complement:
            continue

        sub_op = complement.op
        if sub_op is None or sub_op.op_type != "sub" or sub_op.enclosing_block is not block:
            continue
        if sole_consumer(complement) is not mul_op:
            continue
        one = uniform_scalar_value(sub_op.x)
        if one is None or abs(one - 1.0) > _ONE_TOLERANCE:
            continue

        erf_var = sub_op.y
        erf_op = erf_var.op
        if erf_op is None or erf_op.op_type != "erf" or erf_op.enclosing_block is not block:
            continue
        if sole_consumer(erf_var) is not sub_op:
            continue

        half_op = half_var.op
        if half_op is None or half_op.enclosing_block is not block:
            continue
        if sole_consumer(half_var) is not mul_op:
            continue

        for candidate, factor in peel_to_scaled_input(erf_op.x, block):
            if abs(factor - _FACTOR) > _FACTOR_TOLERANCE:
                continue
            if not _match_half_x(half_var, candidate, block):
                continue
            # A uniform constant of the pattern is accepted whatever its shape, so
            # it may broadcast `x`; `gelu(x)` has `x`'s shape and cannot replace a
            # wider result.
            if not shapes_equal(mul_op.outputs[0].shape, candidate.shape):
                continue
            return candidate

    return None


@register_pass(namespace="common")
class fuse_gelu_erfc(RewritePass):
    """
    Fuse ``0.5 * x * (1 - erf(k * x))`` with ``k == -1/sqrt(2)`` into
    ``gelu(x, mode="EXACT")``.

    ``k`` is recovered by peeling a chain of constant scalings (``mul``,
    ``real_div``) and negations (``sub(0, .)``) off the ``erf`` argument until
    the same var that feeds the ``0.5 *`` multiplication is reached. Every
    intermediate value of the pattern must have exactly one consumer.

    Given:
        %1 = mul(x=%0, y=0.5)
        %2 = sub(x=0.0, y=%0)
        %3 = mul(x=%2, y=0.70710677)
        %4 = erf(x=%3)
        %5 = sub(x=1.0, y=%4)
        %6 = mul(x=%1, y=%5)

    Result:
        %6 = gelu(x=%0, mode="EXACT")
    """

    _REWRITES = "exact GELU(s)"

    def visit(self, op, block) -> bool:
        x = _match(op, block)
        if x is None:
            return False

        gelu = mb.gelu(x=x, mode="EXACT", before_op=op, name=op.outputs[0].name)
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=gelu)
        return True
