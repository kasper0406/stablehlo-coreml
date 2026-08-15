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

import logging
import math

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import remove_dead_ops, sole_consumer, uniform_scalar_value

logger = logging.getLogger(__name__)

# Absolute tolerances on the constants of the pattern.
_HALF_TOLERANCE = 1e-4
_ONE_TOLERANCE = 1e-4
# The erf argument must be -x/sqrt(2); the factor is compared with this tolerance.
_FACTOR = -1.0 / math.sqrt(2.0)
_FACTOR_TOLERANCE = 1e-3

# Upper bound on the number of scaling/negation ops peeled off the erf argument.
# The converter emits at most two (a negate and a multiply); the bound just keeps
# the search finite for hand-written graphs.
_MAX_PEEL_DEPTH = 8


def _other_operand(op, var):
    """The operand of the binary ``op`` that is not ``var`` (``None`` if ``var`` is not one)."""
    if op.x is var:
        return op.y
    if op.y is var:
        return op.x
    return None


def _const_scaled_operand(op):
    """Split a ``mul``/``real_div``/``sub`` op into ``(input_var, factor)``.

    Recognises the ops that scale or negate a value by a compile-time constant:
    ``mul(v, c)``/``mul(c, v)`` -> ``(v, c)``, ``real_div(v, c)`` -> ``(v, 1/c)``
    and the negation ``sub(0, v)`` -> ``(v, -1)``. Returns ``None`` otherwise.
    """
    if op.op_type == "mul":
        for operand in (op.x, op.y):
            other = _other_operand(op, operand)
            factor = uniform_scalar_value(other)
            if factor is not None:
                return operand, factor
        return None

    if op.op_type == "real_div":
        divisor = uniform_scalar_value(op.y)
        if divisor is not None and divisor != 0.0:
            return op.x, 1.0 / divisor
        return None

    if op.op_type == "sub":
        lhs = uniform_scalar_value(op.x)
        if lhs is not None and abs(lhs) < 1e-12:
            return op.y, -1.0
        return None

    return None


def _peel_to_scaled_input(var, block):
    """Walk back over constant scalings/negations, returning ``[(var, factor), ...]``.

    The first entry is ``(var, 1.0)`` itself, then every prefix of the chain with
    the accumulated factor: ``arg == entry_var * factor``.
    """
    chain = [(var, 1.0, [])]
    factor = 1.0
    ops = []
    current = var
    for _ in range(_MAX_PEEL_DEPTH):
        op = current.op
        if op is None or op.enclosing_block is not block:
            break
        if sole_consumer(current) is None:
            # The intermediate value is used elsewhere; peeling further would
            # leave the op in the graph anyway, and the rewrite must not change
            # what that other consumer sees.
            break
        split = _const_scaled_operand(op)
        if split is None:
            break
        current, step = split
        factor *= step
        ops = ops + [op]
        chain.append((current, factor, ops))
    return chain


def _match_half_x(var, x, block) -> bool:
    """True if ``var`` is ``0.5 * x``."""
    op = var.op
    if op is None or op.op_type != "mul" or op.enclosing_block is not block:
        return False
    other = _other_operand(op, x)
    if other is None:
        return False
    half = uniform_scalar_value(other)
    return half is not None and abs(half - 0.5) < _HALF_TOLERANCE


def _match(mul_op, block):
    """Return ``(x, ops)`` if ``mul_op`` computes ``0.5 * x * (1 - erf(-x/sqrt(2)))``."""
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

        for candidate, factor, scale_ops in _peel_to_scaled_input(erf_op.x, block):
            if abs(factor - _FACTOR) > _FACTOR_TOLERANCE:
                continue
            if not _match_half_x(half_var, candidate, block):
                continue
            return candidate, [mul_op, sub_op, erf_op, half_op, *scale_ops]

    return None


@block_context_manager
def _fuse_gelu_erfc(block) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            fused += _fuse_gelu_erfc(nested_block)
        if len(op.blocks) > 0:
            continue

        match = _match(op, block)
        if match is None:
            continue
        x, matched_ops = match

        gelu = mb.gelu(x=x, mode="EXACT", before_op=op, name=op.outputs[0].name)
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=gelu)
        remove_dead_ops(block, matched_ops)
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_gelu_erfc(AbstractGraphPass):
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

    def apply(self, prog):
        for f in prog.functions.values():
            fused = _fuse_gelu_erfc(f)
            if fused:
                logger.debug("fuse_gelu_erfc: fused %d exact GELU(s)", fused)
