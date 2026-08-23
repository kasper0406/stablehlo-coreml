"""MIL pass: fuse JAX's ``tanh``-approximated GELU into ``gelu(mode="TANH_APPROXIMATION")``.

``jax.nn.gelu(x)`` -- ``approximate=True`` is the *default*, so this is what
``flax.nnx.gelu`` and every ``equinox`` MLP with a GELU activation trace to --
is written as ``x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``,
which the converter lowers to::

    %1 = mul(x=%x, y=%x)
    %2 = mul(x=%1, y=%x)              # x**3, spelled as two multiplications
    %3 = mul(x=0.044715, y=%2)
    %4 = add(x=%x, y=%3)
    %5 = mul(x=0.7978845834732056, y=%4)
    %6 = tanh(x=%5)
    %7 = add(x=1.0, y=%6)
    %8 = mul(x=0.5, y=%7)
    %9 = mul(x=%x, y=%8)

coremltools has a pass for this shape, ``fuse_gelu_tanh_approximation``, but it
insists on a literal ``pow(x, 3)`` while JAX emits the ``x * x * x`` chain
above, so nine elementwise ops over the activation tensor survived into every
converted model. This pass recognises both spellings of the cube.

The coefficients are recovered by peeling constant scalings, so the
associativity does not matter: ``x * (0.5 * (1 + tanh(t)))`` and
``(0.5 * x) * (1 + tanh(t))`` are both matched, as is a ``tanh`` argument
written in distributed form (``k*x + k*a*x**3``) rather than factored.
"""

import logging
import math

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import (
    dtype_epsilon,
    peel_to_scaled_input,
    sole_consumer,
    uniform_const_operand,
    uniform_scalar_value,
)

logger = logging.getLogger(__name__)

# The coefficients of the tanh approximation, as the paper writes them.
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_CUBE_COEFFICIENT = 0.044715
# Coefficient of `x**3` inside the tanh once the outer factor is distributed.
_CUBIC_TERM = _SQRT_2_OVER_PI * _CUBE_COEFFICIENT

# Relative tolerance on every coefficient of the pattern. Tight enough to reject
# a merely GELU-shaped formula, loose enough for a constant that was rounded to
# fp32 -- `_tolerance` widens it further for narrower element types.
_RELATIVE_TOLERANCE = 1e-4
# How many ulps of the operand type a coefficient may be off. The literals reach
# MIL already rounded to that type, and the matcher multiplies two of them
# together, so the error compounds.
_TOLERANCE_ULPS = 8.0


def _tolerance(var) -> float:
    """Relative tolerance on the coefficients, for a pattern computed in ``var``'s type."""
    return max(_RELATIVE_TOLERANCE, _TOLERANCE_ULPS * dtype_epsilon(var))


def _close(value: float, expected: float, tolerance: float) -> bool:
    return abs(value - expected) <= tolerance * abs(expected)


def _square_input(var, block):
    """Return ``v`` if ``var`` is ``mul(v, v)``, else ``None``."""
    op = getattr(var, "op", None)
    if op is None or op.op_type != "mul" or op.enclosing_block is not block:
        return None
    return op.x if op.x is op.y else None


def _cube_input(var, block):
    """Return ``v`` if ``var`` is ``v ** 3``, else ``None``.

    Both ``pow(v, 3)`` and the ``mul(mul(v, v), v)`` chain JAX emits are
    recognised; in the latter case the square must not be observed elsewhere.
    """
    op = getattr(var, "op", None)
    if op is None or op.enclosing_block is not block:
        return None

    if op.op_type == "pow":
        exponent = uniform_scalar_value(op.y)
        if exponent is not None and exponent == 3.0:
            return op.x
        return None

    if op.op_type != "mul":
        return None
    for square, base in ((op.x, op.y), (op.y, op.x)):
        if _square_input(square, block) is base and sole_consumer(square) is op:
            return base
    return None


def _match_tanh_argument(arg, block, tolerance):
    """Return ``x`` if ``arg`` is ``sqrt(2/pi) * (x + 0.044715 * x**3)``, else ``None``."""
    for inner, outer_factor in peel_to_scaled_input(arg, block):
        add_op = getattr(inner, "op", None)
        if add_op is None or add_op.op_type != "add" or add_op.enclosing_block is not block:
            continue

        for linear_var, cubic_var in ((add_op.x, add_op.y), (add_op.y, add_op.x)):
            if linear_var is cubic_var:
                continue
            if sole_consumer(cubic_var) is not add_op:
                continue

            for cube_var, cube_factor in peel_to_scaled_input(cubic_var, block):
                x = _cube_input(cube_var, block)
                if x is None:
                    continue
                if not _close(outer_factor * cube_factor, _CUBIC_TERM, tolerance):
                    continue
                for linear_base, linear_factor in peel_to_scaled_input(linear_var, block):
                    if linear_base is not x:
                        continue
                    if _close(outer_factor * linear_factor, _SQRT_2_OVER_PI, tolerance):
                        return x
    return None


def _match(mul_op, block):
    """Return ``x`` if ``mul_op`` computes the tanh-approximated GELU of ``x``."""
    if mul_op.op_type != "mul":
        return None
    tolerance = _tolerance(mul_op.outputs[0])

    for cdf_var, scale_var in ((mul_op.x, mul_op.y), (mul_op.y, mul_op.x)):
        if cdf_var is scale_var:
            continue
        if sole_consumer(cdf_var) is not mul_op:
            continue

        for sum_var, cdf_factor in peel_to_scaled_input(cdf_var, block):
            add_op = getattr(sum_var, "op", None)
            if add_op is None or add_op.op_type != "add" or add_op.enclosing_block is not block:
                continue
            operand = uniform_const_operand(add_op)
            if operand is None:
                continue
            one, tanh_var = operand
            if not _close(one, 1.0, tolerance):
                continue

            tanh_op = getattr(tanh_var, "op", None)
            if tanh_op is None or tanh_op.op_type != "tanh":
                continue
            if tanh_op.enclosing_block is not block:
                continue
            if sole_consumer(tanh_var) is not add_op:
                continue

            x = _match_tanh_argument(tanh_op.x, block, tolerance)
            if x is None:
                continue

            # What is left of the product must be `0.5 * x`.
            for base, factor in peel_to_scaled_input(scale_var, block):
                if base is x and _close(factor * cdf_factor, 0.5, tolerance):
                    return x

    return None


@block_context_manager
def _fuse_gelu_tanh(block) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            fused += _fuse_gelu_tanh(nested_block)
        if len(op.blocks) > 0:
            continue

        x = _match(op, block)
        if x is None:
            continue

        gelu = mb.gelu(x=x, mode="TANH_APPROXIMATION", before_op=op, name=op.outputs[0].name)
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=gelu)
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_gelu_tanh(AbstractGraphPass):
    """
    Fuse ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`` into
    ``gelu(x, mode="TANH_APPROXIMATION")``.

    This is the default spelling of ``jax.nn.gelu`` (and therefore of
    ``flax.nnx.gelu``). coremltools' ``fuse_gelu_tanh_approximation`` covers the
    same formula but requires the cube to be a literal ``pow(x, 3)``, which is
    not what JAX emits.

    Given:
        %1 = mul(x=%0, y=%0)
        %2 = mul(x=%1, y=%0)
        %3 = mul(x=0.044715, y=%2)
        %4 = add(x=%0, y=%3)
        %5 = mul(x=0.7978845834732056, y=%4)
        %6 = tanh(x=%5)
        %7 = add(x=1.0, y=%6)
        %8 = mul(x=0.5, y=%7)
        %9 = mul(x=%0, y=%8)

    Result:
        %9 = gelu(x=%0, mode="TANH_APPROXIMATION")

    The coefficients are recovered by peeling constant scalings off each side,
    so the two associativities of the ``0.5 *`` factor and both the factored and
    the distributed spelling of the ``tanh`` argument are matched. They are
    compared with a relative tolerance that widens with the operand type, since
    fp16 cannot hold ``0.044715`` or ``sqrt(2/pi)`` to fp32 accuracy. Every
    intermediate value of the pattern must have exactly one consumer, apart from
    ``x`` itself, which the pattern reads three times.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            fused = _fuse_gelu_tanh(f)
            if fused:
                logger.debug("fuse_gelu_tanh: fused %d approximated GELU(s)", fused)
