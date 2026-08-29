"""MIL pass: give ``select`` operands the full output shape when a dim is symbolic.

E5RT (Apple's Core ML runtime) cannot propagate shapes through a ``select``
whose operand broadcasts from 1 into a *symbolic* dimension. Loading such a
model fails with

    Failed to PropagateInputTensorShapes: Validation error during type
    inference for select: Incompatible Dimension.

``remove_broadcast_tiles`` already knows about this failure and deliberately
keeps the ``tile``s that feed a ``select``. That is only half the story: it can
only preserve tiles that exist, and JAX never emits one for the value operand of
a whole-tensor cache write such as ``jnp.where(mask, value, cache)`` with
``cache: (1, L, nkv, hd)`` (symbolic ``L``) and ``value: (1, 1, nkv, hd)``.
``jnp.where`` broadcasts implicitly, and an explicit ``jnp.broadcast_to`` in the
traced source is folded away before it reaches MIL, so the missing broadcast has
to be introduced here, on the MIL side.

A ``tile`` is not an option -- its ``reps`` would themselves have to be
symbolic. Each under-shaped operand is instead widened with ``fill_like`` plus
an identity operation: ``fill_like`` takes its shape from a reference operand
that already carries the output shape, so no symbolic arithmetic is needed.
Adding zero is exact in fp16 (and a bool ``cond`` is widened by ``logical_or``
against ``False`` instead), so the rewrite does not change numerics.

Static-shape ``select``s are left alone -- there the runtime knows every size
and propagates shapes fine, and the existing tiles are what keeps that path
working.
"""

import logging

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .pattern_utils import aligned_dim, dims_equal

logger = logging.getLogger(__name__)

# `select`'s operands.
_OPERANDS = ("cond", "a", "b")

# The order in which operands are considered as the `fill_like` reference.
# `fill_like` only takes its *shape* from the reference -- its output dtype
# follows its `value` -- so any of the three works and no `cast` is ever needed.
# Preferring `a`/`b` keeps the emitted zero tensor in the dtype it is added to,
# which makes the graph easier to read.
_REFERENCE_ORDER = ("a", "b", "cond")


def _needs_widening(operand, out_shape) -> bool:
    """True if ``operand`` broadcasts from a static 1 into a symbolic output dim.

    A rank below the output's is not in itself a reason to widen: an operand
    only has to be widened on the axes where broadcasting stretches it *and* the
    output dimension is symbolic. On a static axis both sizes are known at load
    time and E5RT propagates the shape fine. Operands are right-aligned
    NumPy-style, so an axis that broadcasting prepends to a lower-rank operand
    reads as a static 1 and does trigger widening when the output is symbolic
    there.

    A symbolic operand dimension never triggers widening, even when it is a
    different symbol than the output's: MIL mints a fresh symbol at nearly every
    op, so one runtime dimension routinely reaches a ``select`` under several
    names, and only a literal 1 is provably a broadcast.
    """
    if operand is None or operand.shape is None:
        return False
    out_rank = len(out_shape)
    if len(operand.shape) > out_rank:
        return False
    for axis, out_dim in enumerate(out_shape):
        if not is_symbolic(out_dim):
            continue
        dim = aligned_dim(operand, axis, out_rank)
        if dim is not None and not is_symbolic(dim) and int(dim) == 1:
            return True
    return False


def _has_output_shape(operand, out_shape) -> bool:
    """True if ``operand`` already carries the output shape, so it can size ``fill_like``.

    Static dimensions have to match exactly. A symbolic output dimension only
    requires the operand to be symbolic there too (see ``_needs_widening`` on
    why symbols cannot be compared): the output dimension is the broadcast of
    the operands, so an operand that is symbolic on that axis carries the full
    size whatever the symbol is called.
    """
    if operand is None or operand.shape is None:
        return False
    if len(operand.shape) != len(out_shape):
        return False
    for dim, out_dim in zip(operand.shape, out_shape):
        if is_symbolic(out_dim):
            if not is_symbolic(dim):
                return False
        elif not dims_equal(dim, out_dim):
            return False
    return True


def _fill_reference(op, out_shape):
    """The operand ``fill_like`` should take its shape from, or ``None``."""
    for name in _REFERENCE_ORDER:
        operand = op.inputs.get(name)
        if _has_output_shape(operand, out_shape):
            return operand
    return None


def _widen(operand, reference, op, name: str):
    """``operand`` broadcast up to ``reference``'s shape, without changing its value.

    ``select``'s operands are fp16/fp32/int32/bool, all of which ``fill_like``
    can produce, so the identity tensor always comes out in ``operand``'s own
    dtype and no ``cast`` is needed.
    """
    is_bool = operand.dtype == types.bool
    identity = types.nptype_from_builtin(operand.dtype)(0)
    zeros = mb.fill_like(
        ref_tensor=reference,
        value=identity,
        before_op=op,
        name=f"{op.name}_{name}_broadcast_identity",
    )
    combine = mb.logical_or if is_bool else mb.add
    return combine(x=operand, y=zeros, before_op=op, name=f"{op.name}_{name}_broadcast")


@block_context_manager
def _broadcast_select_operands(block) -> int:
    widened = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            widened += _broadcast_select_operands(nested_block)
        if len(op.blocks) > 0:
            continue

        if op.op_type != "select":
            continue

        out_shape = op.outputs[0].shape
        if out_shape is None or not any(is_symbolic(dim) for dim in out_shape):
            continue

        under_shaped = [name for name in _OPERANDS if _needs_widening(op.inputs.get(name), out_shape)]
        if not under_shaped:
            continue

        reference = _fill_reference(op, out_shape)
        if reference is None:
            logger.debug(
                "broadcast_select_operands: no operand of '%s' carries the full output shape %s",
                op.name,
                out_shape,
            )
            continue

        operands = {name: op.inputs[name] for name in _OPERANDS}
        for name in under_shaped:
            operands[name] = _widen(operands[name], reference, op, name)

        # The operands changed shape, so type inference on the new `select` must
        # not be re-run against the old output type (`no_check_var_types`). The
        # output itself is unaffected: widening only replaces implicit
        # broadcasting with explicit, so the broadcast result is the same shape.
        widened_select = mb.select(**operands, before_op=op, name=op.outputs[0].name)
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=widened_select,
            no_check_var_types=True,
        )
        block.remove_ops([op])
        widened += 1

    return widened


@register_pass(namespace="common")
class broadcast_select_operands(AbstractGraphPass):
    """
    Widen the operands of a ``select`` that broadcast into a symbolic dimension.

    A ``select`` is rewritten when all of the following hold:

    1. Its output shape has at least one symbolic dimension.
    2. Some operand (``cond``, ``a`` or ``b``) has a static 1 -- or no axis at
       all -- where the output dimension is symbolic.
    3. Some operand already carries the full output shape, to size ``fill_like``
       from.

    Given (``L`` symbolic):
        %3 = select(cond=%c, a=%v, b=%cache)   # %c, %cache: (1, L, 2, 8), %v: (1, 1, 2, 8)

    Result:
        %1 = fill_like(ref_tensor=%cache, value=0.0)         # (1, L, 2, 8)
        %2 = add(x=%v, y=%1)                                # (1, L, 2, 8)
        %3 = select(cond=%c, a=%2, b=%cache)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            widened = _broadcast_select_operands(f)
            if widened:
                logger.debug("broadcast_select_operands: widened %d select op(s)", widened)
