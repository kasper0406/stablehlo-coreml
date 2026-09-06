"""MIL pass: drop no-op ``concat`` operands and flatten same-axis concat chains.

``concatenate`` lowers one-to-one, so StableHLO built by appending to a list of
tensors reaches MIL as a left-leaning chain of binary ``concat`` ops, every link
of which materialises a full copy of everything concatenated so far. A five-link
chain therefore writes the final tensor more than once over. Padding adds no-ops
on top of that: an edge of width zero lowers to a ``concat`` with a zero-sized
operand, and a single-element concatenation to a ``concat`` with one input --
both plain copies of their input.

coremltools cleans up neither. ``noop_elimination`` does not list ``concat``
among its ``_SUPPORTED_OPS`` at all, ``const_elimination`` only folds concats
whose operands are *all* constants, and ``remove_redundant_ops`` just
deduplicates structurally identical ops. The no-ops consequently survive into
the final model, which is what this pass is for.
"""

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .pattern_utils import RewritePass, normalize_axis, shapes_equal, sole_consumer


def _plain_concat_axis(op, rank: int) -> int | None:
    """``op``'s concat axis, normalised against ``rank``, or ``None``.

    ``None`` whenever the op is not a ``concat`` this pass understands: a
    non-constant or out-of-range ``axis``, or ``interleave=True`` (which
    round-robins the operands instead of appending them, so neither of the
    rewrites below holds).
    """
    if op is None or op.op_type != "concat":
        return None

    interleave = op.inputs.get("interleave")
    if interleave is not None:
        # `interleave` is a `const` by its input spec, so an unknown value here
        # means a graph we do not understand -- give up on it either way.
        value = getattr(interleave, "val", None)
        if value is None or bool(value):
            return None

    axis = getattr(op.inputs.get("axis"), "val", None)
    if axis is None:
        return None
    return normalize_axis(int(axis), rank)


def _has_uniform_rank(values, rank: int) -> bool:
    """True if every value has a known shape of exactly ``rank`` dimensions.

    Rank-0 operands are rejected by construction: ``concat`` promotes scalars to
    length-1 rows, so its output rank is 1 while the operand rank is 0, and
    neither splicing them into another concat nor bypassing the op would keep
    the shapes right.
    """
    return all(value.shape is not None and len(value.shape) == rank for value in values)


def _inline_operands(value, parent, axis: int, block, rank: int):
    """The operands to splice in place of ``value``, or ``None`` to keep it as is.

    ``value`` may be replaced by its own operands when it is produced by a
    ``concat`` on the same axis that nothing else consumes:
    ``concat(concat(a, b), c) == concat(a, b, c)``. The producer has to sit in
    the same block, and ``parent`` has to be its only consumer -- otherwise the
    intermediate result is still needed and the rewrite would only add an op.
    """
    child = value.op
    if _plain_concat_axis(child, rank) != axis:
        return None
    if child.enclosing_block is not block:
        return None
    # `sole_consumer` also rejects a value that leaves the block as an output.
    if sole_consumer(value) is not parent:
        return None
    # A var consumed twice by the same op is a single entry in `child_ops`, so
    # count the occurrences rather than trusting `sole_consumer` alone.
    if sum(1 for operand in parent.values if operand is value) != 1:
        return None
    if not _has_uniform_rank(child.values, rank):
        return None
    return list(child.values)


def _is_empty(value, axis: int) -> bool:
    """True if ``value`` provably contributes nothing along ``axis``.

    Only a literal ``0`` counts: a symbolic dimension may well be zero at
    runtime, but it may just as well not be.
    """
    dim = value.shape[axis]
    return not is_symbolic(dim) and int(dim) == 0


def _concat_shape(values, axis: int):
    """The shape ``concat(values, axis=axis)`` infers, or ``None`` if not provable.

    Mirrors ``concat.type_inference``: the non-axis dimensions come from the
    first operand and the axis dimension is the sum. One symbolic operand makes
    the sum a fresh symbol, which no comparison can match, so those give up.
    """
    shape = list(values[0].shape)
    total = 0
    for value in values:
        dim = value.shape[axis]
        if is_symbolic(dim):
            return None
        total += int(dim)
    shape[axis] = total
    return tuple(shape)


def _breaks_function_outputs(op, new_var) -> bool:
    """True if bypassing ``op`` in favour of ``new_var`` would damage the function outputs.

    coremltools carries the name of a replaced block output over to its
    replacement so that the model keeps its output names
    (``Block.replace_block_output_var``). When ``op``'s output is a function
    output, that rename has two failure modes:

    * ``new_var`` is a function input: the rename is refused outright
      (``ValueError: It is not allowed to modify function inputs name.``), which
      aborts the whole conversion. That is e.g. a single-input ``concat`` of an
      argument returned as the function's result.
    * ``new_var`` is already an output of the same function: the two output
      slots collapse onto one ``Var``, which then takes ``op``'s output name --
      so the converted model is left with one output where the program had two,
      and the other output name silently disappears.

    Neither applies to a nested block: ``replace_block_output_var`` only renames
    on a ``Function``, and a ``cond``/``while_loop`` block may perfectly well
    list the same var in two of its output slots.
    """
    block = op.enclosing_block
    if not isinstance(block, Function):
        return False
    out_var = op.outputs[0]
    if out_var not in block.outputs:
        return False
    if new_var in block.outputs:
        return True
    return new_var in block.inputs.values() and new_var.name != out_var.name


@register_pass(namespace="common")
class simplify_concat(RewritePass):
    """
    Remove ``concat`` operands that contribute nothing, and fold a ``concat``
    chain on one axis into a single n-ary ``concat``.

    A concat is rewritten when, after

    1. replacing every operand that is itself a same-axis ``concat`` in the same
       block with no other consumer by that concat's own operands, and
    2. dropping every operand whose size along the concat axis is a literal 0,

    the operand list has changed, or is down to a single operand -- in which
    case the concat is a copy and its consumers use the operand directly. The
    rewrite is skipped unless the new operand list provably infers the very same
    output shape, which also rules out the cases a symbolic dimension makes
    undecidable (``concat`` mints a fresh symbol for an axis it cannot add up).

    ``interleave=True`` concats take part in neither rewrite: they round-robin
    their operands rather than append them, so they are neither associative nor
    indifferent to an empty operand.

    A copy whose output is a *function* output is only bypassed when that does
    not disturb the model interface; see :func:`_breaks_function_outputs`.

    Given:
        %2 = concat(values=(%0, %1), axis=0)   # %1: (0, 16)
        %4 = concat(values=(%2, %3), axis=0)

    Result:
        %4 = concat(values=(%0, %3), axis=0)
    """

    _REWRITES = "concat op(s)"

    def visit(self, op, block) -> bool:
        out_var = op.outputs[0]
        if out_var.shape is None:
            return False
        rank = len(out_var.shape)

        axis = _plain_concat_axis(op, rank)
        if axis is None:
            return False

        values = list(op.values)
        if not _has_uniform_rank(values, rank):
            return False

        expanded = []
        changed = False
        for value in values:
            inlined = _inline_operands(value, op, axis, block, rank)
            if inlined is None:
                expanded.append(value)
            else:
                expanded += inlined
                changed = True

        new_values = [value for value in expanded if not _is_empty(value, axis)]
        if len(new_values) != len(expanded):
            changed = True

        # Everything was empty: there is no operand left to carry the (equally
        # empty) result, so leave the op alone.
        if len(new_values) == 0:
            return False
        # A single operand left is a copy, worth removing even if nothing else
        # about the operand list changed.
        if not changed and len(new_values) > 1:
            return False

        if not shapes_equal(_concat_shape(new_values, axis), out_var.shape):
            return False

        built_op = None
        if len(new_values) == 1:
            new_var = new_values[0]
            if new_var.dtype != out_var.dtype:
                return False
            if _breaks_function_outputs(op, new_var):
                return False
        else:
            # No `_breaks_function_outputs` check here: this var is brand new, so
            # it is neither a function input nor already an output, and the
            # rename `replace_block_output_var` performs just moves `op`'s output
            # name onto its replacement -- one output slot in, one out.
            new_var = mb.concat(values=new_values, axis=axis, interleave=False, before_op=op)
            built_op = new_var.op

        # `try_...` rather than the unguarded variant: an operand may descend
        # from a var coremltools refuses to replace (a `constexpr_*` weight,
        # say), in which case the rewrite is skipped instead of raising.
        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=out_var, new_var=new_var
        ):
            if built_op is not None:
                built_op.remove_from_block()
            return False

        op.remove_from_block()
        return True
