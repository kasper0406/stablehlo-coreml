"""MIL pass: drop the negative-index guard on values that cannot be negative.

Since iOS17 ``gather``/``gather_nd`` no longer wrap negative indices, so
coremltools' ``guard_negative_gather_indices`` pass makes every one of them wrap
explicitly, by rewriting the indices into::

    cond    = greater_equal(indices, 0)
    plus    = add(indices, <size of the gathered axis>)   # + a shape/slice chain
    indices = select(cond, indices, plus)

StableHLO gathers *clamp* their start indices rather than wrapping them, so the
converter's ``op_gather`` already emits ``minimum(maximum(indices, 0), size - 1)``
ahead of the gather, and the index vectors that reach the remaining gathers come
out of ops that cannot go negative in the first place (``non_zero``, ``shape``).
The guard is then dead weight: its ``select`` provably picks its first operand.

The pass runs right after coremltools' guard and takes it back out wherever the
guarded value is provably non-negative. It only removes the ``select`` itself --
the ``greater_equal``/``add``/``shape``/``slice_*`` ops that fed it are left to
the ``dead_code_elimination`` entry behind the pass.
"""

import numpy as np
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .pattern_utils import RewritePass, shapes_equal, uniform_scalar_value

# How far back `_is_nonnegative` walks. The longest chain the converter produces
# is `squeeze(gather(minimum(maximum(...))))`, so a handful of steps is plenty;
# the bound just keeps the search finite for hand-written graphs.
MAX_PROOF_DEPTH = 16

# Ops whose result is non-negative whatever their inputs are: `non_zero` returns
# indices into its operand, `shape` returns dimensions.
_ALWAYS_NONNEGATIVE = frozenset({"non_zero", "shape"})

# Ops that are non-negative when *every* one of the listed operands is. They
# either move elements around unchanged (`gather` picks elements out of `x`, and
# `reshape` and friends only re-address them) or cannot leave the range spanned
# by their operands (`minimum`, `select`).
#
# Arithmetic (`add`, `mul`, ...) is deliberately absent: two non-negative integers
# can still add up to a negative one by overflowing their type.
_ALL_OPERANDS_NONNEGATIVE = {
    "identity": ("x",),
    "reshape": ("x",),
    "squeeze": ("x",),
    "expand_dims": ("x",),
    "transpose": ("x",),
    "reverse": ("x",),
    "tile": ("x",),
    "slice_by_index": ("x",),
    "slice_by_size": ("x",),
    "gather": ("x",),
    "gather_nd": ("x",),
    "gather_along_axis": ("x",),
    "concat": ("values",),
    "minimum": ("x", "y"),
    "select": ("a", "b"),
}

# Ops that are non-negative as soon as *one* of the listed operands is. The clamp
# the converter emits bottoms out here: `maximum(indices, 0)`.
_ANY_OPERAND_NONNEGATIVE = {
    "maximum": ("x", "y"),
}


def _int_bits(dtype) -> int | None:
    """Width in bits of an integer builtin type, or ``None`` if it is not one."""
    try:
        nptype = types.nptype_from_builtin(dtype)
    except Exception:
        return None
    return np.dtype(nptype).itemsize * 8


def _cast_preserves_nonnegativity(src, dst) -> bool:
    """Whether a non-negative value of type ``src`` stays non-negative as ``dst``.

    Widening never loses the sign, but a narrowing cast wraps: ``int32(70000)``
    is ``int16(4464)`` at 32 bits but ``-15072`` at 16, and a float too large for
    the destination integer type wraps just the same. Only casts that provably
    keep every non-negative value are accepted.
    """
    if types.is_float(dst):
        return True
    if not types.is_int(dst):
        return False
    if types.is_unsigned_int(dst):
        # Handled by the dtype rule in `_is_nonnegative`, but harmless to state:
        # an unsigned result is never read back as a negative number.
        return True
    if not types.is_int(src):
        # A float source has no bit width to compare against.
        return False
    src_bits, dst_bits = _int_bits(src), _int_bits(dst)
    if src_bits is None or dst_bits is None:
        return False
    # Magnitude bits available in the source, plus the destination's sign bit.
    magnitude_bits = src_bits if types.is_unsigned_int(src) else src_bits - 1
    return dst_bits >= magnitude_bits + 1


def _operands(op, names) -> list | None:
    """The vars ``op`` takes under ``names``, flattening list operands.

    ``None`` when one of them is missing, so that callers give up on the proof.
    """
    operands = []
    for name in names:
        operand = op.inputs.get(name)
        if operand is None:
            return None
        if isinstance(operand, (list, tuple)):
            operands.extend(operand)
        else:
            operands.append(operand)
    return operands


def _is_nonnegative(var, depth: int = 0) -> bool:
    """Whether every element of ``var`` is provably ``>= 0``.

    Conservative: anything the walk does not recognise -- a block input, an op
    that is not in the tables above, a chain deeper than
    :data:`MAX_PROOF_DEPTH` -- answers "not provable" rather than "negative".
    """
    if var is None:
        return False

    val = getattr(var, "val", None)
    if val is not None:
        arr = np.asarray(val)
        if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            return False
        return bool(np.all(arr >= 0))

    dtype = getattr(var, "dtype", None)
    if dtype is not None and (types.is_unsigned_int(dtype) or types.is_bool(dtype)):
        return True

    if depth >= MAX_PROOF_DEPTH:
        return False

    op = getattr(var, "op", None)
    if op is None:
        return False

    if op.op_type in _ALWAYS_NONNEGATIVE:
        return True

    if op.op_type == "cast":
        return _cast_preserves_nonnegativity(op.x.dtype, var.dtype) and \
            _is_nonnegative(op.x, depth + 1)

    names = _ALL_OPERANDS_NONNEGATIVE.get(op.op_type)
    if names is not None:
        operands = _operands(op, names)
        return operands is not None and all(_is_nonnegative(x, depth + 1) for x in operands)

    names = _ANY_OPERAND_NONNEGATIVE.get(op.op_type)
    if names is not None:
        operands = _operands(op, names)
        return operands is not None and any(_is_nonnegative(x, depth + 1) for x in operands)

    return False


def _guarded_value(op):
    """The value a ``select`` guards against being negative, or ``None``.

    Matches the shape coremltools' ``guard_negative_gather_indices`` emits,
    ``select(cond=greater_equal(v, 0), a=v, b=<wrapped v>)``, whose result is
    ``v`` itself as soon as ``v`` is non-negative. The result has to *be* ``v``
    and not a broadcast of it: ``b`` and ``cond`` take part in the select's
    broadcasting, so a value that only covers part of the result is rejected
    (which is also what an unprovable symbolic dimension amounts to).
    """
    if op.op_type != "select":
        return None

    guarded, cond = op.inputs.get("a"), op.inputs.get("cond")
    if guarded is None or cond is None:
        return None

    cond_op = getattr(cond, "op", None)
    if cond_op is None or cond_op.op_type != "greater_equal" or cond_op.x is not guarded:
        return None
    if uniform_scalar_value(cond_op.y) != 0.0:
        return None

    out = op.outputs[0]
    if guarded.dtype != out.dtype or not shapes_equal(guarded.shape, out.shape):
        return None
    return guarded


@register_pass(namespace="common")
class remove_nonnegative_index_guard(RewritePass):
    """
    Remove a ``select`` that guards a value against being negative when the
    value provably cannot be.

    This is the guard coremltools' ``guard_negative_gather_indices`` puts on the
    indices of every ``gather``/``gather_nd``, so that the iOS17 ops (which treat
    a negative index as out of bounds) keep wrapping them the way the earlier
    ones did. The indices the converter hands to a gather are clamped into
    ``[0, size - 1]`` already, so the wrap is dead code.

    A guard is removed when all of the following hold:

    1. The ``select`` has the shape the guard emits: ``cond`` is
       ``greater_equal(a, 0)`` on the very same var the select passes through as
       ``a``.
    2. That var is provably non-negative: it is a non-negative constant, has an
       unsigned type, or is produced by a chain of ops that cannot make a
       non-negative value negative (``maximum(x, 0)`` and the ``minimum`` that
       completes the converter's clamp, data movement such as ``gather`` or
       ``reshape``, widening ``cast``s, and the always-non-negative ``non_zero``
       and ``shape``).
    3. Passing the var through in place of the select does not change the shape
       (a symbolic dimension that cannot be proven equal counts as a change) or
       the dtype of the result.

    Given:
        %1 = maximum(x=%0, y=0)
        %2 = minimum(x=%1, y=7)
        %3 = greater_equal(x=%2, y=0)
        %4 = add(x=%2, y=8)
        %5 = select(cond=%3, a=%2, b=%4)
        %6 = gather(x=%data, indices=%5, axis=0)

    Result:
        %1 = maximum(x=%0, y=0)
        %2 = minimum(x=%1, y=7)
        %6 = gather(x=%data, indices=%2, axis=0)
    """

    _REWRITES = "index guard(s)"

    def visit(self, op, block) -> bool:
        guarded = _guarded_value(op)
        if guarded is None or not _is_nonnegative(guarded):
            return False

        out = op.outputs[0]
        # A block output would have to hand its name over to the replacement,
        # which coremltools refuses to do for a function input. Guarded gather
        # indices are never a block output, so there is nothing to gain here.
        if out in block.outputs:
            return False

        # `try_...` rather than the unguarded variant: the guarded value may
        # descend from a var coremltools refuses to replace (a `constexpr_*`
        # weight, say), in which case the rewrite is skipped instead of raising.
        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out, new_var=guarded):
            return False
        op.remove_from_block()
        return True
