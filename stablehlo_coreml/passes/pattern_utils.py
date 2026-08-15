"""Shared helpers for the stablehlo-coreml MIL graph passes.

The passes in this package all have to deal with the same handful of problems:

* recognising constants that hold a single repeated value (either a ``const``
  or a ``fill`` op),
* comparing shapes that may contain symbolic (sympy) dimensions,
* checking that a pattern is safe to rewrite (no consumers outside the match),
* and cleaning up the ops that became dead after a rewrite.

Everything in here is intentionally conservative: whenever a property cannot be
proven (typically because a dimension is symbolic) the helpers report the
"unknown" answer (``None``/``False``) so that callers skip the optimization.
"""

import numpy as np
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

__all__ = [
    "broadcast_shapes",
    "const_int_list",
    "dims_equal",
    "is_large_negative",
    "is_neg_inf",
    "normalize_axis",
    "peel_back",
    "producer_ops",
    "remove_dead_ops",
    "shapes_equal",
    "sole_consumer",
    "uniform_scalar_value",
]


def uniform_scalar_value(var) -> float | None:
    """Return the value of ``var`` if it is a constant with all elements equal.

    Handles both plain constants (``var.val``) and ``fill`` ops whose ``value``
    is such a constant. Returns ``None`` if the value is not known at
    compile time, is empty, or is not uniform.
    """
    if var is None:
        return None

    val = getattr(var, "val", None)
    if val is None:
        op = getattr(var, "op", None)
        if op is not None and op.op_type == "fill":
            return uniform_scalar_value(op.inputs.get("value"))
        return None

    arr = np.asarray(val)
    if arr.size == 0:
        return None
    flat = arr.reshape(-1)
    first = flat[0]
    # NaN never compares equal to itself, so handle it explicitly.
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(first):
        return None
    if arr.size > 1 and not bool(np.all(flat == first)):
        return None
    return float(first)


def is_neg_inf(var) -> bool:
    """True if ``var`` is a uniform constant that acts as negative infinity.

    Values at or below ``-3e38`` (below the fp32 range) are treated as -inf,
    matching how JAX materialises the softmax/masking constants.
    """
    value = uniform_scalar_value(var)
    return value is not None and value <= -3e38


def is_large_negative(var, threshold: float = -1e4) -> bool:
    """True if ``var`` is a uniform constant ``<= threshold`` (e.g. a -1e9 mask fill)."""
    value = uniform_scalar_value(var)
    return value is not None and value <= threshold


def dims_equal(a, b) -> bool:
    """Structural equality of two shape dimensions, symbolic-aware.

    A symbolic dimension is only equal to the very same symbol; a symbolic
    dimension is never considered equal to a concrete one (it might happen to
    take that value at runtime, but we cannot prove it).
    """
    a_symbolic, b_symbolic = is_symbolic(a), is_symbolic(b)
    if a_symbolic != b_symbolic:
        return False
    if a_symbolic:
        return bool(a == b)
    return int(a) == int(b)


def shapes_equal(a, b) -> bool:
    """Structural, symbolic-aware equality of two shapes."""
    if a is None or b is None:
        return False
    a, b = tuple(a), tuple(b)
    if len(a) != len(b):
        return False
    return all(dims_equal(x, y) for x, y in zip(a, b))


def _broadcast_dims(a, b):
    """NumPy broadcast of two dimensions. Returns ``None`` when not provable."""
    a_symbolic, b_symbolic = is_symbolic(a), is_symbolic(b)
    if a_symbolic and b_symbolic:
        return a if bool(a == b) else None
    if a_symbolic:
        # A symbolic dim only broadcasts provably against a literal 1.
        return a if int(b) == 1 else None
    if b_symbolic:
        return b if int(a) == 1 else None
    a, b = int(a), int(b)
    if a == b:
        return a
    if a == 1:
        return b
    if b == 1:
        return a
    return None


def broadcast_shapes(*shapes):
    """NumPy-style broadcast of the given shapes, symbolic-aware.

    Returns the broadcast shape as a tuple, or ``None`` if the shapes are
    incompatible or the result cannot be determined (e.g. a symbolic dimension
    broadcast against a concrete dimension != 1).
    """
    if len(shapes) == 0:
        return ()
    result = tuple(shapes[0])
    for shape in shapes[1:]:
        shape = tuple(shape)
        rank = max(len(result), len(shape))
        lhs = (1,) * (rank - len(result)) + result
        rhs = (1,) * (rank - len(shape)) + shape
        merged = []
        for a, b in zip(lhs, rhs):
            dim = _broadcast_dims(a, b)
            if dim is None:
                return None
            merged.append(dim)
        result = tuple(merged)
    return result


def const_int_list(var) -> list[int] | None:
    """Return ``var``'s value as a flat list of ints, or ``None`` if not a constant."""
    if var is None:
        return None
    val = getattr(var, "val", None)
    if val is None:
        return None
    arr = np.asarray(val).reshape(-1)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.issubdtype(arr.dtype, np.floating):
            return None
        if not bool(np.all(arr == np.round(arr))):
            return None
    return [int(v) for v in arr]


def normalize_axis(axis: int, rank: int) -> int | None:
    """Turn a possibly negative axis into a non-negative one, or ``None`` if out of range."""
    if rank <= 0:
        return None
    axis = int(axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    return axis


def peel_back(var, op_types, max_ops: int | None = None):
    """Walk backward through single-input ops whose type is in ``op_types``.

    Returns ``(var, ops)`` where ``ops`` are the peeled ops in the order they
    were encountered (i.e. from the consumer side towards the producer side).
    """
    ops = []
    while True:
        if max_ops is not None and len(ops) >= max_ops:
            break
        op = getattr(var, "op", None)
        if op is None or op.op_type not in op_types:
            break
        x = op.inputs.get("x")
        if x is None:
            break
        ops.append(op)
        var = x
    return var, ops


def sole_consumer(var, ignored=()):
    """Return the single consumer op of ``var``, or ``None``.

    ``None`` is returned when ``var`` has zero or more than one consumer, or
    when it escapes the pattern by being an output of its enclosing block.
    Consumers whose ``id()`` is in ``ignored`` do not count; callers use that to
    exempt ops they already matched as part of the same pattern.
    """
    if var is None:
        return None
    op = getattr(var, "op", None)
    if op is not None:
        block = op.enclosing_block
        if block is not None and var in block.outputs:
            return None
    child_ops = [child for child in var.child_ops if id(child) not in ignored]
    if len(child_ops) != 1:
        return None
    return child_ops[0]


def producer_ops(var, max_ops: int = 32) -> list:
    """Collect the ops that (transitively) produce ``var``, breadth first.

    Useful to hand :func:`remove_dead_ops` the whole cone behind a rewritten
    value: it only removes the ops that actually became dead, so passing extra
    candidates is harmless.
    """
    collected = []
    seen = set()
    queue = [var]
    while queue and len(collected) < max_ops:
        current = queue.pop(0)
        op = getattr(current, "op", None)
        if op is None or id(op) in seen:
            continue
        seen.add(id(op))
        collected.append(op)
        for value in op.inputs.values():
            if isinstance(value, (list, tuple)):
                queue.extend(value)
            else:
                queue.append(value)
    return collected


def _is_dead(op) -> bool:
    block = op.enclosing_block
    if block is None:
        return False
    for out in op.outputs:
        if len(out.child_ops) > 0:
            return False
        if out in block.outputs:
            return False
    return True


def remove_dead_ops(block, ops) -> int:
    """Remove the ops in ``ops`` that became dead, repeatedly, and return the count.

    ``ops`` is a list of candidate ops (typically the ops that were matched by a
    pattern). Only candidates whose outputs have no remaining consumers and that
    are not block outputs are removed; the removal is repeated until a fixed
    point so that whole chains disappear. Anything left over is picked up by
    coremltools' ``dead_code_elimination`` later in the pipeline.
    """
    candidates = [op for op in ops if op is not None and op.enclosing_block is block]
    removed = 0
    changed = True
    while changed:
        changed = False
        for op in list(candidates):
            if op.enclosing_block is None:
                # Already removed from the graph.
                candidates.remove(op)
                continue
            if _is_dead(op):
                op.remove_from_block()
                candidates.remove(op)
                removed += 1
                changed = True
    return removed
