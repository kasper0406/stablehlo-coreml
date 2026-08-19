"""Shared helpers for the stablehlo-coreml MIL graph passes.

The passes in this package all have to deal with the same handful of problems:

* recognising constants that hold a single repeated value (either a ``const``
  or a ``fill`` op),
* comparing shapes that may contain symbolic (sympy) dimensions,
* and checking that a pattern is safe to rewrite (no consumers outside the
  match).

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
    "is_broadcast_tile",
    "normalize_axis",
    "shapes_equal",
    "sole_consumer",
    "uniform_const_operand",
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


def uniform_const_operand(op):
    """For a binary op, return ``(const_value, other_var)``, or ``None``.

    The pair is only returned when exactly one of ``x``/``y`` is a compile-time
    constant with all elements equal (see :func:`uniform_scalar_value`).
    """
    x, y = op.inputs["x"], op.inputs["y"]
    x_val, y_val = uniform_scalar_value(x), uniform_scalar_value(y)
    if x_val is not None and y_val is None:
        return x_val, y
    if y_val is not None and x_val is None:
        return y_val, x
    return None


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


def is_broadcast_tile(op) -> bool:
    """True if ``op`` is a ``tile`` that only replicates size-1 dimensions.

    A tile of a dimension that is not 1 is *not* a broadcast:
    ``tile([1, 2], reps=[2]) == [1, 2, 1, 2]`` cannot be expressed by implicit
    broadcasting. ``reps`` must be known at compile time and match the input rank.
    """
    if op.op_type != "tile":
        return False
    reps = const_int_list(op.inputs.get("reps"))
    x_shape = op.inputs["x"].shape
    if reps is None or x_shape is None or len(reps) != len(x_shape):
        return False
    # `dim` may be symbolic; only a literal 1 is safe to broadcast.
    return all(rep == 1 or dims_equal(dim, 1) for dim, rep in zip(x_shape, reps))
