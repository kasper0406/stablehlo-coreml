"""MIL pass: drop ``slice_update`` ops that overwrite the whole destination tensor.

The converter builds several results by allocating a buffer and writing into it
with ``slice_update`` (``op_dynamic_update_slice``, and the accumulator loops in
``reductions.py`` / ``DotGeneralOp``). When the written slice happens to cover
the entire buffer the write is just a copy, and the update tensor can be used
directly.
"""

import numpy as np
import sympy as sm
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

from .generated import remove_noop_slice_update_rule as rule


def _const_vector(var, rank, *, kind, default=None):
    """Strictly normalize a rank-length MIL constant vector."""
    if var is None:
        return None if default is None else [default] * rank
    val = getattr(var, "val", None)
    if val is None:
        return None
    array = np.asarray(val)
    if array.ndim != 1 or len(array) != rank:
        return None
    if kind == "int":
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            return None
        return [int(value) for value in array]
    if kind == "bool":
        if not np.issubdtype(array.dtype, np.bool_):
            return None
        return [bool(value) for value in array]
    raise AssertionError(f"unknown constant vector kind {kind!r}")


class _SymbolInterner:
    """Give structurally equal SymPy symbols the same opaque rule identifier."""

    def __init__(self):
        self._symbols = []

    def dim(self, value):
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            value = int(value)
            return rule.Dim("fixed", value) if value > 0 else None
        # Composite SymPy expressions are outside this adapter's trusted input
        # language. MIL's ordinary dynamic dimensions are single Symbols.
        if not isinstance(value, sm.Symbol):
            return None
        for index, known in enumerate(self._symbols):
            if bool(value == known):
                return rule.Dim("symbol", f"s{index}")
        self._symbols.append(value)
        return rule.Dim("symbol", f"s{len(self._symbols) - 1}")


def _dtype_id(var):
    try:
        return types.builtin_to_string(var.dtype)
    except Exception:
        return None


def _normalize_match(op):
    """Translate a well-formed MIL candidate into the generated rule language.

    This adapter is intentionally strict and is part of the trusted boundary:
    malformed attributes, non-constant vectors, empty ranks, and composite
    symbolic shape expressions cause the optimization to be skipped.
    """
    if op.op_type != "slice_update" or len(op.outputs) != 1:
        return None
    variables = (op.x, op.update, op.outputs[0])
    if any(var.shape is None for var in variables):
        return None
    shapes = tuple(tuple(var.shape) for var in variables)
    rank = len(shapes[0])
    if rank == 0 or any(len(shape) != rank for shape in shapes[1:]):
        return None

    begin = _const_vector(op.begin, rank, kind="int")
    stop = _const_vector(op.end, rank, kind="int")
    stride = _const_vector(op.stride, rank, kind="int", default=1)
    begin_mask = _const_vector(op.begin_mask, rank, kind="bool", default=False)
    end_mask = _const_vector(op.end_mask, rank, kind="bool", default=False)
    squeeze_mask = _const_vector(op.squeeze_mask, rank, kind="bool", default=False)
    vectors = (begin, stop, stride, begin_mask, end_mask, squeeze_mask)
    if any(vector is None for vector in vectors):
        return None

    interner = _SymbolInterner()
    normalized_shapes = tuple(tuple(interner.dim(dim) for dim in shape) for shape in shapes)
    if any(dim is None for shape in normalized_shapes for dim in shape):
        return None
    axes = tuple(
        rule.Axis(
            dim=normalized_shapes[0][axis],
            update_dim=normalized_shapes[1][axis],
            output_dim=normalized_shapes[2][axis],
            begin=begin[axis],
            stop=stop[axis],
            stride=stride[axis],
            begin_mask=begin_mask[axis],
            end_mask=end_mask[axis],
            squeeze_mask=squeeze_mask[axis],
        )
        for axis in range(rank)
    )
    dtype_ids = tuple(_dtype_id(var) for var in variables)
    if any(dtype_id is None for dtype_id in dtype_ids):
        return None
    return rule.Match(
        x_dtype=dtype_ids[0],
        update_dtype=dtype_ids[1],
        output_dtype=dtype_ids[2],
        axes=axes,
    )


def _needs_output_name_bridge(slice_update_op, new_var) -> bool:
    """Whether direct replacement would rename an existing value or input."""
    block = slice_update_op.enclosing_block
    out_var = slice_update_op.outputs[0]
    return out_var in block.outputs and new_var.name != out_var.name


def _replace_if_rule_matches(slice_update_op):
    """Check the generated rule immediately before performing its mutation."""
    candidate = _normalize_match(slice_update_op)
    if candidate is None or not rule.matches(candidate):
        return False

    block = slice_update_op.enclosing_block
    out_var = slice_update_op.outputs[0]

    new_var = slice_update_op.update
    bridge_op = None
    if _needs_output_name_bridge(slice_update_op, new_var):
        # Block output replacement transfers the old public name to the new
        # value. A bridge prevents that from renaming a function input or
        # clobbering the name of another output that already uses `new_var`.
        new_var = mb.identity(x=new_var, before_op=slice_update_op)
        bridge_op = new_var.op

    # `update` is an input of the matched op, so MIL SSA guarantees it is visible
    # at the anchor. `try_...` also checks coremltools' replacement restrictions
    # before mutating the block.
    if not block.try_replace_uses_of_var_after_op(
        anchor_op=slice_update_op, old_var=out_var, new_var=new_var
    ):
        # A failed `try_replace` is pre-mutation. Remove the unused bridge we
        # just inserted so a skipped optimization leaves the graph unchanged.
        if bridge_op is not None and bridge_op.enclosing_block is block:
            bridge_out = bridge_op.outputs[0]
            if len(bridge_out.child_ops) == 0 and bridge_out not in block.outputs:
                bridge_op.remove_from_block()
        return False
    slice_update_op.remove_from_block()
    return True


@block_context_manager
def _remove_noop_slice_update(block):
    did_optimize = False
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for b in op.blocks:
            did_optimize |= _remove_noop_slice_update(b)
        if len(op.blocks) > 0:
            continue

        if _replace_if_rule_matches(op):
            did_optimize = True
    return did_optimize


@register_pass(namespace="common")
class remove_noop_slice_update(AbstractGraphPass):
    """
    If a slice_update is called on the full tensor with an update of the same shape,
    simply use the update tensor going forward.

    This optimization is very useful for the way the HLO DotGeneralOp is implemented,
    in case the DotGeneralOp reduces to a single matrix multiplication.

    Given:
        %1 = <buffer tensor of shape S>
        %2 = <update tensor of shape S>
        %2 = slice_update(x=%buffer, update=%2, begin=[0] * rank(%1), end=S, stride=[1] * rank(%1))
        %3 = some_op(%2)

    Result:
        %1 = <tensor of shape S>
        %3 = some_op(%1)
        ...

    A symbolic dimension of S is only ever covered through `end_mask`, since a
    constant `end` is never provably equal to a symbol.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _remove_noop_slice_update(f)
