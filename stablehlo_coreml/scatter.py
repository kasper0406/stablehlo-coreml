"""Static OOB-drop lowering for scatter over complete trailing slices."""
from math import prod

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .utils import clamp_index


def static_scatter_sizes(operand_shape, indices_shape, indexed_rank):
    """Return (flattened front, update rows), or None for symbolic routing dimensions.

    Use Python integers so the eligibility check itself cannot overflow. Trailing
    slice dimensions need not be static: only the routing domain must be known.
    """
    front = operand_shape[:indexed_rank]
    batch = indices_shape[:-1]
    if any(is_symbolic(d) for d in (*front, *batch)):
        return None
    size, rows = prod(int(d) for d in front), prod(int(d) for d in batch)
    if size + rows > np.iinfo(np.int32).max:
        raise ValueError("scatter operand front too large for int32 indexing")
    return size, rows


def scatter_column(x, column, squeeze=False):
    """Slice one last-axis column, retaining arbitrary static or symbolic batches."""
    return mb.slice_by_index(
        x=x, begin=(0,) * (x.rank - 1) + (column,),
        end=(0,) * (x.rank - 1) + (column + 1,),
        end_mask=(True,) * (x.rank - 1) + (False,),
        squeeze_mask=(False,) * (x.rank - 1) + (squeeze,),
    )


def static_scatter_indices(indices, front, rows, dim_mapping):
    """Linearize original indices, assigning each OOB row a unique slot.

    Slice columns directly: add_int16_cast can narrow gather's int32 DATA and
    corrupt large index values before bounds checking, even for identity gathers.
    """
    indices = mb.reshape(x=indices, shape=(rows, len(front)))
    valid, linear = None, None
    for axis, extent in enumerate(front):
        column = dim_mapping.index(axis)
        component = scatter_column(indices, column, squeeze=True)
        in_bounds = mb.logical_and(
            x=mb.greater_equal(x=component, y=0), y=mb.less(x=component, y=extent),
        )
        valid = in_bounds if valid is None else mb.logical_and(x=valid, y=in_bounds)
        # Clamp BEFORE multiplying: even INT_MIN/INT_MAX must not overflow the
        # flattened index expression, including values in the discarded branch.
        component = clamp_index(component, extent, 1)
        linear = component if linear is None else mb.add(x=mb.mul(x=linear, y=extent), y=component)
    dummy = np.arange(prod(front), prod(front) + rows, dtype=np.int32)
    return mb.select(cond=valid, a=linear, b=dummy)


def static_scatter(operand, indices, updates, mode, size, rows):
    """Use the updates themselves as discarded padding; retain operand once."""
    data = mb.reshape(x=operand, shape=(size, -1))
    updates = mb.reshape(x=updates, shape=(rows, -1))
    data = mb.concat(values=(data, updates), axis=0)
    result = mb.scatter(data=data, indices=indices, updates=updates, axis=0, mode=mode)
    result = mb.slice_by_size(x=result, begin=(0, 0), size=(size, -1))
    shape = mb.shape(x=operand) if any(is_symbolic(d) for d in operand.shape) else operand.shape
    return mb.reshape(x=result, shape=shape)
