from coremltools.converters.mil.mil import Builder as mb

from dataclasses import dataclass
from typing import List

@dataclass
class ResolvedSliceSpec:
    start_indices: List[int]
    end_indices: List[int]
    strides: List[int]

def index_by_slices(tensor, slice_spec):
    resolved_slices = _resolve_slice_spec(tensor, slice_spec)

    return mb.slice_by_index(
        x=tensor,
        begin=resolved_slices.start_indices,
        end=resolved_slices.end_indices,
        stride=resolved_slices.strides
    )

def update_tensor_by_slice(tensor, slice_spec, value):
    resolved_slices = _resolve_slice_spec(tensor, slice_spec)

    reshaped_value = [
        (end - start) // stride for start, end, stride
        in zip(resolved_slices.start_indices, resolved_slices.end_indices, resolved_slices.strides)
    ]
    value = mb.reshape(x=value, shape=reshaped_value)
    return mb.slice_update(
        x=tensor,
        update=value,
        begin=resolved_slices.start_indices,
        end=resolved_slices.end_indices,
        stride=resolved_slices.strides
    )

def _resolve_slice_spec(tensor, slice_spec) -> ResolvedSliceSpec:
    start_indices = []
    end_indices = []
    strides = []

    def value_or_default(val, default):
        if val:
            return val
        return default

    if len(slice_spec) == 0:
        # Special case for scalar
        slice_spec = [ slice(None) ]

    tensor_rank = len(tensor.shape)
    ellipsis_encountered = False
    dim_counter = 0
    for i, spec in enumerate(slice_spec):
        if isinstance(spec, type(slice(None))):
            start_indices.append(value_or_default(spec.start, 0))
            end_indices.append(value_or_default(spec.stop, tensor.shape[dim_counter]))
            strides.append(value_or_default(spec.step, 1))
            dim_counter += 1
        elif isinstance(spec, type(Ellipsis)):
            if ellipsis_encountered:
                raise ValueError("Only supports one ellipsis when indexing")
            ellipsis_encountered = True

            dims_before = i
            dims_after = len(slice_spec) - i - 1
            num_ellipsis_dims = tensor_rank - (dims_before + dims_after)

            ellipsis_starts = [ 0 ] * num_ellipsis_dims
            ellipsis_ends = [ tensor.shape[dim] for dim in range(i, i + num_ellipsis_dims) ]
            ellipsis_strides = [ 1 ] * num_ellipsis_dims

            start_indices += ellipsis_starts
            end_indices += ellipsis_ends
            strides += ellipsis_strides

            dim_counter += num_ellipsis_dims
        else:
            # Assume it is an integer index
            idx = int(spec)
            start_indices.append(idx)
            end_indices.append(idx + 1)
            strides.append(1)
            dim_counter += 1
    
    if len(start_indices) != tensor_rank:
        raise ValueError("Slice does not line up!")
    
    return ResolvedSliceSpec(
        start_indices=start_indices,
        end_indices=end_indices,
        strides=strides,
    )
