from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.var import Var

from dataclasses import dataclass
from typing import List
from functools import reduce
import itertools


@dataclass
class ResolvedSliceSpec:
    start_indices: List[int] | Var
    end_indices: List[int] | Var
    strides: List[int]
    shape: List[int]


def index_by_slices(tensor, slice_spec):
    tensor = fix_scalar_tensor(tensor)
    resolved_slices = _resolve_slice_spec(tensor, slice_spec)

    return mb.slice_by_index(
        x=tensor,
        begin=resolved_slices.start_indices,
        end=resolved_slices.end_indices,
        stride=resolved_slices.strides
    )


def update_tensor_by_slice(tensor, slice_spec, value):
    tensor = fix_scalar_tensor(tensor)
    resolved_slices = _resolve_slice_spec(tensor, slice_spec)

    value = mb.reshape(x=value, shape=resolved_slices.shape)
    return mb.slice_update(
        x=tensor,
        update=value,
        begin=resolved_slices.start_indices,
        end=resolved_slices.end_indices,
        stride=resolved_slices.strides
    )


def fix_scalar_tensor(tensor):
    """
    From a numpy scalar type, CoreML will create a rank 0 tensor, which it will
    later struggle to do operations on. We will re-shape it to a rank 1 tensor
    with dimension 1.
    """
    if len(tensor.shape) == 0:
        tensor = mb.reshape(x=tensor, shape=(1,))
    return tensor


def _flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            flat_list += _flatten_list(item)
        else:
            flat_list.append(item)
    return flat_list


def _count_dimensions(slice_spec):
    dim_count = 0
    for spec in slice_spec:
        if isinstance(spec, Var):
            if spec.rank != 1:
                raise ValueError("The Var spec must have rank 1!")
            dim_count += spec.shape[0]
        elif isinstance(spec, type(Ellipsis)):
            raise ValueError("Can not count dimensions for slice spec containing Ellipsis")
        else:
            dim_count += 1
    return dim_count


def _resolve_slice_spec(tensor, slice_spec) -> ResolvedSliceSpec:
    start_indices = []
    end_indices = []
    strides = []
    shape = []

    # We allow the slice_spec to have nested lists. In that case we flatten it
    slice_spec = _flatten_list(slice_spec)
    if len(slice_spec) == 0:
        # Special case for scalar
        slice_spec = [slice(None)]

    tensor_rank = len(tensor.shape)
    contains_var_type = False
    dim_counter = 0
    for i, spec in enumerate(slice_spec):
        if isinstance(spec, type(slice(None))):
            start_indices.append(spec.start or 0)
            end_indices.append(spec.stop or tensor.shape[dim_counter])
            strides.append(spec.step or 1)
            shape.append(end_indices[-1] - start_indices[-1] // strides[-1])
            dim_counter += 1
        elif isinstance(spec, type(Ellipsis)):
            if any([isinstance(s, type(Ellipsis)) for s in slice_spec[i+1:]]):
                raise ValueError("Only supports one ellipsis when indexing")

            dims_before = dim_counter
            dims_after = _count_dimensions(slice_spec[i+1:])
            num_ellipsis_dims = tensor_rank - (dims_before + dims_after)

            ellipsis_starts = [0] * num_ellipsis_dims
            ellipsis_ends = [tensor.shape[dim] for dim in range(dim_counter, dim_counter + num_ellipsis_dims)]
            ellipsis_strides = [1] * num_ellipsis_dims
            ellipsis_shape = [
                (end - start) // stride for start, end, stride
                in zip(ellipsis_starts, ellipsis_ends, ellipsis_strides)
            ]

            start_indices += ellipsis_starts
            end_indices += ellipsis_ends
            strides += ellipsis_strides
            shape += ellipsis_shape

            dim_counter += num_ellipsis_dims
        elif isinstance(spec, Var):
            if spec.rank != 1:
                raise ValueError("The Var spec must have rank 1!")
            contains_var_type = True
            start_indices.append(spec)
            end_indices.append(mb.add(x=spec, y=1))
            strides += [1] * spec.shape[0]
            shape += [1] * spec.shape[0]
            dim_counter += spec.shape[0]
        else:
            # Assume it is an integer index
            idx = int(spec)
            start_indices.append(idx)
            end_indices.append(idx + 1)
            strides.append(1)
            shape.append(1)
            dim_counter += 1

    # If slice_spec contained any Var types, we will need to concatenate the full result
    # to be one big Var type
    if contains_var_type:
        def partition_list(lst):
            parts = [[]]
            for element in lst:
                if isinstance(element, Var):
                    if len(parts[-1]) == 0:
                        parts.pop()
                    parts.append(element)
                    parts.append([])
                else:
                    parts[-1].append(element)
            if len(parts[-1]) == 0:
                # The last partition may be empty, if so we skip it
                return parts[:-1]
            return parts

        def concat_to_var(lst):
            parts = partition_list(lst)
            return mb.concat(values=parts, axis=0)

        start_indices = concat_to_var(start_indices)
        end_indices = concat_to_var(end_indices)

    if len(strides) != tensor_rank or len(shape) != tensor_rank:
        raise ValueError("Slice does not line up!")

    return ResolvedSliceSpec(
        start_indices=start_indices,
        end_indices=end_indices,
        strides=strides,
        shape=shape,
    )


def iterate_indexes_in_shapes(func, shapes: List, init_values: List, unroll_limit: int = 25):
    """
    Given a list of `shapes`, fx [(3, 2, 3), (5, 2, 3)] this method will iterate
    the product of all valid indexes into the given shapes.
    The function `func: Idx1, Idx2, ..., Idxn, Acc1, ..., Acck -> Res1, ..., Resk` is expected to given the
    list of indexes and the accumulated result so far, to update the result based
    on the index.
    The init_values is a list of [InitVal1, ..., InitValk], and the function `func`
    must return a list of `k` values [Res1, ..., Resk].
    The indexes, Idx1, ..., Idn, may be either a mil mb.Var type of a python
    tuple depending on if the loop is unrolled or not. `func` is expected to be
    able to handle this.

    If the total number of traversed indexes is <`unroll_limit`, the loop will be
    fully un-rolled into MIL instructions. Otherwise it will be constructed as
    a dynamic while loop executed at runtime.
    """
    shapes_elements = [reduce(lambda a, b: a * b, shape, 1) for shape in shapes]
    total_iterations = reduce(lambda a, b: a * b, shapes_elements, 1)

    results = init_values
    if total_iterations <= unroll_limit:
        # Fully unroll the loop
        ranges = [itertools.product(*[range(dim) for dim in shape]) for shape in shapes]
        for indexes in itertools.product(*ranges):
            results = func(*indexes, *results)
    else:
        # Dynamically compute the loop
        def suffix_product(lst):
            res = []
            acc = 1
            for i in reversed(lst):
                res.append(acc)
                acc *= i
            return list(reversed(res))
        integer_index_strides = suffix_product(shapes_elements)
        index_strides = [suffix_product(shape) for shape in shapes]

        # Attempt at looping over result indexes without fully unrolling
        def cond(i, *acc):
            return mb.less(x=i, y=total_iterations)

        def body(i, *acc):
            # Split out the index i to an integer index into the individual shapes.
            integer_indexes = [
                mb.mod(x=mb.floor_div(x=i, y=stride), y=elements)
                for stride, elements in zip(integer_index_strides, shapes_elements)
            ]
            # Map the integer index in the shapes to an actual shaped index
            indexes = [
                mb.concat(values=[
                    mb.mod(x=mb.floor_div(x=idx, y=stride), y=dim) for stride, dim in zip(strides, shape)
                ], axis=0)
                if len(shape) > 0 else []
                for idx, strides, shape in zip(integer_indexes, index_strides, shapes)
            ]

            results = func(*indexes, *acc)
            return [mb.add(x=i, y=1)] + results

        # TODO: Fix this in a nicer way!!!
        fixed_init_values = []
        for init_value in init_values:
            if len(init_value.shape) == 0:
                fixed_init_values.append(mb.reshape(x=init_value, shape=(1,)))
            else:
                fixed_init_values.append(init_value)

        results = mb.while_loop(_cond=cond, _body=body, loop_vars=[0] + fixed_init_values)[1:]  # Skip the counter

    return results


def inverse_permutation(perm):
    """
    Given a permutation `perm`, compute the inverse of the permutation
    """
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[j] = i
    return inv
