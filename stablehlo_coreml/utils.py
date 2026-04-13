import itertools
from dataclasses import dataclass
from functools import reduce, wraps

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.var import Var
from jaxlib.mlir import ir


@dataclass
class ResolvedSliceSpec:
    start_indices: list[int] | Var
    end_indices: list[int] | Var
    strides: list[int]
    shape: list[int]


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


def iterate_indexes_in_shapes(func, shapes: list, init_values: list, unroll_limit: int = 25):
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

        fixed_init_values = [fix_scalar_tensor(init_value) for init_value in init_values]
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


def get_mil_type_from_ir(element_type):
    if isinstance(element_type, ir.IntegerType):
        match (element_type.width, element_type.is_unsigned):
            case (32, False):
                return types.int32
            case (32, True):
                return types.uint32
            case (16, False):
                return types.int16
            case (16, True):
                return types.uint16
            case (8, False):
                return types.int8
            case (8, True):
                return types.uint8
            case (1, _):
                return types.bool
    if isinstance(element_type, ir.F16Type):
        return types.fp16
    if isinstance(element_type, ir.F32Type):
        return types.fp32
    raise ValueError(f"Unsupported type {element_type}")


def get_mil_type(obj):
    if isinstance(obj, ir.Type):
        if hasattr(obj, 'element_type'):
            return get_mil_type_from_ir(obj.element_type)
        return get_mil_type_from_ir(obj)
    if isinstance(obj, np.ndarray):
        return types.numpy_type_to_builtin_type(obj.dtype)
    return obj.dtype


def get_numpy_type(obj):
    return types.nptype_from_builtin(get_mil_type(obj))


def dtype_str(type):
    return {
        types.int64: "int64",
        types.uint64: "uint64",
        types.int32: "int32",
        types.uint32: "uint32",
        types.int16: "int16",
        types.uint16: "uint16",
        types.int8: "int8",
        types.uint8: "uint8",
        types.fp64: "fp64",
        types.fp32: "fp32",
        types.fp16: "fp16",
        types.bool: "bool",
    }[type]


def safe_cast_to_int32(array_like, name=""):
    """Cast an array-like to int32, raising ValueError on overflow."""
    arr = np.array(array_like, dtype=np.int64)
    if np.any((arr > np.iinfo(np.int32).max) | (arr < np.iinfo(np.int32).min)):
        raise ValueError(f"{name} array overflows int32 limits: {arr}")
    return arr.astype(np.int32)


def clamp_index(index, shape, size):
    """
    Clamps start indices to ensure they are within bounds: [0, operand_dim - slice_size]
    This is required by the StableHLO specification
    """
    max_start_indices = mb.sub(x=shape, y=size)
    index = mb.minimum(x=index, y=max_start_indices)
    index = mb.maximum(x=index, y=0)
    return index


def range_along_dim(shape, axis, dtype):
    axis = len(shape) + axis if axis < 0 else axis
    vec_shape = [shape[dim] if dim == axis else 1 for dim in range(len(shape))]
    vec_reps = [1 if dim == axis else shape[dim] for dim in range(len(shape))]
    arange = mb.range_1d(start=dtype(0), end=dtype(shape[axis]), step=dtype(1))
    return mb.tile(x=mb.reshape(x=arange, shape=vec_shape), reps=vec_reps)


def auto_cast_bool(target_dtype="int32"):
    """
    Unfortunately scatter/gather operations in CoreML do not support boolean inputs.
    It is fixed by automatically casting boolean inputs to `target_dtype` (e.g. int32)
    before passing them to the operation, and casting the results back to boolean
    if needed.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, context, op):
            class CastingContext:
                def __init__(self, wrapped_context):
                    self._wrapped = wrapped_context

                def __getitem__(self, name):
                    val = self._wrapped[name]
                    if val.dtype == types.bool:
                        return mb.cast(x=val, dtype=target_dtype)
                    return val

                def add_result(self, hlo_result, result):
                    # Check if the expected result type is bool
                    expected_type = get_mil_type_from_ir(hlo_result.type.element_type)
                    if expected_type == types.bool and result.dtype != types.bool:
                        result = mb.cast(x=result, dtype="bool")
                    self._wrapped.add_result(hlo_result, result)

                def __getattr__(self, name):
                    return getattr(self._wrapped, name)

            return func(self, CastingContext(context), op)
        return wrapper
    return decorator
