from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
from jaxlib.mlir.dialects.stablehlo import (
    AddOp, MulOp, MinOp, MaxOp, OrOp, AndOp, ReturnOp, SubtractOp, DivOp
)

from .utils import (
    index_by_slices, update_tensor_by_slice, iterate_indexes_in_shapes,
    get_numpy_type, dtype_str
)
from .translation_context import TranslationContext


def match_computation(hlo_body):
    if len(hlo_body.blocks) != 1:
        return None, None, None
    args = list(hlo_body.blocks[0].arguments)
    ops = list(hlo_body.blocks[0].operations)

    # Check for the special "update" mode (overwrite)
    # This corresponds to returning the second argument (the update value)
    if len(ops) == 1 and isinstance(ops[0], ReturnOp) and ops[0].operands[0] == args[1]:
        # This is the "update" mode: return args[1] (the update value)
        # We define a lambda that just returns the update value
        def mil_binary_op(x, y):
            return y
        mode = "update"
        return None, mil_binary_op, mode

    # Simple matches are where the `hlo_body` is on the form
    #   return _generic_reduction_op_type_(`args`)
    # In that case, if MIL has an equivalent of `_generic_reduction_op_`, we simply delegate to that
    simple_matches = {
        MaxOp: (mb.reduce_max, mb.maximum, "max"),
        MinOp: (mb.reduce_min, mb.minimum, "min"),
        AddOp: (mb.reduce_sum, mb.add, "add"),
        MulOp: (mb.reduce_prod, mb.mul, "mul"),
        SubtractOp: (None, mb.sub, "sub"),
        DivOp: (None, mb.real_div, "div"),
        OrOp: (None, mb.logical_or, "or"),
        AndOp: (None, mb.logical_and, "and"),
    }

    for generic_reduce_op_type, mil_equivalents in simple_matches.items():
        if len(ops) == 2 and isinstance(ops[0], generic_reduce_op_type) and isinstance(ops[1], ReturnOp):
            if list(ops[0].operands) == args and list(ops[1].operands) == list(ops[0].results):
                return mil_equivalents

    return None, None, None


def match_simple_reduce_window(body, inputs, init_values, window_dimensions, window_strides):
    _, _, mode = match_computation(body)
    if mode not in ("min", "max", "add"):
        return None
    if len(inputs) != 1:
        return None

    window_dimensions = list(window_dimensions)
    window_strides = list(window_strides)

    x = inputs[0]
    rank = len(window_dimensions)

    # CoreML pool/conv instructions support up to 3 spatial dimensions for sliding windows.
    spatial_rank = 0
    for i in range(rank):
        if window_dimensions[i] > 1 or window_strides[i] > 1:
            spatial_rank = rank - i
            break

    if spatial_rank == 0:
        return x

    if spatial_rank > 3:
        return None

    spatial_kernel = window_dimensions[rank - spatial_rank:rank]
    spatial_strides = window_strides[rank - spatial_rank:rank]

    x_shape = mb.shape(x=x)

    # Flatten all leading dimensions into a single batch dimension, and set channel dimension to 1.
    b_shape = np.array([-1], dtype=np.int32)
    c_shape = np.array([1], dtype=np.int32)
    s_shape = mb.slice_by_size(x=x_shape, begin=[rank - spatial_rank], size=[spatial_rank])
    new_shape = mb.concat(values=[b_shape, c_shape, s_shape], axis=0)

    x_reshaped = mb.reshape(x=x, shape=new_shape)

    # mb.max_pool and mb.conv do not support integer types.
    x_dtype = x.dtype
    is_int_or_bool = types.is_int(x_dtype) or types.is_bool(x_dtype)
    if is_int_or_bool:
        x_reshaped = mb.cast(x=x_reshaped, dtype="fp32")

    match mode:
        case "max":
            pool_res = mb.max_pool(x=x_reshaped, kernel_sizes=spatial_kernel, strides=spatial_strides, pad_type="valid")
        case "min":
            x_neg = mb.sub(x=0.0, y=x_reshaped)
            pool_res = mb.max_pool(x=x_neg, kernel_sizes=spatial_kernel, strides=spatial_strides, pad_type="valid")
            pool_res = mb.sub(x=0.0, y=pool_res)
        case "add":
            weight = np.ones((1, 1) + tuple(spatial_kernel), dtype=np.float32 if is_int_or_bool else get_numpy_type(x))
            pool_res = mb.conv(x=x_reshaped, weight=weight, strides=spatial_strides, pad_type="valid")
        case _:
            return None

    if is_int_or_bool:
        pool_res = mb.cast(x=mb.round(x=pool_res), dtype=dtype_str(x_dtype))

    # Reshape back to the output dims mapping
    if rank == spatial_rank:
        out_shape = mb.slice_by_size(x=mb.shape(x=pool_res), begin=[2], size=[spatial_rank])
    else:
        leading_shape = mb.slice_by_size(x=x_shape, begin=[0], size=[rank - spatial_rank])
        pool_res_shape = mb.shape(x=pool_res)
        out_spatial_shape = mb.slice_by_size(x=pool_res_shape, begin=[2], size=[spatial_rank])
        out_shape = mb.concat(values=[leading_shape, out_spatial_shape], axis=0)

    final_res = mb.reshape(x=pool_res, shape=out_shape)

    return final_res


def compute_reduction(converter, context: TranslationContext, inputs, dimensions, body, init_values, result_types):
    mil_reduction, mil_single_reduction, mode = match_computation(body)
    if mil_reduction and mil_single_reduction and len(inputs) == 1:
        res = mil_reduction(x=inputs[0], axes=np.array(dimensions, dtype=np.int32))
        # Handle initial value
        res = mil_single_reduction(x=res, y=init_values[0])
        return [res]

    # Boolean Or/And: MIL has no reduce_or/reduce_and, so lower via
    # cast(bool→int32) → reduce_max/reduce_min → cast(→bool) → logical combine.
    # Skip this fast path when the input is a compile-time constant, because
    # the explicit ops propagate constness into downstream operations, causing
    # the MIL const_elimination pass to aggressively fold large subgraphs.
    # The while-loop fallback naturally breaks this const chain.
    if mode in ("or", "and") and len(inputs) == 1 and types.is_bool(inputs[0].dtype) \
            and not inputs[0].can_be_folded_to_const():
        axes = np.array(dimensions, dtype=np.int32)
        x_int = mb.cast(x=inputs[0], dtype="int32")
        if mode == "or":
            res = mb.reduce_max(x=x_int, axes=axes)
        else:
            res = mb.reduce_min(x=x_int, axes=axes)
        res = mb.cast(x=res, dtype="bool")
        res = mil_single_reduction(x=res, y=init_values[0])
        return [res]

    # Fall back to loop implementation
    logger.warning("Falling back to while-loop implementation for reduction. This may be slower than expected!")

    input_rank = len(inputs[0].shape)
    # Notice for the loops we treat both `reduce_shape` and `result_shape` as being
    # of the input rank. This is to make computing element indexes easier.
    # When updating the result, we later pick out just the result indices
    # we care about in the actual result.
    reduce_shape = [inputs[0].shape[dim] if dim in dimensions else 1 for dim in range(input_rank)]
    result_shape = [inputs[0].shape[dim] if dim not in dimensions else 1 for dim in range(input_rank)]

    def compute_reduction_loop(result_idx, *partial_results):
        def compute_inner(element_idx, *acc):
            element_idx = mb.add(x=result_idx, y=element_idx)
            elements = [mb.reshape(x=index_by_slices(input, [element_idx]), shape=(1,)) for input in inputs]

            args = list(acc) + elements
            hlo_params = list(body.blocks[0].arguments)
            outputs = converter.invoke_hlo_function(context, "reduce_body", hlo_params, body, args)

            return outputs

        reduction_results = iterate_indexes_in_shapes(compute_inner, [reduce_shape], init_values)

        # The result rank is likely less than the input shape.
        # We need to pick the indexes in the result shape we want to update
        result_indices = [dim for dim in range(input_rank) if dim not in dimensions]
        if len(result_indices) != 0:
            result_idx = [mb.gather(x=result_idx, indices=result_indices)]
        else:
            result_idx = []

        return [
            update_tensor_by_slice(acc, result_idx, result)
            for acc, result in zip(partial_results, reduction_results)
        ]

    mil_results = [
        np.zeros(result_type.shape, dtype=get_numpy_type(result_type))
        if len(result_type.shape) == 0 else
        mb.tile(x=np.zeros((1,) * len(result_type.shape), dtype=get_numpy_type(result_type)), reps=result_type.shape)
        for result_type in result_types
    ]
    mil_results = iterate_indexes_in_shapes(compute_reduction_loop, [result_shape], mil_results, unroll_limit=5)
    return mil_results


def compute_windowed_reduction(
    converter,
    context: TranslationContext,
    inputs,
    window_dimensions,
    window_strides,
    body,
    init_values,
    result_types
):
    def move_axis_last(arr, axis):
        permutation = list(range(len(arr.shape)))
        permutation.append(permutation.pop(axis))
        return mb.transpose(x=arr, perm=permutation)

    # First group all the dimensions being reduced over in a group at the end
    inputs_rank = len(window_dimensions)
    partitioned_inputs = []
    for input in inputs:
        transformed = mb.sliding_windows(
            x=input,
            axis=0,
            size=window_dimensions[0],
            stride=window_strides[0]
        )
        transformed = move_axis_last(transformed, 1)
        for axis in range(1, inputs_rank):
            transformed = mb.sliding_windows(
                x=transformed, axis=axis, size=window_dimensions[axis], stride=window_strides[axis])
            transformed = move_axis_last(transformed, axis + 1)
            # Contract the two last dimensions into one
            transformed_rank = len(transformed.shape)
            new_shape = mb.concat(values=[
                mb.slice_by_size(x=mb.shape(x=transformed), begin=[0], size=[transformed_rank - 2]),
                np.array([-1], dtype=np.int32)
            ], axis=0)
            transformed = mb.reshape(x=transformed, shape=new_shape)
        partitioned_inputs.append(transformed)

    # Then use the normal reduce implementation to compute the result
    reduction_dimension = len(partitioned_inputs[0].shape) - 1
    reduction_results = compute_reduction(
        converter=converter,
        context=context,
        inputs=partitioned_inputs,
        dimensions=[reduction_dimension],
        body=body,
        init_values=init_values,
        result_types=result_types,
    )
    return reduction_results
