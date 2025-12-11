from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
import numpy as np
from jaxlib.mlir.dialects.stablehlo import (
    AddOp, MulOp, MinOp, MaxOp, ReturnOp, SubtractOp, DivOp
)

from .utils import (
    index_by_slices, update_tensor_by_slice, iterate_indexes_in_shapes,
    get_numpy_type
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
    }

    for generic_reduce_op_type, mil_equivalents in simple_matches.items():
        if len(ops) == 2 and isinstance(ops[0], generic_reduce_op_type) and isinstance(ops[1], ReturnOp):
            if list(ops[0].operands) == args and list(ops[1].operands) == list(ops[0].results):
                return mil_equivalents

    return None, None, None


def compute_reduction(converter, context: TranslationContext, inputs, dimensions, body, init_values, result_types):
    mil_reduction, mil_single_reduction, _ = match_computation(body)
    if mil_reduction and mil_single_reduction and len(inputs) == 1:
        res = mil_reduction(x=inputs[0], axes=np.array(dimensions, dtype=np.int32))
        # Handle initial value
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
        mb.tile(
            x=np.zeros((1,) * len(result_type.shape), dtype=get_numpy_type(result_type)),
            reps=result_type.shape
        ) for result_type in result_types
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
