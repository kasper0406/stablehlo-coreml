from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.ops.defs._utils import (
    promote_input_dtypes,
)
from .utils import (
    index_by_slices, update_tensor_by_slice, iterate_indexes_in_shapes,
    inverse_permutation, get_mil_type, dtype_str, get_mil_type_from_ir, get_numpy_type,
    clamp_index, range_along_dim
)
from .passes.utils import register_optimizations
from .translation_context import TranslationContext
from .ops_register import StableHloOpsRegistry, register_stablehlo_op
from .sort_utils import match_sort, stable_argsort
from .reductions import (
    compute_reduction, compute_windowed_reduction, match_computation
)
from .padding import pad_with_cast

from jaxlib.mlir import ir
from jaxlib.mlir.dialects.func import FuncOp, CallOp, ReturnOp as FuncReturnOp
from jaxlib.mlir.dialects.stablehlo import (
    AddOp, SubtractOp, MulOp, DivOp, NegOp, SignOp, AbsOp, ExpOp, Expm1Op, LogOp,
    Log1pOp, SqrtOp, ConstantOp, DotGeneralOp, ReshapeOp, BroadcastInDimOp, WhileOp,
    CompareOp, ConvertOp, SelectOp, DynamicSliceOp, ReturnOp, ConvolutionOp, MinOp,
    MaxOp, RsqrtOp, TanhOp, SineOp, CosineOp, TanOp, Atan2Op, ConcatenateOp, TransposeOp,
    DynamicUpdateSliceOp, SliceOp, CustomCallOp, IotaOp, ReduceOp, ReduceWindowOp,
    OrOp, AndOp, NotOp, ReverseOp, IsFiniteOp, GatherOp, PowOp, PadOp, RemOp,
    ScatterOp, FloorOp, CeilOp, SortOp, ClampOp, CaseOp,
)
from jaxlib.mlir.dialects.mhlo import (TopKOp, AsinOp, AcosOp, SinhOp, CoshOp, AsinhOp, AcoshOp, AtanhOp)
from jax._src.lib.mlir.dialects import hlo

import numpy as np

from typing import List, Optional
from functools import partial, reduce


def convert(module, minimum_deployment_target: AvailableTarget):
    if minimum_deployment_target < AvailableTarget.iOS18:
        raise ValueError("Converting to <iOS18 is not supported")

    register_optimizations()

    converter = StableHloConverter(opset_version=minimum_deployment_target)
    return converter.convert(module)


class StableHloConverter(metaclass=StableHloOpsRegistry):

    def __init__(self, opset_version: Optional[int] = None):
        self.opset_version = AvailableTarget(opset_version) if opset_version is not None else None
        self.prog = mil.Program()
        self.func_index = {}

    def convert(self, module: ir.Module) -> Program:
        logger.info("Converting graph.")

        # Build function index to resolve/inline HLO function calls
        for func in module.body:
            self.func_index[func.name.value] = func

        for func in module.body:
            if func.sym_visibility is None or "public" == func.sym_visibility.value:
                self.build_func(func)

        return self.prog

    def build_func(self, hlo_func: FuncOp):
        context = TranslationContext()  # Map from results to created variables

        func_inputs = {}
        for arg in hlo_func.arguments:
            shape = arg.type.shape
            if shape == []:
                shape = [1]

            func_inputs[arg.get_name()] = mb.placeholder(
                shape=shape, dtype=get_mil_type_from_ir(arg.type.element_type)
            )

        with Function(func_inputs, opset_version=self.opset_version) as ssa_func:
            for name in func_inputs.keys():
                context.add_variable(name, ssa_func.inputs[name])

            ssa_func.set_outputs(self.process_block(context, hlo_func.body.blocks[0]))
            self.prog.add_function(hlo_func.name.value, ssa_func)

    def process_block(self, context: TranslationContext, block: ir.Block):
        outputs = None
        for op in block:
            # Convention: Only the "return" op is returning from its building function
            # TODO: Check that "return" is always the last node!
            ret = self.dispatch_op(self, context, op)
            if ret is not None:
                if outputs is not None:
                    raise ValueError("More than 1 return op in block!")
                outputs = ret
        return outputs

    @register_stablehlo_op
    def op_call(self, context: TranslationContext, op: CallOp):
        # We can not do function calls in MIL, so we have to inline the function

        # Get the argument mapping prior to entering the function context
        context_args = []

        for arg in op.operands:
            context_args.append(context[arg.get_name()])

        func_name = op.callee.value
        hlo_func = self.func_index[op.callee.value]
        params = hlo_func.arguments
        outputs = self.invoke_hlo_function(context, func_name, params, hlo_func.body, context_args)

        # Configure return value
        for result, output in zip(op.results, outputs):
            context.add_result(result, output)

    @register_stablehlo_op
    def op_return(self, context: TranslationContext, op: ReturnOp):
        return [context[result.get_name()] for result in op.operands]

    @register_stablehlo_op
    def op_func_return(self, context: TranslationContext, op: FuncReturnOp):
        # The HLO / MLIR types for function return ops seem to be both in use
        # The behaviour and fields of the two types should be similar, so we
        # simply delegate to the HLO version
        return self.op_return(context, op)

    @register_stablehlo_op
    def op_add(self, context: TranslationContext, op: AddOp):
        self.__simple_binary_op(context, mb.add, op)

    @register_stablehlo_op
    def op_or(self, context: TranslationContext, op: OrOp):
        self.__simple_binary_op(context, mb.logical_or, op)

    @register_stablehlo_op
    def op_and(self, context: TranslationContext, op: AndOp):
        self.__simple_binary_op(context, mb.logical_and, op)

    @register_stablehlo_op
    def op_not(self, context: TranslationContext, op: NotOp):
        self.__simple_unary_op(context, mb.logical_not, op)

    @register_stablehlo_op
    def op_subtract(self, context: TranslationContext, op: SubtractOp):
        self.__simple_binary_op(context, mb.sub, op)

    @register_stablehlo_op
    def op_mul(self, context: TranslationContext, op: MulOp):
        self.__simple_binary_op(context, mb.mul, op)

    @register_stablehlo_op
    def op_div(self, context: TranslationContext, op: DivOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]

        # From HLO constraints we know the base-types should line up
        lhs_type = get_mil_type(lhs)
        rhs_type = get_mil_type(rhs)
        if lhs_type != rhs_type:
            raise ValueError(f"Division not supported for different types. lhs type: {lhs_type}, rhs type: {rhs_type}")
        if types.is_complex(lhs_type):
            raise ValueError("Complex numbers are not supported in MIL")

        if types.is_float(lhs_type):
            cml_op = mb.real_div(x=lhs, y=rhs)
        elif types.is_int(lhs_type):
            cml_op = mb.floor_div(x=lhs, y=rhs)
        else:
            raise ValueError(f"Unknown dtype {lhs_type}")

        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_neg(self, context: TranslationContext, op: NegOp):
        # TODO(knielsen): Consider unsigned and more exotic types
        operand = context[op.operand.get_name()]
        minus_one = np.array([-1], dtype=get_numpy_type(operand))
        cml_op = mb.mul(x=minus_one, y=operand)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_sign(self, context: TranslationContext, op: SignOp):
        self.__simple_unary_op(context, mb.sign, op)

    @register_stablehlo_op
    def op_abs(self, context: TranslationContext, op: AbsOp):
        self.__simple_unary_op(context, mb.abs, op)

    @register_stablehlo_op
    def op_log(self, context: TranslationContext, op: LogOp):
        self.__simple_unary_op(context, mb.log, op)

    @register_stablehlo_op
    def op_log1p(self, context: TranslationContext, op: Log1pOp):
        operand = context[op.operand.get_name()]
        one = np.array([1], dtype=get_numpy_type(operand))
        x_plus_one = mb.add(x=one, y=operand)
        cml_op = mb.log(x=x_plus_one)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_exp(self, context: TranslationContext, op: ExpOp):
        self.__simple_unary_op(context, mb.exp, op)

    @register_stablehlo_op
    def op_pow(self, context: TranslationContext, op: PowOp):
        self.__simple_binary_op(context, mb.pow, op)

    @register_stablehlo_op
    def op_expm1(self, context: TranslationContext, op: Expm1Op):
        operand = context[op.operand.get_name()]
        cml_op = mb.add(x=mb.exp(x=operand), y=-1.0)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_transpose(self, context: TranslationContext, op: TransposeOp):
        operand = context[op.operand.get_name()]
        perm = np.array(op.permutation, dtype=np.int32)
        cml_op = mb.transpose(x=operand, perm=perm)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_pad(self, context: TranslationContext, op: PadOp):
        operand = context[op.operand.get_name()]

        if not np.all(np.array(op.interior_padding) == 0):
            raise ValueError("Interior padding is not supported")

        operand_rank = len(op.operand.type.shape)
        indices = np.arange(2 * operand_rank, dtype=np.int32)
        pad = np.zeros_like(indices)
        pad = mb.scatter_along_axis(
            data=pad,
            indices=indices[::2],
            mode="update",
            updates=np.array(op.edge_padding_low, dtype=np.int32)
        )
        pad = mb.scatter_along_axis(
            data=pad,
            indices=indices[1::2],
            mode="update",
            updates=np.array(op.edge_padding_high, dtype=np.int32)
        )

        cml_padding_value = context[op.padding_value.get_name()]
        cml_op = pad_with_cast(x=operand, pad=pad, mode="constant", constant_val=cml_padding_value)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_sqrt(self, context: TranslationContext, op: SqrtOp):
        self.__simple_unary_op(context, mb.sqrt, op)

    @register_stablehlo_op
    def op_constant(self, context: TranslationContext, op: ConstantOp):
        constant = np.array(op.value)
        constant = np.reshape(constant, op.result.type.shape)
        context.add_result(op.result, constant)

    @register_stablehlo_op
    def op_dot_general(self, context: TranslationContext, op: DotGeneralOp):
        # This roughly follows the steps from https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
        # but uses that we have a matrix multiplication primitive, instead of just a dot-product primitive.
        lhs_rank = len(op.lhs.type.shape)
        rhs_rank = len(op.rhs.type.shape)
        dot_dim_numbers = hlo.DotDimensionNumbers(op.dot_dimension_numbers)

        lhs_contracting_dim = dot_dim_numbers.lhs_contracting_dimensions
        rhs_contracting_dim = dot_dim_numbers.rhs_contracting_dimensions
        lhs_batching_dim = dot_dim_numbers.lhs_batching_dimensions
        rhs_batching_dim = dot_dim_numbers.rhs_batching_dimensions

        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]

        def multiply(lst: List):
            return reduce(lambda a, b: int(a) * int(b), lst, 1)

        def last_column_dot(lhs, rhs):
            # TODO: Figure out if we need to special case broadcasting dims
            return mb.matmul(x=lhs, y=rhs, transpose_y=True)

        # Remark: There is a potential performance optimization here:
        #         If we move the largest result dimensions of the tensor towards
        #         the end of the array, we may save a lot of work when iterating
        #         over the result indexes later, as the last dims will be handled
        #         by matrix multiplication
        lhs_result_dim = [dim for dim in range(lhs_rank) if dim not in lhs_batching_dim + lhs_contracting_dim]
        rhs_result_dim = [dim for dim in range(rhs_rank) if dim not in rhs_batching_dim + rhs_contracting_dim]

        # For both the lhs and rhs, put the dimensions being contracted last
        transposed_lhs = mb.transpose(x=lhs, perm=lhs_batching_dim + lhs_result_dim + lhs_contracting_dim)
        transposed_rhs = mb.transpose(x=rhs, perm=rhs_batching_dim + rhs_result_dim + rhs_contracting_dim)

        # Calculate the result by looping over the contracting dims in order
        result_shape = [lhs.shape[dim] for dim in lhs_batching_dim]
        result_shape += [lhs.shape[dim] for dim in lhs_result_dim]
        result_shape += [rhs.shape[dim] for dim in rhs_result_dim]
        if len(result_shape) == 0:
            # Special case for scalar result
            result_shape = [1]

        # Allocate memory of the correct type for the result
        result_dtype = get_mil_type_from_ir(op.result.type.element_type)
        result = mb.fill(shape=result_shape, value=mb.cast(x=0, dtype=dtype_str(result_dtype)))

        def calculate_result_index(lhs_idx, rhs_idx, acc):
            contracted_element_count = multiply([lhs.shape[dim] for dim in lhs_contracting_dim])
            # print(f"contracted_element_count = {contracted_element_count}")
            batch_selector = tuple([slice(None) for _i in range(len(lhs_batching_dim))])
            batch_shape = tuple([lhs.shape[dim] for dim in lhs_batching_dim])

            # Reshape the lhs and rhs to have all the contracting dimensions in the end.
            # We will always make them have the shape `(batch_shape, last_dim_shape, contraction_count)``
            # where we may have to set `last_dim_shape` to 1, if the dimension does not exist.
            lhs_for_result_idx = index_by_slices(transposed_lhs, list(batch_selector) + [lhs_idx, ...])
            if len(lhs_result_dim) > 0:
                lhs_reshape_shape = batch_shape + (lhs.shape[lhs_result_dim[-1]],) + (contracted_element_count, )
            else:
                lhs_reshape_shape = batch_shape + (1, contracted_element_count)
            contracted_lhs = mb.reshape(x=lhs_for_result_idx, shape=lhs_reshape_shape)

            rhs_for_result_idx = index_by_slices(transposed_rhs, list(batch_selector) + [rhs_idx, ...])
            if len(rhs_result_dim) > 0:
                rhs_reshape_shape = batch_shape + (rhs.shape[rhs_result_dim[-1]],) + (contracted_element_count, )
            else:
                rhs_reshape_shape = batch_shape + (1, contracted_element_count)
            contracted_rhs = mb.reshape(x=rhs_for_result_idx, shape=rhs_reshape_shape)

            # print(f"contracted_lhs shape: {contracted_lhs.shape}")
            # print(f"contracted_rhs shape: {contracted_rhs.shape}")

            idx_result = last_column_dot(contracted_lhs, contracted_rhs)

            # If we added a fake dimension, we will make sure to squeeze it away
            if len(lhs_result_dim) == 0 and len(rhs_result_dim) == 0:
                if len(idx_result.shape) == 2:
                    assert idx_result.shape == (1, 1)
                    # This is a special case, where the result is a scalar of shape (1, 1)
                    # In order to not end up with a 0-rank tensor, we only contract one dimension
                    idx_result = mb.reshape(x=idx_result, shape=(1,))
                else:
                    idx_result = mb.squeeze(x=idx_result, axes=(-1, -2))
            elif len(lhs_result_dim) == 0:
                idx_result = mb.squeeze(x=idx_result, axes=(-2,))
            elif len(rhs_result_dim) == 0:
                idx_result = mb.squeeze(x=idx_result, axes=(-1,))

            # TODO: Consider making this work on iOS<18 by using concatenation
            # We may have to add an extra slice for the skipped dimension
            result_idx = []
            result_idx.append(lhs_idx)
            if len(lhs_result_dim) > 0:
                result_idx.append(slice(None))
            result_idx.append(rhs_idx)
            if len(rhs_result_dim) > 0:
                result_idx.append(slice(None))

            return [update_tensor_by_slice(acc, list(batch_selector) + result_idx, idx_result)]

        # We can utilize that we have a full matrix multiply primitive available, compared to having only
        # a dot-product primitive. Therefore we can avoid iterating over the last dimension in respectively
        # the lhs and rhs tensors
        lhs_shape = [lhs.shape[dim] for dim in lhs_result_dim[:-1]]
        rhs_shape = [rhs.shape[dim] for dim in rhs_result_dim[:-1]]
        # In principle all of the matrix multiplications generated here, could be done in parallel.
        # MIL does not seem to support this.
        # We could try to combine the matrix multiplications when the shapes allow it, but for now
        # we will just loop through them sequentially.
        result, = iterate_indexes_in_shapes(calculate_result_index, [lhs_shape, rhs_shape], [result])

        context.add_result(op.result, result)

    @register_stablehlo_op
    def op_rem(self, context: TranslationContext, op: RemOp):
        self.__simple_binary_op(context, mb.mod, op)

    @register_stablehlo_op
    def op_floor(self, context: TranslationContext, op: FloorOp):
        self.__simple_unary_op(context, mb.floor, op)

    @register_stablehlo_op
    def op_ceil(self, context: TranslationContext, op: CeilOp):
        self.__simple_unary_op(context, mb.ceil, op)

    @register_stablehlo_op
    def op_clamp(self, context: TranslationContext, op: ClampOp):
        min = context[op.min.get_name()]
        max = context[op.max.get_name()]
        operand = context[op.operand.get_name()]
        result = mb.minimum(x=mb.maximum(x=operand, y=min), y=max)
        context.add_result(op.results[0], result)

    @register_stablehlo_op
    def op_sort(self, context: TranslationContext, op: SortOp):
        # StableHLO defines sorting via a comparator region (a small function) that returns true if
        # element A < element B. CoreML, however, uses high-level primitives.
        # To bridge this gap, we must analyze the comparator's structure to reverse-engineer
        # the sorting criteria (which keys to sort by and in what direction).
        inputs = [context[operand.get_name()] for operand in op.inputs]

        if len(op.comparator.blocks) != 1:
            raise ValueError("Unsupported comparator format: must have exactly one block")

        comparator_block = op.comparator.blocks[0]
        return_op = comparator_block.operations[-1]

        if not isinstance(return_op, ReturnOp):
            raise ValueError("Unsupported comparator format: last operation must be a return")

        # We start tracing from the return value of the comparator to understand the logic
        comparator_root = return_op.operands[0].owner.opview
        args = list(comparator_block.arguments)

        # Try to match known sorting patterns
        sort_keys = match_sort(comparator_root, args, inputs)
        if sort_keys is None:
            raise ValueError("Unrecognized comparator format")

        # Apply the sort
        sort_dim, (key, ascending) = op.dimension.value, sort_keys[-1]
        if len(inputs) == 1:
            if types.is_int(key.dtype):
                indices = mb.argsort(x=key, axis=sort_dim, ascending=ascending)
            else:
                # MIL puts NaN as equal to -inf instead of above +inf like StableHLO
                nans = mb.not_equal(x=key, y=key)
                inf_masking_nans = mb.select(cond=nans, a=np.inf, b=key)
                indices = mb.argsort(x=inf_masking_nans, axis=sort_dim, ascending=ascending)
                gathered_key = mb.gather_along_axis(x=inf_masking_nans, indices=indices, axis=sort_dim)
                n_masked = mb.reduce_sum(x=mb.cast(x=nans, dtype="int32"), axes=(sort_dim,), keep_dims=True)
                arange = np.indices(key.shape)[sort_dim]
                nan_mask = mb.greater_equal(x=arange, y=mb.sub(x=key.shape[sort_dim], y=n_masked))
                nan_full = np.full(key.shape, np.nan)
                res = mb.add(x=gathered_key, y=mb.select(cond=nan_mask, a=nan_full, b=0.))
                context.add_result(op.results[0], res)
                return
        else:
            indices = stable_argsort(x=key, axis=sort_dim, ascending=ascending)

        for key, ascending in sort_keys[-2::-1]:
            gathered_key = mb.gather_along_axis(x=key, indices=indices, axis=sort_dim)
            relative_indices = stable_argsort(x=gathered_key, axis=sort_dim, ascending=ascending)
            indices = mb.gather_along_axis(x=indices, indices=relative_indices, axis=sort_dim)

        for i, tensor in enumerate(inputs):
            context.add_result(op.results[i], mb.gather_along_axis(x=tensor, indices=indices, axis=sort_dim))

    @register_stablehlo_op
    def op_case(self, context: TranslationContext, op: CaseOp):
        index = context[op.index.get_name()]

        def params(i):
            closure, args = [], []
            for j in op.branches[i].blocks[0].operations:
                for k in j.operands:
                    if k.get_name() in context.variables[context.path()]:
                        closure.append(k)
                        args.append(context[k.get_name()])
            return (closure, op.branches[i], args)

        def build_branch(i):
            if i == len(op.branches) - 1:
                # Default/Last branch
                return self.invoke_hlo_function(context, "branch_default", *params(i))

            def true_fn():
                return self.invoke_hlo_function(context, f"branch_{i}", *params(i))

            def false_fn():
                return build_branch(i + 1)

            return mb.cond(
                pred=mb.equal(x=index, y=i),
                _true_fn=true_fn,
                _false_fn=false_fn
            )

        results = build_branch(0)
        if not isinstance(results, (list, tuple)):
            results = [results]
        for i, result in enumerate(results):
            context.add_result(op.results[i], result)

    @register_stablehlo_op
    def op_reshape(self, context: TranslationContext, op: ReshapeOp):
        x = context[op.operand.get_name()]
        new_shape = op.result.type.shape
        if len(new_shape) == 0:
            reshape_res = mb.squeeze(x=x)
        else:
            reshape_res = mb.reshape(x=x, shape=new_shape)
        context.add_result(op.result, reshape_res)

    @register_stablehlo_op
    def op_broadcast_in_dim(self, context: TranslationContext, op: BroadcastInDimOp):
        x = context[op.operand.get_name()]

        result_shape = op.result.type.shape
        if len(result_shape) == 0:
            # Cast a scalar shape to a (1,) shape
            result_shape = [1]
        result_shape_rank = len(result_shape)

        reshaped_operand_shape = [1] * result_shape_rank
        for i, op_shape in enumerate(op.operand.type.shape):
            result_idx = op.broadcast_dimensions[i]
            reshaped_operand_shape[result_idx] = op_shape

        x = mb.reshape(x=x, shape=reshaped_operand_shape)

        result_tiling = [1] * result_shape_rank
        for result_dim, current_shape in enumerate(reshaped_operand_shape):
            # Replicate data along dimension `dim` until the result dimension matches
            assert result_shape[result_dim] % current_shape == 0
            result_tiling[result_dim] = result_shape[result_dim] // current_shape
        x = mb.tile(x=x, reps=result_tiling)

        context.add_result(op.result, x)

    @register_stablehlo_op
    def op_while(self, context: TranslationContext, op: WhileOp):
        def cond(*loop_args):
            params = [param for param in op.cond.blocks[0].arguments]
            outputs = self.invoke_hlo_function(context, "while_cond", params, op.cond, loop_args)
            if len(outputs) != 1:
                raise ValueError("The output of while_cond should always be a single boolean!")
            # TODO(knielsen): Add a check that the output is in fact a single boolean value

            return outputs[0]

        def body(*body_args):
            params = [param for param in op.body.blocks[0].arguments]
            return self.invoke_hlo_function(context, "while_body", params, op.body, body_args)

        loop_vars = [context[arg.get_name()] for arg in op.operands]
        while_results = mb.while_loop(_cond=cond, _body=body, loop_vars=loop_vars)

        for result_var, while_result in zip(op.results, while_results):
            context.add_result(result_var, while_result)

    @register_stablehlo_op
    def op_compare(self, context: TranslationContext, op: CompareOp):
        comparison_direction = hlo.ComparisonDirectionAttr(op.comparison_direction).value
        cml_op_builder = {
            "EQ": mb.equal,
            "NE": mb.not_equal,
            "GE": mb.greater_equal,
            "GT": mb.greater,
            "LE": mb.less_equal,
            "LT": mb.less,
        }[comparison_direction]

        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        if types.is_bool(lhs.dtype):
            if comparison_direction == "EQ":
                cml_op = mb.logical_not(x=mb.logical_xor(x=lhs, y=rhs))
            elif comparison_direction == "NE":
                cml_op = mb.logical_xor(x=lhs, y=rhs)
            else:
                raise ValueError(
                    f"Boolean comparison operations other than EQ and NE (such as GT, LT, GE, LE) are not supported! "
                    f"Attempted operation: {comparison_direction}"
                )
        else:
            cml_op = cml_op_builder(x=lhs, y=rhs)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_convert(self, context: TranslationContext, op: ConvertOp):
        x = context[op.operand.get_name()]
        new_dtype = get_mil_type_from_ir(op.result.type.element_type)
        cml_op = mb.cast(x=x, dtype=dtype_str(new_dtype))
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_select(self, context: TranslationContext, op: SelectOp):
        cond = context[op.pred.get_name()]
        a = context[op.on_true.get_name()]
        b = context[op.on_false.get_name()]
        cml_op = mb.select(cond=cond, a=a, b=b)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_dynamic_slice(self, context: TranslationContext, op: DynamicSliceOp):
        x = context[op.operand.get_name()]

        # The HLO DynamicSliceOp gives the start indices as seperate 0-dimensional integer variables
        # We need to convert them to a tensor to be compatible with mb.slice_by_size
        start_idx_variables = [context[i.get_name()] for i in op.start_indices]
        begin = mb.concat(values=start_idx_variables, axis=0)

        # The slice sizes in HLO are given by a signed integer with 64 bits
        # This is not supported by MIL, so we convert it to a MIL int32 type
        # TODO(knielsen): Overflow check?
        sizes = np.array(op.slice_sizes, dtype=np.int32)

        # Clamp start indices to ensure they are within bounds: [0, operand_dim - slice_size]
        # This is required by the StableHLO specification
        shape = mb.shape(x=x)
        begin = clamp_index(begin, shape, sizes)

        cml_op = mb.slice_by_size(x=x, begin=begin, size=sizes)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_slice(self, context: TranslationContext, op: SliceOp):
        x = context[op.operand.get_name()]

        begin = np.array(op.start_indices, dtype=np.int32)
        end = np.array(op.limit_indices, dtype=np.int32)
        stride = np.array(op.strides, dtype=np.int32)

        cml_op = mb.slice_by_index(
            x=x,
            begin=begin,
            end=end,
            stride=stride,
        )
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_dynamic_update_slice(self, context: TranslationContext, op: DynamicUpdateSliceOp):
        x = context[op.operand.get_name()]
        updates = context[op.update.get_name()]

        start_indices = [context[i.get_name()] for i in op.start_indices]
        start_indices = mb.concat(values=start_indices, axis=0)

        # Clamp start indices to ensure they are within bounds: [0, operand_dim - update_dim]
        # This is required by the StableHLO specification
        shape = mb.shape(x=x)
        update_shape = mb.shape(x=updates)
        start_indices = clamp_index(start_indices, shape, update_shape)

        end_indices = mb.add(x=start_indices, y=op.update.type.shape)

        update_res = mb.slice_update(
            x=x,
            update=updates,
            begin=start_indices,
            end=end_indices,
        )
        context.add_result(op.result, update_res)

    @register_stablehlo_op
    def op_convolution(self, context: TranslationContext, op: ConvolutionOp):
        dim_spec = hlo.ConvDimensionNumbers(op.dimension_numbers)
        # TODO(knielsen): It should be possible to remove this batch dimension check, but
        #                 there should be a unit test testing it.
        if dim_spec.input_batch_dimension != 0 or dim_spec.output_batch_dimension != 0:
            raise ValueError(f"Only the first dimension is currently supported for batch dimension. Got {dim_spec}")
        if len(dim_spec.input_spatial_dimensions) > 3 or len(dim_spec.output_spatial_dimensions) > 3:
            raise ValueError("MIL only supports convolutions with dim <= 3")

        if op.batch_group_count.value != 1:
            raise ValueError(f"Only a batch group count of 1 is supported. Got {op.batch_group_count.value}")

        # MIL expects it on the form [input_batch_dimension, input_feature_dimension, spatial_dimensions*]
        input_permutation = [
            dim_spec.input_batch_dimension,
            dim_spec.input_feature_dimension,
            *dim_spec.input_spatial_dimensions
        ]
        x = context[op.lhs.get_name()]  # The inputs comes from vars
        x = mb.transpose(x=x, perm=input_permutation)

        strides = None
        if op.window_strides is not None:
            strides = np.array(op.window_strides, dtype=np.int32)

        kernel_dilation = None
        if op.rhs_dilation is not None:
            kernel_dilation = np.array(op.rhs_dilation, dtype=np.int32)

        groups = op.feature_group_count.value

        # Handle padding
        # TODO(knielsen): Consider moving splat/non-splat handling to some utility
        in_rank = x.rank - 2
        if op.padding is None:
            pad = np.zeros((2 * in_rank), dtype=np.int32)
        elif op.padding.is_splat:
            pad = op.padding.get_splat_value().value * np.ones((2 * in_rank), dtype=np.int32)
        else:
            # We need to reshape the array to a linear array to match MILs expectation
            pad = np.reshape(np.array(op.padding, dtype=np.int32), (2 * in_rank, ))

        # We switch the convolution to a transposed convolution if we have lhs_dilation
        conv_type = mb.conv
        if op.lhs_dilation:
            lhs_dilations = np.array(op.lhs_dilation, dtype=np.int32)
            if np.any(lhs_dilations > 1):
                # This is a transpoed convolution
                if strides is not None:
                    raise ValueError("For a conv with lhs dilation we expect the stride to be not set! "
                                     "Because convolution with input dilation d is equivalent to transposed "
                                     "convolution with stride d.")
                # Convolution with input dilation d is equivalent to transposed convolution with stride d
                strides = lhs_dilations

                output_shape = [op.result.type.shape[dim_spec.output_batch_dimension],
                                op.result.type.shape[dim_spec.output_feature_dimension]]
                for d in dim_spec.output_spatial_dimensions:
                    output_shape.append(op.result.type.shape[d])

                conv_type = partial(
                    mb.conv_transpose,
                    output_shape=output_shape
                )

                # Calculate the padding for the transposed convolution
                # We need to invert the padding: p_transpose = K - 1 - p_original
                # If the target padding is negative, we need to pad the input x
                kernel_spatial_dims = dim_spec.kernel_spatial_dimensions
                raw_weight_shape = context[op.rhs.get_name()].shape
                kernel_sizes = [raw_weight_shape[d] for d in kernel_spatial_dims]

                new_pad_out = []
                pad_in = []

                for i in range(len(kernel_sizes)):
                    k = kernel_sizes[i]
                    s = strides[i]
                    d = kernel_dilation[i] if kernel_dilation is not None else 1
                    k_eff = (k - 1) * d + 1

                    p_low = pad[2*i]
                    p_high = pad[2*i+1]

                    # Target crop
                    t_low = k_eff - 1 - p_low
                    t_high = k_eff - 1 - p_high

                    # Calculate input padding needed to satisfy non-negative crop
                    # pad_in >= ceil(-t / s)
                    pi_low = max(0, (-t_low + s - 1) // s)
                    pi_high = max(0, (-t_high + s - 1) // s)

                    # Calculate output crop
                    po_low = t_low + pi_low * s
                    po_high = t_high + pi_high * s

                    new_pad_out.extend([po_low, po_high])
                    pad_in.extend([pi_low, pi_high])

                pad = np.array(new_pad_out, dtype=np.int32)
                pad_in = np.array(pad_in, dtype=np.int32)

                if np.any(pad_in > 0):
                    # Apply padding to x
                    # x is [batch, channel, spatial...]
                    x_rank = len(x.shape)
                    full_pad_in = np.zeros(2 * x_rank, dtype=np.int32)
                    # Fill spatial padding starting at dimension 2
                    for i in range(len(pad_in)):
                        full_pad_in[4 + i] = pad_in[i]
                    x = pad_with_cast(x=x, pad=full_pad_in)

                if np.any(pad < 0):
                    raise ValueError("The case where the padding turns negative when translating to a "
                                     "transposed convolution is not supported.")

        # The MIL weights should be on form:
        #  - normal convolutions: [output_features, input_features / groups, spatial kernels*]
        #  - transposed convolutions: [input_features, output_features / groups, spatial kernels*]
        weight = context[op.rhs.get_name()]  # The weights are numpy arrays
        weight_permutation = []
        if conv_type == mb.conv:
            weight_permutation = [
                dim_spec.kernel_output_feature_dimension,
                dim_spec.kernel_input_feature_dimension,
                *dim_spec.kernel_spatial_dimensions
            ]
        else:
            weight_permutation = [
                dim_spec.kernel_input_feature_dimension,
                dim_spec.kernel_output_feature_dimension,
                *dim_spec.kernel_spatial_dimensions
            ]
        weight = mb.transpose(x=weight, perm=weight_permutation)

        # TODO(knielsen): Make this check more readable!
        # It is executed for conv transpose
        if conv_type != mb.conv:
            # MIL expects the weights to be reversed along the kernel dimensions
            kernel_dimensions = [i + 2 for i in range(len(weight.shape) - 2)]
            weight = mb.reverse(x=weight, axes=kernel_dimensions)

        cml_conv = conv_type(
            x=x,
            weight=weight,
            strides=strides,
            pad_type="custom",
            pad=pad,
            dilations=kernel_dilation,
            groups=groups,
        )

        # Re-arrange output dimensions to match expectation
        # MIL outputs on the form [batch, features, spatial dims*]
        output_permutation = inverse_permutation([
            dim_spec.output_batch_dimension,
            dim_spec.output_feature_dimension,
            *dim_spec.output_spatial_dimensions
        ])
        cml_conv = mb.transpose(x=cml_conv, perm=output_permutation)

        context.add_result(op.result, cml_conv)

    @register_stablehlo_op
    def op_max(self, context: TranslationContext, op: MaxOp):
        self.__simple_binary_op(context, mb.maximum, op)

    @register_stablehlo_op
    def op_min(self, context: TranslationContext, op: MinOp):
        self.__simple_binary_op(context, mb.minimum, op)

    @register_stablehlo_op
    def op_rsqrt(self, context: TranslationContext, op: RsqrtOp):
        self.__simple_unary_op(context, mb.rsqrt, op)

    @register_stablehlo_op
    def op_tanh(self, context: TranslationContext, op: TanhOp):
        self.__simple_unary_op(context, mb.tanh, op)

    @register_stablehlo_op
    def op_sine(self, context: TranslationContext, op: SineOp):
        self.__simple_unary_op(context, mb.sin, op)

    @register_stablehlo_op
    def op_cosine(self, context: TranslationContext, op: CosineOp):
        self.__simple_unary_op(context, mb.cos, op)

    @register_stablehlo_op
    def op_tan(self, context: TranslationContext, op: TanOp):
        self.__simple_unary_op(context, mb.tan, op)

    @register_stablehlo_op
    def op_atan2(self, context: TranslationContext, op: Atan2Op):
        y = context[op.lhs.get_name()]
        x = context[op.rhs.get_name()]
        # Notice the fraction may be +-inf
        fraction = mb.real_div(x=y, y=x)
        atan2_res = mb.atan(x=fraction)
        # We need to adjust for negative x, based on the sign of y
        atan2_res_adjusted = mb.add(x=atan2_res, y=mb.mul(x=mb.sign(x=y), y=np.pi))
        atan2_res = mb.select(
            cond=mb.less(x=x, y=0.0),
            a=atan2_res_adjusted,
            b=atan2_res,
        )
        context.add_result(op.result, atan2_res)

    @register_stablehlo_op
    def op_concatenate(self, context: TranslationContext, op: ConcatenateOp):
        values = [context[input.get_name()] for input in op.inputs]
        values = promote_input_dtypes(values)
        mil_res = mb.concat(values=values, axis=op.dimension.value)
        context.add_result(op.result, mil_res)

    @register_stablehlo_op
    def op_reverse(self, context: TranslationContext, op: ReverseOp):
        x = context[op.operand.get_name()]
        mil_res = mb.reverse(x=x, axes=np.array(op.dimensions, dtype=np.int32))
        context.add_result(op.result, mil_res)

    @register_stablehlo_op
    def op_isfinite(self, context: TranslationContext, op: IsFiniteOp):
        x = context[op.x.get_name()]
        # All finite numbers will have abs(x) < inf
        infinity = np.array(np.inf, dtype=get_numpy_type(x))
        mil_res = mb.less(x=mb.abs(x=x), y=infinity)
        context.add_result(op.result, mil_res)

    @register_stablehlo_op
    def op_reduce(self, context: TranslationContext, op: ReduceOp):
        # HLO reductions can be arbitrarily complex and defines a custom function
        # specifying the reduction.
        # Unforunately this level of granularity is not supported through MIL.
        # We try to detect some simple cases for reductions mapping to native MIL
        # instructions, and otherwise fall back to a MIL while-loop based implementation.
        inputs = [context[input.get_name()] for input in op.inputs]
        init_values = [context[init_value.get_name()] for init_value in op.init_values]
        result_types = [result.type for result in op.results]

        mil_results = compute_reduction(self, context, inputs, op.dimensions, op.body, init_values, result_types)
        for (res, mil_res) in zip(op.results, mil_results):
            context.add_result(res, mil_res)

    @register_stablehlo_op
    def op_reduce_window(self, context: TranslationContext, op: ReduceWindowOp):
        if op.window_dilations and not np.all(op.window_dilations == 1):
            raise ValueError("Window dilations are currently unsupported for windowed reduce")
        if op.base_dilations and not np.all(op.base_dilations == 1):
            raise ValueError("Base dilations are currently unsupported for windowed reduce")

        inputs_rank = len(op.window_dimensions)
        window_strides = op.window_strides
        if not window_strides:
            window_strides = np.ones((inputs_rank,), dtype=np.int32)

        inputs = [context[input.get_name()] for input in op.inputs]
        init_values = [context[init_value.get_name()] for init_value in op.init_values]

        # Pad the inputs if required
        if op.padding:
            padding = np.reshape(np.array(op.padding, dtype=np.int32), (2 * inputs_rank,))
            inputs = [
                pad_with_cast(x=input, pad=padding, constant_val=mb.reduce_max(x=init_value))
                for input, init_value in zip(inputs, init_values)
            ]

        # Unfortunately CoreML only supports tensors with rank <= 6.
        # Due to the re-shaping and windowing operations inside `__compute_windowed_reduction`, this
        # means the function can not be called with tensors of rank >= 4.
        # To work around this problem, we have to iterate over the leading dimensions not being
        # windowed over, and calculate the result values incrementally.
        fixed_dimensions = []
        reduction_dimensions = []
        for axis in range(inputs_rank):
            if op.window_dimensions[axis] == 1 and window_strides[axis] == 1:
                fixed_dimensions.append(axis)
            else:
                reduction_dimensions.append(axis)
        permutation = fixed_dimensions + reduction_dimensions

        # We will put as few dimensions as possible in the loop_dimensions (i.e. we may
        # choose to put some of the `fixedf_dimensions` inside the reduction itself)
        max_dims = 3
        if len(reduction_dimensions) > max_dims:
            raise ValueError("Due to CoreML's rank <= 5 restriction, it is not supported to reduce on more then 3 dimensions!")
        loop_dimensions = fixed_dimensions[:max(0, inputs_rank - max_dims)]
        loop_shapes = [inputs[0].shape[dim] for dim in loop_dimensions]
        loop_shape_rank = len(loop_shapes)

        # Transpose the input so they are easily indexable inside the loop
        transposed_inputs = [mb.transpose(x=input, perm=permutation) for input in inputs]

        def compute_reduction(result_idx, *partial_results):
            # Pick out the attributes from the dimensions we are reducing over for this index
            idx_dims = permutation[loop_shape_rank:]
            idx_inputs = [index_by_slices(input, [result_idx] + [...]) for input in transposed_inputs]
            idx_window_dimensions = [op.window_dimensions[dim] for dim in idx_dims]
            idx_window_strides = [window_strides[dim] for dim in idx_dims]
            idx_result_types = [
                index_by_slices(partial_result, [result_idx] + [...])
                for partial_result in partial_results
            ]

            if loop_shape_rank > 0:
                # We need to squeeze out the loop (result_idx) dimensions
                idx_inputs = [
                    mb.reshape(x=input, shape=mb.slice_by_size(x=mb.shape(x=input), begin=[loop_shape_rank], size=[-1]))
                    for input in idx_inputs
                ]
                idx_result_types = [
                    mb.reshape(x=result, shape=mb.slice_by_size(x=mb.shape(x=result), begin=[loop_shape_rank], size=[-1]))
                    for result in idx_result_types
                ]

            results = compute_windowed_reduction(
                converter=self,
                context=context,
                inputs=idx_inputs,
                window_dimensions=idx_window_dimensions,
                window_strides=idx_window_strides,
                body=op.body,
                init_values=init_values,
                result_types=idx_result_types,
            )

            result_rank = inputs_rank - loop_shape_rank
            return [
                update_tensor_by_slice(acc, [result_idx] + [slice(None)] * result_rank, result)
                for acc, result in zip(partial_results, results)
            ]

        result_types = [result.type for result in op.results]
        reduction_results = [
            mb.transpose(
                x=np.zeros(result_type.shape, dtype=get_numpy_type(result_type.element_type)),
                perm=permutation,
            )
            for result_type in result_types
        ]
        reduction_results = iterate_indexes_in_shapes(compute_reduction, [loop_shapes], reduction_results, unroll_limit=5)
        reduction_results = [
            mb.transpose(x=reduction_result, perm=inverse_permutation(permutation))
            for reduction_result in reduction_results
        ]

        for (res, mil_res) in zip(op.results, reduction_results):
            context.add_result(res, mil_res)

    @register_stablehlo_op
    def op_iota(self, context: TranslationContext, op: IotaOp):
        res = range_along_dim(op.result.type.shape, int(op.iota_dimension), get_numpy_type(op.result.type.element_type))
        context.add_result(op.result, res)

    @register_stablehlo_op
    def op_gather(self, context: TranslationContext, op: GatherOp):
        """
        Calculates special cases of the GatherOp. Assumes no backing dims, and
        that the index_vector_dim is always the last indexing dimension.

        TODO(knielsen): Consider if this can be done in a more efficient way
        """
        start_indices = context[op.start_indices.get_name()]
        operand = context[op.operand.get_name()]

        operand_rank = len(operand.shape)
        start_indices_rank = len(start_indices.shape)

        dim_numbers = hlo.GatherDimensionNumbers(op.dimension_numbers)
        dim_mapping = dim_numbers.start_index_map
        dim_batches = dim_numbers.operand_batching_dims

        if dim_numbers.index_vector_dim != start_indices_rank - 1:
            raise ValueError("The `index_vector_dim` is only supported to be the last dimension")

        # Handle simple gather cases directly, avoiding the while-loop below
        inferred_sizes = np.array([
            1 if i in dim_mapping or i in dim_batches else
            operand.shape[i] for i in range(operand_rank)]
        )
        if dim_batches == dim_numbers.start_indices_batching_dims and \
                (not dim_batches or np.max(dim_batches) < len(dim_batches)) and \
                np.all(np.array(op.slice_sizes) == inferred_sizes):
            upper, lower = [operand.shape[i] - 1 for i in dim_mapping], [0] * len(dim_mapping)

            def broadcastable(x):
                return np.array(x)[(None,) * (start_indices_rank - 1)]
            clamped_indices = mb.minimum(x=mb.maximum(x=start_indices, y=broadcastable(lower)), y=broadcastable(upper))
            clamped_indices = mb.gather(x=clamped_indices, indices=np.argsort(dim_mapping), axis=-1)
            if len(dim_mapping) == 1:
                if start_indices_rank > 1:
                    clamped_indices = mb.squeeze(x=clamped_indices, axes=(start_indices_rank - 1,))
                result = mb.gather(x=operand, indices=clamped_indices, axis=dim_mapping[0], batch_dims=len(dim_batches))
                context.add_result(op.result, result)
                return
            elif np.max(dim_mapping) < len(dim_mapping) + len(dim_batches):
                result = mb.gather_nd(x=operand, indices=clamped_indices, batch_dims=len(dim_batches))
                window_outputs = [
                    i for i in range(operand_rank)
                    if i not in dim_batches and i not in dim_numbers.collapsed_slice_dims
                ]
                window_outputs = [j for i, j in zip(window_outputs, dim_numbers.offset_dims) if op.slice_sizes[i] == 1]
                if window_outputs:
                    result = mb.expand_dims(x=result, axes=window_outputs)
                context.add_result(op.result, result)
                return

        result_rank = len(op.result.type.shape)
        slice_sizes = op.slice_sizes
        result_iteration_axes = [axis for axis in range(result_rank) if axis not in dim_numbers.offset_dims]

        def compute_index_slice(slice_idx, *partial_results):
            partial_results = partial_results[0]

            slice_start = []
            slice_end = []

            for operand_dim in range(operand_rank):
                if operand_dim in dim_numbers.start_index_map:
                    start_index_dim = dim_numbers.start_index_map.index(operand_dim)
                    elements = operand.shape[operand_dim]

                    start_index = index_by_slices(start_indices, [slice_idx] + [start_index_dim])
                    start_index = mb.reshape(x=start_index, shape=(1,))

                    actual_start_index = mb.maximum(x=mb.minimum(x=start_index, y=elements - slice_sizes[operand_dim]), y=0)
                    end_index = mb.add(x=actual_start_index, y=slice_sizes[operand_dim])
                    slice_start.append(actual_start_index)
                    slice_end.append(end_index)
                elif operand_dim in dim_numbers.operand_batching_dims:
                    batch_index = dim_numbers.operand_batching_dims.index(operand_dim)
                    slice_batch = dim_numbers.start_indices_batching_dims[batch_index]
                    start_index = mb.slice_by_size(x=slice_idx, begin=(slice_batch,), size=(1,))
                    slice_start.append(start_index)
                    slice_end.append(mb.add(x=start_index, y=1))
                elif operand_dim in dim_numbers.collapsed_slice_dims:
                    slice_start.append(mb.reshape(x=0, shape=(1,)))
                    slice_end.append(mb.reshape(x=1, shape=(1,)))
                else:
                    slice_start.append(mb.reshape(x=0, shape=(1,)))
                    slice_end.append(mb.reshape(x=slice_sizes[operand_dim], shape=(1,)))

            selected_slice = mb.slice_by_index(
                x=operand,
                begin=mb.concat(values=slice_start, axis=0),
                end=mb.concat(values=slice_end, axis=0),
            )
            if len(dim_numbers.collapsed_slice_dims) > 0:
                selected_slice = mb.squeeze(x=selected_slice, axes=dim_numbers.collapsed_slice_dims)

            # Figure out which result to update
            update_slice_spec = []
            stack_axes_idx = 0
            for output_dim in range(result_rank):
                if output_dim in result_iteration_axes:
                    result_idx = mb.gather(x=slice_idx, indices=[stack_axes_idx])
                    update_slice_spec.append(result_idx)
                    stack_axes_idx += 1
                else:
                    update_slice_spec.append(slice(None))
            return [update_tensor_by_slice(partial_results, update_slice_spec, selected_slice)]

        result_dtype = get_mil_type_from_ir(op.result.type.element_type)
        result = mb.fill(shape=op.result.type.shape, value=mb.cast(x=0, dtype=dtype_str(result_dtype)))
        result_iteration_shape = [result.shape[stack_axis] for stack_axis in result_iteration_axes]
        result, = iterate_indexes_in_shapes(compute_index_slice, [result_iteration_shape], [result], unroll_limit=5)

        context.add_result(op.result, result)

    @register_stablehlo_op
    def op_scatter(self, context: TranslationContext, op: ScatterOp):
        dim_numbers = hlo.ScatterDimensionNumbers(op.scatter_dimension_numbers)
        dim_mapping = dim_numbers.scattered_dims_to_operand_dims
        operand = context[op.inputs[0].get_name()]
        scatter_indices = context[op.scatter_indices.get_name()]
        updates = context[op.updates[0].get_name()]

        if len(dim_numbers.input_batching_dims) > 0:
            raise ValueError("Scatter batching index is not supported!")
        if len(op.inputs) != 1 or len(op.updates) != 1:
            raise ValueError("Scatter with multiple operands is not supported!")

        scatter_indices_rank = len(scatter_indices.shape)
        if scatter_indices_rank == 0 or 0 in scatter_indices.shape:
            # Special case for empty scatter indices
            context.add_result(op.results[0], operand)
            return

        if np.max(dim_mapping) >= len(dim_mapping):
            raise ValueError("Scatter windows are only supported with dimension numbers contiguous with the rank!")
        # MIL only supports scatter window update sizes that match the operand shape
        #     updates must be the shape as `indices.shape[:-1] + data.shape[indices.shape[-1]:]`
        # [sic] via
        #     https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather.scatter_nd
        if scatter_indices.shape != (1,) and \
                updates.shape != scatter_indices.shape[:-1] + operand.shape[scatter_indices.shape[-1]:]:
            raise ValueError("Scatter windows that only partially fill dimensions are not supported!")

        # this can be done pre-emptively because of the constraint on scatter windows
        scatter_indices = mb.gather(x=scatter_indices, indices=np.argsort(dim_mapping), axis=-1)

        # StableHLO supports arbitrary scatter computations, but MIL has a fixed set
        # We try to match the update computation to a known binary operation
        _, mil_binary_op, mode = match_computation(op.update_computation)

        if mil_binary_op is None:
            raise ValueError("Unsupported update mode for scatter operation")

        upper_bound = np.array(operand.shape[:len(dim_mapping)], dtype=np.int32)[(None,) * (scatter_indices_rank - 1)]
        valid = mb.logical_and(
            x=mb.greater_equal(x=scatter_indices, y=0),
            y=mb.less(x=scatter_indices, y=upper_bound)
        )

        def along(n):
            return mb.slice_by_index(
                x=valid, begin=(0,) * (scatter_indices_rank - 1) + (n,),
                end=scatter_indices.shape[:-1] + (n + 1,)
            )

        # unrolling O(scatter_indices.rank)
        reduction = along(0)
        for i in range(1, scatter_indices.shape[-1]):
            reduction = mb.logical_and(x=reduction, y=along(i))
        reduction = mb.squeeze(x=reduction, axes=(scatter_indices_rank - 1,))

        # Special handling for rank-0 reduction (single index update).
        # It supports updating a window that is a subset of the dimension (partial update),
        # which `scatter_nd` does not support (it only supports full slice updates).
        if reduction.rank == 0:
            assert scatter_indices.shape == (1,), \
                    f"unexpected input shape for scatter indices of {scatter_indices.shape}"
            assert updates.rank == operand.rank

            # The index to update
            update_index = scatter_indices

            # Helper to construct end indices for slicing
            # If rank <= 1, it's just the bound. Otherwise, it's [bound, dim1, dim2, ...]
            def get_end_indices(operand, bound):
                if operand.rank <= 1:
                    return bound
                return mb.concat(values=(bound, operand.shape[1:]), axis=0)

            # 1. Slice before the update index
            # operand[:update_index]
            before = mb.slice_by_index(
                x=operand,
                begin=(0,) * operand.rank,
                end=get_end_indices(operand, update_index)
            )

            # 2. Slice after the update index
            # operand[update_index+window_size:]
            # We need to clamp the start index for 'after' to be at most operand.shape[0]
            # to avoid out of bounds.
            update_window_size = updates.shape[0]
            update_end_index = mb.minimum(
                x=mb.add(x=update_index, y=update_window_size),
                y=operand.shape[0]
            )
            after = mb.slice_by_index(
                x=operand,
                begin=get_end_indices(operand, update_end_index),
                end=operand.shape
            )

            # 3. The update value itself
            # We need to extract the current value at the update index to apply the update operation
            # operand[update_index:update_index+window_size]
            current_value_slice = mb.slice_by_index(
                x=operand,
                begin=get_end_indices(operand, update_index),
                end=get_end_indices(operand, update_end_index)
            )

            # Apply the update computation (add, mul, etc.)
            new_value_slice = mil_binary_op(x=current_value_slice, y=updates)

            # 4. Concatenate parts to form the result
            # [before, new_value, after]
            # We only do this if the index is valid (reduction condition)
            # 'reduction' here is actually a boolean scalar indicating if the index is valid
            is_valid_index = reduction

            result_if_valid = mb.concat(values=(before, new_value_slice, after), axis=0)
            result = mb.select(
                cond=is_valid_index,
                a=result_if_valid,
                b=operand
            )
        else:
            where = mb.non_zero(x=reduction)
            scatter_indices = mb.gather_nd(x=scatter_indices, indices=where)
            updates = mb.gather_nd(x=updates, indices=where)
            result = mb.scatter_nd(data=operand, indices=scatter_indices, updates=updates, mode=mode)
        context.add_result(op.results[0], result)

    @register_stablehlo_op
    def op_custom_call(self, context: TranslationContext, op: CustomCallOp):
        if op.call_target_name.value.startswith("mhlo."):
            mapped_op = None
            op_impl = None
            match op.call_target_name.value:
                case "mhlo.topk":
                    mapped_op = TopKOp
                    op_impl = self._op_mhlo_topk
                case "mhlo.asin":
                    mapped_op = AsinOp
                    op_impl = self._op_mhlo_asin
                case "mhlo.sinh":
                    mapped_op = SinhOp
                    op_impl = self._op_mhlo_sinh
                case "mhlo.asinh":
                    mapped_op = AsinhOp
                    op_impl = self._op_mhlo_asinh
                case "mhlo.acos":
                    mapped_op = AcosOp
                    op_impl = self._op_mhlo_acos
                case "mhlo.cosh":
                    mapped_op = CoshOp
                    op_impl = self._op_mhlo_cosh
                case "mhlo.acosh":
                    mapped_op = AcoshOp
                    op_impl = self._op_mhlo_acosh
                case "mhlo.atanh":
                    mapped_op = AtanhOp
                    op_impl = self._op_mhlo_atanh

            if not mapped_op:
                raise ValueError(f"mhlo op '{op.call_target_name.value}' is not implemented")
            if not op_impl:
                raise ValueError(f"mhlo op '{op.call_target_name.value}' does not have an implementation")

            mhlo_attributes = {attr.name: attr.attr for attr in list(op.attributes["mhlo.attributes"])}
            delegate_op = partial(mapped_op, **mhlo_attributes, loc=op.location)(*op.operands)

            # We manually have to handle the results, as the current API does not allow naming
            # the `delegate_op` results according to the custom call results
            mil_results = op_impl(context, delegate_op)
            for (custom_call_result, mil_result) in zip(op.results, mil_results):
                context.add_result(custom_call_result, mil_result)

            return

        raise ValueError(f"Custom call is not supported: {op.call_target_name}")

    def _op_mhlo_topk(self, context: TranslationContext, op: TopKOp):
        """
        This is a MHLO op, and follows a slightly different pattern, since it is unvoked by a
        custom call. It will return the results, as we currently can not rename the results
        in the TopKOp
        """
        x = context[op.operand.get_name()]
        descending = op.largest is None or op.largest.value
        mil_res = mb.topk(x=x, k=op.k.value, ascending=not descending)
        return mil_res

    def _op_mhlo_asin(self, context: TranslationContext, op: AsinOp):
        x = context[op.operand.get_name()]
        mil_res = mb.asin(x=x)
        return [mil_res]

    def _op_mhlo_sinh(self, context: TranslationContext, op: SinhOp):
        x = context[op.operand.get_name()]
        mil_res = mb.sinh(x=x)
        return [mil_res]

    def _op_mhlo_asinh(self, context: TranslationContext, op: AsinhOp):
        x = context[op.operand.get_name()]
        # asinh(x) = log(x + sqrt(x^2 + 1))
        x_sq = mb.mul(x=x, y=x)
        x_sq_plus_1 = mb.add(x=x_sq, y=1.0)
        sqrt_part = mb.sqrt(x=x_sq_plus_1)
        log_arg = mb.add(x=x, y=sqrt_part)
        mil_res = mb.log(x=log_arg)
        return [mil_res]

    def _op_mhlo_acos(self, context: TranslationContext, op: AcosOp):
        x = context[op.operand.get_name()]
        mil_res = mb.acos(x=x)
        return [mil_res]

    def _op_mhlo_cosh(self, context: TranslationContext, op: CoshOp):
        x = context[op.operand.get_name()]
        mil_res = mb.cosh(x=x)
        return [mil_res]

    def _op_mhlo_acosh(self, context: TranslationContext, op: AcoshOp):
        x = context[op.operand.get_name()]
        # acosh(x) = log(x + sqrt(x^2 - 1))
        x_sq = mb.mul(x=x, y=x)
        x_sq_minus_1 = mb.sub(x=x_sq, y=1.0)
        sqrt_part = mb.sqrt(x=x_sq_minus_1)
        log_arg = mb.add(x=x, y=sqrt_part)
        mil_res = mb.log(x=log_arg)
        return [mil_res]

    def _op_mhlo_atanh(self, context: TranslationContext, op: AtanhOp):
        x = context[op.operand.get_name()]
        # atanh(x) = 0.5 * log((1 + x) / (1 - x))
        one_plus_x = mb.add(x=1.0, y=x)
        one_minus_x = mb.sub(x=1.0, y=x)
        div_res = mb.real_div(x=one_plus_x, y=one_minus_x)
        log_res = mb.log(x=div_res)
        mil_res = mb.mul(x=0.5, y=log_res)
        return [mil_res]

    def invoke_hlo_function(self, context: TranslationContext, func_name: str, hlo_params, hlo_func_body, cml_args):
        # Enter variable context for the function call
        context.push_function(func_name)

        # Setup arguments for the function
        for hlo_func_param, actual_arg in zip(hlo_params, cml_args):
            context.add_result(hlo_func_param, actual_arg)

        # Process the function
        if len(hlo_func_body.blocks) != 1:
            raise ValueError(f"Unsupported function with {len(hlo_func_body.blocks)} blocks")
        outputs = self.process_block(context, hlo_func_body.blocks[0])

        # Exit the function context
        context.pop_function()

        return outputs

    def __simple_unary_op(self, context: TranslationContext, mil_op, hlo_op):
        operand = context[hlo_op.operand.get_name()]
        cml_op = mil_op(x=operand)
        context.add_result(hlo_op.result, cml_op)

    def __simple_binary_op(self, context: TranslationContext, mil_op, hlo_op):
        lhs = context[hlo_op.lhs.get_name()]
        rhs = context[hlo_op.rhs.get_name()]
        cml_op = mil_op(x=lhs, y=rhs)
        context.add_result(hlo_op.result, cml_op)
