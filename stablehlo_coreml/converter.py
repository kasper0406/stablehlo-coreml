from functools import partial, reduce

import numpy as np
from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil.mil.ops.defs._utils import (
    promote_input_dtypes,
)
from jax._src.lib.mlir.dialects import hlo
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import stablehlo as stablehlo_dialect
from jaxlib.mlir.dialects.func import CallOp, FuncOp
from jaxlib.mlir.dialects.func import ReturnOp as FuncReturnOp
from jaxlib.mlir.dialects.stablehlo import (
    AbsOp,
    AddOp,
    AndOp,
    Atan2Op,
    BroadcastInDimOp,
    CaseOp,
    CeilOp,
    ClampOp,
    CompareOp,
    CompositeOp,
    ConcatenateOp,
    ConstantOp,
    ConvertOp,
    ConvolutionOp,
    CosineOp,
    CustomCallOp,
    DivOp,
    DotGeneralOp,
    DynamicSliceOp,
    DynamicUpdateSliceOp,
    Expm1Op,
    ExpOp,
    FloorOp,
    GatherOp,
    IotaOp,
    IsFiniteOp,
    Log1pOp,
    LogOp,
    MaxOp,
    MinOp,
    MulOp,
    NegOp,
    NotOp,
    OrOp,
    PadOp,
    PowOp,
    ReduceOp,
    ReduceWindowOp,
    RemOp,
    ReshapeOp,
    ReturnOp,
    ReverseOp,
    RoundOp,
    RsqrtOp,
    ScatterOp,
    SelectOp,
    SignOp,
    SineOp,
    SliceOp,
    SortOp,
    SqrtOp,
    SubtractOp,
    TanhOp,
    TanOp,
    TransposeOp,
    WhileOp,
    XorOp,
)

from .ops_register import StableHloOpsRegistry, register_composite_op, register_stablehlo_op
from .padding import pad_with_cast
from .passes.utils import register_optimizations
from .reductions import compute_reduction, compute_windowed_reduction, match_computation, match_simple_reduce_window
from .sort_utils import match_sort
from .translation_context import TranslationContext
from .utils import (
    auto_cast_bool,
    clamp_index,
    dtype_str,
    fix_scalar_tensor,
    get_mil_type,
    get_mil_type_from_ir,
    get_numpy_type,
    index_by_slices,
    inverse_permutation,
    iterate_indexes_in_shapes,
    range_along_dim,
    safe_cast_to_int32,
    update_tensor_by_slice,
)


def convert(module, minimum_deployment_target: AvailableTarget):
    if minimum_deployment_target < AvailableTarget.iOS18:
        raise ValueError("Converting to <iOS18 is not supported")

    register_optimizations()

    _normalize_module(module)

    converter = StableHloConverter(opset_version=minimum_deployment_target)
    return converter.convert(module)


def _normalize_module(module: ir.Module) -> None:
    """Normalize an incoming StableHLO module in-place before conversion.

    Two passes are applied in sequence:

    1. ``chlo-legalize-to-stablehlo`` – lowers any raw CHLO ops that appear
       *outside* a ``stablehlo.composite`` wrapper (e.g. from the
       ``jax.jit().lower().compiler_ir('stablehlo')`` path) to their StableHLO
       equivalents.  Ops already wrapped in a composite are intentionally left
       untouched; the composite handlers in the converter map them directly to
       CoreML primitives, bypassing the (potentially unsupported) stablehlo
       decompositions.

    2. ``stablehlo-legalize-deprecated-ops`` – rewrites any deprecated
       StableHLO ops to their current equivalents, keeping the converter
       insulated from older serialized modules.
    """
    stablehlo_dialect.register_stablehlo_passes()
    pm = passmanager.PassManager.parse(
        "builtin.module(func.func(chlo-legalize-to-stablehlo, stablehlo-legalize-deprecated-ops))",
        context=module.context,
    )
    pm.run(module.operation)


class StableHloConverter(metaclass=StableHloOpsRegistry):

    def __init__(self, opset_version: int | None = None):
        self.opset_version = AvailableTarget(opset_version) if opset_version is not None else None
        self.prog = mil.Program()
        self.func_index = {}

    def convert(self, module: ir.Module) -> Program:
        logger.info("Converting graph.")

        # Build function index to resolve/inline HLO function calls
        for func in module.body:
            self.func_index[func.name.value] = func

        for func in module.body:
            if func.sym_visibility is None or func.sym_visibility.value == "public":
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
            for name in func_inputs:
                context.add_variable(name, ssa_func.inputs[name])

            ssa_func.set_outputs(self.process_block(context, hlo_func.body.blocks[0]))
            self.prog.add_function(hlo_func.name.value, ssa_func)

    def process_block(self, context: TranslationContext, block: ir.Block):
        outputs = None
        for op in block:
            if outputs is not None:
                raise ValueError("The 'return' op must be the last operation in the block.")
            ret = self.dispatch_op(self, context, op)
            if ret is not None:
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
    def op_composite(self, context: TranslationContext, op: CompositeOp):
        # Dispatch to a named handler if one is registered for this composite op.
        # Named handlers can map directly to CoreML primitives, which is preferable
        # when the decomposition uses features that are not supported.
        composite_name = op.name.value
        handler = self._composite_ops_registry.get(composite_name)
        if handler is not None:
            return handler(self, context, op)

        # Default: inline the decomposition function.
        context_args = [context[arg.get_name()] for arg in op.inputs]

        func_name = op.decomposition.value
        hlo_func = self.func_index[func_name]
        params = hlo_func.arguments
        outputs = self.invoke_hlo_function(context, func_name, params, hlo_func.body, context_args)

        for result, output in zip(op.results, outputs):
            context.add_result(result, output)

    @register_composite_op("chlo.top_k")
    def _op_composite_chlo_top_k(self, context: TranslationContext, op: CompositeOp):
        # Map directly to mb.topk rather than inlining the decomposition, which
        # uses a stable multi-input sort that is not supported.
        x = context[op.inputs[0].get_name()]
        k = ir.IntegerAttr(op.composite_attributes["k"]).value
        # Default to descending (largest=True) as per the chlo.top_k spec; honour
        # an explicit 'largest' attribute if present.
        largest = (ir.BoolAttr(op.composite_attributes["largest"]).value
                   if "largest" in op.composite_attributes else True)
        values, indices = mb.topk(x=x, k=k, ascending=not largest)
        context.add_result(op.results[0], values)
        context.add_result(op.results[1], indices)

    @register_composite_op("chlo.asin")
    def _op_composite_chlo_asin(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        context.add_result(op.results[0], mb.asin(x=x))

    @register_composite_op("chlo.acos")
    def _op_composite_chlo_acos(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        context.add_result(op.results[0], mb.acos(x=x))

    @register_composite_op("chlo.sinh")
    def _op_composite_chlo_sinh(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        context.add_result(op.results[0], mb.sinh(x=x))

    @register_composite_op("chlo.cosh")
    def _op_composite_chlo_cosh(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        context.add_result(op.results[0], mb.cosh(x=x))

    @register_composite_op("chlo.atanh")
    def _op_composite_chlo_atanh(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        # atanh(x) = 0.5 * log((1 + x) / (1 - x))
        ratio = mb.real_div(x=mb.add(x=1.0, y=x), y=mb.sub(x=1.0, y=x))
        context.add_result(op.results[0], mb.mul(x=0.5, y=mb.log(x=ratio)))

    @register_composite_op("chlo.asinh")
    def _op_composite_chlo_asinh(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        # asinh(x) = log(x + sqrt(x^2 + 1))
        x_sq = mb.mul(x=x, y=x)
        x_sq_plus_1 = mb.add(x=x_sq, y=1.0)
        context.add_result(op.results[0], mb.log(x=mb.add(x=x, y=mb.sqrt(x=x_sq_plus_1))))

    @register_composite_op("chlo.acosh")
    def _op_composite_chlo_acosh(self, context: TranslationContext, op: CompositeOp):
        x = context[op.inputs[0].get_name()]
        # acosh(x) = log(x + sqrt(x^2 - 1))
        x_sq = mb.mul(x=x, y=x)
        x_sq_minus_1 = mb.sub(x=x_sq, y=1.0)
        context.add_result(op.results[0], mb.log(x=mb.add(x=x, y=mb.sqrt(x=x_sq_minus_1))))

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
    def op_xor(self, context: TranslationContext, op: XorOp):
        self.__simple_binary_op(context, mb.logical_xor, op)

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
    def op_round(self, context: TranslationContext, op: RoundOp):
        operand = context[op.operand.get_name()]
        if op.OPERATION_NAME == 'stablehlo.round_nearest_afz':
            magnitude = mb.abs(x=operand)
            shifted = mb.add(x=magnitude, y=0.5)
            rounded_magnitude = mb.floor(x=shifted)
            result = mb.mul(x=rounded_magnitude, y=mb.sign(x=operand))
            context.add_result(op.result, result)
        # elif op.OPERATION_NAME == 'stablehlo.round_nearest_even':
        else:
            raise ValueError(f"Unsupported RoundOp type of {op.OPERATION_NAME}")

    @register_stablehlo_op
    def op_neg(self, context: TranslationContext, op: NegOp):
        operand = context[op.operand.get_name()]
        numpy_dtype = get_numpy_type(operand)

        # CoreML's `sub` operator expects signed integers or floats.
        # CoreML's `cast` also does not support casting from uint32/uint64 directly.
        # Negation on unsigned types is rarely used, but when it is, it's virtually unsupported in MIL.
        is_unsigned = numpy_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]
        if is_unsigned:
            raise ValueError(
                f"CoreML does not support negation (or casting) for unsigned integer type {numpy_dtype}."
            )

        zero_val = np.array([0], dtype=numpy_dtype)
        result = mb.sub(x=zero_val, y=operand)

        context.add_result(op.result, result)

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
        context.add_result(op.result, mb.const(val=constant))

    @register_stablehlo_op
    def op_dot_general(self, context: TranslationContext, op: DotGeneralOp):
        # Implements dot_general via transpose → reshape → single matmul → reshape.
        # See https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
        lhs_rank = len(op.lhs.type.shape)
        rhs_rank = len(op.rhs.type.shape)
        dot_dim_numbers = hlo.DotDimensionNumbers(op.dot_dimension_numbers)

        lhs_contracting_dim = dot_dim_numbers.lhs_contracting_dimensions
        rhs_contracting_dim = dot_dim_numbers.rhs_contracting_dimensions
        lhs_batching_dim = dot_dim_numbers.lhs_batching_dimensions
        rhs_batching_dim = dot_dim_numbers.rhs_batching_dimensions

        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]

        lhs_result_dim = [d for d in range(lhs_rank) if d not in lhs_batching_dim + lhs_contracting_dim]
        rhs_result_dim = [d for d in range(rhs_rank) if d not in rhs_batching_dim + rhs_contracting_dim]

        # Transpose to [batch…, result…, contract…]
        lhs_perm = lhs_batching_dim + lhs_result_dim + lhs_contracting_dim
        rhs_perm = rhs_batching_dim + rhs_result_dim + rhs_contracting_dim
        t_lhs = lhs if lhs_perm == list(range(lhs_rank)) else mb.transpose(x=lhs, perm=lhs_perm)
        t_rhs = rhs if rhs_perm == list(range(rhs_rank)) else mb.transpose(x=rhs, perm=rhs_perm)

        batch_shape = tuple(int(lhs.shape[d]) for d in lhs_batching_dim)

        def _product(dims):
            return int(reduce(lambda a, b: int(a) * int(b), dims, 1))

        lhs_result_count = _product([lhs.shape[d] for d in lhs_result_dim]) if lhs_result_dim else 1
        rhs_result_count = _product([rhs.shape[d] for d in rhs_result_dim]) if rhs_result_dim else 1
        contracted_count = _product([lhs.shape[d] for d in lhs_contracting_dim]) if lhs_contracting_dim else 1

        # Reshape to (batch…, M, K) and (batch…, N, K)
        lhs_3d = list(batch_shape) + [lhs_result_count, contracted_count]
        rhs_3d = list(batch_shape) + [rhs_result_count, contracted_count]
        c_lhs = mb.reshape(x=t_lhs, shape=lhs_3d)
        c_rhs = mb.reshape(x=t_rhs, shape=rhs_3d)

        # (batch…, M, K) × (batch…, N, K)^T → (batch…, M, N)
        result = mb.matmul(x=c_lhs, y=c_rhs, transpose_y=True)

        # Squeeze fake dims when a side has no result dims
        if len(lhs_result_dim) == 0 and len(rhs_result_dim) == 0:
            if batch_shape:
                result = mb.squeeze(x=result, axes=[-2, -1])
            else:
                result = mb.reshape(x=result, shape=(1,))
        elif len(lhs_result_dim) == 0:
            result = mb.squeeze(x=result, axes=[len(batch_shape)])
        elif len(rhs_result_dim) == 0:
            result = mb.squeeze(x=result, axes=[-1])

        # Reshape from (batch…, M, N) to (batch…, L1, L2, …, R1, R2, …)
        final_shape = list(batch_shape)
        final_shape += [int(lhs.shape[d]) for d in lhs_result_dim]
        final_shape += [int(rhs.shape[d]) for d in rhs_result_dim]
        if not final_shape:
            final_shape = [1]

        if list(result.shape) != final_shape:
            result = mb.reshape(x=result, shape=final_shape)

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
        if op.is_stable and len(inputs) > 1:
            raise ValueError("Stable sorting is not supported for multi-input sorting")

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
        indices = mb.argsort(x=key, axis=sort_dim, ascending=ascending)

        # Given CoreML's argsort is unstable we are not able to handle multiple sort keys
        if len(sort_keys) > 1:
            raise ValueError("Having more than one sort key is not supported because MIL's argsort is not supported")
        # The following code would be used if CoreML had a stable argsort
        # for key, ascending in sort_keys[-2::-1]:
        #     gathered_key = mb.gather_along_axis(x=key, indices=indices, axis=sort_dim)
        #     relative_indices = mb.argsort(x=gathered_key, axis=sort_dim, ascending=ascending)
        #     indices = mb.gather_along_axis(x=indices, indices=relative_indices, axis=sort_dim)

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
        elif any(i == 0 for i in result_shape):
            res = mb.const(
                val=np.empty(result_shape, get_numpy_type(op.result.type))
            )
            context.add_result(op.result, res)
            return
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
            outputs = self.invoke_hlo_function(context, "while_body", params, op.body, body_args)
            return [fix_scalar_tensor(output) for output in outputs]

        loop_vars = [context[arg.get_name()] for arg in op.operands]
        while_results = mb.while_loop(_cond=cond, _body=body, loop_vars=loop_vars)

        if not isinstance(while_results, (list, tuple)):
            while_results = [while_results]

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
            elif comparison_direction == "GT":
                cml_op = mb.logical_and(x=lhs, y=mb.logical_not(x=rhs))
            elif comparison_direction == "LT":
                cml_op = mb.logical_and(x=mb.logical_not(x=lhs), y=rhs)
            elif comparison_direction == "GE":
                cml_op = mb.logical_or(x=lhs, y=mb.logical_not(x=rhs))
            elif comparison_direction == "LE":
                cml_op = mb.logical_or(x=mb.logical_not(x=lhs), y=rhs)
            else:
                raise ValueError(
                    f"Unexpected operation: {comparison_direction}"
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
        sizes = safe_cast_to_int32(op.slice_sizes, "slice_sizes")

        # Clamp start indices to ensure they are within bounds: [0, operand_dim - slice_size]
        # This is required by the StableHLO specification
        shape = mb.shape(x=x)
        begin = clamp_index(begin, shape, sizes)

        cml_op = mb.slice_by_size(x=x, begin=begin, size=sizes)
        context.add_result(op.result, cml_op)

    @register_stablehlo_op
    def op_slice(self, context: TranslationContext, op: SliceOp):
        x = context[op.operand.get_name()]

        begin = safe_cast_to_int32(op.start_indices, "start_indices")
        end = safe_cast_to_int32(op.limit_indices, "limit_indices")
        stride = safe_cast_to_int32(op.strides, "strides")

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
        in_rank = x.rank - 2
        pad_attr = op.padding
        if pad_attr is None:
            pad = np.zeros((2 * in_rank), dtype=np.int32)
        elif pad_attr.is_splat:
            pad = np.full(2 * in_rank, pad_attr.get_splat_value().value, dtype=np.int32)
        else:
            pad = np.array(pad_attr, dtype=np.int32).reshape(2 * in_rank)

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
        weight = context[op.rhs.get_name()]
        is_transposed = conv_type != mb.conv

        weight_permutation = [
            dim_spec.kernel_input_feature_dimension if is_transposed else dim_spec.kernel_output_feature_dimension,
            dim_spec.kernel_output_feature_dimension if is_transposed else dim_spec.kernel_input_feature_dimension,
            *dim_spec.kernel_spatial_dimensions
        ]
        weight = mb.transpose(x=weight, perm=weight_permutation)

        # For transposed convolutions, MIL requires the weights to be reversed along all spatial dimensions
        if is_transposed:
            num_spatial_dims = len(dim_spec.kernel_spatial_dimensions)
            spatial_axes = [i + 2 for i in range(num_spatial_dims)]
            weight = mb.reverse(x=weight, axes=spatial_axes)

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
        np_dtype = get_numpy_type(y)
        atan2_res_adjusted = mb.add(x=atan2_res, y=mb.mul(x=mb.sign(x=y), y=np_dtype(np.pi)))
        atan2_res = mb.select(
            cond=mb.less(x=x, y=np_dtype(0.0)),
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

        # Pad the inputs if required before attempting simple match
        if op.padding:
            padding = np.reshape(np.array(op.padding, dtype=np.int32), (2 * inputs_rank,))
            inputs = [
                pad_with_cast(x=input, pad=padding, constant_val=mb.reduce_max(x=init_value))
                for input, init_value in zip(inputs, init_values)
            ]

        fixed_dimensions = []
        reduction_dimensions = []
        for axis in range(inputs_rank):
            if op.window_dimensions[axis] == 1 and window_strides[axis] == 1:
                fixed_dimensions.append(axis)
            else:
                reduction_dimensions.append(axis)
        permutation = fixed_dimensions + reduction_dimensions
        is_identity_perm = all(i == p for i, p in enumerate(permutation))

        transposed_inputs = inputs
        transposed_window_dimensions = list(op.window_dimensions)
        transposed_window_strides = list(window_strides)
        if not is_identity_perm:
            transposed_inputs = [mb.transpose(x=input, perm=permutation) for input in inputs]
            transposed_window_dimensions = [op.window_dimensions[i] for i in permutation]
            transposed_window_strides = [window_strides[i] for i in permutation]

        res = match_simple_reduce_window(
            op.body, transposed_inputs, init_values, transposed_window_dimensions, transposed_window_strides
        )
        if res is not None:
            if not is_identity_perm:
                res = mb.transpose(x=res, perm=inverse_permutation(permutation))
            context.add_result(op.result, res)
            return

        # Unfortunately CoreML only supports tensors with rank <= 6.
        # Due to the re-shaping and windowing operations inside `__compute_windowed_reduction`, this
        # means the function can not be called with tensors of rank >= 4.
        # To work around this problem, we have to iterate over the leading dimensions not being
        # windowed over, and calculate the result values incrementally.

        # We will put as few dimensions as possible in the loop_dimensions (i.e. we may
        # choose to put some of the `fixedf_dimensions` inside the reduction itself)
        max_dims = 3
        if len(reduction_dimensions) > max_dims:
            raise ValueError("Due to CoreML's rank <= 5 restriction, it is not supported to reduce on more then 3 dimensions!")
        loop_dimensions = fixed_dimensions[:max(0, inputs_rank - max_dims)]
        loop_shapes = [inputs[0].shape[dim] for dim in loop_dimensions]
        loop_shape_rank = len(loop_shapes)

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
    @auto_cast_bool(target_dtype="uint8")
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
                if start_indices_rank > 1 or len(dim_numbers.collapsed_slice_dims):
                    clamped_indices = mb.squeeze(x=clamped_indices, axes=(start_indices_rank - 1,))
                result = mb.gather(x=operand, indices=clamped_indices, axis=dim_mapping[0], batch_dims=len(dim_batches))
                if start_indices_rank > 1 and not len(dim_numbers.collapsed_slice_dims):
                    result = mb.expand_dims(x=result, axes=(start_indices_rank - 1,))
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
    @auto_cast_bool(target_dtype="int32")
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
        raise ValueError(f"Custom call is not supported: {op.call_target_name}")

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
