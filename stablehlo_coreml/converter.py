from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from .utils import index_by_slices, update_tensor_by_slice

from jaxlib.mlir import ir
from jaxlib.mlir.dialects.func import FuncOp, CallOp, ReturnOp as FuncReturnOp
from jaxlib.mlir.dialects.stablehlo import AddOp, SubtractOp, MulOp, DivOp, NegOp, SignOp, AbsOp, ExpOp, Log1pOp, SqrtOp, ConstantOp, DotGeneralOp, ReshapeOp, BroadcastInDimOp, WhileOp, CompareOp, ConvertOp, SelectOp, DynamicSliceOp, ReturnOp, ConvolutionOp, MaxOp, RsqrtOp, TanhOp, ConcatenateOp, TransposeOp, DynamicUpdateSliceOp, SliceOp
from jax._src.lib.mlir.dialects import hlo

import numpy as np

from typing import List, Optional
import inspect
from functools import partial, reduce
import operator
import itertools

def convert(module, minimum_deployment_target: AvailableTarget):
    if minimum_deployment_target < AvailableTarget.iOS18:
        raise ValueError("Converting to <iOS18 is not supported")

    converter = StableHloConverter(opset_version=minimum_deployment_target)
    return converter.convert(module)

class TranscriptionContext:
    def __init__(self):
        self._path = []
        self.seen_paths = set()
        self.variables = {} # Nested map: path -> variable -> mil var

    def push_function(self, name: str):
        counter = 0
        ctx_name = name
        while True:
            new_path = self._path + [ctx_name]
            if "/".join(new_path) in self.seen_paths:
                # Ensure that the new context name is in fact unique
                # A collision can happen if the same function is called twice
                ctx_name = f"{name}_{counter}"
                counter += 1
            else:
                self._path.append(ctx_name)
                self.seen_paths.add(self.path())
                return ctx_name
    
    def pop_function(self):
        self.variables.pop(self.path())
        self._path.pop()
    
    def add_variable(self, name: str, mil_var):
        path = self.path()
        if path not in self.variables:
            self.variables[path] = {}
        
        if name in self.variables[path]:
            raise ValueError(f"Variable {name} is already defined in path {path}")
        self.variables[path][name] = mil_var

    def __getitem__(self, name: str):
        path = self.path()
        ctx = self.variables[path]
        if name not in ctx:
            raise ValueError(f"Variable with name {name} is not defined in path {path}")
        return ctx[name]

    def path(self) -> str:
        return "/".join(self._path)

def register_stablehlo_op(func):
    # Check the signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Exclude 'self' from the parameters
    params = params[1:]

    error_msg = "HLO op implementations should take parameters of exactly (context: TranscriptionContext, op: <HLO_OP_TYPE>)"
    if len(params) != 2:
        raise TypeError(error_msg)
    
    if not issubclass(params[0].annotation, TranscriptionContext):
        raise TypeError(error_msg)

    # We identify the function by the type of operation it implements
    func._implements_hlo_op = params[1].annotation
    return func

class StableHloOpsRegistry(type):
    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)

        cls._stablehlo_ops_registry = {}
        for name, method in clsdict.items():
            op_type = getattr(method, '_implements_hlo_op', False)
            if callable(method) and op_type:
                if op_type in cls._stablehlo_ops_registry:
                    raise TypeError(f"StableHLO op {op_type} has been registered more than once!")
                cls._stablehlo_ops_registry[op_type] = method
    
    def _dispatch_op(cls, self, context: TranscriptionContext, op):
        if type(op) not in self._stablehlo_ops_registry:
            raise TypeError(f"The StableHLO op {type(op)} has not been implemented!")
        
        op_method = self._stablehlo_ops_registry[type(op)]
        return op_method(self, context, op)

    def __call__(cls, *args, **kwargs):
        # Register the dispatch_op method
        instance = super().__call__(*args, **kwargs)
        setattr(instance, 'dispatch_op', cls._dispatch_op)
        return instance

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
            if "public" == func.visibility.value:
                self.build_func(func)

        return self.prog

    def build_func(self, hlo_func: FuncOp):
        context = TranscriptionContext() # Map from results to created variables

        func_inputs = {}
        for arg in hlo_func.arguments:
            shape = arg.type.shape
            if shape == []:
                shape = [1]

            func_inputs[arg.get_name()] = mb.placeholder(
                shape=shape, dtype=self.__get_dtype(arg.type.element_type)
            )

        with Function(func_inputs, opset_version=self.opset_version) as ssa_func:
            for name in func_inputs.keys():
                context.add_variable(name, ssa_func.inputs[name])

            ssa_func.set_outputs(self.process_block(context, hlo_func.body.blocks[0]))
            self.prog.add_function(hlo_func.name.value, ssa_func)

    def process_block(self, context: TranscriptionContext, block: ir.Block):
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
    def op_call(self, context: TranscriptionContext, op: CallOp):
        # We can not do function calls in MIL, so we have to inline the function

        # Get the argument mapping prior to entering the function context
        context_args = []

        for arg in op.operands:
            context_args.append(context[arg.get_name()])

        func_name = op.callee.value
        hlo_func = self.func_index[op.callee.value]
        params = hlo_func.arguments
        outputs = self.__invoke_hlo_function(context, func_name, params, hlo_func.body, context_args)

        # Configure return value
        for result, output in zip(op.results, outputs):
            context.add_variable(result.get_name(), output)

    @register_stablehlo_op
    def op_return(self, context: TranscriptionContext, op: ReturnOp):
        return [ context[result.get_name()] for result in op.operands ]

    @register_stablehlo_op
    def op_func_return(self, context: TranscriptionContext, op: FuncReturnOp):
        # The HLO / MLIR types for function return ops seem to be both in use
        # The behaviour and fields of the two types should be similar, so we
        # simply delegate to the HLO version
        return self.op_return(context, op)

    @register_stablehlo_op
    def op_add(self, context: TranscriptionContext, op: AddOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.add(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_subtract(self, context: TranscriptionContext, op: SubtractOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.sub(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_mul(self, context: TranscriptionContext, op: MulOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.mul(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_div(self, context: TranscriptionContext, op: DivOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]

        # From HLO constraints we know the base-types should line up
        lhs_type = self.__resolve_type(lhs)
        rhs_type = self.__resolve_type(rhs)
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

        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_neg(self, context: TranscriptionContext, op: NegOp):
        # TODO(knielsen): Consider unsigned and more exotic types
        operand = context[op.operand.get_name()]
        minus_one = np.array([-1], dtype=types.nptype_from_builtin(operand.dtype))
        cml_op = mb.mul(x=minus_one, y=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_sign(self, context: TranscriptionContext, op: SignOp):
        operand = context[op.operand.get_name()]
        cml_op = mb.sign(x=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_abs(self, context: TranscriptionContext, op: AbsOp):
        operand = context[op.operand.get_name()]
        cml_op = mb.abs(x=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_log1p(self, context: TranscriptionContext, op: Log1pOp):
        operand = context[op.operand.get_name()]
        one = np.array([1], dtype=types.nptype_from_builtin(self.__resolve_type(operand)))
        x_plus_one = mb.add(x=one, y=operand)
        cml_op = mb.log(x=x_plus_one)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_exp(self, context: TranscriptionContext, op: ExpOp):
        operand = context[op.operand.get_name()]
        cml_op = mb.exp(x=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_transpose(self, context: TranscriptionContext, op: TransposeOp):
        operand = context[op.operand.get_name()]
        perm = np.array(op.permutation, dtype=np.int32)
        cml_op = mb.transpose(x=operand, perm=perm)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_sqrt(self, context: TranscriptionContext, op: SqrtOp):
        operand = context[op.operand.get_name()]
        cml_op = mb.sqrt(x=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_constant(self, context: TranscriptionContext, op: ConstantOp):
        constant = np.array(op.value)
        context.add_variable(op.result.get_name(), constant)

    @register_stablehlo_op
    def op_dot_general(self, context: TranscriptionContext, op: DotGeneralOp):
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
            if len(lst) == 0:
                return 1
            return reduce(lambda a, b: int(a) * int(b), lst)

        def last_column_dot(lhs, rhs):
            # TODO: Figure out if we need to special case broadcasting dims
            return mb.matmul(x=lhs, y=rhs, transpose_y=True)

        lhs_result_dim = [ dim for dim in range(lhs_rank) if dim not in lhs_batching_dim + lhs_contracting_dim  ]
        rhs_result_dim = [ dim for dim in range(rhs_rank) if dim not in rhs_batching_dim + rhs_contracting_dim  ]

        # For both the lhs and rhs, put the dimensions being contracted last
        transposed_lhs = mb.transpose(x=lhs, perm=lhs_batching_dim + lhs_result_dim + lhs_contracting_dim)
        transposed_rhs = mb.transpose(x=rhs, perm=rhs_batching_dim + rhs_result_dim + rhs_contracting_dim)

        # Calculate the result by looping over the contracting dims in order
        result_shape = [ lhs.shape[dim] for dim in lhs_batching_dim ] + [ lhs.shape[dim] for dim in lhs_result_dim ] + [ rhs.shape[dim] for dim in rhs_result_dim ]
        if len(result_shape) == 0:
            # Special case for scalar result
            result_shape = [1]
        result = mb.fill(shape=result_shape)

        # We can utilize that we have a full matrix multiply primitive available, compared to having only
        # a dot-product primitive. Therefore we can avoid iterating over the last dimension in respectively
        # the lhs and rhs tensors
        lhs_result_indices = itertools.product(*[ range(lhs.shape[dim]) for dim in lhs_result_dim[:-1] ])
        rhs_result_indices = itertools.product(*[ range(rhs.shape[dim]) for dim in rhs_result_dim[:-1] ])
        for lhs_idx, rhs_idx in itertools.product(lhs_result_indices, rhs_result_indices):
            # We may have to add an extra slice for the skipped dimension
            result_idx = ()
            result_idx += lhs_idx
            if len(lhs_result_dim) > 0:
                result_idx += (slice(None), )
            result_idx += rhs_idx
            if len(rhs_result_dim) > 0:
                result_idx += (slice(None), )
            # print(f"Calculating result for index {result_idx}, lhs = {lhs_idx}, rhs = {rhs_idx}")

            contracted_element_count = multiply([ lhs.shape[dim] for dim in lhs_contracting_dim ])
            # print(f"contracted_element_count = {contracted_element_count}")
            batch_selector = tuple([ slice(None) for _i in range(len(lhs_batching_dim)) ])
            batch_shape = tuple([ lhs.shape[dim] for dim in lhs_batching_dim ])

            # Reshape the lhs and rhs to have all the contracting dimensions in the end.
            # We will always make them have the shape `(batch_shape, last_dim_shape, contraction_count)``
            # where we may have to set `last_dim_shape` to 1, if the dimension does not exist.
            lhs_for_result_idx = index_by_slices(transposed_lhs, batch_selector + lhs_idx + ( ..., ))
            if len(lhs_result_dim) > 0:
                lhs_reshape_shape = batch_shape + (lhs.shape[lhs_result_dim[-1]],) + (contracted_element_count, )
            else:
                lhs_reshape_shape = batch_shape + (1, contracted_element_count)
            contracted_lhs = mb.reshape(x=lhs_for_result_idx, shape=lhs_reshape_shape)

            rhs_for_result_idx = index_by_slices(transposed_rhs, batch_selector + rhs_idx + ( ..., ))
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
                idx_result = mb.squeeze(x=idx_result, axes=(-1, -2))
            elif len(lhs_result_dim) == 0:
                idx_result = mb.squeeze(x=idx_result, axes=(-2,))
            elif len(rhs_result_dim) == 0:
                idx_result = mb.squeeze(x=idx_result, axes=(-1,))

            # TODO: Consider making this work on iOS<18 by using concatenation
            result = update_tensor_by_slice(result, batch_selector + result_idx, idx_result)

        context.add_variable(op.result.get_name(), result)


    @register_stablehlo_op
    def op_reshape(self, context: TranscriptionContext, op: ReshapeOp):
        x = context[op.operand.get_name()]
        new_shape = op.result.type.shape
        reshape_res = mb.reshape(x=x, shape=new_shape)
        context.add_variable(op.result.get_name(), reshape_res)

    @register_stablehlo_op
    def op_broadcast_in_dim(self, context: TranscriptionContext, op: BroadcastInDimOp):
        # TODO(knielsen): Consider if this is actually correct!
        # CoreML seems to auto-broadcast along the lines of numpy. Therefore this
        # explicit broadcasting op is not necessary.
        x = context[op.operand.get_name()]

        # We handle one special case where the broadcast functions as a reshape
        op_elements = reduce(operator.mul, op.result.type.shape, 1)
        x_elements = reduce(operator.mul, x.shape, 1)
        if op_elements == x_elements:
            # We know that the only possibility is for data to be added, so this is likely a reshape
            x = mb.reshape(x=x, shape=op.result.type.shape)

        # Another special case. If we are broadcasting a constant in all directions, just change the shape
        if len(op.broadcast_dimensions) == 0 and len(x.shape) == 0:
            dtype = types.nptype_from_builtin(self.__resolve_type(x))
            x = mb.mul(x=x, y=np.ones(op.result.type.shape, dtype=dtype))

        context.add_variable(op.result.get_name(), x)

    @register_stablehlo_op
    def op_while(self, context: TranscriptionContext, op: WhileOp):
        def cond(*loop_args):
            params = [ param for param in op.cond.blocks[0].arguments ]
            outputs = self.__invoke_hlo_function(context, "while_cond", params, op.cond, loop_args)
            if len(outputs) != 1:
                raise ValueError("The output of while_cond should always be a single boolean!")
            # TODO(knielsen): Add a check that the output is in fact a single boolean value

            return outputs[0]
        
        def body(*body_args):
            params = [ param for param in op.body.blocks[0].arguments ]
            return self.__invoke_hlo_function(context, "while_body", params, op.body, body_args)

        loop_vars = [ context[arg.get_name()] for arg in op.operands ]
        while_results = mb.while_loop(_cond=cond, _body=body, loop_vars=loop_vars)

        for result_var, while_result in zip(op.results, while_results):
            context.add_variable(result_var.get_name(), while_result)

    @register_stablehlo_op
    def op_compare(self, context: TranscriptionContext, op: CompareOp):
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
        cml_op = cml_op_builder(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_convert(self, context: TranscriptionContext, op: ConvertOp):
        x = context[op.operand.get_name()]
        new_dtype = self.__get_dtype(op.result.type.element_type)
        cml_op = mb.cast(x=x, dtype=self.__dtype_str(new_dtype))
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_select(self, context: TranscriptionContext, op: SelectOp):
        cond = context[op.pred.get_name()]
        a = context[op.on_true.get_name()]
        b = context[op.on_false.get_name()]
        cml_op = mb.select(cond=cond, a=a, b=b)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_dynamic_slice(self, context: TranscriptionContext, op: DynamicSliceOp):
        x = context[op.operand.get_name()]

        # The HLO DynamicSliceOp gives the start indices as seperate 0-dimensional integer variables
        # We need to convert them to a tensor to be compatible with mb.slice_by_size
        start_idx_variables = [ context[i.get_name()] for i in op.start_indices ]
        begin = mb.concat(values=start_idx_variables, axis=0)

        # The slice sizes in HLO are given by a signed integer with 64 bits
        # This is not supported by MIL, so we convert it to a MIL int32 type
        # TODO(knielsen): Overflow check?
        sizes = np.array(op.slice_sizes, dtype=np.int32)

        cml_op = mb.slice_by_size(x=x, begin=begin, size=sizes)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_slice(self, context: TranscriptionContext, op: SliceOp):
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
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_dynamic_update_slice(self, context: TranscriptionContext, op: DynamicUpdateSliceOp):
        x = context[op.operand.get_name()]
        updates = context[op.update.get_name()]

        start_indices = [ context[i.get_name()] for i in op.start_indices ]
        start_indices = mb.concat(values=start_indices, axis=0)
        end_indices = mb.add(x=start_indices, y=op.update.type.shape)

        update_res = mb.slice_update(
            x=x,
            update=updates,
            begin=start_indices,
            end=end_indices,
        )
        context.add_variable(op.result.get_name(), update_res)

    @register_stablehlo_op
    def op_convolution(self, context: TranscriptionContext, op: ConvolutionOp):
        # TODO(knielsen): Support additional dimension specifications
        dim_spec = hlo.ConvDimensionNumbers(op.dimension_numbers)
        if dim_spec.input_batch_dimension != 0 or dim_spec.output_batch_dimension != 0:
            raise ValueError(f"Only the first dimension is currently supported for batch dimension. Got {dim_spec}")
        if dim_spec.input_feature_dimension != 2:
            raise ValueError("Only dimension 2 is currently supported as input feature dimension")
        if dim_spec.output_feature_dimension != 2:
            raise ValueError("Only dimension 2 is currently supported as output feature dimension")

        if op.batch_group_count.value != 1:
            raise ValueError(f"Only a batch group count of 1 is supported. Got {op.batch_group_count.value}")

        # The op.lhs has dimension [batch, d_in*, channels]
        # MIL expects it on the form [batch, channels, d_in*]
        x = context[op.lhs.get_name()] # The inputs comes from vars
        perm = list(range(x.rank))
        # Move the second axis to the end
        perm.append(perm.pop(1))
        x = mb.transpose(x=x, perm=perm)

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
                    raise ValueError("For a conv with lhs dilation we expect the stride to be not set! Because convolution with input dilation d is equivalent to transposed convolution with stride d.")
                # Convolution with input dilation d is equivalent to transposed convolution with stride d
                strides = lhs_dilations

                output_shape = op.result.type.shape
                output_shape.append(output_shape.pop(1)) # Match the format of MIL

                conv_type = partial(
                    mb.conv_transpose,
                    output_shape=output_shape
                )

                # We need to subtract 1 from the padding to make the dimensions line up
                pad -= 1
                if np.any(pad < 0):
                    raise ValueError("The case where the padding turns negative when translating to a transposed convolution is not supported.")

        # The MIL weights should be on form:
        #  - normal convolutions: [C_out, C_in / groups, Kernel*]
        #  - transposed convolutions: [C_in, C_out / groups, Kernel*]
        # HLO has the form [Kernel*, C_in / groups, C_out]
        weight = context[op.rhs.get_name()] # The weights are numpy arrays
        perm = []
        # Move the channel dims
        if conv_type == mb.conv:
            perm.append(len(weight.shape) - 1)
            perm.append(len(weight.shape) - 2)
        else:
            perm.append(len(weight.shape) - 2)
            perm.append(len(weight.shape) - 1)
        for i in range(len(weight.shape) - 2):
            # Kernel perms moved to after the channels
            perm.append(i)
        weight = mb.transpose(x=weight, perm=perm)

        # TODO(knielsen): Make this check more readable!
        # It is executed for conv transpose
        if conv_type != mb.conv:
            # MIL expects the weights to be reversed along the kernel dimensions
            kernel_dimensions = [ i + 2 for i in range(len(weight.shape) - 2) ]
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
        # MIL outputs on the form [batch, channels, d_in*]
        # In the HLO program we expect [batch, d_in*, channels]
        perm = list(range(x.rank))
        # Move the second axis to the end
        perm.append(perm.pop(1))
        cml_conv = mb.transpose(x=cml_conv, perm=perm)

        context.add_variable(op.result.get_name(), cml_conv)

    @register_stablehlo_op
    def op_max(self, context: TranscriptionContext, op: MaxOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_res = mb.maximum(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_res)

    @register_stablehlo_op
    def op_rsqrt(self, context: TranscriptionContext, op: RsqrtOp):
        x = context[op.operand.get_name()]
        mil_res = mb.rsqrt(x=x)
        context.add_variable(op.result.get_name(), mil_res)

    @register_stablehlo_op
    def op_tanh(self, context: TranscriptionContext, op: TanhOp):
        x = context[op.operand.get_name()]
        mil_res = mb.tanh(x=x)
        context.add_variable(op.result.get_name(), mil_res)

    @register_stablehlo_op
    def op_concatenate(self, context: TranscriptionContext, op: ConcatenateOp):
        values = [ context[input.get_name()] for input in op.inputs ]
        mil_res = mb.concat(values=values, axis=op.dimension.value)
        context.add_variable(op.result.get_name(), mil_res)

    def __invoke_hlo_function(self, context: TranscriptionContext, func_name: str, hlo_params, hlo_func_body, cml_args):
        # Enter variable context for the function call
        context.push_function(func_name)

        # Setup arguments for the function
        for hlo_func_param, actual_arg in zip(hlo_params, cml_args):
            context.add_variable(hlo_func_param.get_name(), actual_arg)
        
        # Process the function
        if len(hlo_func_body.blocks) != 1:
            raise ValueError(f"Unsupported function with {len(hlo_func_body.blocks)} blocks")
        outputs = self.process_block(context, hlo_func_body.blocks[0])

        # Exit the function context
        context.pop_function()

        return outputs

    def __resolve_type(self, obj):
        if isinstance(obj, np.ndarray):
            return types.numpy_type_to_builtin_type(obj.dtype)
        return obj.dtype

    def __dtype_str(self, type):
        # TODO(knielsen): Add additional types
        return {
            types.int32: "int32",
            types.fp16: "fp16",
            types.fp32: "fp32",
        }[type]

    def __get_dtype(self, element_type):
        if isinstance(element_type, ir.IntegerType):
            # TODO(knielsen): Handle different kinds of integer types
            return types.int32
        if isinstance(element_type, ir.F16Type):
            return types.fp16
        if isinstance(element_type, ir.F32Type):
            return types.fp32
        raise ValueError(f"Unsupported type {element_type}")
