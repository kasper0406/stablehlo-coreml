"""Small, deliberately strict SMT model for bounded MIL pass proof fixtures.

This is not a general MIL verifier.  It models exactly the operations needed by
the current proof fixtures and raises :class:`UnsupportedMILGraph` for everything
else.  The interpreter has no dependency on this project's pass registry, so a
fixture can run and check any coremltools MIL graph pass whose input and output
stay within the supported semantic subset.
Tensor elements are represented by their raw bits.  Pointwise operations are
uninterpreted functions of those bits, which is stronger than choosing one
floating-point arithmetic model: a proof may only rely on the operation seeing
the same operand bits at the same output index before and after a rewrite.
Reductions similarly use an uninterpreted function over the canonical ordered
input slice.  Their equivalence is conditional on coremltools using the same
reduction kernel and element order when only ``keep_dims`` changes; it does not
claim a real-number or IEEE-754 reduction identity.
``scaled_tanh`` is expanded into the operation order in coremltools' value
inference, ``alpha * tanh(x * beta)``.  Its proof is conditional on the fused
kernel using the same typed multiply rounding and tanh implementation; native
backend precision is outside this model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count, product
from math import prod

import numpy as np
import z3
from coremltools.converters.mil.mil import types


class ProofFailure(AssertionError):
    """Base class for a failed or inconclusive proof."""


class VacuousProof(ProofFailure):
    """The theorem assumptions have no model."""


class InconclusiveProof(ProofFailure):
    """Z3 returned ``unknown`` instead of proving or refuting the goal."""


class InequivalentMILPrograms(ProofFailure):
    """Z3 found an input/index where two MIL programs differ."""


class UnsupportedMILGraph(ProofFailure):
    """The bounded MIL interpreter cannot model the supplied graph exactly."""


class MILGraphContractError(ProofFailure):
    """The compared MIL programs disagree on inputs, outputs, shapes, or dtypes."""


_PROOF_IDS = count()


def prove_for_all(
    assumptions: Sequence[z3.BoolRef],
    conclusion: z3.BoolRef,
    *,
    theorem: str,
    solver_factory: Callable[[], z3.Solver] = z3.Solver,
    counterexample_error: type[ProofFailure] = ProofFailure,
    timeout_ms: int = 10_000,
) -> None:
    """Prove ``assumptions => conclusion``, rejecting vacuity and ``unknown``.

    Free Z3 constants are implicitly universally quantified by checking that
    the assumptions conjoined with the negated conclusion are unsatisfiable.
    """
    solver = solver_factory()
    solver.set(timeout=timeout_ms)
    solver.add(*assumptions)
    status = solver.check()
    if status == z3.unknown:
        raise InconclusiveProof(f"{theorem}: assumptions check returned unknown: {solver.reason_unknown()}")
    if status == z3.unsat:
        raise VacuousProof(f"{theorem}: assumptions are inconsistent")

    solver.add(z3.Not(conclusion))
    status = solver.check()
    if status == z3.unknown:
        raise InconclusiveProof(f"{theorem}: proof returned unknown: {solver.reason_unknown()}")
    if status == z3.sat:
        raise counterexample_error(f"{theorem}: counterexample: {solver.model()}")


def prove_full_slice_axis() -> None:
    """Prove the per-axis index rule used by ``remove_noop_slice_update``.

    The dimension and output index are arbitrary integers.  A positive
    dimension, begin=0, stride=1, and effective end=dimension imply that every
    output index is selected and maps to the identically numbered update index.
    ``end_mask=True`` has the same effective end, so this theorem covers both
    accepted forms of the pass.
    """
    dim, index = z3.Ints("slice_dim slice_index")
    selected = z3.And(index >= 0, index < dim, index % 1 == 0)
    update_index = index / 1
    prove_for_all(
        [dim > 0, index >= 0, index < dim],
        z3.And(selected, update_index == index),
        theorem="full slice_update axis is the update axis",
    )


def _broadcast_index(index: z3.ArithRef, dim: z3.ArithRef) -> z3.ArithRef:
    return z3.If(dim == 1, 0, index)


def prove_broadcast_tile_axis() -> None:
    """Prove one arbitrary axis of ``remove_broadcast_tiles``.

    ``dim`` and ``reps`` describe the tile input.  The tile is a broadcast only
    when either the repetition is one or the input dimension is one.  The other
    consumer operand must be broadcast-compatible with the tiled dimension,
    and the pass additionally requires it to supply that dimension whenever an
    axis is actually replicated.  Under those conditions bypassing the tile
    preserves both output size and the source index read by the tiled operand.
    """
    dim, reps, other, index = z3.Ints("tile_dim tile_reps other_dim tile_index")
    tiled_dim = dim * reps
    before_compatible = z3.Or(other == 1, tiled_dim == 1, other == tiled_dim)
    broadcast_tile = z3.Or(reps == 1, dim == 1)
    pass_shape_guard = z3.Or(reps == 1, other == tiled_dim)
    before_dim = z3.If(tiled_dim == 1, other, tiled_dim)
    after_dim = z3.If(dim == 1, other, dim)
    tiled_operand_index = _broadcast_index(index, tiled_dim)
    tiled_source_index = tiled_operand_index % dim
    bypassed_source_index = _broadcast_index(index, dim)

    prove_for_all(
        [
            dim > 0,
            reps > 0,
            other > 0,
            before_compatible,
            broadcast_tile,
            pass_shape_guard,
            index >= 0,
            index < before_dim,
        ],
        z3.And(after_dim == before_dim, tiled_source_index == bypassed_source_index),
        theorem="broadcast tile axis may be bypassed",
    )


def prove_singleton_insertion_preserves_flat_index() -> None:
    """Prove row-major flattening is unchanged by one inserted size-1 axis.

    ``prefix`` and ``suffix`` stand for products of any number of positive
    dimensions on either side of the inserted axis.  Repeated application
    therefore covers any number of singleton insertions without bounding rank
    or dimension size.
    """
    prefix, suffix, prefix_index, suffix_index = z3.Ints(
        "reshape_prefix reshape_suffix reshape_prefix_index reshape_suffix_index"
    )
    original_flat = prefix_index * suffix + suffix_index
    expanded_flat = (prefix_index * 1 + 0) * suffix + suffix_index
    prove_for_all(
        [
            prefix > 0,
            suffix > 0,
            prefix_index >= 0,
            prefix_index < prefix,
            suffix_index >= 0,
            suffix_index < suffix,
        ],
        original_flat == expanded_flat,
        theorem="singleton insertion preserves row-major flat index",
    )


_DTYPE_BITS = {
    "bool": 1,
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "uint16": 16,
    "int32": 32,
    "fp16": 16,
    "fp32": 32,
}

_BINARY_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "real_div",
        "maximum",
        "minimum",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "pow",
        "floor_div",
        "mod",
    }
)
_REDUCE_OPS = frozenset(
    {
        "reduce_l1_norm",
        "reduce_l2_norm",
        "reduce_log_sum",
        "reduce_log_sum_exp",
        "reduce_max",
        "reduce_mean",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "reduce_sum_square",
    }
)
_UNARY_OPS = frozenset({"tanh"})
_SUPPORTED_OPS = _BINARY_OPS | _REDUCE_OPS | _UNARY_OPS | {
    "const",
    "expand_dims",
    "identity",
    "reshape",
    "scaled_tanh",
    "slice_update",
    "tile",
}
_MAX_CONST_ELEMENTS = 4096
_MAX_REDUCTION_ELEMENTS = 64


def _dtype_name(var) -> str:
    try:
        name = types.builtin_to_string(var.dtype)
    except Exception as exc:  # pragma: no cover - defensive against coremltools API changes
        raise UnsupportedMILGraph(f"cannot identify dtype of {var.name!r}") from exc
    if name not in _DTYPE_BITS:
        raise UnsupportedMILGraph(f"unsupported dtype {name!r} on {var.name!r}")
    return name


def _concrete_shape(var) -> tuple[int, ...]:
    if var.shape is None:
        raise UnsupportedMILGraph(f"unknown shape on {var.name!r}")
    shape = []
    for dim in var.shape:
        if not isinstance(dim, (int, np.integer)):
            raise UnsupportedMILGraph(
                f"symbolic fixture shape on {var.name!r}; symbolic dimensions are covered by the integer lemmas"
            )
        dim = int(dim)
        if dim <= 0:
            raise UnsupportedMILGraph(f"non-positive dimension {dim} on {var.name!r}")
        shape.append(dim)
    return tuple(shape)


def _flat_index(coords: Sequence[z3.ArithRef], shape: Sequence[int]) -> z3.ArithRef:
    if len(coords) != len(shape):
        raise UnsupportedMILGraph(f"index rank {len(coords)} does not match shape rank {len(shape)}")
    flat = z3.IntVal(0)
    for coord, dim in zip(coords, shape):
        flat = flat * dim + coord
    return flat


def _unflatten_index(flat: z3.ArithRef, shape: Sequence[int]) -> tuple[z3.ArithRef, ...]:
    strides = [prod(shape[axis + 1 :]) for axis in range(len(shape))]
    return tuple((flat / stride) % dim for dim, stride in zip(shape, strides))


def _broadcast_coords(
    output_coords: Sequence[z3.ArithRef], input_shape: Sequence[int], output_shape: Sequence[int]
) -> tuple[z3.ArithRef, ...]:
    if len(input_shape) > len(output_shape):
        raise UnsupportedMILGraph(f"cannot broadcast rank {len(input_shape)} to rank {len(output_shape)}")
    offset = len(output_shape) - len(input_shape)
    result = []
    for axis, dim in enumerate(input_shape):
        out_dim = output_shape[offset + axis]
        if dim not in (1, out_dim):
            raise UnsupportedMILGraph(f"incompatible broadcast dimensions {dim} and {out_dim}")
        result.append(z3.IntVal(0) if dim == 1 else output_coords[offset + axis])
    return tuple(result)


def _broadcast_shape(lhs: Sequence[int], rhs: Sequence[int]) -> tuple[int, ...]:
    rank = max(len(lhs), len(rhs))
    lhs = (1,) * (rank - len(lhs)) + tuple(lhs)
    rhs = (1,) * (rank - len(rhs)) + tuple(rhs)
    result = []
    for left_dim, right_dim in zip(lhs, rhs):
        if left_dim == right_dim or right_dim == 1:
            result.append(left_dim)
        elif left_dim == 1:
            result.append(right_dim)
        else:
            raise UnsupportedMILGraph(f"incompatible broadcast dimensions {left_dim} and {right_dim}")
    return tuple(result)


def _const_bits(value, dtype_name: str) -> int:
    if dtype_name == "bool":
        return int(bool(value))
    numpy_dtype = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "fp16": np.float16,
        "fp32": np.float32,
    }[dtype_name]
    unsigned_dtype = {
        1: np.uint8,
        2: np.uint16,
        4: np.uint32,
    }[np.dtype(numpy_dtype).itemsize]
    scalar = np.asarray(value, dtype=numpy_dtype).reshape(()).view(unsigned_dtype)
    return int(scalar)


@dataclass(frozen=True)
class _TensorExpr:
    shape: tuple[int, ...]
    dtype: str
    at: Callable[[tuple[z3.ArithRef, ...]], z3.BitVecRef]


class _MILInterpreter:
    def __init__(self, before, after):
        self._programs = (before, after)
        self._proof_id = next(_PROOF_IDS)
        self._input_functions: dict[tuple[str, tuple[int, ...], str], z3.FuncDeclRef] = {}
        self._op_functions: dict[tuple[str, str, str, str], z3.FuncDeclRef] = {}
        self._unary_functions: dict[tuple[str, str, str], z3.FuncDeclRef] = {}
        self._reduce_functions: dict[tuple, z3.FuncDeclRef] = {}
        self._cache: dict[int, _TensorExpr] = {}

    def _validate_program(self, prog) -> None:
        if set(prog.functions) != {"main"}:
            raise UnsupportedMILGraph("proof fixtures must contain exactly the 'main' function")
        for op in prog.functions["main"].operations:
            if op.blocks:
                raise UnsupportedMILGraph(f"nested blocks are unsupported (operation {op.op_type!r})")
            if op.op_type not in _SUPPORTED_OPS:
                raise UnsupportedMILGraph(f"unsupported operation {op.op_type!r}")
            if len(op.outputs) != 1:
                raise UnsupportedMILGraph(f"multi-output operation {op.op_type!r} is unsupported")
            _concrete_shape(op.outputs[0])
            _dtype_name(op.outputs[0])

    def validate_contract(self) -> None:
        for prog in self._programs:
            self._validate_program(prog)
        before_fn, after_fn = (prog.functions["main"] for prog in self._programs)
        if tuple(before_fn.inputs) != tuple(after_fn.inputs):
            raise MILGraphContractError("function input names or order changed")
        for name in before_fn.inputs:
            before_var, after_var = before_fn.inputs[name], after_fn.inputs[name]
            if _concrete_shape(before_var) != _concrete_shape(after_var):
                raise MILGraphContractError(f"input {name!r} changed shape")
            if _dtype_name(before_var) != _dtype_name(after_var):
                raise MILGraphContractError(f"input {name!r} changed dtype")
        if len(before_fn.outputs) != len(after_fn.outputs):
            raise MILGraphContractError("function output count changed")
        if len(before_fn.outputs) == 0:
            raise MILGraphContractError("functions with no outputs cannot be compared")
        for output_number, (before_var, after_var) in enumerate(zip(before_fn.outputs, after_fn.outputs)):
            if before_var.name != after_var.name:
                raise MILGraphContractError(f"output {output_number} changed name")
            if _concrete_shape(before_var) != _concrete_shape(after_var):
                raise MILGraphContractError(f"output {output_number} changed shape")
            if _dtype_name(before_var) != _dtype_name(after_var):
                raise MILGraphContractError(f"output {output_number} changed dtype")

    def _input(self, var) -> _TensorExpr:
        shape, dtype = _concrete_shape(var), _dtype_name(var)
        key = (var.name, shape, dtype)
        function = self._input_functions.get(key)
        if function is None:
            function = z3.Function(
                f"p{self._proof_id}_input_{var.name}_{dtype}", z3.IntSort(), z3.BitVecSort(_DTYPE_BITS[dtype])
            )
            self._input_functions[key] = function
        return _TensorExpr(shape, dtype, lambda coords: function(_flat_index(coords, shape)))

    def _const(self, var) -> _TensorExpr:
        shape, dtype = _concrete_shape(var), _dtype_name(var)
        value = getattr(var, "val", None)
        if value is None:
            raise UnsupportedMILGraph(f"non-constant value on const {var.name!r}")
        array = np.asarray(value)
        size = prod(shape)
        if array.size != size:
            raise UnsupportedMILGraph(f"const {var.name!r} value does not match its shape")
        if size > _MAX_CONST_ELEMENTS:
            raise UnsupportedMILGraph(f"const {var.name!r} exceeds {_MAX_CONST_ELEMENTS} proof elements")
        values = [_const_bits(item, dtype) for item in array.reshape(-1)]
        width = _DTYPE_BITS[dtype]

        def at(coords):
            flat = _flat_index(coords, shape)
            expression = z3.BitVecVal(values[-1], width)
            for position in range(len(values) - 2, -1, -1):
                expression = z3.If(flat == position, z3.BitVecVal(values[position], width), expression)
            return expression

        return _TensorExpr(shape, dtype, at)

    def tensor(self, var) -> _TensorExpr:
        cached = self._cache.get(id(var))
        if cached is not None:
            return cached
        op = getattr(var, "op", None)
        if op is None:
            result = self._input(var)
        elif op.op_type == "const":
            result = self._const(var)
        elif op.op_type == "identity":
            source = self.tensor(op.x)
            result = _TensorExpr(_concrete_shape(var), _dtype_name(var), source.at)
            if (result.shape, result.dtype) != (source.shape, source.dtype):
                raise UnsupportedMILGraph("identity changed shape or dtype")
        elif op.op_type == "reshape":
            result = self._reshape(op, var)
        elif op.op_type == "expand_dims":
            result = self._expand_dims(op, var)
        elif op.op_type == "tile":
            result = self._tile(op, var)
        elif op.op_type == "slice_update":
            result = self._slice_update(op, var)
        elif op.op_type in _REDUCE_OPS:
            result = self._reduce(op, var)
        elif op.op_type in _UNARY_OPS:
            result = self._unary(op, var)
        elif op.op_type == "scaled_tanh":
            result = self._scaled_tanh(op, var)
        elif op.op_type in _BINARY_OPS:
            result = self._binary(op, var)
        else:  # pragma: no cover - _validate_program reports this first
            raise UnsupportedMILGraph(f"unsupported producer operation {op.op_type!r}")
        self._cache[id(var)] = result
        return result

    def _reshape(self, op, output_var) -> _TensorExpr:
        source = self.tensor(op.x)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        shape_value = getattr(op.shape, "val", None)
        if shape_value is None:
            raise UnsupportedMILGraph("reshape shape must be constant")
        requested = tuple(int(dim) for dim in np.asarray(shape_value).reshape(-1))
        if requested.count(-1) > 1:
            raise UnsupportedMILGraph("reshape shape may contain at most one -1")
        if any(dim < -1 for dim in requested):
            raise UnsupportedMILGraph("reshape proof subset supports nonnegative dimensions and one optional -1")
        if 0 in requested:
            if len(requested) != len(source.shape):
                raise UnsupportedMILGraph("reshape zero-copy dimensions require the input and output ranks to match")
            requested = tuple(source.shape[axis] if dim == 0 else dim for axis, dim in enumerate(requested))
        source_volume = prod(source.shape)
        known_volume = prod(dim for dim in requested if dim != -1)
        if -1 in requested:
            if known_volume <= 0 or source_volume % known_volume:
                raise UnsupportedMILGraph("reshape -1 cannot be inferred exactly")
            inferred = source_volume // known_volume
            resolved = tuple(inferred if dim == -1 else dim for dim in requested)
        else:
            resolved = requested
        if resolved != output_shape or prod(resolved) != source_volume:
            raise UnsupportedMILGraph("reshape shape operand disagrees with input/output volume")
        if output_dtype != source.dtype:
            raise UnsupportedMILGraph("reshape changed dtype")

        def at(coords):
            return source.at(_unflatten_index(_flat_index(coords, output_shape), source.shape))

        return _TensorExpr(output_shape, output_dtype, at)

    def _unary_function(self, op_type: str, input_dtype: str, output_dtype: str):
        key = (op_type, input_dtype, output_dtype)
        function = self._unary_functions.get(key)
        if function is None:
            function = z3.Function(
                f"p{self._proof_id}_op_{'_'.join(key)}",
                z3.BitVecSort(_DTYPE_BITS[input_dtype]),
                z3.BitVecSort(_DTYPE_BITS[output_dtype]),
            )
            self._unary_functions[key] = function
        return function

    def _binary_function(self, op_type: str, lhs_dtype: str, rhs_dtype: str, output_dtype: str):
        key = (op_type, lhs_dtype, rhs_dtype, output_dtype)
        function = self._op_functions.get(key)
        if function is None:
            function = z3.Function(
                f"p{self._proof_id}_op_{'_'.join(key)}",
                z3.BitVecSort(_DTYPE_BITS[lhs_dtype]),
                z3.BitVecSort(_DTYPE_BITS[rhs_dtype]),
                z3.BitVecSort(_DTYPE_BITS[output_dtype]),
            )
            self._op_functions[key] = function
        return function

    def _unary(self, op, output_var) -> _TensorExpr:
        source = self.tensor(op.x)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        if output_shape != source.shape or output_dtype != source.dtype:
            raise UnsupportedMILGraph(f"{op.op_type} changed shape or dtype")
        function = self._unary_function(op.op_type, source.dtype, output_dtype)
        return _TensorExpr(output_shape, output_dtype, lambda coords: function(source.at(coords)))

    def _scaled_tanh(self, op, output_var) -> _TensorExpr:
        source = self.tensor(op.x)
        if getattr(op.alpha, "val", None) is None or getattr(op.beta, "val", None) is None:
            raise UnsupportedMILGraph("scaled_tanh alpha and beta must be compile-time constants")
        alpha, beta = self.tensor(op.alpha), self.tensor(op.beta)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        if output_shape != source.shape or output_dtype != source.dtype:
            raise UnsupportedMILGraph("scaled_tanh changed shape or dtype")
        if alpha.shape != () or beta.shape != ():
            raise UnsupportedMILGraph("scaled_tanh alpha and beta must be rank-0 scalars")
        if source.dtype not in {"fp16", "fp32"}:
            raise UnsupportedMILGraph("scaled_tanh input must be fp16 or fp32")
        if alpha.dtype != source.dtype or beta.dtype != source.dtype:
            raise UnsupportedMILGraph("scaled_tanh parameter dtype differs from its input dtype")
        multiply = self._binary_function("mul", source.dtype, source.dtype, source.dtype)
        hyperbolic_tangent = self._unary_function("tanh", source.dtype, source.dtype)
        alpha_bits = alpha.at(tuple(z3.IntVal(0) for _ in alpha.shape))
        beta_bits = beta.at(tuple(z3.IntVal(0) for _ in beta.shape))

        def at(coords):
            inner = multiply(source.at(coords), beta_bits)
            return multiply(alpha_bits, hyperbolic_tangent(inner))

        return _TensorExpr(output_shape, output_dtype, at)

    def _expand_dims(self, op, output_var) -> _TensorExpr:
        source = self.tensor(op.x)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        axes_value = getattr(op.axes, "val", None)
        if axes_value is None:
            raise UnsupportedMILGraph("expand_dims axes must be constant")
        raw_axes = tuple(int(axis) for axis in np.asarray(axes_value).reshape(-1))
        output_rank = len(source.shape) + len(raw_axes)
        if any(axis < -output_rank or axis >= output_rank for axis in raw_axes):
            raise UnsupportedMILGraph("expand_dims axis is out of range")
        axes = tuple(sorted(axis + output_rank if axis < 0 else axis for axis in raw_axes))
        if len(set(axes)) != len(axes):
            raise UnsupportedMILGraph("duplicate expand_dims axes are outside the proof subset")
        expected_shape = list(source.shape)
        for axis in axes:
            expected_shape.insert(axis, 1)
        if tuple(expected_shape) != output_shape:
            raise UnsupportedMILGraph("expand_dims axes disagree with its output shape")
        if output_dtype != source.dtype:
            raise UnsupportedMILGraph("expand_dims changed dtype")
        axes_set = set(axes)
        return _TensorExpr(
            output_shape,
            output_dtype,
            lambda coords: source.at(tuple(coord for axis, coord in enumerate(coords) if axis not in axes_set)),
        )

    @staticmethod
    def _reduction_axes(op, rank: int) -> tuple[int, ...]:
        axes_var = op.inputs.get("axes")
        if axes_var is None:
            return tuple(range(rank))
        axes_value = getattr(axes_var, "val", None)
        if axes_value is None:
            raise UnsupportedMILGraph("reduction axes must be constant")
        raw_axes = tuple(int(axis) for axis in np.asarray(axes_value).reshape(-1))
        if rank == 0 and raw_axes:
            raise UnsupportedMILGraph("a scalar reduction cannot name an axis")
        if any(axis < -rank or axis >= rank for axis in raw_axes):
            raise UnsupportedMILGraph("reduction axis is out of range")
        axes = tuple(sorted(axis + rank if axis < 0 else axis for axis in raw_axes))
        if len(set(axes)) != len(axes):
            raise UnsupportedMILGraph("duplicate reduction axes are outside the proof subset")
        return axes

    def _reduce(self, op, output_var) -> _TensorExpr:
        source = self.tensor(op.x)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        axes = self._reduction_axes(op, len(source.shape))
        keep_dims_var = op.inputs.get("keep_dims")
        if keep_dims_var is None:
            keep_dims = False
        else:
            keep_dims_value = getattr(keep_dims_var, "val", None)
            if keep_dims_value is None:
                raise UnsupportedMILGraph("reduction keep_dims must be constant")
            keep_dims = bool(np.asarray(keep_dims_value).reshape(()))
        axes_set = set(axes)
        if keep_dims:
            expected_shape = tuple(1 if axis in axes_set else dim for axis, dim in enumerate(source.shape))
        else:
            expected_shape = tuple(dim for axis, dim in enumerate(source.shape) if axis not in axes_set)
        if output_shape != expected_shape:
            raise UnsupportedMILGraph("reduction axes/keep_dims disagree with its output shape")
        if output_dtype != source.dtype:
            raise UnsupportedMILGraph("reduction changed dtype")
        reduced_shape = tuple(source.shape[axis] for axis in axes)
        reduction_volume = prod(reduced_shape)
        if reduction_volume > _MAX_REDUCTION_ELEMENTS:
            raise UnsupportedMILGraph(f"reduction exceeds {_MAX_REDUCTION_ELEMENTS} proof elements per output")
        key = (op.op_type, source.dtype, output_dtype, axes, source.shape)
        function = self._reduce_functions.get(key)
        if function is None:
            bit_sort = z3.BitVecSort(_DTYPE_BITS[source.dtype])
            function = z3.Function(
                f"p{self._proof_id}_{op.op_type}_{source.dtype}"
                f"_axes{'_'.join(map(str, axes)) or 'none'}_shape{'_'.join(map(str, source.shape)) or 'scalar'}",
                *([bit_sort] * reduction_volume),
                z3.BitVecSort(_DTYPE_BITS[output_dtype]),
            )
            self._reduce_functions[key] = function

        def at(coords):
            base_coords = []
            output_position = 0
            for axis in range(len(source.shape)):
                if axis in axes_set:
                    base_coords.append(None)
                    if keep_dims:
                        output_position += 1
                else:
                    base_coords.append(coords[output_position])
                    output_position += 1
            values = []
            for reduced_coords in product(*(range(dim) for dim in reduced_shape)):
                input_coords = list(base_coords)
                for axis, reduced_coord in zip(axes, reduced_coords):
                    input_coords[axis] = z3.IntVal(reduced_coord)
                values.append(source.at(tuple(input_coords)))
            return function(*values)

        return _TensorExpr(output_shape, output_dtype, at)

    def _tile(self, op, output_var) -> _TensorExpr:
        source = self.tensor(op.x)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        reps_val = getattr(op.reps, "val", None)
        if reps_val is None:
            raise UnsupportedMILGraph("tile reps must be constant")
        reps = tuple(int(value) for value in np.asarray(reps_val).reshape(-1))
        if len(reps) != len(source.shape) or any(rep <= 0 for rep in reps):
            raise UnsupportedMILGraph("tile reps must be positive and match the input rank")
        if output_shape != tuple(dim * rep for dim, rep in zip(source.shape, reps)):
            raise UnsupportedMILGraph("tile output shape disagrees with reps")
        if output_dtype != source.dtype:
            raise UnsupportedMILGraph("tile changed dtype")
        return _TensorExpr(
            output_shape,
            output_dtype,
            lambda coords: source.at(tuple(coord % dim for coord, dim in zip(coords, source.shape))),
        )

    @staticmethod
    def _const_vector(var, *, name: str, length: int, default=None) -> tuple:
        if var is None:
            if default is None:
                raise UnsupportedMILGraph(f"slice_update {name} is absent")
            return tuple(default for _ in range(length))
        value = getattr(var, "val", None)
        if value is None:
            raise UnsupportedMILGraph(f"slice_update {name} must be constant")
        result = tuple(np.asarray(value).reshape(-1).tolist())
        if len(result) != length:
            raise UnsupportedMILGraph(f"slice_update {name} rank mismatch")
        return result

    def _slice_update(self, op, output_var) -> _TensorExpr:
        source, update = self.tensor(op.x), self.tensor(op.update)
        rank = len(source.shape)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        begin = tuple(int(x) for x in self._const_vector(op.begin, name="begin", length=rank))
        end = tuple(int(x) for x in self._const_vector(op.end, name="end", length=rank))
        stride = tuple(int(x) for x in self._const_vector(op.stride, name="stride", length=rank, default=1))
        begin_mask = tuple(bool(x) for x in self._const_vector(op.begin_mask, name="begin_mask", length=rank, default=False))
        end_mask = tuple(bool(x) for x in self._const_vector(op.end_mask, name="end_mask", length=rank, default=False))
        squeeze_mask = tuple(
            bool(x) for x in self._const_vector(op.squeeze_mask, name="squeeze_mask", length=rank, default=False)
        )
        if any(begin_mask) or any(squeeze_mask):
            raise UnsupportedMILGraph("proof fixtures only support plain, non-squeezing slice_update begins")
        effective_end = tuple(dim if masked else stop for dim, masked, stop in zip(source.shape, end_mask, end))
        if any(start < 0 for start in begin) or any(stop < 0 for stop in effective_end):
            raise UnsupportedMILGraph("negative slice_update indices are unsupported")
        if any(start > dim or stop > dim or start > stop for start, stop, dim in zip(begin, effective_end, source.shape)):
            raise UnsupportedMILGraph("unclipped slice_update indices must lie within the source shape")
        if any(step <= 0 for step in stride):
            raise UnsupportedMILGraph("non-positive slice_update strides are unsupported")
        expected_update_shape = tuple(
            max(0, (stop - start + step - 1) // step)
            for start, stop, step in zip(begin, effective_end, stride)
        )
        if expected_update_shape != update.shape:
            raise UnsupportedMILGraph("slice_update update shape disagrees with its index vectors")
        if output_shape != source.shape or output_dtype != source.dtype or update.dtype != source.dtype:
            raise UnsupportedMILGraph("slice_update changed tensor shape or dtype")

        def at(coords):
            selected = z3.And(
                *(
                    z3.And(coord >= start, coord < stop, (coord - start) % step == 0)
                    for coord, start, stop, step in zip(coords, begin, effective_end, stride)
                )
            )
            update_coords = tuple((coord - start) / step for coord, start, step in zip(coords, begin, stride))
            return z3.If(selected, update.at(update_coords), source.at(coords))

        return _TensorExpr(output_shape, output_dtype, at)

    def _binary(self, op, output_var) -> _TensorExpr:
        lhs, rhs = self.tensor(op.x), self.tensor(op.y)
        output_shape, output_dtype = _concrete_shape(output_var), _dtype_name(output_var)
        inferred_shape = _broadcast_shape(lhs.shape, rhs.shape)
        if inferred_shape != output_shape:
            raise UnsupportedMILGraph(
                f"{op.op_type} output shape {output_shape} disagrees with broadcast shape {inferred_shape}"
            )
        function = self._binary_function(op.op_type, lhs.dtype, rhs.dtype, output_dtype)

        def at(coords):
            lhs_coords = _broadcast_coords(coords, lhs.shape, output_shape)
            rhs_coords = _broadcast_coords(coords, rhs.shape, output_shape)
            return function(lhs.at(lhs_coords), rhs.at(rhs_coords))

        return _TensorExpr(output_shape, output_dtype, at)


def prove_mil_programs_equivalent(before, after) -> None:
    """Prove equality at every output index for two bounded MIL fixtures.

    The shapes are concrete only to keep translation small; tensor values and
    the selected output index remain symbolic.  The production pass itself is
    run by the tests before this function is called.
    """
    interpreter = _MILInterpreter(before, after)
    interpreter.validate_contract()
    before_outputs = before.functions["main"].outputs
    after_outputs = after.functions["main"].outputs
    for output_number, (before_var, after_var) in enumerate(zip(before_outputs, after_outputs)):
        before_expr, after_expr = interpreter.tensor(before_var), interpreter.tensor(after_var)
        coords = tuple(
            z3.Int(f"p{interpreter._proof_id}_out{output_number}_axis{axis}")
            for axis in range(len(before_expr.shape))
        )
        bounds = [z3.And(coord >= 0, coord < dim) for coord, dim in zip(coords, before_expr.shape)]
        prove_for_all(
            bounds,
            before_expr.at(coords) == after_expr.at(coords),
            theorem=f"MIL output {output_number} is bit-identical",
            counterexample_error=InequivalentMILPrograms,
        )
