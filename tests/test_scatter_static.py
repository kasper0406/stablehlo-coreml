"""Static scatter semantics, routing safety, and dynamic fallback coverage."""
from math import prod

import coremltools as ct
import jax
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.types.symbolic import is_symbolic, k_used_symbols
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

from stablehlo_coreml.converter import convert
from stablehlo_coreml.scatter import static_scatter_indices, static_scatter_sizes
from tests.utils import run_and_compare_hlo_module, run_and_compare_specific_input, run_and_compare_symbolic

MODES = ('update', 'add', 'sub', 'mul', 'div', 'min', 'max')


@pytest.fixture(autouse=True)
def _restore_symbol_registry():
    # Expected conversion failures bypass ct.convert's normal symbol cleanup.
    before = k_used_symbols.copy()
    yield
    k_used_symbols.clear()
    k_used_symbols.update(before)


def _module(shape, index_shape, mode='add', dtype='f32', mapping=None):
    k = index_shape[-1]
    mapping = tuple(range(k)) if mapping is None else mapping
    rest = tuple(shape[i] for i in range(len(shape)) if i not in mapping)
    update_shape = (*index_shape[:-1], *rest)
    def tensor(s, t):
        return 'tensor<' + 'x'.join((*map(str, s), t)) + '>'
    data_t, index_t, update_t = tensor(shape, dtype), tensor(index_shape, 'i32'), tensor(update_shape, dtype)
    window = list(range(len(index_shape) - 1, len(update_shape)))
    scalar = tensor((), dtype)
    if mode == 'update':
        body = f'stablehlo.return %b : {scalar}'
    else:
        op = {'min': 'minimum', 'max': 'maximum', 'sub': 'subtract', 'mul': 'multiply', 'div': 'divide'}.get(mode, mode)
        body = f'%r = stablehlo.{op} %a, %b : {scalar}\n stablehlo.return %r : {scalar}'
    return ir.Module.parse(f'''module {{
      func.func @main(%data: {data_t}, %indices: {index_t}, %updates: {update_t}) -> {data_t} {{
        %out = "stablehlo.scatter"(%data, %indices, %updates) ({{
          ^bb0(%a: {scalar}, %b: {scalar}):
            {body}
        }}) {{indices_are_sorted = false, unique_indices = false,
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = {window}, inserted_window_dims = {sorted(mapping)},
                scatter_dims_to_operand_dims = {list(mapping)}, index_vector_dim = {len(index_shape)-1}>
        }} : ({data_t}, {index_t}, {update_t}) -> {data_t}
        return %out : {data_t}
      }}
    }}''', context=jax_mlir.make_ir_context())


def _walk(block):
    for op in block.operations:
        yield op
        for child in op.blocks:
            yield from _walk(child)


def _audit(model):
    ops = list(_walk(model._mil_program.functions['main']))
    assert all(op.op_type not in ('non_zero', 'gather', 'gather_nd', 'gather_along_axis') for op in ops)
    assert all(not is_symbolic(d) for op in ops for v in op.outputs for d in v.shape)
    return ops


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
@pytest.mark.parametrize('shape,mapping,index_batch', [
    ((4,), (0,), (6,)),
    ((3, 4), (0,), (2, 3)),
    ((3, 4), (1, 0), (6,)),
    ((3, 4, 2), (0,), (6,)),
    ((3, 4, 2), (1, 0), (2, 3)),
])
def test_modes_and_full_slices(mode, dtype, shape, mapping, index_batch):
    """Nonzero operand, duplicates, and arbitrary OOB int32 values.

    Duplicate overwrite values are equal; no conflicting-overwrite order is
    required. Values are exactly representable in all tested arithmetic modes.
    """
    k = len(mapping)
    indices = np.zeros((prod(index_batch), k), dtype=np.int32)
    indices[:2] = np.arange(k)
    indices[2] = 1 + np.arange(k)
    indices[3] = -1
    indices[4] = np.iinfo(np.int32).max
    indices[5] = np.iinfo(np.int32).min
    indices = indices.reshape((*index_batch, k))
    rest = shape[k:]
    data = np.full(shape, 16, dtype=dtype)
    updates = np.full((*index_batch, *rest), 2, dtype=dtype)
    updates.reshape((-1, *rest))[2] = 32
    expected = data.copy()
    for row, value in zip(indices.reshape(-1, k), updates.reshape((-1, *rest))):
        dest = tuple(row[mapping.index(axis)] for axis in range(k))
        if not all(0 <= d < shape[i] for i, d in enumerate(dest)):
            continue
        old = expected[dest]
        expected[dest] = {
            'update': value, 'add': old + value,
            'sub': old - value, 'mul': old * value,
            'div': old / value, 'min': np.minimum(old, value),
            'max': np.maximum(old, value),
        }[mode]
    model = run_and_compare_hlo_module(
        _module(shape, indices.shape, mode, 'f32' if dtype == np.float32 else 'i32', mapping),
        (data, indices, updates), expected, atol=0, rtol=0,
    )
    ops = _audit(model)
    assert any(op.op_type == 'scatter' and op.mode.val == mode for op in ops)


@pytest.mark.parametrize('indices', [np.array([[1]], np.int32), np.array([[-1], [5], [9]], np.int32)])
def test_single_row_and_all_oob(indices):
    data = np.full(4, 16, np.float32)
    updates = np.full(indices.shape[0], 2, np.float32)
    expected = data.copy()
    if indices[0, 0] == 1:
        expected[1] = 18
    model = run_and_compare_hlo_module(_module(data.shape, indices.shape), (data, indices, updates), expected)
    _audit(model)


def test_bool_cast_and_empty():
    def fn(data, indices, updates):
        dims = jax.lax.ScatterDimensionNumbers((), (0,), (0,))
        return jax.lax.scatter(data, indices, updates, dims, mode='drop')
    data = np.array([True, False, True, False])
    for indices in [np.array([[1], [1], [-1], [10]], np.int32), np.empty((0, 1), np.int32)]:
        model = run_and_compare_specific_input(fn, (data, indices, np.ones(len(indices), bool)))
        _audit(model)


def test_unique_dummy_routing():
    @mb.program(input_specs=[mb.TensorSpec(shape=(5, 2), dtype=types.int32)])
    def program(indices):
        return static_scatter_indices(indices, (3, 4), 5, (0, 1))
    model = ct.convert(program, source='milinternal', compute_units=ct.ComputeUnit.CPU_ONLY,
                       minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32)
    indices = np.array([[2, 3], [-1, 0], [0, 4], [2**31-1, 0], [0, -2**31]], np.int32)
    actual = next(iter(model.predict({'indices': indices}).values()))
    np.testing.assert_array_equal(actual, [11, 13, 14, 15, 16])


def test_size_guard_without_allocations():
    limit = np.iinfo(np.int32).max
    assert static_scatter_sizes((limit - 2,), (2, 1), 1) == (limit - 2, 2)
    for shape, indices, k in [((limit - 1,), (2, 1), 1), ((2**40, 2**40), (2, 2), 2)]:
        with pytest.raises(ValueError, match="scatter operand front too large for int32 indexing"):
            static_scatter_sizes(shape, indices, k)
    symbol = get_new_symbol()
    assert static_scatter_sizes((4,), (symbol, 1), 1) is None
    assert static_scatter_sizes((symbol,), (2, 1), 1) is None
    assert static_scatter_sizes((4, symbol), (2, 1), 1) == (4, 2)
    assert static_scatter_sizes((0, 4), (2, 1), 1) == (0, 2)


def test_large_valid_index_not_narrowed_to_int16():
    data = np.full((40000, 2), 16, np.float32)
    indices = np.array([[35000], [35000], [-2**31], [2**31 - 1]], np.int32)
    updates = np.full((4, 2), 2, np.float32)
    expected = data.copy()
    expected[35000] += 4
    model = run_and_compare_hlo_module(_module(data.shape, indices.shape), (data, indices, updates), expected, atol=0, rtol=0)
    _audit(model)


@pytest.mark.parametrize('symbolic_front', [False, True])
def test_symbolic_fallback(symbolic_front):
    def fn(data, indices, updates):
        dims = jax.lax.ScatterDimensionNumbers((), (0,), (0,))
        return jax.lax.scatter_add(data, indices, updates, dims, mode='drop')
    n, = jax.export.symbolic_shape('n')
    specs = [jax.ShapeDtypeStruct((n if symbolic_front else 4,), np.float32),
             jax.ShapeDtypeStruct((2 if symbolic_front else n, 1), np.int32),
             jax.ShapeDtypeStruct((2 if symbolic_front else n,), np.float32)]
    inputs = []
    for size in [2, 5]:
        rows = 2 if symbolic_front else size
        indices = np.arange(rows, dtype=np.int32)[:, None] - 1
        inputs.append((np.full(size if symbolic_front else 4, 16, np.float32), indices, np.full(rows, 2, np.float32)))
    run_and_compare_symbolic(fn, specs, inputs)


def test_noncontiguous_mapping_still_rejected():
    module = _module((3, 4), (2, 1), mapping=(1,))
    with pytest.raises(ValueError, match='contiguous'):
        convert(module, minimum_deployment_target=ct.target.iOS18)


def test_captured_indices_in_sibling_blocks():
    def fn(data, indices, flag):
        dims = jax.lax.ScatterDimensionNumbers((), (0,), (0,))

        def branch(scale):
            return jax.lax.scatter_add(data, indices, jax.numpy.full((3,), scale), dims, mode='drop')

        return jax.lax.cond(flag, lambda: branch(2.), lambda: branch(4.))

    for flag in [True, False]:
        model = run_and_compare_specific_input(
            fn, (np.full(4, 16, np.float32), np.array([[1], [1], [-1]], np.int32), np.array(flag)),
        )
        _audit(model)


@pytest.mark.parametrize('aliased', [False, True])
def test_symbolic_trailing_full_slice(aliased):
    def fn(data, indices, updates):
        dims = jax.lax.ScatterDimensionNumbers((1,), (0,), (0,))
        return jax.lax.scatter_add(data, indices, data if aliased else updates, dims, mode='drop')

    n, = jax.export.symbolic_shape('n')
    specs = [jax.ShapeDtypeStruct((4, n), np.float32), jax.ShapeDtypeStruct((4, 1), np.int32),
             jax.ShapeDtypeStruct((4, n), np.float32)]
    inputs = [(np.full((4, width), 16, np.float32), np.array([[1], [-1], [1], [9]], np.int32),
               np.full((4, width), 2, np.float32)) for width in [2, 5]]
    if aliased:
        run_and_compare_symbolic(fn, specs, inputs)
    else:
        module = ir.Module.parse(jax.export.export(jax.jit(fn))(*specs).mlir_module(), context=jax_mlir.make_ir_context())
        with pytest.raises(ValueError, match='unproven symbolic extents'):
            convert(module, minimum_deployment_target=ct.target.iOS18)



def test_zero_trailing_window_returns_operand():
    program = convert(_module((4, 0), (3, 1)), minimum_deployment_target=ct.target.iOS18)
    main = program.functions['main']
    assert main.outputs[0] is next(iter(main.inputs.values()))
    assert not any(op.op_type in ('scatter', 'scatter_nd') for op in _walk(main))


@pytest.mark.parametrize('indices', [np.array([1, 2], np.int32), np.array([-1, 2], np.int32)])
def test_rank_one_index_vector(indices):
    data = np.full((3, 4, 2), 16, np.float32)
    updates = np.array([2, 3], np.float32)
    expected = data.copy()
    if indices[0] >= 0:
        expected[1, 2] += updates
    model = run_and_compare_hlo_module(_module(data.shape, indices.shape), (data, indices, updates), expected)
    _audit(model)


def test_symbolic_partial_window_rejected_clearly():
    def fn(data, indices, updates):
        dims = jax.lax.ScatterDimensionNumbers((0,), (), (0,))
        return jax.lax.scatter_add(data, indices, updates, dims, mode='drop')

    n, = jax.export.symbolic_shape('n', constraints=('n >= 2',))
    exported = jax.export.export(jax.jit(fn))(
        jax.ShapeDtypeStruct((n,), np.float32), jax.ShapeDtypeStruct((1,), np.int32),
        jax.ShapeDtypeStruct((2,), np.float32),
    )
    module = ir.Module.parse(exported.mlir_module(), context=jax_mlir.make_ir_context())
    with pytest.raises(ValueError, match='partial-window operand front dimension must be static'):
        convert(module, minimum_deployment_target=ct.target.iOS18)


def test_extra_nontrailing_window_axes_rejected():
    def fn(data, indices, updates):
        dims = jax.lax.ScatterDimensionNumbers((0, 2), (), (0,))
        return jax.lax.scatter_add(data, indices, updates, dims, mode='drop')
    specs = [jax.ShapeDtypeStruct((4, 2), np.float32), jax.ShapeDtypeStruct((2, 1), np.int32),
             jax.ShapeDtypeStruct((2, 2, 1), np.float32)]
    module = ir.Module.parse(jax.export.export(jax.jit(fn))(*specs).mlir_module(), context=jax_mlir.make_ir_context())
    with pytest.raises(ValueError, match='canonical full-slice'):
        convert(module, minimum_deployment_target=ct.target.iOS18)


def test_nontrailing_index_vector_rejected():
    # Square indices keep input/update shapes valid while changing index_vector_dim.
    text = str(_module((3, 4), (2, 2))).replace('index_vector_dim = 1', 'index_vector_dim = 0')
    assert 'index_vector_dim = 0' in text
    module = ir.Module.parse(text, context=jax_mlir.make_ir_context())
    with pytest.raises(ValueError, match='trailing index vectors'):
        convert(module, minimum_deployment_target=ct.target.iOS18)


def test_symbolic_partial_trailing_extent_rejected():
    def fn(data, indices, updates):
        dims = jax.lax.ScatterDimensionNumbers((1,), (0,), (0,))
        return jax.lax.scatter_add(data, indices, updates, dims, mode='drop')
    n, = jax.export.symbolic_shape('n', constraints=('n >= 2',))
    specs = [jax.ShapeDtypeStruct((4, n), np.float32), jax.ShapeDtypeStruct((2, 1), np.int32),
             jax.ShapeDtypeStruct((2, n - 1), np.float32)]
    module = ir.Module.parse(jax.export.export(jax.jit(fn))(*specs).mlir_module(), context=jax_mlir.make_ir_context())
    with pytest.raises(ValueError, match='unproven symbolic extents'):
        convert(module, minimum_deployment_target=ct.target.iOS18)
