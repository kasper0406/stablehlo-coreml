"""Pipeline regressions for upstream int32 data/index narrowing."""
import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types

from stablehlo_coreml import build_pass_pipeline
from stablehlo_coreml.passes.utils import _int16_cast_is_safe
from tests.utils import run_and_compare_specific_input


def _convert(program, fp32=False):
    pipeline = build_pass_pipeline()
    pipeline.remove_passes(['common::add_fp16_cast'])
    return ct.convert(program, source='milinternal', minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT32 if fp32 else None, compute_units=ct.ComputeUnit.CPU_ONLY,
                      pass_pipeline=pipeline)


def test_large_jax_take():
    table = np.arange(80000, dtype=np.int32).reshape(40000, 2)
    run_and_compare_specific_input(lambda x, i: jnp.take(x, i, axis=0, mode='clip'),
                                   (table, np.array([35000, 3], np.int32)), max_complexity=100000, atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['gather_nd', 'squeeze', 'topk'])
def test_int32_data_preserved(kind):
    shape = {'gather_nd': (40000, 2), 'squeeze': (1, 4), 'topk': (4,)}[kind]

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.int32)], opset_version=ct.target.iOS18)
    def program(x):
        if kind == 'gather_nd':
            return mb.gather_nd(x=x, indices=np.array([[35000], [3]], np.int32))
        if kind == 'squeeze':
            return mb.squeeze(x=x, axes=[0])
        return mb.topk(x=x, k=2)[0]

    data = (np.arange(80000, dtype=np.int32).reshape(shape) if kind == 'gather_nd'
            else np.array([40000, -40000, 35000, 2], np.int32).reshape(shape))
    expected = {'gather_nd': lambda: data[[35000, 3]], 'squeeze': lambda: data[0],
                'topk': lambda: np.array([40000, 35000], np.int32)}[kind]()
    model = _convert(program)
    np.testing.assert_array_equal(next(iter(model.predict({'x': data}).values())), expected)
    assert not any(op.op_type == 'cast' and op.dtype.val in ('int16', 'uint16') and op.x.dtype == types.int32
                   and op.x.val is None for op in model._mil_program.functions['main'].operations)


def test_jax_topk_int32():
    model = run_and_compare_specific_input(lambda x: jax.lax.top_k(x, 2),
                                           (np.array([40000, -40000, 35000, 2], np.int32),), atol=0, rtol=0)
    assert any(op.op_type == 'topk' for op in model._mil_program.functions['main'].operations)


def test_safe_small_gather_still_narrows():
    # Negative constant entries select signed int16 rather than uint16. Dynamic
    # indices also permit a surviving int16 cast to be inspected after folding.
    @mb.program(input_specs=[mb.TensorSpec(shape=(2,), dtype=types.int32)], opset_version=ct.target.iOS18)
    def program(indices):
        return mb.gather(x=np.array([-3, 2, 7], np.int32), indices=indices, axis=0)

    model = _convert(program)
    np.testing.assert_array_equal(next(iter(model.predict({'indices': np.array([0, 2], np.int32)}).values())), [-3, 7])
    assert any(op.op_type == 'cast' and op.dtype.val == 'int16' for op in model._mil_program.functions['main'].operations)

    @mb.program(input_specs=[mb.TensorSpec(shape=(3,))], opset_version=ct.target.iOS18)
    def const_indexed(x):
        return mb.gather(x=x, indices=np.array([0, 2], np.int32), axis=0)

    model = _convert(const_indexed)
    gather = next(op for op in model._mil_program.functions['main'].operations if op.op_type == 'gather')
    # Constant cast folds to a constant, so inspect the actual narrowed dtype.
    assert gather.indices.dtype in (types.int16, types.uint16)
    np.testing.assert_array_equal(next(iter(model.predict({'x': np.array([1., 2., 3.], np.float32)}).values())), [1., 3.])


def test_explicit_fp32_conversion_removes_cast_options():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2,))])
    def program(x):
        return mb.add(x=x, y=1.)
    model = _convert(program, fp32=True)
    np.testing.assert_array_equal(next(iter(model.predict({'x': np.array([1., 2.], np.float32)}).values())), [2., 3.])


@pytest.mark.parametrize('kind', ['gather', 'gather_along_axis', 'gather_nd'])
@pytest.mark.parametrize('size,safe', [(32767, True), (32768, False), (None, False)])
def test_dynamic_index_domain(kind, size, safe):
    extent = get_new_symbol() if size is None else size
    index_shape = (2, 1) if kind == 'gather_nd' else (2,)

    @mb.program(input_specs=[mb.TensorSpec(shape=(extent,)), mb.TensorSpec(shape=index_shape, dtype=types.int32)],
                opset_version=ct.target.iOS18)
    def program(x, indices):
        if kind == 'gather_nd':
            return mb.gather_nd(x=x, indices=indices)
        return getattr(mb, kind)(x=x, indices=indices, axis=0)

    op = next(op for op in program.functions['main'].operations if op.op_type == kind)
    assert _int16_cast_is_safe(op) is safe


@pytest.mark.parametrize('shape,safe', [((1, 80000), False), ((80000, 3), True)])
def test_batched_gather_nd_domain(shape, safe):
    @mb.program(input_specs=[mb.TensorSpec(shape=shape),
                            mb.TensorSpec(shape=(shape[0], 1, 1), dtype=types.int32)], opset_version=ct.target.iOS18)
    def program(x, indices):
        return mb.gather_nd(x=x, indices=indices, batch_dims=1)

    op = next(op for op in program.functions['main'].operations if op.op_type == 'gather_nd')
    assert _int16_cast_is_safe(op) is safe
    model = _convert(program)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    index = 2 if safe else 70000
    actual = next(iter(model.predict({'x': x, 'indices': np.full((shape[0], 1, 1), index, np.int32)}).values()))
    np.testing.assert_array_equal(actual, x[:, index:index + 1])


def test_batched_gather_axis_is_absolute():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 80000)), mb.TensorSpec(shape=(1, 1), dtype=types.int32)],
                opset_version=ct.target.iOS18)
    def program(x, indices):
        return mb.gather(x=x, indices=indices, axis=-1, batch_dims=1)
    op = next(op for op in program.functions['main'].operations if op.op_type == 'gather')
    assert not _int16_cast_is_safe(op)


def test_nonnegative_table_preserves_negative_indices():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int32)], opset_version=ct.target.iOS18)
    def program(indices):
        return mb.gather(x=np.array([3, 2, 7], np.int32), indices=indices, axis=0)
    model = _convert(program)
    np.testing.assert_array_equal(next(iter(model.predict({'indices': np.array([-1], np.int32)}).values())), [7])


@pytest.mark.parametrize('size', [80000, None])
def test_large_float_topk_keeps_int32_indices(size):
    extent = get_new_symbol() if size is None else size

    @mb.program(input_specs=[mb.TensorSpec(shape=(extent,))], opset_version=ct.target.iOS18)
    def program(x):
        return mb.topk(x=x, k=1, output_indices_dtype='int32')
    op = next(op for op in program.functions['main'].operations if op.op_type == 'topk')
    assert not _int16_cast_is_safe(op)
    if size is not None:
        model = _convert(program)
        topk = next(op for op in model._mil_program.functions['main'].operations if op.op_type == 'topk')
        assert topk.output_indices_dtype.val == 'int32'
        x = np.zeros(size, np.float32)
        x[70000] = 1
        out = list(model.predict({'x': x}).values())
        assert any(np.array_equal(v, [70000]) for v in out)


def test_caller_selector_still_disables_casts():
    base = ct.PassPipeline.DEFAULT
    base.remove_passes(['common::add_fp16_cast'])
    base.set_options('common::add_int16_cast', {'op_selector': lambda op: False})

    @mb.program(input_specs=[mb.TensorSpec(shape=(3,)), mb.TensorSpec(shape=(1,), dtype=types.int32)],
                opset_version=ct.target.iOS18)
    def program(x, indices):
        return mb.gather(x=x, indices=indices, axis=0)
    model = ct.convert(program, source='milinternal', minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_ONLY, pass_pipeline=build_pass_pipeline(base))
    assert not any(op.op_type == 'cast' and op.dtype.val in ('int16', 'uint16')
                   for op in model._mil_program.functions['main'].operations)


def test_remove_cast_pass_by_index_cleans_last_options():
    pipeline = build_pass_pipeline()
    name = 'common::add_int16_cast'
    while name in pipeline.passes:
        pipeline.remove_pass(pipeline.passes.index(name))
        if name in pipeline.passes:
            assert pipeline.get_options(name)
    assert pipeline.get_options(name) is None
    pipeline.validate()
