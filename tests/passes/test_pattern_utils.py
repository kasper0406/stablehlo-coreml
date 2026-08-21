import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol

from stablehlo_coreml.passes.pattern_utils import (
    broadcast_shapes,
    dims_equal,
    shapes_equal,
    sole_consumer,
    uniform_scalar_value,
)


def _build(builder_fn, input_shape=(4, 8)):
    """Build a tiny MIL program, returning (prog, captured_vars).

    ``builder_fn(x, captured)`` gets the program input and a dict it can stash
    intermediate vars in, and must return the program output.
    """
    captured = {}

    @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
    def prog(x):
        captured["x"] = x
        return builder_fn(x, captured)

    return prog, captured


class TestUniformScalarValue:

    def test_scalar_const(self):
        _, captured = _build(lambda x, c: c.setdefault("v", mb.add(x=x, y=np.float32(3.5))))
        assert uniform_scalar_value(captured["v"].op.inputs["y"]) == pytest.approx(3.5)

    def test_uniform_tensor_const(self):
        const = np.full((4, 8), -2.25, dtype=np.float32)
        _, captured = _build(lambda x, c: c.setdefault("v", mb.add(x=x, y=const)))
        assert uniform_scalar_value(captured["v"].op.inputs["y"]) == pytest.approx(-2.25)

    def test_non_uniform_const_is_none(self):
        const = np.arange(32, dtype=np.float32).reshape(4, 8)
        _, captured = _build(lambda x, c: c.setdefault("v", mb.add(x=x, y=const)))
        assert uniform_scalar_value(captured["v"].op.inputs["y"]) is None

    def test_fill_op(self):
        def build(x, captured):
            captured["fill"] = mb.fill(shape=[4, 8], value=-1e9)
            return mb.add(x=x, y=captured["fill"])
        _, captured = _build(build)
        assert uniform_scalar_value(captured["fill"]) == pytest.approx(-1e9)

    def test_non_const_var_is_none(self):
        _, captured = _build(lambda x, c: mb.add(x=x, y=1.0))
        assert uniform_scalar_value(captured["x"]) is None

    def test_none_is_none(self):
        assert uniform_scalar_value(None) is None

    def test_empty_const_is_none(self):
        empty = np.zeros((0,), dtype=np.float32)
        _, captured = _build(lambda x, c: c.setdefault("v", mb.concat(values=[mb.reshape(x=x, shape=(32,)), empty], axis=0)))
        assert uniform_scalar_value(captured["v"].op.inputs["values"][1]) is None

    def test_nan_const_is_none(self):
        _, captured = _build(lambda x, c: c.setdefault("v", mb.add(x=x, y=np.float32("nan"))))
        assert uniform_scalar_value(captured["v"].op.inputs["y"]) is None


class TestShapeHelpers:

    def test_dims_equal(self):
        symbol = get_new_symbol()
        other = get_new_symbol()
        assert dims_equal(4, 4)
        assert dims_equal(np.int64(4), 4)
        assert not dims_equal(4, 5)
        assert dims_equal(symbol, symbol)
        assert not dims_equal(symbol, other)
        # A symbolic dim is never provably equal to a concrete one
        assert not dims_equal(symbol, 4)
        assert not dims_equal(4, symbol)

    def test_shapes_equal(self):
        symbol = get_new_symbol()
        assert shapes_equal((2, 3), [2, 3])
        assert not shapes_equal((2, 3), (2, 3, 1))
        assert not shapes_equal((2, 3), (2, 4))
        assert shapes_equal((symbol, 3), (symbol, 3))
        assert not shapes_equal((symbol, 3), (2, 3))
        assert not shapes_equal(None, (2, 3))

    def test_broadcast_static(self):
        assert broadcast_shapes((4, 8), (4, 8)) == (4, 8)
        assert broadcast_shapes((4, 1), (1, 8)) == (4, 8)
        assert broadcast_shapes((8,), (4, 8)) == (4, 8)
        assert broadcast_shapes((), (4, 8)) == (4, 8)
        assert broadcast_shapes((4, 8)) == (4, 8)
        assert broadcast_shapes() == ()
        assert broadcast_shapes((4, 1), (1, 8), (4, 8)) == (4, 8)

    def test_broadcast_incompatible(self):
        assert broadcast_shapes((4, 3), (4, 8)) is None
        assert broadcast_shapes((2,), (3,)) is None

    def test_broadcast_symbolic(self):
        symbol = get_new_symbol()
        other = get_new_symbol()
        assert broadcast_shapes((symbol, 8), (symbol, 8)) == (symbol, 8)
        assert broadcast_shapes((symbol, 1), (symbol, 8)) == (symbol, 8)
        assert broadcast_shapes((symbol, 8), (1, 8)) == (symbol, 8)
        # Not provable: `symbol` might be 1, or might be 4
        assert broadcast_shapes((symbol, 8), (4, 8)) is None
        assert broadcast_shapes((symbol, 8), (other, 8)) is None


class TestSoleConsumer:

    def test_single_consumer(self):
        def build(x, captured):
            captured["mid"] = mb.mul(x=x, y=2.0)
            return mb.add(x=captured["mid"], y=1.0)
        _, captured = _build(build)
        consumer = sole_consumer(captured["mid"])
        assert consumer is not None and consumer.op_type == "add"

    def test_two_consumers(self):
        def build(x, captured):
            captured["mid"] = mb.mul(x=x, y=2.0)
            return mb.add(x=captured["mid"], y=captured["mid"])
        _, captured = _build(build)
        assert sole_consumer(captured["mid"]) is None

    def test_no_consumer(self):
        def build(x, captured):
            captured["dead"] = mb.mul(x=x, y=2.0)
            return mb.add(x=x, y=1.0)
        _, captured = _build(build)
        assert sole_consumer(captured["dead"]) is None

    def test_block_output_is_not_sole_consumer(self):
        def build(x, captured):
            captured["mid"] = mb.mul(x=x, y=2.0)
            return [mb.add(x=captured["mid"], y=1.0), captured["mid"]]
        _, captured = _build(build)
        assert sole_consumer(captured["mid"]) is None

    def test_none(self):
        assert sole_consumer(None) is None
