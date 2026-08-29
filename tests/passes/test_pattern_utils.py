import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types

from stablehlo_coreml.passes.pattern_utils import (
    broadcast_shapes,
    dims_equal,
    dtype_epsilon,
    peel_to_scaled_input,
    rewrite_leaf_ops,
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


class TestPeelToScaledInput:

    def test_peels_a_chain_of_constant_scalings(self):
        def build(x, captured):
            scaled = mb.mul(x=x, y=np.float32(3.0))
            negated = mb.sub(x=np.float32(0.0), y=scaled)
            captured["out"] = mb.real_div(x=negated, y=np.float32(2.0))
            # `sole_consumer` refuses a block output, so keep `out` internal.
            return mb.identity(x=captured["out"])

        prog, captured = _build(build)
        block = prog.functions["main"]
        chain = peel_to_scaled_input(captured["out"], block)

        assert [var for var, _ in chain][-1] is captured["x"]
        assert [factor for _, factor in chain] == pytest.approx([1.0, 0.5, -0.5, -1.5])

    def test_stops_at_a_value_with_two_consumers(self):
        def build(x, captured):
            scaled = mb.mul(x=x, y=np.float32(3.0))
            captured["out"] = mb.mul(x=scaled, y=np.float32(2.0))
            return mb.add(x=captured["out"], y=scaled)

        prog, captured = _build(build)
        chain = peel_to_scaled_input(captured["out"], prog.functions["main"])

        # `scaled` escapes to the `add`, so peeling past it is not allowed.
        assert len(chain) == 2
        assert chain[-1][0] is not captured["x"]

    def test_stops_at_a_non_scaling_op(self):
        def build(x, captured):
            captured["out"] = mb.mul(x=mb.tanh(x=x), y=np.float32(3.0))
            return mb.identity(x=captured["out"])

        prog, captured = _build(build)
        chain = peel_to_scaled_input(captured["out"], prog.functions["main"])
        assert len(chain) == 2
        assert chain[-1][0].op.op_type == "tanh"

    def test_a_non_constant_multiplication_is_not_a_scaling(self):
        def build(x, captured):
            captured["out"] = mb.mul(x=x, y=x)
            return mb.identity(x=captured["out"])

        prog, captured = _build(build)
        assert peel_to_scaled_input(captured["out"], prog.functions["main"]) == [
            (captured["out"], 1.0)
        ]


class TestDtypeEpsilon:

    def test_matches_numpy(self):
        _, captured = _build(lambda x, c: c.setdefault("v", mb.tanh(x=x)))
        assert dtype_epsilon(captured["v"]) == pytest.approx(np.finfo(np.float32).eps)

    def test_fp16(self):
        captured = {}

        @mb.program(input_specs=[mb.TensorSpec(shape=(4,), dtype=types.fp16)])
        def prog(x):
            captured["v"] = mb.tanh(x=x)
            return captured["v"]

        assert dtype_epsilon(captured["v"]) == pytest.approx(np.finfo(np.float16).eps)

    def test_non_float_and_none(self):
        captured = {}

        @mb.program(input_specs=[mb.TensorSpec(shape=(4,), dtype=types.int32)])
        def prog(x):
            captured["v"] = mb.add(x=x, y=np.int32(1))
            return captured["v"]

        assert dtype_epsilon(captured["v"]) == 0.0
        assert dtype_epsilon(None) == 0.0


class TestRewriteLeafOps:

    @staticmethod
    def _cond_prog():
        """A program whose only ``cond`` holds an ``add`` and a ``mul``."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            pred = mb.squeeze(x=mb.less(x=x, y=1.0))
            return mb.cond(
                pred=pred,
                _true_fn=lambda: mb.add(x=x, y=1.0),
                _false_fn=lambda: mb.mul(x=x, y=2.0),
            )

        return prog

    def test_descends_into_nested_blocks_instead_of_visiting_their_parent(self):
        main = self._cond_prog().functions["main"]
        seen = []

        def visit(op, block):
            seen.append((op.op_type, block is main))

        assert rewrite_leaf_ops(main, visit) == 0
        assert ("cond", True) not in seen
        assert ("less", True) in seen
        # The ops of the two branches are visited with their own block.
        assert ("add", False) in seen
        assert ("mul", False) in seen

    def test_counts_the_truthy_visits_of_every_block(self):
        main = self._cond_prog().functions["main"]
        assert rewrite_leaf_ops(main, lambda op, block: op.op_type in ("add", "mul")) == 2

    def test_skips_ops_a_previous_visit_removed(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4,))])
        def prog(x):
            a = mb.identity(x=x, name="a")
            b = mb.identity(x=a, name="b")
            return mb.identity(x=b, name="c")

        seen = []

        def visit(op, block):
            """Fuse ``a -> b`` into a single op, the way a real pass would."""
            seen.append(op.name)
            if op.name != "a":
                return False
            consumer = op.outputs[0].child_ops[0]
            fused = mb.identity(x=op.x, before_op=op, name="fused")
            block.replace_uses_of_var_after_op(
                anchor_op=consumer, old_var=consumer.outputs[0], new_var=fused
            )
            block.remove_ops([consumer, op])
            return True

        main = prog.functions["main"]
        assert rewrite_leaf_ops(main, visit) == 1
        # `b` is gone by the time the walk reaches it, and the replacement op is
        # not visited either -- the walk iterates over a snapshot of the block.
        assert seen == ["a", "c"]
        assert [op.name for op in main.operations] == ["fused", "c"]
