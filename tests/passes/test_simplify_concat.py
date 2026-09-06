import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
)

from tests.passes.helpers import apply_pass, count_ops, ops_of_type, predict
from tests.utils import get_model_instruction_types, run_and_compare

PASS_NAME = "common::simplify_concat"


def _apply(prog):
    """Apply the pass (plus DCE, which clears the folded-away concats)."""
    return apply_pass(prog, PASS_NAME)


def _concat_operands(prog):
    """The operand shapes of every remaining concat, in program order."""
    return [
        [tuple(value.shape) for value in op.values]
        for op in ops_of_type(prog, "concat", recurse=True)
    ]


def _empty(rows: int = 0, cols: int = 4):
    return np.zeros((rows, cols), dtype=np.float32)


class TestSimplifyConcat:
    """Unit tests on hand-built MIL programs."""

    def test_drops_a_zero_sized_operand(self):
        """One operand left after the drop, so the concat itself is a copy."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            padded = mb.concat(values=[x, _empty()], axis=0)
            return mb.relu(x=padded)

        assert get_op_types_in_program(prog) == ["concat", "relu"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["relu"]
        assert prog.functions["main"].outputs[0].shape == (3, 4)
        assert_model_is_valid(
            prog, {"x": (3, 4)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_drops_a_zero_sized_operand_and_keeps_the_concat(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(x, y):
            padded = mb.concat(values=[x, _empty(), y], axis=0)
            return mb.relu(x=padded)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["concat", "relu"]
        assert _concat_operands(prog) == [[(3, 4), (2, 4)]]
        assert prog.functions["main"].outputs[0].shape == (5, 4)

    def test_drops_a_zero_sized_operand_on_a_non_zero_axis(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 3))])
        def prog(x):
            padded = mb.concat(values=[x, np.zeros((4, 0), dtype=np.float32)], axis=1)
            return mb.relu(x=padded)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["relu"]
        assert prog.functions["main"].outputs[0].shape == (4, 3)

    def test_removes_a_single_input_concat(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            copied = mb.concat(values=[x], axis=0)
            return mb.relu(x=copied)

        assert get_op_types_in_program(prog) == ["concat", "relu"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["relu"]
        assert prog.functions["main"].outputs[0].shape == (3, 4)

    def test_flattens_a_same_axis_chain(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 4)),
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(3, 4)),
            mb.TensorSpec(shape=(4, 4)),
        ])
        def prog(a, b, c, d):
            ab = mb.concat(values=[a, b], axis=0)
            abc = mb.concat(values=[ab, c], axis=0)
            abcd = mb.concat(values=[abc, d], axis=0)
            return mb.relu(x=abcd)

        assert count_ops(prog, "concat") == 3
        _apply(prog)
        assert _concat_operands(prog) == [[(1, 4), (2, 4), (3, 4), (4, 4)]]
        assert prog.functions["main"].outputs[0].shape == (10, 4)
        assert_model_is_valid(
            prog,
            {"a": (1, 4), "b": (2, 4), "c": (3, 4), "d": (4, 4)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_flattens_a_chain_whose_child_is_not_the_first_operand(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 4)),
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(3, 4)),
        ])
        def prog(a, b, c):
            bc = mb.concat(values=[b, c], axis=0)
            abc = mb.concat(values=[a, bc], axis=0)
            return mb.relu(x=abc)

        _apply(prog)
        assert _concat_operands(prog) == [[(1, 4), (2, 4), (3, 4)]]

    def test_flattens_a_chain_on_a_negative_axis(self):
        """A negative axis names the same dimension, so the chain still folds."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(4, 1)),
            mb.TensorSpec(shape=(4, 2)),
            mb.TensorSpec(shape=(4, 3)),
        ])
        def prog(a, b, c):
            ab = mb.concat(values=[a, b], axis=-1)
            abc = mb.concat(values=[ab, c], axis=1)
            return mb.relu(x=abc)

        _apply(prog)
        assert _concat_operands(prog) == [[(4, 1), (4, 2), (4, 3)]]
        assert prog.functions["main"].outputs[0].shape == (4, 6)

    def test_flattens_a_five_deep_chain_into_one_op(self):
        """The shape of the chains the sphere model builds: one concat per scale."""
        sizes = [120, 527, 2205, 9216, 37668, 152292]

        @mb.program(input_specs=[mb.TensorSpec(shape=(size,)) for size in sizes])
        def prog(a, b, c, d, e, f):
            acc = a
            for scale in (b, c, d, e, f):
                acc = mb.concat(values=[acc, scale], axis=0)
            return mb.relu(x=acc)

        assert count_ops(prog, "concat") == 5
        _apply(prog)
        assert _concat_operands(prog) == [[(size,) for size in sizes]]
        assert prog.functions["main"].outputs[0].shape == (sum(sizes),)

    def test_drops_a_zero_sized_operand_of_a_flattened_chain(self):
        """The empty operand comes in through the splice, and still goes."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(x, y):
            padded = mb.concat(values=[x, _empty()], axis=0)
            joined = mb.concat(values=[padded, y], axis=0)
            return mb.relu(x=joined)

        _apply(prog)
        assert _concat_operands(prog) == [[(3, 4), (2, 4)]]

    def test_not_flattened_when_the_child_has_another_consumer(self):
        """The intermediate result is needed anyway; folding would only add an op."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(a, b):
            ab = mb.concat(values=[a, b], axis=0)
            abb = mb.concat(values=[ab, b], axis=0)
            return mb.add(x=abb, y=mb.concat(values=[ab, b], axis=0))

        before = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == before

    def test_not_flattened_when_the_child_is_a_block_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(a, b):
            ab = mb.concat(values=[a, b], axis=0)
            abb = mb.concat(values=[ab, b], axis=0)
            return ab, abb

        _apply(prog)
        assert count_ops(prog, "concat") == 2

    def test_not_flattened_across_different_axes(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(a, b):
            ab = mb.concat(values=[a, b], axis=1)
            stacked = mb.concat(values=[ab, mb.concat(values=[a, b], axis=1)], axis=0)
            return mb.relu(x=stacked)

        _apply(prog)
        assert count_ops(prog, "concat") == 3

    def test_not_flattened_when_the_child_interleaves(self):
        """`interleave` round-robins the operands, so the chain is not associative."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(a, b):
            ab = mb.concat(values=[a, b], axis=0, interleave=True)
            abb = mb.concat(values=[ab, b], axis=0)
            return mb.relu(x=abb)

        _apply(prog)
        assert count_ops(prog, "concat") == 2

    def test_not_rewritten_when_the_parent_interleaves(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(a, b):
            ab = mb.concat(values=[a, b], axis=0)
            interleaved = mb.concat(values=[ab, ab], axis=0, interleave=True)
            return mb.relu(x=interleaved)

        _apply(prog)
        assert count_ops(prog, "concat") == 2

    def test_not_rewritten_when_the_same_operand_is_used_twice(self):
        """`concat(c, c)` keeps `c`: splicing it in would drop one of the copies."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(a, b):
            ab = mb.concat(values=[a, b], axis=0)
            doubled = mb.concat(values=[ab, ab], axis=0)
            return mb.relu(x=doubled)

        _apply(prog)
        assert count_ops(prog, "concat") == 2
        assert prog.functions["main"].outputs[0].shape == (6, 4)

    def test_not_dropped_for_a_symbolic_operand(self):
        """A symbolic dimension may be 0 at runtime, but it may just as well not be."""
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(length, 4))])
        def prog(x, y):
            joined = mb.concat(values=[x, y], axis=0)
            return mb.relu(x=joined)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["concat", "relu"]
        assert _concat_operands(prog) == [[(3, 4), (length, 4)]]

    def test_not_flattened_when_a_symbolic_dim_would_be_renamed(self):
        """`concat` mints a fresh symbol per op, so the folded shape is unprovable.

        The chain below really is associative, but its output dimension is a
        symbol that no compile-time comparison can match against the one the
        rebuilt concat would infer, so the pass leaves it alone.
        """
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(length, 4)),
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(3, 4)),
        ])
        def prog(a, b, c):
            ab = mb.concat(values=[a, b], axis=0)
            abc = mb.concat(values=[ab, c], axis=0)
            return mb.relu(x=abc)

        _apply(prog)
        assert count_ops(prog, "concat") == 2

    def test_all_empty_operands_are_left_alone(self):
        """Nothing is left to carry the (equally empty) result."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            joined = mb.concat(values=[_empty(), _empty()], axis=0)
            return mb.concat(values=[x, joined], axis=0)

        _apply(prog)
        assert count_ops(prog, "concat") == 2

    def test_scalar_concat_is_left_alone(self):
        """`concat` promotes rank-0 operands to length 1, so it is not a copy."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3,))])
        def prog(x):
            scalar = mb.reduce_sum(x=x, axes=[0], keep_dims=False)
            promoted = mb.concat(values=[scalar], axis=0)
            return mb.relu(x=promoted)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "concat", "relu"]
        assert prog.functions["main"].outputs[0].shape == (1,)

    def test_single_input_concat_of_a_function_input_is_left_alone(self):
        """coremltools refuses to rename a function input to the output's name."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            return mb.concat(values=[x], axis=0)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["concat"]

    def test_rewritten_inside_a_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                return mb.concat(values=[x, _empty()], axis=0)

            def false_fn():
                return mb.relu(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        assert count_ops(prog, "concat", recurse=True) == 1
        _apply(prog)
        assert count_ops(prog, "concat", recurse=True) == 0

    def test_not_flattened_when_the_child_is_consumed_from_a_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            ab = mb.concat(values=[x, x], axis=0)

            def true_fn():
                return mb.concat(values=[ab, x], axis=0)

            def false_fn():
                return mb.concat(values=[x, ab], axis=0)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        assert count_ops(prog, "concat", recurse=True) == 3

    def test_is_idempotent(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 4)),
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(3, 4)),
        ])
        def prog(a, b, c):
            ab = mb.concat(values=[a, b, _empty()], axis=0)
            abc = mb.concat(values=[ab, c], axis=0)
            return mb.relu(x=abc)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        operands_after_first = _concat_operands(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first
        assert _concat_operands(prog) == operands_after_first


class TestSimplifyConcatNumerics:
    """The rewritten program computes exactly what the original one did."""

    def test_flattened_chain_matches(self):
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 4)),
            mb.TensorSpec(shape=(2, 4)),
            mb.TensorSpec(shape=(3, 4)),
        ])
        def prog(a, b, c):
            ab = mb.concat(values=[a, b], axis=0)
            abc = mb.concat(values=[ab, c], axis=0)
            return mb.relu(x=abc)

        values = {
            "a": np.random.randn(1, 4).astype(np.float32),
            "b": np.random.randn(2, 4).astype(np.float32),
            "c": np.random.randn(3, 4).astype(np.float32),
        }
        prev_prog = _apply(prog)
        assert count_ops(prog, "concat") == 1
        np.testing.assert_array_equal(predict(prog, **values), predict(prev_prog, **values))

    def test_dropped_empty_operand_matches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(2, 4))])
        def prog(x, y):
            padded = mb.concat(values=[x, _empty(), y], axis=0)
            return mb.relu(x=padded)

        values = {
            "x": np.random.randn(3, 4).astype(np.float32),
            "y": np.random.randn(2, 4).astype(np.float32),
        }
        prev_prog = _apply(prog)
        assert _concat_operands(prog) == [[(3, 4), (2, 4)]]
        np.testing.assert_array_equal(predict(prog, **values), predict(prev_prog, **values))


class TestSimplifyConcatEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_repeated_concatenate_leaves_a_single_concat(self):
        def f(a, b, c, d):
            acc = a
            for part in (b, c, d):
                acc = jnp.concatenate([acc, part], axis=0)
            return acc * 2.0

        cml_model = run_and_compare(
            f,
            [
                jax.ShapeDtypeStruct((1, 8), jnp.float32),
                jax.ShapeDtypeStruct((2, 8), jnp.float32),
                jax.ShapeDtypeStruct((3, 8), jnp.float32),
                jax.ShapeDtypeStruct((4, 8), jnp.float32),
            ],
        )
        assert get_model_instruction_types(cml_model).count("concat") == 1
