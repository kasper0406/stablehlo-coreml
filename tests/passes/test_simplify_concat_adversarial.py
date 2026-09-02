"""Adversarial tests for ``common::simplify_concat``.

The cases in here go after the paths the pass has to get right for *any*
graph rather than for the sphere model: concats whose output is a block
output, nested blocks, loop-carried values, non-float dtypes, concats that
feed a ``reshape``'s shape vector, and symbolic non-axis dimensions.

The ``xfail(strict=True)`` tests pin down confirmed defects: they start
failing on purpose (and must have their marker removed) once the pass is
fixed.
"""

import coremltools as ct
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipelineManager
from coremltools.converters.mil.testing_utils import get_op_types_in_program

import stablehlo_coreml.passes.utils  # noqa: F401  (registers the pass)
from tests.passes.helpers import DCE_PASS_NAME, apply_pass, count_ops, predict

PASS_NAME = "common::simplify_concat"


def _apply(prog, **kwargs):
    return apply_pass(prog, PASS_NAME, **kwargs)


def _apply_unchecked(prog):
    """Apply the pass without coremltools' basic checks (which would already trip)."""
    PassPipelineManager.apply_pipeline(prog, ct.PassPipeline([PASS_NAME, DCE_PASS_NAME], "adversarial"))


def _convert(prog, **kwargs):
    return ct.convert(
        prog,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT32,
        pass_pipeline=ct.PassPipeline.DEFAULT,
        **kwargs,
    )


def _output_names(prog):
    return [var.name for var in prog.functions["main"].outputs]


def _empty(rows: int = 0, cols: int = 4):
    return np.zeros((rows, cols), dtype=np.float32)


class TestBlockOutputs:
    """Concats whose output leaves the function."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "simplify_concat bypasses a single-input concat that is a function output while its "
            "operand is *also* a function output: Block.replace_block_output_var then renames the "
            "operand to the concat's name, so the model ends up with two outputs both called 'c' and "
            "the output 'y' silently disappears."
        ),
    )
    def test_copy_of_a_var_that_is_also_returned_directly_keeps_both_outputs(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            y = mb.relu(x=x, name="y")
            c = mb.concat(values=[y], axis=0, name="c")
            return y, c

        names_before = _output_names(prog)
        assert names_before == ["y", "c"]
        _apply_unchecked(prog)
        assert _output_names(prog) == names_before

        model = _convert(prog)
        spec_outputs = [feature.name for feature in model.get_spec().description.output]
        assert spec_outputs == ["y", "c"]
        result = model.predict({"x": np.ones((3, 4), dtype=np.float32)})
        assert set(result) == {"y", "c"}

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "two no-op concats of the same var, both returned by the function, collapse into one "
            "var: the function then has two outputs that are the same Var with one name, and the "
            "converted model exposes a single output."
        ),
    )
    def test_two_copies_of_one_var_stay_two_outputs(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            y = mb.relu(x=x)
            first = mb.concat(values=[y], axis=0, name="first")
            second = mb.concat(values=[y, _empty()], axis=0, name="second")
            return first, second

        _apply_unchecked(prog)
        assert _output_names(prog) == ["first", "second"]
        outputs = prog.functions["main"].outputs
        assert outputs[0] is not outputs[1]

        model = _convert(prog)
        result = model.predict({"x": np.ones((3, 4), dtype=np.float32)})
        assert set(result) == {"first", "second"}

    def test_copy_returned_next_to_an_unrelated_output_keeps_the_names(self):
        """The rename is harmless when the operand is not itself an output."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            y = mb.relu(x=x)
            c = mb.concat(values=[y], axis=0, name="c")
            return c, mb.sigmoid(x=y, name="s")

        _apply(prog)
        assert _output_names(prog) == ["c", "s"]
        assert get_op_types_in_program(prog) == ["relu", "sigmoid"]


class TestNestedBlocks:
    """The rewrite inside `cond` / `while_loop` blocks, run end to end."""

    def test_cond_branch_that_becomes_an_outer_var_runs(self):
        """After the rewrite the true branch returns the outer `x` directly."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(1,))])
        def prog(x, pred):
            def true_fn():
                return mb.concat(values=[x, _empty()], axis=0)

            def false_fn():
                return mb.sigmoid(x=x)

            return mb.cond(pred=mb.squeeze(x=mb.cast(x=pred, dtype="bool")), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        assert count_ops(prog, "concat", recurse=True) == 0
        true_block = next(op for op in prog.functions["main"].operations if op.op_type == "cond").blocks[0]
        assert true_block.outputs[0] is prog.functions["main"].inputs["x"]

        model = _convert(prog)
        x = np.random.randn(3, 4).astype(np.float32)
        for pred, expected in ((1.0, x), (0.0, 1.0 / (1.0 + np.exp(-x)))):
            result = next(iter(model.predict({"x": x, "pred": np.array([pred], dtype=np.float32)}).values()))
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_cond_branch_returning_a_var_twice_runs(self):
        """A branch output that collapses onto another output of the same branch."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4)), mb.TensorSpec(shape=(1,))])
        def prog(x, pred):
            def true_fn():
                y = mb.relu(x=x)
                return y, mb.concat(values=[y], axis=0)

            def false_fn():
                return mb.sigmoid(x=x), mb.tanh(x=x)

            a, b = mb.cond(pred=mb.squeeze(x=mb.cast(x=pred, dtype="bool")), _true_fn=true_fn, _false_fn=false_fn)
            return mb.add(x=a, y=b)

        _apply(prog)
        assert count_ops(prog, "concat", recurse=True) == 0

        model = _convert(prog)
        x = np.random.randn(3, 4).astype(np.float32)
        expected = {1.0: 2 * np.maximum(x, 0), 0.0: 1.0 / (1.0 + np.exp(-x)) + np.tanh(x)}
        for pred, value in expected.items():
            result = next(iter(model.predict({"x": x, "pred": np.array([pred], dtype=np.float32)}).values()))
            np.testing.assert_allclose(result, value, rtol=1e-6, atol=1e-6)

    def test_while_body_copy_of_a_loop_var(self):
        """The body then returns its own block input; the loop stays correct."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):
            def cond(i, acc):
                return mb.less(x=i, y=3)

            def body(i, acc):
                return mb.add(x=i, y=1), mb.concat(values=[acc, _empty()], axis=0)

            i, acc = mb.while_loop(_cond=cond, _body=body, loop_vars=(np.int32(0), x))
            return mb.add(x=acc, y=mb.cast(x=i, dtype="fp32"))

        _apply(prog)
        assert count_ops(prog, "concat", recurse=True) == 0

        x = np.random.randn(3, 4).astype(np.float32)
        np.testing.assert_allclose(predict(prog, x=x), x + 3.0, rtol=1e-6)

    def test_loop_carried_chain_with_a_symbolic_accumulator_is_left_alone(self):
        """A growing accumulator has a fresh symbol per link; nothing is provable."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4))])
        def prog(x):
            def cond(i, acc):
                return mb.less(x=i, y=3)

            def body(i, acc):
                grown = mb.concat(values=[acc, x], axis=0)
                return mb.add(x=i, y=1), mb.concat(values=[grown, _empty()], axis=0)

            _, acc = mb.while_loop(_cond=cond, _body=body, loop_vars=(np.int32(0), x))
            return acc

        _apply(prog, skip_output_shape_check=True)
        assert count_ops(prog, "concat", recurse=True) == 2


class TestDtypesAndValues:
    def test_int32_chain_matches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(1, 3))])
        def prog(a, b):
            a = mb.cast(x=a, dtype="int32")
            b = mb.cast(x=b, dtype="int32")
            ab = mb.concat(values=[a, b], axis=0)
            abb = mb.concat(values=[ab, b, np.zeros((0, 3), dtype=np.int32)], axis=0)
            return mb.cast(x=mb.add(x=abb, y=np.int32(1)), dtype="fp32")

        values = {
            "a": np.random.randint(-5, 5, (2, 3)).astype(np.float32),
            "b": np.random.randint(-5, 5, (1, 3)).astype(np.float32),
        }
        prev_prog = _apply(prog)
        assert count_ops(prog, "concat") == 1
        np.testing.assert_array_equal(predict(prog, **values), predict(prev_prog, **values))

    def test_bool_chain_matches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(1, 3))])
        def prog(a, b):
            a = mb.greater(x=a, y=0.0)
            b = mb.greater(x=b, y=0.0)
            ab = mb.concat(values=[a, b], axis=0)
            abb = mb.concat(values=[ab, b, np.zeros((0, 3), dtype=np.bool_)], axis=0)
            return mb.cast(x=mb.logical_not(x=abb), dtype="fp32")

        values = {
            "a": np.random.randn(2, 3).astype(np.float32),
            "b": np.random.randn(1, 3).astype(np.float32),
        }
        prev_prog = _apply(prog)
        assert count_ops(prog, "concat") == 1
        np.testing.assert_array_equal(predict(prog, **values), predict(prev_prog, **values))

    def test_child_with_a_repeated_operand_is_spliced(self):
        """`concat(concat(a, a), b)` -> `concat(a, a, b)`: the duplicate is inside the child."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(1, 3))])
        def prog(a, b):
            aa = mb.concat(values=[a, a], axis=0)
            return mb.relu(x=mb.concat(values=[aa, b], axis=0))

        values = {
            "a": np.random.randn(2, 3).astype(np.float32),
            "b": np.random.randn(1, 3).astype(np.float32),
        }
        prev_prog = _apply(prog)
        assert count_ops(prog, "concat") == 1
        concat = next(op for op in prog.functions["main"].operations if op.op_type == "concat")
        assert [value.name for value in concat.values] == ["a", "a", "b"]
        np.testing.assert_array_equal(predict(prog, **values), predict(prev_prog, **values))

    def test_shape_vector_concat_keeps_its_symbolic_value(self):
        """A `reshape` whose shape is a chain of concats over shape pieces still type-infers."""
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(length, 4))])
        def prog(x):
            n = mb.slice_by_index(x=mb.shape(x=x), begin=[0], end=[1])
            inner = mb.concat(values=[n, np.array([2], dtype=np.int32)], axis=0)
            shape = mb.concat(values=[inner, np.array([2], dtype=np.int32), np.zeros((0,), dtype=np.int32)], axis=0)
            return mb.reshape(x=x, shape=shape)

        assert prog.functions["main"].outputs[0].shape == (length, 2, 2)
        _apply(prog)
        assert count_ops(prog, "concat") == 1
        assert prog.functions["main"].outputs[0].shape == (length, 2, 2)

        model = _convert(prog, inputs=[ct.TensorType(name="x", shape=(ct.RangeDim(1, 16), 4))])
        x = np.random.randn(5, 4).astype(np.float32)
        result = next(iter(model.predict({"x": x}).values()))
        np.testing.assert_array_equal(result, x.reshape(5, 2, 2))


class TestSymbolicNonAxisDims:
    """`concat` takes the non-axis dims from its first operand, symbolic or not."""

    def test_empty_first_operand_with_a_symbolic_non_axis_dim_is_kept(self):
        """Dropping it would turn the output's `(3, s)` into `(3, 4)`."""
        width = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(0, width)), mb.TensorSpec(shape=(3, 4))])
        def prog(empty, y):
            return mb.relu(x=mb.concat(values=[empty, y], axis=0))

        assert prog.functions["main"].outputs[0].shape == (3, width)
        _apply(prog)
        assert get_op_types_in_program(prog) == ["concat", "relu"]
        assert prog.functions["main"].outputs[0].shape == (3, width)

    def test_empty_second_operand_with_a_symbolic_non_axis_dim_is_dropped(self):
        width = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(0, width)), mb.TensorSpec(shape=(3, 4))])
        def prog(empty, y):
            return mb.relu(x=mb.concat(values=[y, empty], axis=0))

        assert prog.functions["main"].outputs[0].shape == (3, 4)
        _apply(prog)
        assert get_op_types_in_program(prog) == ["relu"]
        assert prog.functions["main"].outputs[0].shape == (3, 4)
