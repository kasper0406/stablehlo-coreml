import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.testing_utils import get_op_types_in_program

from stablehlo_coreml.passes.pattern_utils import dims_equal, shapes_equal
from tests.passes.helpers import apply_pass, ops_of_type, predict
from tests.utils import get_model_instruction_types, run_and_compare, run_and_compare_symbolic

PASS_NAME = "common::broadcast_select_operands"

_OPERANDS = ("cond", "a", "b")


def _apply(prog):
    """Apply the pass, returning a deep copy of the program as it was before."""
    return apply_pass(prog, PASS_NAME, dce=False, skip_output_shape_check=True)


def _sole_select(prog):
    selects = ops_of_type(prog, "select", recurse=True)
    assert len(selects) == 1, f"expected exactly one select, got {len(selects)}"
    return selects[0]


def _operand_shapes(select_op):
    return {name: tuple(select_op.inputs[name].shape) for name in _OPERANDS}


def _ct_inputs(prog, symbolic_dim: int, range_max: int = 64):
    """``ct.TensorType``s for ``prog``, with a ``RangeDim`` on every symbolic axis."""
    func = next(iter(prog.functions.values()))
    return [
        ct.TensorType(
            name=name,
            shape=[
                ct.RangeDim(1, range_max, default=symbolic_dim) if is_symbolic(dim) else int(dim)
                for dim in var.shape
            ],
        )
        for name, var in func.inputs.items()
    ]


def _predict(prog, values, symbolic_dim: int):
    """Convert ``prog`` and run it on ``values``, with ``symbolic_dim`` for the symbolic axes."""
    return predict(prog, ct_inputs=_ct_inputs(prog, symbolic_dim), **values)


class TestBroadcastSelectOperands:
    """Unit tests on hand-built MIL programs."""

    def test_widens_the_value_operand(self):
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, length, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        assert get_op_types_in_program(prog) == ["select"]
        _apply(prog)

        ops = get_op_types_in_program(prog)
        assert ops == ["fill_like", "add", "select"]

        select = _sole_select(prog)
        out_shape = tuple(select.outputs[0].shape)
        assert shapes_equal(out_shape, (1, length, 1, 8))
        for name, shape in _operand_shapes(select).items():
            assert shapes_equal(shape, out_shape), f"{name} has shape {shape}, expected {out_shape}"

    def test_widens_the_cond_operand(self):
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)

        # A bool operand is widened with `logical_or` against `False`, not `add`.
        assert get_op_types_in_program(prog) == ["fill_like", "logical_or", "select"]

        select = _sole_select(prog)
        out_shape = tuple(select.outputs[0].shape)
        assert select.inputs["cond"].dtype == types.bool
        for shape in _operand_shapes(select).values():
            assert shapes_equal(shape, out_shape)

    def test_widens_cond_and_value_together(self):
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)

        ops = get_op_types_in_program(prog)
        assert ops.count("fill_like") == 2
        assert ops.count("logical_or") == 1
        assert ops.count("add") == 1

        select = _sole_select(prog)
        out_shape = tuple(select.outputs[0].shape)
        for shape in _operand_shapes(select).values():
            assert shapes_equal(shape, out_shape)

    def test_widens_a_lower_rank_operand_that_hits_a_symbolic_axis(self):
        """The axes broadcasting prepends read as 1, so they need widening too."""
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(8,)),
            mb.TensorSpec(shape=(1, length, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)

        assert get_op_types_in_program(prog) == ["fill_like", "add", "select"]
        select = _sole_select(prog)
        out_shape = tuple(select.outputs[0].shape)
        for shape in _operand_shapes(select).values():
            assert shapes_equal(shape, out_shape)

    def test_not_widened_when_only_a_static_axis_broadcasts(self):
        """``(1, L, 1, 1)`` carries the symbolic axis, so the runtime can size it."""
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, length, 1, 1)),
            mb.TensorSpec(shape=(1, length, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)

        assert get_op_types_in_program(prog) == ["select"]
        assert _operand_shapes(_sole_select(prog))["a"] == (1, length, 1, 1)

    def test_static_shapes_are_untouched(self):
        """No symbolic dimension, no problem: E5RT sizes a static broadcast fine."""
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(2, 4, 8)),
            mb.TensorSpec(shape=(1, 4, 8)),
            mb.TensorSpec(shape=(2, 4, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        before = _apply(prog)

        assert get_op_types_in_program(prog) == get_op_types_in_program(before) == ["select"]
        assert _operand_shapes(_sole_select(prog)) == _operand_shapes(_sole_select(before))

    def test_not_widened_without_a_full_shape_reference(self):
        """No operand carries the whole output shape, so ``fill_like`` has nothing to size from."""
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 1)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        before = _apply(prog)

        assert get_op_types_in_program(prog) == ["select"]
        assert _operand_shapes(_sole_select(prog)) == _operand_shapes(_sole_select(before))
        assert shapes_equal(_sole_select(prog).outputs[0].shape, (1, length, 1, 8))

    def test_widened_inside_nested_block(self):
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, length, 1, 8), dtype=types.bool),
            mb.TensorSpec(shape=(1,), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask, pred):
            def true_fn():
                return mb.select(cond=mask, a=value, b=cache)

            def false_fn():
                return mb.identity(x=cache)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)

        ops = get_op_types_in_program(prog, recurse=True)
        assert ops.count("fill_like") == 1
        assert ops.count("add") == 1

        select = _sole_select(prog)
        out_shape = tuple(select.outputs[0].shape)
        for shape in _operand_shapes(select).values():
            assert shapes_equal(shape, out_shape)

    def test_widened_operands_keep_the_output_symbol(self):
        """The widened operand carries the reference's symbol, not a fresh one."""
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, length, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)

        widened = _sole_select(prog).inputs["a"]
        assert dims_equal(widened.shape[1], length)

    def test_is_idempotent(self):
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, length, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first

    def test_widening_does_not_change_the_result(self):
        """Widening is ``+ 0`` / ``or False``, so the values are exactly the ones ``select`` gave.

        The reference is ``np.where``, i.e. the implicit-broadcast semantics of
        the ``select`` before the rewrite. Predicting the *un-widened* program
        would be the more direct comparison, but that is the construction this
        pass exists to remove: running it makes Core ML log

            E5RT encountered an STL exception. msg = Failed to
            PropagateInputTensorShapes: Validation error during type inference
            for select: at unknown location: Incompatible Dimension.

        The runtime happens to recover from it here (it is a multifunction
        ``.mlpackage`` load that fails outright), so the message can neither be
        asserted on nor relied upon not to become a hard failure.
        """
        length = get_new_symbol()

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, length, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8)),
            mb.TensorSpec(shape=(1, 1, 1, 8), dtype=types.bool),
        ], opset_version=ct.target.iOS18)
        def prog(cache, value, mask):
            return mb.select(cond=mask, a=value, b=cache)

        _apply(prog)
        # Both a float (`add`) and a bool (`logical_or`) widening are exercised.
        assert get_op_types_in_program(prog).count("fill_like") == 2

        concrete = 5
        rng = np.random.default_rng(0)
        mask = rng.random((1, 1, 1, 8)) > 0.5
        values = {
            "cache": rng.standard_normal((1, concrete, 1, 8)).astype(np.float32),
            "value": rng.standard_normal((1, 1, 1, 8)).astype(np.float32),
            # Core ML has no bool model input; it is exposed as fp32 and cast back.
            "mask": mask.astype(np.float32),
        }
        expected = np.where(mask, values["value"], values["cache"])

        np.testing.assert_allclose(_predict(prog, values, concrete), expected, atol=1e-6)


class TestBroadcastSelectOperandsEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    def test_where_widens_the_value_operand_under_symbolic_shapes(self):
        """``jnp.where`` against a symbolic-length cache: the value is never tiled.

        JAX emits a ``dynamic_broadcast_in_dim`` for the value operand, which the
        converter cannot turn into a ``tile`` (the reps would be symbolic), so the
        ``select`` reaching MIL broadcasts a static 1 into the symbolic length.

        The E5RT load failure this guards against ("Failed to
        PropagateInputTensorShapes ... Incompatible Dimension") only shows up when
        a multifunction ``.mlpackage`` is loaded on device, which CI cannot
        reproduce; the test asserts the graph shape that avoids it instead.
        """
        (length,) = jax.export.symbolic_shape("(L,)")

        def f(cache, value, mask):
            return jnp.where(mask, value, cache)

        def sample(concrete, seed):
            rng = np.random.default_rng(seed)
            return (
                rng.standard_normal((1, concrete, 2, 8)).astype(np.float32),
                rng.standard_normal((1, 1, 2, 8)).astype(np.float32),
                rng.random((1, concrete, 2, 8)) > 0.5,
            )

        cml_model = run_and_compare_symbolic(
            f,
            [
                jax.ShapeDtypeStruct((1, length, 2, 8), jnp.float32),
                jax.ShapeDtypeStruct((1, 1, 2, 8), jnp.float32),
                jax.ShapeDtypeStruct((1, length, 2, 8), jnp.bool_),
            ],
            [sample(3, seed=0), sample(7, seed=1)],
        )

        ops = get_model_instruction_types(cml_model)
        assert "select" in ops
        assert "fill_like" in ops

        selects = ops_of_type(cml_model._mil_program, "select", recurse=True)
        assert len(selects) == 1
        for select in selects:
            out_shape = tuple(select.outputs[0].shape)
            assert any(is_symbolic(dim) for dim in out_shape)
            for name, operand in _operand_shapes(select).items():
                assert len(operand) == len(out_shape), f"{name}: {operand} vs {out_shape}"
                for dim, out_dim in zip(operand, out_shape):
                    assert not (is_symbolic(out_dim) and not is_symbolic(dim) and int(dim) == 1), (
                        f"{name} broadcasts from 1 into the symbolic dim of {out_shape}"
                    )

    def test_static_where_is_left_to_remove_broadcast_tiles(self):
        """The static path is untouched: its tile already gives `select` full-shape operands.

        ``test_where_keeps_its_tile`` in ``test_remove_broadcast_tiles.py`` covers
        the tile itself; this asserts the new pass adds nothing on top of it.
        """
        def f(scores, mask):
            return jnp.where(mask, scores, jnp.float32(-1e9))

        cml_model = run_and_compare(
            f,
            [jax.ShapeDtypeStruct((2, 4, 8), jnp.float32), jax.ShapeDtypeStruct((2, 4, 8), jnp.bool_)],
        )

        assert "fill_like" not in get_model_instruction_types(cml_model)
        select = _sole_select(cml_model._mil_program)
        out_shape = tuple(select.outputs[0].shape)
        assert all(shape == out_shape for shape in _operand_shapes(select).values())
