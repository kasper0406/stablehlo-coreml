import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
)

from tests.passes.helpers import apply_pass, count_ops, ops_of_type
from tests.utils import get_model_instruction_types, run_and_compare, run_and_compare_symbolic

PASS_NAME = "common::remove_broadcast_tiles"


def _apply(prog):
    apply_pass(prog, PASS_NAME, dce=False, skip_output_shape_check=True)


def _count_tiles(prog, recurse: bool = True) -> int:
    return count_ops(prog, "tile", recurse=recurse)


class TestRemoveBroadcastTiles:
    """Unit tests on hand-built MIL programs."""

    def test_removed_for_scalar_const_operand(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8))])
        def prog(x):
            scalar = np.full((1, 1), 2.0, dtype=np.float32)
            tiled = mb.tile(x=scalar, reps=[4, 8])
            return mb.mul(x=x, y=tiled)

        assert get_op_types_in_program(prog) == ["tile", "mul"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["mul"]
        assert_model_is_valid(
            prog, {"x": (4, 8)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_removed_for_tensor_operand(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(4, 1))])
        def prog(x, y):
            tiled = mb.tile(x=y, reps=[1, 8])
            return mb.add(x=x, y=tiled)

        assert get_op_types_in_program(prog) == ["tile", "add"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["add"]
        assert_model_is_valid(
            prog, {"x": (4, 8), "y": (4, 1)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_removed_when_both_operands_are_tiled(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 1)), mb.TensorSpec(shape=(1, 8))])
        def prog(x, y):
            tiled_x = mb.tile(x=x, reps=[1, 8])
            tiled_y = mb.tile(x=y, reps=[4, 1])
            return mb.add(x=tiled_x, y=tiled_y)

        assert get_op_types_in_program(prog) == ["tile", "tile", "add"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["add"]
        assert_model_is_valid(
            prog, {"x": (4, 1), "y": (1, 8)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_removed_with_symbolic_batch_dim(self):
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 4, 8)), mb.TensorSpec(shape=(batch, 4, 1))])
        def prog(x, y):
            tiled = mb.tile(x=y, reps=[1, 1, 8])
            return mb.add(x=x, y=tiled)

        assert get_op_types_in_program(prog) == ["tile", "add"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["add"]

    def test_removed_when_the_operands_carry_different_symbols(self):
        """MIL mints a fresh symbol per op, so one runtime dimension gets several names.

        `is5` and `dim_0` below are the same dimension at runtime, but no
        compile-time comparison can say so. Only the axis the tile actually
        replicated matters, and that one is a literal.
        """
        batch = get_new_symbol()
        renamed_batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 8)), mb.TensorSpec(shape=(renamed_batch, 1))])
        def prog(x, y):
            tiled = mb.tile(x=y, reps=[1, 8])
            return mb.mul(x=x, y=tiled)

        assert get_op_types_in_program(prog) == ["tile", "mul"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["mul"]

    def test_not_removed_when_the_other_operand_lacks_the_replicated_size(self):
        """Both operands are size 1 on the replicated axis, so the output would shrink."""
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 1)), mb.TensorSpec(shape=(batch, 1))])
        def prog(x, y):
            tiled = mb.tile(x=y, reps=[1, 8])
            return mb.add(x=x, y=tiled)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile", "add"]
        assert prog.functions["main"].outputs[0].shape == (batch, 8)

    def test_not_removed_when_the_tile_is_both_operands(self):
        """Bypassing the tile on both sides takes the output back down to (4, 1)."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 1))])
        def prog(x):
            tiled = mb.tile(x=x, reps=[1, 8])
            return mb.mul(x=tiled, y=tiled)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile", "mul"]
        assert prog.functions["main"].outputs[0].shape == (4, 8)

    def test_removed_when_the_other_operand_has_a_lower_rank(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(8,)), mb.TensorSpec(shape=(4, 1))])
        def prog(x, y):
            tiled = mb.tile(x=y, reps=[1, 8])
            return mb.add(x=tiled, y=x)

        assert get_op_types_in_program(prog) == ["tile", "add"]
        _apply(prog)
        assert get_op_types_in_program(prog) == ["add"]
        assert prog.functions["main"].outputs[0].shape == (4, 8)

    @pytest.mark.xfail(
        strict=True,
        reason="`matmul` broadcasts its batch dimensions natively (verified against the "
               "runtime), but it is not in `_BROADCAST_OPS`: unlike the elementwise ops, "
               "only its leading axes broadcast while the trailing two are contracted, so "
               "it needs its own axis rule. `jnp.broadcast_to(x, (B, ...)) @ y` therefore "
               "still materialises the full batch.",
    )
    def test_batch_broadcast_ahead_of_matmul_is_removed(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8)), mb.TensorSpec(shape=(2, 8, 4))])
        def prog(x, y):
            tiled = mb.tile(x=x, reps=[2, 1, 1])
            return mb.matmul(x=tiled, y=y)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["matmul"]

    def test_not_removed_when_symbolic_dim_is_tiled(self):
        """A symbolic dim cannot be proven to be 1, so tiling it is not a broadcast."""
        batch = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 8)), mb.TensorSpec(shape=(4, 8))])
        def prog(x, y):
            tiled = mb.tile(x=x, reps=[4, 1])
            return mb.add(x=tiled, y=y)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile", "add"]

    def test_not_removed_for_real_tile(self):
        """`tile([1, 2], reps=[3]) == [1, 2, 1, 2, 1, 2]` is not a broadcast."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(2,)), mb.TensorSpec(shape=(6,))])
        def prog(x, y):
            tiled = mb.tile(x=x, reps=[3])
            return mb.add(x=tiled, y=y)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile", "add"]

    def test_not_removed_for_select_consumer(self):
        """E5RT cannot propagate shapes through `select` with implicit broadcasting."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(4, 8), dtype=types.bool)])
        def prog(x, cond):
            fill = np.full((1, 1), -1e9, dtype=np.float32)
            tiled = mb.tile(x=fill, reps=[4, 8])
            return mb.select(cond=cond, a=x, b=tiled)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile", "select"]

    def test_not_removed_for_mixed_consumers(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(4, 8), dtype=types.bool)])
        def prog(x, cond):
            fill = np.full((1, 1), -1e9, dtype=np.float32)
            tiled = mb.tile(x=fill, reps=[4, 8])
            added = mb.add(x=x, y=tiled)
            return mb.select(cond=cond, a=added, b=tiled)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile", "add", "select"]

    def test_not_removed_when_output_shape_would_shrink(self):
        """Both operands are rank-1 broadcasts; removing both would shrink the output."""
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            const = np.zeros((1,), dtype=np.float32)
            tiled_x = mb.tile(x=x, reps=[8])
            tiled_const = mb.tile(x=const, reps=[8])
            return mb.add(x=tiled_x, y=tiled_const)

        _apply(prog)
        # Exactly one of the two tiles can go; the other has to stay to keep the (8,) output.
        assert _count_tiles(prog) == 1
        assert prog.functions["main"].outputs[0].shape == (8,)
        assert_model_is_valid(
            prog, {"x": (1,)}, minimum_deployment_target=ct.target.iOS18, backend=("mlprogram", "fp32")
        )

    def test_not_removed_when_tile_is_block_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 1))])
        def prog(x):
            return mb.tile(x=x, reps=[1, 8])

        _apply(prog)
        assert get_op_types_in_program(prog) == ["tile"]

    def test_removed_inside_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            def true_fn():
                scalar = np.full((1, 1), 2.0, dtype=np.float32)
                tiled = mb.tile(x=scalar, reps=[4, 8])
                return mb.mul(x=x, y=tiled)

            def false_fn():
                return mb.identity(x=x)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        assert _count_tiles(prog) == 1
        _apply(prog)
        assert _count_tiles(prog) == 0

    def test_not_removed_when_consumed_from_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(1,), dtype=types.bool)])
        def prog(x, pred):
            scalar = np.full((1, 1), 2.0, dtype=np.float32)
            tiled = mb.tile(x=scalar, reps=[4, 8])

            def true_fn():
                return mb.mul(x=x, y=tiled)

            def false_fn():
                return mb.identity(x=tiled)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        assert _count_tiles(prog) == 1

    def test_is_idempotent(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8)), mb.TensorSpec(shape=(4, 1))])
        def prog(x, y):
            tiled = mb.tile(x=y, reps=[1, 8])
            return mb.add(x=x, y=tiled)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestRemoveBroadcastTilesEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    @staticmethod
    def _large_constants(cml_model, max_elements: int = 16):
        large = []
        for op in ops_of_type(cml_model._mil_program, "const"):
            val = op.outputs[0].val
            if val is not None and np.asarray(val).size > max_elements:
                large.append((op.name, np.asarray(val).shape))
        return large

    def test_scalar_broadcast_leaves_no_tile_and_no_large_const(self):
        def f(x):
            return x * 2.0

        cml_model = run_and_compare(f, [jax.ShapeDtypeStruct((256, 1024), jnp.float32)])
        ops = get_model_instruction_types(cml_model)
        assert "tile" not in ops
        assert self._large_constants(cml_model) == []

    def test_tensor_broadcast_leaves_no_tile(self):
        def f(x, bias):
            return x + bias

        cml_model = run_and_compare(
            f,
            [jax.ShapeDtypeStruct((2, 16, 64), jnp.float32), jax.ShapeDtypeStruct((1, 1, 64), jnp.float32)],
        )
        assert "tile" not in get_model_instruction_types(cml_model)

    def test_symbolic_broadcast_leaves_no_tile(self):
        """Dynamic shapes: the two `mul` operands reach MIL with different symbols."""
        symbolic_shape = jax.export.symbolic_shape("(b, 8)")

        def f(x):
            return jnp.mean(x, axis=-1, keepdims=True) * x

        cml_model = run_and_compare_symbolic(
            f,
            [jax.ShapeDtypeStruct(symbolic_shape, jnp.float32)],
            [
                (np.random.randn(3, 8).astype(np.float32),),
                (np.random.randn(5, 8).astype(np.float32),),
            ],
        )
        assert "tile" not in get_model_instruction_types(cml_model)

    def test_where_keeps_its_tile(self):
        """`jnp.where` lowers to `select`, which must keep its explicit broadcast tile."""
        def f(scores, mask):
            return jnp.where(mask, scores, jnp.float32(-1e9))

        cml_model = run_and_compare(
            f,
            [jax.ShapeDtypeStruct((2, 4, 8), jnp.float32), jax.ShapeDtypeStruct((2, 4, 8), jnp.bool_)],
        )
        assert "select" in get_model_instruction_types(cml_model)

        # The tile feeding `select` survives our pass (coremltools' const_elimination
        # then folds it into a full-shape constant), so `select` never sees operands
        # that need implicit broadcasting.
        selects = [
            op
            for func in cml_model._mil_program.functions.values()
            for op in func.operations
            if op.op_type == "select"
        ]
        assert len(selects) == 1
        out_shape = tuple(selects[0].outputs[0].shape)
        for operand in ("cond", "a", "b"):
            assert tuple(selects[0].inputs[operand].shape) == out_shape
