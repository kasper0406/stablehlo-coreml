import jax
import jax.numpy as jnp
from functools import partial
import pytest

from tests.utils import run_and_compare, run_and_compare_specific_input, get_model_instruction_types


def test_addition():
    run_and_compare(jnp.add, (jnp.float32(1), jnp.float32(1)))
    run_and_compare(jnp.add, (jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2))))


def test_div():
    run_and_compare(jnp.divide, (jnp.float32(1), jnp.float32(1)))

    dim_size = 20
    run_and_compare(jnp.divide, (jnp.zeros((dim_size, dim_size)), jnp.zeros((dim_size, dim_size))))
    run_and_compare(jnp.divide, (jnp.zeros((dim_size, dim_size)), jnp.float32(1)))
    run_and_compare(jnp.divide, (jnp.float32(1), jnp.zeros((dim_size, dim_size))))
    run_and_compare(jnp.divide, (jnp.zeros((dim_size, dim_size)), jnp.zeros((dim_size, dim_size))))

    run_and_compare(jnp.divide, (jnp.int32(1), jnp.int32(1)))
    run_and_compare(jnp.divide, (
        jnp.zeros((dim_size, dim_size), dtype=jnp.int32),
        jnp.zeros((dim_size, dim_size), dtype=jnp.int32)
    ))
    run_and_compare(jnp.divide, (jnp.zeros((dim_size, dim_size), dtype=jnp.int32), jnp.int32(1)))
    run_and_compare(jnp.divide, (jnp.int32(1), jnp.zeros((dim_size, dim_size), dtype=jnp.int32)))


def test_tensor_multiplication():
    def scalar_product(lhs, rhs):
        return jnp.einsum("a,a", lhs, rhs)

    def scalar_with_vector(lhs, rhs):
        return jnp.einsum("a,b->ab", lhs, rhs)

    def scalar_with_matrix(lhs, rhs):
        return jnp.einsum("a,bc->abc", lhs, rhs)

    def vector_with_matrix(lhs, rhs):
        return jnp.einsum("a,ab->b", lhs, rhs)

    def matrix_multiplication(lhs, rhs):
        return jnp.einsum("ij,jk -> ik", lhs, rhs)

    def outer_product_with_single_batch_dim(lhs, rhs):
        return jnp.einsum("abc,ajk->abcjk", lhs, rhs)

    def single_contraction_single_batch(lhs, rhs):
        return jnp.einsum("abcd,ackl->abdkl", lhs, rhs)

    def two_contractions_single_batch(lhs, rhs):
        return jnp.einsum("abcd,ackd->abk", lhs, rhs)

    def three_contractions_single_batch(lhs, rhs):
        return jnp.einsum("abcd,acbd->a", lhs, rhs)

    def contract_all(lhs, rhs):
        return jnp.einsum("abcd,acbd", lhs, rhs)

    def full_tensor_product(lhs, rhs):
        return jnp.einsum("ab,ihj->abihj", lhs, rhs)

    def full_tensor_product_1_4(lhs, rhs):
        return jnp.einsum("a,ihjk->aihjk", lhs, rhs)

    def full_tensor_product_3_2(lhs, rhs):
        return jnp.einsum("abc,ih->abcih", lhs, rhs)

    def full_tensor_product_4_1(lhs, rhs):
        return jnp.einsum("abcd,i->abcdi", lhs, rhs)

    run_and_compare(scalar_product, (jnp.zeros((1)), jnp.zeros((1))))
    run_and_compare(scalar_with_vector, (jnp.zeros((1)), jnp.zeros((5))))
    run_and_compare(scalar_with_matrix, (jnp.zeros((1)), jnp.zeros((5, 3))))
    run_and_compare(vector_with_matrix, (jnp.zeros((5)), jnp.zeros((5, 3))))
    run_and_compare(matrix_multiplication, (jnp.zeros((3, 4)), jnp.zeros((4, 5))))
    run_and_compare(outer_product_with_single_batch_dim, (jnp.zeros((2, 3, 4)), jnp.zeros((2, 4, 5))))
    run_and_compare(single_contraction_single_batch, (jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 4, 2, 5))))
    run_and_compare(two_contractions_single_batch, (jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 4, 2, 5))))
    run_and_compare(three_contractions_single_batch, (jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 4, 3, 5))))
    run_and_compare(full_tensor_product, (jnp.zeros((2, 3)), jnp.zeros((2, 4, 3))))
    run_and_compare(contract_all, (jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 4, 3, 5))))

    # # Test the full tensor product with a big dimensions, and ensure that the program gets handled by a dynamic loop
    run_and_compare(full_tensor_product, (jnp.zeros((10, 3)), jnp.zeros((15, 20, 3))))
    run_and_compare(full_tensor_product_1_4, (jnp.zeros((10,)), jnp.zeros((15, 20, 5, 3))))
    run_and_compare(full_tensor_product_1_4, (jnp.zeros((2,)), jnp.zeros((2, 2, 2, 3))))
    run_and_compare(full_tensor_product_3_2, (jnp.zeros((20, 10, 3)), jnp.zeros((15, 3))))
    run_and_compare(full_tensor_product_3_2, (jnp.zeros((2, 2, 3)), jnp.zeros((2, 3))))
    run_and_compare(full_tensor_product_4_1, (jnp.zeros(((15, 20, 5, 3))), jnp.zeros((10,))))
    run_and_compare(full_tensor_product_4_1, (jnp.zeros(((2, 2, 2, 3))), jnp.zeros((2,))))


def test_simple_reductions():
    def compare_and_ensure_no_loops(jax_func, input_spec):
        cml_model = run_and_compare(jax_func, input_spec)
        assert "while_loop" not in get_model_instruction_types(cml_model)

    compare_and_ensure_no_loops(partial(jnp.max, axis=1), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.max, axis=1, keepdims=True), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.sum, axis=0), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.sum, axis=1), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.sum, axis=2), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.sum, axis=(0, 2)), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.sum, axis=(0, 1, 2)), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.min, axis=0), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.min, axis=(1, 2)), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.mean, axis=0), (jnp.zeros((2, 3, 4)),))
    compare_and_ensure_no_loops(partial(jnp.prod, axis=1), (jnp.zeros((2, 3, 4)),))


def test_complex_reductions():
    """
    These reductions are complicated, and will be handled using while loops (potentially unrolled)
    """
    run_and_compare(jnp.argmax, (jnp.zeros((2, 3, 3)),))
    run_and_compare(partial(jnp.argmax, keepdims=True), (jnp.zeros((2, 3, 3)),))
    run_and_compare(jnp.argmax, (jnp.zeros((20, 30, 40)),))
    run_and_compare(partial(jnp.argmax, keepdims=True), (jnp.zeros((20, 30, 40)),))
    run_and_compare(partial(jnp.argmax, axis=1), (jnp.zeros((2, 3, 3)),))
    run_and_compare(partial(jnp.argmax, axis=0, keepdims=True), (jnp.zeros((2, 3, 3)),))
    run_and_compare(partial(jnp.argmax, axis=2), (jnp.zeros((2, 3, 3)),))
    run_and_compare(partial(jnp.argmax, axis=1), (jnp.zeros((20, 30, 40)),))

    run_and_compare(jnp.argmin, (jnp.zeros((2, 3, 3)),))
    run_and_compare(partial(jnp.argmin, axis=1), (jnp.zeros((2, 3, 3)),))
    run_and_compare(partial(jnp.argmin, axis=1), (jnp.zeros((20, 100, 40)),))
    run_and_compare(partial(jnp.argmin, axis=1, keepdims=True), (jnp.zeros((20, 100, 40)),))


def test_topk():
    input_shape = (3, 5, 10)
    run_and_compare(partial(jax.lax.top_k, k=3), (jnp.zeros(input_shape),))


def test_reverse():
    run_and_compare(jnp.flip, (jnp.zeros((5,)),))
    run_and_compare(jnp.flip, (jnp.zeros((5, 5, 5)),))
    run_and_compare(partial(jnp.flip, axis=(0, 2)), (jnp.zeros((5, 5, 5)),))
    run_and_compare(partial(jnp.flip, axis=(1,)), (jnp.zeros((5, 5, 5)),))


def test_trigonmetry():
    run_and_compare(jnp.sin, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.cos, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.tan, (jnp.zeros((5, 6)),))

    run_and_compare(jnp.arcsin, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.arccos, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.arctan, (jnp.zeros((5, 6)),))

    run_and_compare(jnp.sinh, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.cosh, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.tanh, (jnp.zeros((5, 6)),))

    run_and_compare(jnp.arcsinh, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.arccosh, (jnp.zeros((5, 6)),))
    run_and_compare(jnp.arctanh, (jnp.zeros((5, 6)),))

    run_and_compare(jnp.atan2, (jnp.zeros((50, 20)), jnp.zeros((50, 20)),))


def test_is_finite():
    input = (jnp.array([20.0, -12.23, jnp.inf, -jnp.inf, jnp.nan], dtype=jnp.float16), )
    run_and_compare_specific_input(jnp.isfinite, input)
    run_and_compare_specific_input(jnp.isinf, input)
    run_and_compare_specific_input(jnp.isnan, input)


def test_take():
    run_and_compare_specific_input(jnp.take, (jnp.reshape(jnp.arange(24), (4, 6)), jnp.array([
        [[0, 0], [1, 1], [2, 2]]
    ], dtype=jnp.int32)))


def test_gather():
    from jax.lax import GatherDimensionNumbers

    def wrapped_gather(dimension_numbers, slice_sizes):
        @jax.jit
        def internal_gather(operand, start_indices):
            return jax.lax.gather(
                operand=operand,
                start_indices=start_indices,
                dimension_numbers=dimension_numbers,
                slice_sizes=slice_sizes,
            )
        return internal_gather

    operand = jnp.reshape(jnp.arange(8000), (10, 8, 5, 20))
    start_indices = jnp.array([
        [1, 1], [3, 1], [1, 10], [4, 15],
    ], dtype=jnp.int32)

    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(0, 1),
        collapsed_slice_dims=(0, 2,),
        start_index_map=(1, 3, )
    )

    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 2, 1, 3)), (operand, start_indices))
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 3, 1, 4)), (operand, start_indices))
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 3, 1, 7)), (operand, start_indices))
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 8, 1, 2)), (operand, start_indices))

    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 2,),
        start_index_map=(1, 3, )
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 3, 1, 4)), (operand, start_indices))

    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(0, 2),
        collapsed_slice_dims=(0, 2,),
        start_index_map=(1, 3, )
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 3, 1, 4)), (operand, start_indices))

    operand = jnp.reshape(jnp.arange(50), (5, 10))
    start_indices = jnp.array([
        [0, 1], [1, 0], [0, 0], [2, 6], [4, 2]
    ], dtype=jnp.int32)
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=tuple(),
        start_index_map=(0, 1)
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 10)), (operand, start_indices))

    operand = jnp.reshape(jnp.arange(500), (5, 2, 5, 10))
    start_indices = jnp.array([
        [0, 1, 4], [1, 0, 8], [0, 0, 2], [2, 6, 1], [4, 2, 0]
    ], dtype=jnp.int32)
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(0, 1, 2, 3),
        collapsed_slice_dims=tuple(),
        start_index_map=(0, 1, 2)
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 1, 2, 4)), (operand, start_indices))

    operand = jnp.reshape(jnp.arange(50), (10, 5))
    start_indices = jnp.array([
        [[3], [1], [7]],
        [[4], [0], [9]]
    ], dtype=jnp.int32)  # (2, 3, 1)

    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )

    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 5)), (operand, start_indices))


def test_complex_gather():
    from jax.lax import GatherDimensionNumbers

    def wrapped_gather(dimension_numbers, slice_sizes):
        @jax.jit
        def internal_gather(operand, start_indices):
            return jax.lax.gather(
                operand=operand,
                start_indices=start_indices,
                dimension_numbers=dimension_numbers,
                slice_sizes=slice_sizes,
            )
        return internal_gather

    start_indices = [
                     [
                      [[0, 0], [1, 0], [2, 1]],
                      [[0, 1], [1, 1], [0, 9]]
                     ],
                     [
                      [[0, 0], [2, 1], [2, 2]],
                      [[1, 2], [0, 1], [1, 0]]
                     ]
                    ]
    start_indices = jnp.array(start_indices, dtype=jnp.int32)
    operand = jnp.arange(1, 49).reshape((2, 3, 4, 2))
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(3, 4),
        collapsed_slice_dims=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
        start_index_map=(2, 1),
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 1, 1, 2)), (operand, start_indices))

    operand = jnp.arange(1, 25).reshape((3, 4, 2))
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0,),
        start_index_map=(1, 0),
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 2, 2)), (operand, start_indices[0]))

    operand = jnp.arange(1, 49).reshape((2, 3, 4, 2))
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(3, 4),
        collapsed_slice_dims=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(1,),
        start_index_map=(2, 1),
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 1, 1, 2)), (operand, start_indices))

    start_indices = jnp.concatenate((start_indices, start_indices[::-1, 1:]), 1)
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(3, 4),
        collapsed_slice_dims=(),
        operand_batching_dims=(0, 1),
        start_indices_batching_dims=(0, 1),
        start_index_map=(3, 2),
    )
    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 1, 1, 1)), (operand, start_indices))


def test_large_gather():
    # Test gather with large indices to verify if optimization is necessary
    # and if it works correctly.
    from jax.lax import GatherDimensionNumbers

    def wrapped_gather(dimension_numbers, slice_sizes):
        @jax.jit
        def internal_gather(operand, start_indices):
            return jax.lax.gather(
                operand=operand,
                start_indices=start_indices,
                dimension_numbers=dimension_numbers,
                slice_sizes=slice_sizes,
            )
        return internal_gather

    # Create a large operand and indices
    # Operand: (10, 20, 30)
    operand = jnp.reshape(jnp.arange(6000), (10, 20, 30))

    # Indices: (100, 2) -> gathering 100 slices
    # We want enough indices to potentially trigger unroll limits if not optimized
    num_indices = 100
    start_indices = jnp.zeros((num_indices, 2), dtype=jnp.int32)
    # Fill with some valid indices
    for i in range(num_indices):
        start_indices = start_indices.at[i, 0].set(i % 10)
        start_indices = start_indices.at[i, 1].set(i % 20)

    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )

    # slice_sizes: (1, 1, 30)
    # We gather from dim 0 and 1. Dim 2 is kept.
    # Result shape: (100, 30)

    run_and_compare_specific_input(wrapped_gather(dimension_numbers, (1, 1, 30)), (operand, start_indices))


def test_simple_scatter():
    def scatter_set(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].set(updates)
    run_and_compare(scatter_set, (jnp.zeros((30,)),))

    def scatter_add(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].add(updates)
    run_and_compare(scatter_add, (jnp.zeros((30,)),))

    def scatter_sub(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].subtract(updates)
    run_and_compare(scatter_sub, (jnp.zeros((30,)),))

    def scatter_mul(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].multiply(updates)
    run_and_compare(scatter_mul, (jnp.zeros((30,)),))

    def scatter_div(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].divide(updates)
    run_and_compare(scatter_div, (jnp.zeros((30,)),))

    def scatter_max(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].max(updates)
    run_and_compare(scatter_max, (jnp.zeros((30,)),))

    def scatter_min(arr):
        indices = jnp.arange(arr.shape[0] // 2) * 2
        updates = jnp.arange(indices.shape[0])
        return arr.at[indices].min(updates)
    run_and_compare(scatter_min, (jnp.zeros((30,)),))


def test_scatter_with_dimension_numbers():
    from jax.lax import ScatterDimensionNumbers

    def wrapped_scatter_add(dimension_numbers):
        @jax.jit
        def internal_scatter_add(operand, scatter_indices, updates):
            return jax.lax.scatter_add(
                operand=operand,
                scatter_indices=scatter_indices,
                updates=updates,
                dimension_numbers=dimension_numbers,
            )
        return internal_scatter_add

    # https://raw.githubusercontent.com/openxla/stablehlo/bd8d708/docs/images/spec/scatter.svg
    # original test case features partially filled update dimension windows

    scatter_indices = [[[0, 2], [1, 0], [2, 1]], [[0, 1], [1, 0], [0, 9]]]
    scatter_indices = jnp.array(scatter_indices)
    operand = jnp.arange(1, 25).reshape((3, 4, 2))
    update = jnp.ones((2, 3, 2), dtype=jnp.int32)
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(2,),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(1, 0),
    )

    run_and_compare_specific_input(wrapped_scatter_add(dimension_numbers), (operand, scatter_indices, update))


@pytest.mark.parametrize("op_fn,op_name", [
    (lambda x, y, z, dnums: jax.lax.scatter_add(x, y, z, dnums), "add"),
    (lambda x, y, z, dnums: jax.lax.scatter_mul(x, y, z, dnums), "mul"),
    (lambda x, y, z, dnums: jax.lax.scatter_min(x, y, z, dnums), "min"),
    (lambda x, y, z, dnums: jax.lax.scatter_max(x, y, z, dnums), "max"),
    # scatter_apply (set) is slightly different in JAX, usually just scatter
    (lambda x, y, z, dnums: jax.lax.scatter(x, y, z, dnums), "set"),
], ids=["add", "mul", "min", "max", "set"])
def test_scatter_middle_update_rank1(op_fn, op_name):
    # Update a slice in the middle, leaving the end untouched.
    # Operand: [0, 1, 2, 3, 4]
    # Indices: [1] (shape (1,))
    # Updates: [10, 11] (shape (2,))
    # Expected (add): [0, 11, 13, 3, 4]

    def scatter_func(operand, indices, updates):
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(0,),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,)
        )
        return op_fn(operand, indices, updates, dnums)

    operand = jnp.array([0, 1, 2, 3, 4], dtype=jnp.float32)
    indices = jnp.array([1], dtype=jnp.int32)  # Shape (1,)
    updates = jnp.array([10, 11], dtype=jnp.float32)  # Shape (2,)

    run_and_compare_specific_input(scatter_func, (operand, indices, updates))


@pytest.mark.parametrize("op_fn,op_name", [
    (lambda x, y: x.at[y].set, "set"),
    (lambda x, y: x.at[y].add, "add"),
    (lambda x, y: x.at[y].subtract, "subtract"),
    (lambda x, y: x.at[y].multiply, "multiply"),
    (lambda x, y: x.at[y].divide, "divide"),
    (lambda x, y: x.at[y].max, "max"),
    (lambda x, y: x.at[y].min, "min"),
], ids=["set", "add", "subtract", "multiply", "divide", "max", "min"])
@pytest.mark.parametrize("shape", [
    (30,),
    (10, 3),
    (5, 2, 3),
], ids=["rank1", "rank2", "rank3"])
def test_scatter(op_fn, op_name, shape):
    arr = jnp.zeros(shape)

    axis_len = arr.shape[0]
    indices = jnp.arange(axis_len // 2) * 2
    updates_shape = (indices.shape[0],) + arr.shape[1:]
    updates = jnp.arange(jnp.prod(jnp.array(updates_shape))).reshape(updates_shape)

    def scatter_op(arr):
        return op_fn(arr, indices)(updates)

    run_and_compare(scatter_op, (arr,))


@pytest.mark.parametrize("op_fn,op_name", [
    (lambda x, y: x.at[y].set, "set"),
    (lambda x, y: x.at[y].add, "add"),
    (lambda x, y: x.at[y].subtract, "subtract"),
    (lambda x, y: x.at[y].multiply, "multiply"),
    (lambda x, y: x.at[y].divide, "divide"),
    (lambda x, y: x.at[y].max, "max"),
    (lambda x, y: x.at[y].min, "min"),
], ids=["set", "add", "subtract", "multiply", "divide", "max", "min"])
def test_scatter_2d_indices(op_fn, op_name):
    arr = jnp.zeros((5, 5))

    indices = jnp.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 0],
    ])
    updates = jnp.arange(arr.shape[1])

    def scatter_op(arr):
        return op_fn(arr, indices)(updates)

    run_and_compare(scatter_op, (arr,))


@pytest.mark.parametrize("op_fn,op_name", [
    (lambda x, y: x.at[y].set, "set"),
    (lambda x, y: x.at[y].add, "add"),
    (lambda x, y: x.at[y].subtract, "subtract"),
    (lambda x, y: x.at[y].multiply, "multiply"),
    (lambda x, y: x.at[y].divide, "divide"),
    (lambda x, y: x.at[y].max, "max"),
    (lambda x, y: x.at[y].min, "min"),
], ids=["set", "add", "subtract", "multiply", "divide", "max", "min"])
def test_scatter_3d_indices(op_fn, op_name):
    arr = jnp.zeros((4, 3, 2))

    indices = jnp.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 0],
        [3, 0, 1],
    ])
    updates = jnp.arange(arr.shape[2])

    def scatter_op(arr):
        return op_fn(arr, indices)(updates)

    run_and_compare(scatter_op, (arr,))


@pytest.mark.parametrize("op_fn,op_name", [
    (lambda x, y: x.at[y].set, "set"),
    (lambda x, y: x.at[y].add, "add"),
    (lambda x, y: x.at[y].subtract, "subtract"),
    (lambda x, y: x.at[y].multiply, "multiply"),
    (lambda x, y: x.at[y].divide, "divide"),
    (lambda x, y: x.at[y].max, "max"),
    (lambda x, y: x.at[y].min, "min"),
], ids=["set", "add", "subtract", "multiply", "divide", "max", "min"])
def test_scatter_replace_vector(op_fn, op_name):
    arr = jnp.zeros((3, 4, 5, 6))

    indices = jnp.array([
        [0, 1, 3],
        [1, 2, 4],
    ])

    updates = jnp.reshape(jnp.arange(arr.shape[3] * indices.shape[0]), (indices.shape[0], 1, 1, 1, arr.shape[3]))

    def scatter_op(arr):
        return op_fn(arr, indices)(updates)

    run_and_compare(scatter_op, (arr,))


@pytest.mark.parametrize("op_fn,op_name", [
    (lambda x, y: x.at[y].set, "set"),
    (lambda x, y: x.at[y].add, "add"),
    (lambda x, y: x.at[y].subtract, "subtract"),
    (lambda x, y: x.at[y].multiply, "multiply"),
    (lambda x, y: x.at[y].divide, "divide"),
    (lambda x, y: x.at[y].max, "max"),
    (lambda x, y: x.at[y].min, "min"),
], ids=["set", "add", "subtract", "multiply", "divide", "max", "min"])
def test_scatter_replace_matrix(op_fn, op_name):
    arr = jnp.zeros((3, 4, 5, 6))

    indices = jnp.array([
        [0, 1],
        [1, 2],
    ])

    updates = jnp.arange(arr.shape[2])[None, :] @ jnp.arange(arr.shape[2])[:, None]

    def scatter_op(arr):
        return op_fn(arr, indices)(updates)

    run_and_compare(scatter_op, (arr,))


def test_scatter_empty_indices():
    def scatter_add(operand, indices, updates):
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        return jax.lax.scatter_add(operand, indices, updates, dnums)

    operand = jnp.zeros((4, 4), dtype=jnp.float32)
    indices = jnp.zeros((0, 1), dtype=jnp.int32)
    updates = jnp.ones((0, 4), dtype=jnp.float32)

    run_and_compare_specific_input(scatter_add, (operand, indices, updates))


def test_pad():
    run_and_compare(partial(jnp.pad, pad_width=((0, 0), (10, 5))), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((0, 10), (5, 0), (2, 1))), (jnp.zeros((10, 20, 15)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="empty"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((1, 2), (3, 4)), constant_values=12.3), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="reflect"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="wrap"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="edge"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="linear_ramp"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="maximum"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="mean"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="median"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="minimum"), (jnp.zeros((10, 20)),))
    run_and_compare(partial(jnp.pad, pad_width=((5, 10), (10, 5)), mode="symmetric"), (jnp.zeros((10, 20)),))


def test_pad_int32():
    run_and_compare(partial(jnp.pad, pad_width=((1, 1), (2, 2)), constant_values=10), (jnp.zeros((5, 5), dtype=jnp.int32),))
    run_and_compare(partial(jnp.pad, pad_width=((1, 1), (2, 2))), (jnp.zeros((5, 5), dtype=jnp.int32),))


def test_remainder():
    run_and_compare(jnp.remainder, (
        jnp.array([10, 20, 30], dtype=jnp.int32), jnp.array([3, 7, 11], dtype=jnp.int32)
    ))
    run_and_compare(jnp.remainder, (
        jnp.array([10.5, 20.2, 30.1], dtype=jnp.float32), jnp.array([3.1, 7.2, 11.3], dtype=jnp.float32)
    ))


def test_floor():
    run_and_compare(jnp.floor, (jnp.array([1.1, 2.9, -1.1, -2.9], dtype=jnp.float32),))


def test_ceil():
    run_and_compare(jnp.ceil, (jnp.array([1.1, 2.9, -1.1, -2.9], dtype=jnp.float32),))


def test_clamp():
    run_and_compare(partial(jnp.clip, a_min=0.0, a_max=1.0), (jnp.array([-1.0, 0.5, 2.0], dtype=jnp.float32),))
    run_and_compare(partial(jnp.clip, a_min=-5, a_max=5), (jnp.array([-10, 0, 10], dtype=jnp.int32),))


def test_sort():
    run_and_compare(jnp.sort, (jnp.array([3, 1, 2], dtype=jnp.int32),))
    run_and_compare(jnp.argsort, (jnp.array([3, 1, 2, 0x8002], dtype=jnp.uint16),))
    run_and_compare(jnp.sort, (jnp.array([[3, 1, 2], [6, 5, 4]], dtype=jnp.float32),))
    run_and_compare(partial(jnp.sort, axis=0), (jnp.array([[3, 1, 2], [6, 5, 4]], dtype=jnp.float32),))

    # Test with larger random input
    run_and_compare(jnp.sort, (jnp.zeros((100, 50), dtype=jnp.float32),))
    run_and_compare(partial(jnp.sort, axis=0), (jnp.zeros((100, 50), dtype=jnp.float32),))
    run_and_compare(partial(jnp.sort, descending=True), (jnp.zeros((100, 50), dtype=jnp.float32),))
    run_and_compare(partial(jnp.sort, axis=0, descending=True), (jnp.zeros((100, 50), dtype=jnp.float32),))

    # Test with NaNs and negative zeros to trigger total sort logic
    # Total sort order for floats: -Inf < ... < -0.0 == 0.0 < ... < Inf < NaN
    # Note: CoreML handles NaNs differently than JAX (CoreML puts them at the beginning, JAX at the end)
    data = jnp.array([0.0, -0.0, 1.0, -1.0, jnp.nan, jnp.inf, -jnp.inf, jnp.nan], dtype=jnp.float32)
    run_and_compare_specific_input(jnp.sort, (data,))
    run_and_compare_specific_input(jnp.argsort, (data,))

    # Test with subnormals
    # Smallest normal float32 is 1.17549435e-38
    # Largest subnormal float32 is 1.17549421e-38
    subnormals = jnp.array([1.17549435e-38, 1.17549421e-38, -1.17549421e-38, 0.0], dtype=jnp.float32)
    run_and_compare_specific_input(jnp.sort, (subnormals,))


def test_argsort():
    run_and_compare_specific_input(partial(jnp.argsort, stable=True),
                                   (jnp.array([10, 5, 10, 5, 10, 5, 10, 5, 10, 5], dtype=jnp.int32),))
    run_and_compare_specific_input(partial(jnp.argsort, stable=True),
                                   (jnp.array([10, 5, -10, 5, 10, -5, 10, -5, -10, 5], dtype=jnp.int32),))
    run_and_compare_specific_input(partial(jnp.argsort, stable=True), (jnp.array(range(-8, 17), dtype=jnp.float32),))
    run_and_compare_specific_input(partial(jnp.argsort, stable=True),
                                   (jnp.array([-2, -1, -0.6, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.6, 1],
                                              dtype=jnp.float32),))


def test_multikey_sort():
    # Test lexicographical sort with multiple keys
    def sort_dim_0(k1, k2):
        return jax.lax.sort([k1, k2], dimension=0, num_keys=2)

    k1 = jnp.array([1, 3, 2, 4], dtype=jnp.int32)
    k2 = jnp.array([3, 1, 2, 4], dtype=jnp.int32)
    run_and_compare_specific_input(sort_dim_0, (k1, k2))

    k1 = jnp.array([1, 5, 1, 4, 3, 4, 4], dtype=jnp.int32)
    k2 = jnp.array([9, 4, 0, 4, 0, 2, 1], dtype=jnp.int32)
    run_and_compare_specific_input(sort_dim_0, (k1, k2))

    k1 = jnp.array([1, 3, 1, 4, 3, 5, 4], dtype=jnp.int32)
    k2 = jnp.array([0, 4, 0, 4, 0, -21, -12], dtype=jnp.int32)
    run_and_compare_specific_input(sort_dim_0, (k1, k2))

    k1_2d = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    k2_2d = jnp.array([[3, 1], [2, 4]], dtype=jnp.int32)

    def sort_dim_1(k1, k2):
        return jax.lax.sort([k1, k2], dimension=1, num_keys=2)

    run_and_compare_specific_input(sort_dim_1, (k1_2d, k2_2d))

    # Larger random inputs
    def sort_dim_0_large(k1, k2):
        return jax.lax.sort([k1, k2], dimension=0, num_keys=2)

    run_and_compare(sort_dim_0_large, (jnp.zeros((100, 50), dtype=jnp.int32), jnp.zeros((100, 50), dtype=jnp.int32)))
    run_and_compare(sort_dim_0_large, (jnp.zeros((100, 50), dtype=jnp.float32), jnp.zeros((100, 50), dtype=jnp.float32)))


def test_unstable_argsort():
    def unstable_argsort(x, **kwargs):
        return jnp.argsort(x, stable=False, **kwargs)

    run_and_compare_specific_input(unstable_argsort, (jnp.array([3, 1, 2], dtype=jnp.int32),))
    run_and_compare_specific_input(unstable_argsort, (jnp.array([[3, 1, 2], [6, 5, 4]], dtype=jnp.float32),))
    run_and_compare_specific_input(partial(unstable_argsort, axis=0), (jnp.array([[3, 1, 2], [6, 5, 4]], dtype=jnp.float32),))

    # Test with larger random input
    run_and_compare(unstable_argsort, (jnp.zeros((100, 50), dtype=jnp.float32),))
    run_and_compare(partial(unstable_argsort, axis=0), (jnp.zeros((100, 50), dtype=jnp.float32),))


def test_multi_input_argsort():
    # Because argsort is unstable, we cannot directly compare the output indices.
    # Instead, we perform argsort followed by gather to retrieve the sorted values,
    # which can then be compared.
    def unstable_argsort_and_lookup(sort_array, lookup_array, lookup_values):
        _sorted_array, ordered_lookup_idx = jax.lax.sort([sort_array, lookup_array], dimension=0, num_keys=1, is_stable=False)
        gathered = jnp.take(lookup_values, ordered_lookup_idx)
        return gathered

    run_and_compare_specific_input(unstable_argsort_and_lookup, (
        jnp.array([3, 1, 2, 3, 1, 2, 3, 1, 2], dtype=jnp.int32),
        jnp.array([2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=jnp.int32),
        jnp.array([0, 1, 2], dtype=jnp.int32)
    ))

    run_and_compare_specific_input(unstable_argsort_and_lookup, (
        jnp.array([3, 1, 2, 3, 1, 2, 3, 1, 2], dtype=jnp.float32),
        jnp.array([2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=jnp.int32),
        jnp.array([0, 1, 2], dtype=jnp.float32)
    ))


def test_sort_int16_casting():
    run_and_compare_specific_input(partial(jnp.argsort, stable=True),
                                   (jnp.array([10, 5, 10, 0x1_0010, 10, 0x5_0000, 10, 5, 10, 0x1_0000, 5], dtype=jnp.int32),))

    def sort_dim_0(k1, k2):
        return jax.lax.sort([k1, k2], dimension=0, num_keys=2)

    def check_size(size):
        key = jax.random.PRNGKey(0)
        subkey1, subkey2 = jax.random.split(key)
        k1 = jax.random.randint(subkey1, (size,), 0, 1_000, dtype=jnp.int32)
        k2 = jax.random.uniform(subkey2, (size,), dtype=jnp.float32)
        run_and_compare_specific_input(sort_dim_0, (k1, k2))

    check_size(0x100)
    check_size(0x1_0010)
    check_size(0x40_0000)


def test_case():
    def switch_fn(index, x):
        return jax.lax.switch(index, [
            lambda x: x + 1,
            lambda x: x * 2,
            lambda x: x - 1
        ], x)

    run_and_compare_specific_input(switch_fn, (
        jnp.array(0, dtype=jnp.int32), jnp.array(10.0, dtype=jnp.float32)
    ))
    run_and_compare_specific_input(switch_fn, (
        jnp.array(1, dtype=jnp.int32), jnp.array(10.0, dtype=jnp.float32)
    ))
    run_and_compare_specific_input(switch_fn, (
        jnp.array(2, dtype=jnp.int32), jnp.array(10.0, dtype=jnp.float32)
    ))


def test_reshape_scalar():
    # Test reshaping to scalar (0-rank tensor)
    def reshape_to_scalar(x):
        return jnp.reshape(x, ())

    run_and_compare(reshape_to_scalar, (jnp.array([5.0], dtype=jnp.float32),))


def test_compare_bool():
    run_and_compare_specific_input(jnp.equal, (
        jnp.array([True, False, True], dtype=jnp.bool_),
        jnp.array([True, True, False], dtype=jnp.bool_)
    ))
    run_and_compare_specific_input(jnp.not_equal, (
        jnp.array([True, False, True], dtype=jnp.bool_),
        jnp.array([True, True, False], dtype=jnp.bool_)
    ))


def test_logical_not():
    run_and_compare(jnp.logical_not, (jnp.array([True, False]),))


def test_power():
    run_and_compare(jnp.power, (jnp.array([2.0, 3.0]), jnp.array([3.0, 2.0])))


def test_dynamic_slice_oob():
    # Test dynamic slice with out of bounds indices
    # StableHLO spec requires that the start indices are clamped to ensure the slice remains within bounds
    # start_index = clamp(start_index, 0, operand_dim - slice_size)
    def dynamic_slice(operand, start_indices):
        return jax.lax.dynamic_slice(operand, start_indices, slice_sizes=(2, 2))

    operand = jnp.zeros((5, 5))
    # Valid index
    run_and_compare_specific_input(dynamic_slice, (operand, jnp.array([1, 1], dtype=jnp.int32)))
    # Out of bounds index (too large) -> should be clamped to 5-2 = 3
    run_and_compare_specific_input(dynamic_slice, (operand, jnp.array([4, 4], dtype=jnp.int32)))
    # Out of bounds index (negative) -> should be clamped to 0
    run_and_compare_specific_input(dynamic_slice, (operand, jnp.array([10, 10], dtype=jnp.int32)))


def test_dynamic_update_slice_oob():
    # Test dynamic update slice with out of bounds indices
    # StableHLO spec requires that the start indices are clamped to ensure the slice remains within bounds
    # start_index = clamp(start_index, 0, operand_dim - update_dim)
    def dynamic_update_slice(operand, update, start_indices):
        return jax.lax.dynamic_update_slice(operand, update, start_indices)

    operand = jnp.zeros((5, 5))
    update = jnp.ones((2, 2))
    # Valid index
    run_and_compare_specific_input(dynamic_update_slice, (operand, update, jnp.array([1, 1], dtype=jnp.int32)))
    # Out of bounds index (too large) -> should be clamped to 5-2 = 3
    run_and_compare_specific_input(dynamic_update_slice, (operand, update, jnp.array([4, 4], dtype=jnp.int32)))
    # Out of bounds index (negative) -> should be clamped to 0
    run_and_compare_specific_input(dynamic_update_slice, (operand, update, jnp.array([10, 10], dtype=jnp.int32)))


def test_transposed_conv_large_padding():
    input_shape = (1, 1, 4, 4)
    kernel_shape = (1, 1, 3, 3)

    def transposed_conv(img, kernel):
        return jax.lax.conv_general_dilated(
            lhs=img,
            rhs=kernel,
            window_strides=(1, 1),
            padding=((3, 3), (3, 3)),
            lhs_dilation=(2, 2),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )

    run_and_compare(transposed_conv, (jnp.zeros(input_shape), jnp.zeros(kernel_shape)))
