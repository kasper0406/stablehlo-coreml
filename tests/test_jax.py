import jax
import jax.numpy as jnp
from functools import partial

from tests.utils import run_and_compare, run_and_compare_specific_input, get_model_instruction_types


def test_addition():
    run_and_compare(jnp.add, (jnp.float32(1), jnp.float32(1)))
    run_and_compare(jnp.add, (jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2))))


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

    # Currently the `contract_all` test is failing, due to a runtime error in CoreML
    # crashing Python entirely. Reported to Apple in https://feedbackassistant.apple.com/feedback/15643467
    # run_and_compare(contract_all, (jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 4, 3, 5))))

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
    run_and_compare(partial(jnp.argmin, axis=1), (jnp.zeros((20, 30, 40)),))
    run_and_compare(partial(jnp.argmin, axis=1, keepdims=True), (jnp.zeros((20, 30, 40)),))


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


# Unfortunately this test currently fails due to https://github.com/llvm/llvm-project/pull/113064
# def test_take():
#     run_and_compare_specific_input(jnp.take, (jnp.reshape(jnp.arange(24), (4, 6)), jnp.array([
#         [[0, 0], [1, 1], [2, 2]]
#     ], dtype=jnp.int32)))


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
