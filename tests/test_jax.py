import jax
import jax.export
import jax.numpy as jnp
from jax.experimental import export
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
import numpy as np

from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE
import coremltools as ct
from coremltools.converters.mil.testing_utils import compare_backend
from coremltools.converters.mil.mil import Program, Block

from functools import partial
from typing import List


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
    run_and_compare(contract_all, (jnp.zeros((2, 3, 4, 5)), jnp.zeros((2, 4, 3, 5))))
    run_and_compare(full_tensor_product, (jnp.zeros((2, 3)), jnp.zeros((2, 4, 3))))

    # Test the full tensor product with a big dimensions, and ensure that the program gets handled by a dynamic loop
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


def jax_export(jax_func, input_spec):
    def compute_input_shapes(input_specs):
        shapes = []
        for input_spec in input_specs:
            if isinstance(input_spec, (list, tuple)):
                # We only unwrap the shapes for one level
                shapes.append(input_spec)
            else:
                shapes.append(jax.ShapeDtypeStruct(input_spec.shape, input_spec.dtype))
        return shapes
    input_shapes = compute_input_shapes(input_spec)
    jax_exported = export.export(jax.jit(jax_func))(*input_shapes)
    return jax_exported


def generate_random_from_shape(input_spec, key=jax.random.PRNGKey):
    shape = input_spec.shape
    dtype = input_spec.dtype
    output = jax.random.uniform(key=key, shape=shape, dtype=dtype, minval=-10, maxval=10)
    return output


def flatten(nested_list):
    def visit(lst):
        flat = []
        for element in lst:
            if isinstance(element, (list, tuple)):
                flat += visit(element)
            else:
                flat.append(element)
        return flat
    return visit(nested_list)


def __nest_flat_jax_input_to_input_spec(input_spec, flat_input):
    idx = 0

    def visit(lst):
        nonlocal idx
        result = []
        for element in lst:
            if isinstance(element, (list, tuple)):
                result.append(visit(element))
            else:
                if idx >= len(flat_input):
                    raise ValueError(
                        "flat_input had too many inputs to fit input_spec. "
                        f"Input spec: {input_spec}, Flat input: {flat_input}")
                result.append(flat_input[idx])
                idx += 1
        return result

    structured_input = visit(input_spec)
    if idx != len(flat_input):
        raise ValueError("flat_input had too few inputs to fill input_spec. "
                         f"Input spec: {input_spec}, Flat input: {flat_input}")

    return structured_input


def _count_program_complexity(mil_program: Program):
    """
    Counts the number of instructions in the given `mil_program`
    This is used to ensure we don't generate crazy big programs
    """
    def count_block(block: Block):
        complexity = 0
        for op in block.operations:
            for child_block in op.blocks:
                complexity += count_block(child_block)
            complexity += 1
        return complexity

    total_complexity = 0
    for func in mil_program.functions.values():
        total_complexity += count_block(func)
    return total_complexity


def run_and_compare(jax_func, input_spec, max_complexity: int = 10_000):
    """
    Converts the given `jax_func` to a CoreML model.
    Both models will be run on random input data with shapes specified by `input_spec`.
    If the CoreML model and `jax_func` does not agree on the output, an error will be raised.
    The resulting CoreML model will be returned.
    """

    jax_func = jax.jit(jax_func)
    exported = jax_export(jax_func, input_spec)
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)
    # print(f"HLO module: {hlo_module}")

    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
    program_complexity = _count_program_complexity(mil_program)
    if program_complexity > max_complexity:
        raise ValueError(
            f"Generated a MIL program with complexity {program_complexity}, "
            "max allowed complexity is {max_complexity}"
        )

    pipeline = DEFAULT_HLO_PIPELINE
    # We temporarily avoid fp16 conversions in tests because of https://github.com/apple/coremltools/issues/2324
    passes_to_remove = [
         'common::add_fp16_cast'
    ]
    pipeline.remove_passes(passes_to_remove)

    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pipeline,
    )

    # Generate random inputs that matches cml_model input spec
    cml_input_key_values = {}
    jax_input_values = []
    key = jax.random.PRNGKey(0)
    for input_name, input_shape in zip(cml_model.input_description, exported.in_avals):
        key, value_key = jax.random.split(key, num=2)
        input_value = generate_random_from_shape(input_shape, value_key)
        cml_input_key_values[input_name] = input_value
        jax_input_values.append(input_value)

    # Transfor the input to match the Jax model, and call it
    jax_input_values = __nest_flat_jax_input_to_input_spec(input_spec, jax_input_values)
    expected_output = jax_func(*jax_input_values)

    # TODO(knielsen): Is there a nicer way of doing this?
    if not isinstance(expected_output, (list, tuple)):
        expected_output = (expected_output, )

    # Prepare the output for comparison
    cml_expected_outputs = {}
    for output_name, output_value in zip(cml_model.output_description, flatten(expected_output)):
        cml_expected_outputs[output_name] = np.asarray(output_value)

    compare_backend(cml_model, cml_input_key_values, cml_expected_outputs)

    return cml_model


def get_model_instruction_types(cml_model) -> List[str]:
    def collect_ops(ops: List) -> List[str]:
        collected_ops = []
        for op in ops:
            collected_ops.append(op.op_type)
            for block in op.blocks:
                collected_ops += collect_ops(block.operations)

        return collected_ops

    mil_program = cml_model._mil_program
    all_ops = []
    for func in mil_program.functions.values():
        all_ops += collect_ops(func.operations)
    return all_ops
