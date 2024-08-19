import jax
import jax.export
import jax.numpy as jnp
from jax.experimental import export
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
import numpy as np

from stablehlo_coreml.converter import convert
import coremltools as ct
from coremltools.converters.mil.testing_utils import compare_backend

def test_addition():
    def plus(x, y):
        return jnp.add(x, y)

    run_and_compare(plus, (jnp.float32(1), jnp.float32(1)))
    run_and_compare(plus, (jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2))))

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


def jax_export(jax_func, input_spec):
    def compute_input_shapes(input_specs):
        shapes = []
        for input_spec in input_specs:
            if isinstance(input_spec, (list, tuple)):
                shapes.append(compute_input_shapes(input_spec))
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
                    raise ValueError(f"flat_input had too many inputs to fit input_spec. Input spec: {input_spec}, Flat input: {flat_input}")
                result.append(flat_input[idx])
                idx += 1
        return result

    structured_input = visit(input_spec)
    if idx != len(flat_input):
        raise ValueError("flat_input had too few inputs to fill input_spec. Input spec: {input_spec}, Flat input: {flat_input}")

    return structured_input

def run_and_compare(jax_func, input_spec):
    jax_func = jax.jit(jax_func)
    exported = jax_export(jax_func, input_spec)
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)
    # print(f"HLO module: {hlo_module}")

    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
    cml_model = ct.convert(mil_program, source="milinternal", minimum_deployment_target=ct.target.iOS18)

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
