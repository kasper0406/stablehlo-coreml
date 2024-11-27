import jax
from jax.export import export as _jax_export
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir

import numpy as np

from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

import coremltools as ct
from coremltools.converters.mil.testing_utils import compare_backend
from coremltools.converters.mil.mil import Program, Block

from typing import List


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
    jax_exported = _jax_export(jax.jit(jax_func))(*input_shapes)
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


def run_and_compare_hlo_module(hlo_module, inputs, expected_outputs, max_complexity: int = 10_000):
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
    program_complexity = _count_program_complexity(mil_program)
    if program_complexity > max_complexity:
        raise ValueError(
            f"Generated a MIL program with complexity {program_complexity}, "
            f"max allowed complexity is {max_complexity}"
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
    for input_name, input_value in zip(cml_model.input_description, flatten(inputs)):
        cml_input_key_values[input_name] = input_value

    # TODO(knielsen): Is there a nicer way of doing this?
    if not isinstance(expected_outputs, (list, tuple)):
        expected_outputs = (expected_outputs, )

    # Prepare the output for comparison
    cml_expected_outputs = {}
    for output_name, output_value in zip(cml_model.output_description, flatten(expected_outputs)):
        cml_expected_outputs[output_name] = np.asarray(output_value)

    compare_backend(cml_model, cml_input_key_values, cml_expected_outputs)

    return cml_model


def run_and_compare_specific_input(jax_func, inputs, max_complexity: int = 10_000):
    """
    Converts the given `jax_func` to a CoreML model.
    If the CoreML model and `jax_func` does not agree on the output, an error will be raised.
    The resulting CoreML model will be returned.
    """

    jax_func = jax.jit(jax_func)
    exported = jax_export(jax_func, inputs)
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)
    # print(f"HLO module: {hlo_module}")

    # Transfor the input to match the Jax model, and call it
    jax_input_values = __nest_flat_jax_input_to_input_spec(inputs, flatten(inputs))
    expected_output = jax_func(*jax_input_values)

    return run_and_compare_hlo_module(hlo_module, inputs, expected_output, max_complexity=max_complexity)


def run_and_compare(jax_func, input_specification, max_complexity: int = 10_000):
    """
    Converts the given `jax_func` to a CoreML model.
    The model will be tested with randomly generated data with the shapes of `input_specification`.
    If the CoreML model and `jax_func` does not agree on the output, an error will be raised.
    The resulting CoreML model will be returned.
    """
    flat_inputs = []
    key = jax.random.PRNGKey(0)
    for input_spec in flatten(input_specification):
        key, value_key = jax.random.split(key, num=2)
        input_value = generate_random_from_shape(input_spec, value_key)
        flat_inputs.append(input_value)

    inputs = __nest_flat_jax_input_to_input_spec(input_specification, flat_inputs)
    return run_and_compare_specific_input(jax_func, inputs, max_complexity=max_complexity)


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
