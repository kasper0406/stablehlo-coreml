import copy
from collections.abc import Mapping

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
from coremltools.converters.mil.mil import Block, Program, Symbol
from coremltools.converters.mil.testing_utils import compare_backend
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax.export import export as _jax_export

from stablehlo_coreml import DEFAULT_HLO_PIPELINE
from stablehlo_coreml.converter import convert
from stablehlo_coreml.state import resolve_state_map


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
    if jnp.issubdtype(dtype, jnp.integer):
        output = jax.random.randint(key=key, shape=shape, minval=-100, maxval=100, dtype=dtype)
    elif jnp.issubdtype(dtype, jnp.bool_):
        output = jax.random.bernoulli(key=key, shape=shape).astype(dtype)
    else:
        output = jax.random.normal(key=key, shape=shape, dtype=dtype)
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


def _convert_mil_to_coreml(
    mil_program,
    *,
    max_complexity: int = 10_000,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    ct_inputs=None,
):
    """Convert a MIL program to a CoreML model with standard test pipeline.

    Checks complexity, removes the fp16 cast pass (see
    https://github.com/apple/coremltools/issues/2324), and runs ct.convert.
    """
    program_complexity = _count_program_complexity(mil_program)
    if program_complexity > max_complexity:
        raise ValueError(
            f"Generated a MIL program with complexity {program_complexity}, "
            f"max allowed complexity is {max_complexity}"
        )

    pipeline = copy.deepcopy(DEFAULT_HLO_PIPELINE)
    pipeline.remove_passes(['common::add_fp16_cast'])

    convert_kwargs = dict(
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pipeline,
        compute_units=compute_units,
    )
    if ct_inputs is not None:
        convert_kwargs["inputs"] = ct_inputs

    return ct.convert(mil_program, **convert_kwargs)


def _as_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _as_numpy(value):
    """Detach JAX arrays into a Core ML-friendly contiguous ndarray."""
    return np.array(value, copy=True)


def _convert_hlo_module(
    hlo_module,
    *,
    states=None,
    max_complexity: int = 10_000,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    ct_inputs=None,
):
    """Convert a StableHLO module to ``(mil_program, cml_model)``.

    ``states`` is forwarded to :func:`convert`. ``ct_inputs`` is
    either a list of ``ct.TensorType``s, or a callable that builds one from the
    converted MIL program.
    """
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states=states)

    if callable(ct_inputs):
        ct_inputs = ct_inputs(mil_program)

    cml_model = _convert_mil_to_coreml(
        mil_program,
        max_complexity=max_complexity,
        compute_units=compute_units,
        ct_inputs=ct_inputs,
    )
    return mil_program, cml_model


def _compare_model_outputs(
    cml_model,
    inputs,
    expected_outputs,
    *,
    atol,
    rtol,
    state=None,
):
    """Predict with ``inputs`` and compare the results to ``expected_outputs``.

    Inputs and expected outputs are matched positionally against the model
    input/output descriptions. Only the outputs covered by ``expected_outputs``
    are compared, so any extra model output (such as ``state_update_token``) is
    ignored.
    """
    cml_input_key_values = {
        input_name: _as_numpy(input_value)
        for input_name, input_value in zip(cml_model.input_description, flatten(inputs))
    }
    cml_expected_outputs = {
        output_name: np.asarray(output_value)
        for output_name, output_value
        in zip(cml_model.output_description, flatten(_as_tuple(expected_outputs)))
    }

    compare_backend(
        cml_model,
        cml_input_key_values,
        cml_expected_outputs,
        atol=atol,
        rtol=rtol,
        state=state,
    )


def run_and_compare_hlo_module(
    hlo_module,
    inputs,
    expected_outputs,
    *,
    max_complexity: int = 10_000,
    atol=1e-04,
    rtol=1e-05,
    compute_units=ct.ComputeUnit.CPU_ONLY,
):
    _, cml_model = _convert_hlo_module(
        hlo_module,
        max_complexity=max_complexity,
        compute_units=compute_units,
    )

    _compare_model_outputs(cml_model, inputs, expected_outputs, atol=atol, rtol=rtol)

    return cml_model


def run_and_compare_specific_input(
    jax_func,
    inputs,
    max_complexity: int = 10_000,
    atol=1e-04,
    rtol=1e-05,
    compute_units=ct.ComputeUnit.CPU_ONLY,
):
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

    return run_and_compare_hlo_module(
        hlo_module,
        inputs,
        expected_output,
        max_complexity=max_complexity,
        atol=atol,
        rtol=rtol,
        compute_units=compute_units,
    )


def run_and_compare(
    jax_func,
    input_specification,
    max_complexity: int = 10_000,
    atol=1e-04,
    rtol=1e-05,
    compute_units=ct.ComputeUnit.CPU_ONLY,
):
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
    return run_and_compare_specific_input(
        jax_func,
        inputs,
        max_complexity=max_complexity,
        atol=atol,
        rtol=rtol,
        compute_units=compute_units,
    )


def get_model_instruction_types(cml_model) -> list[str]:
    def collect_ops(ops: list) -> list[str]:
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


def export_hlo_module(jax_func, inputs):
    """Export ``jax_func`` traced at ``inputs`` to a parsed StableHLO module."""
    jax_func = jax.jit(jax_func)
    exported = jax_export(jax_func, inputs)
    context = jax_mlir.make_ir_context()
    return ir.Module.parse(exported.mlir_module(), context=context)


def run_and_compare_stateful(
    jax_func,
    initial_inputs,
    states,
    subsequent_inputs=(),
    *,
    max_complexity: int = 10_000,
    atol=1e-04,
    rtol=1e-05,
    compute_units=ct.ComputeUnit.CPU_ONLY,
):
    """Convert ``jax_func`` with ``states`` and compare multi-step predictions.

    ``states`` is either a single-function state mapping or a function-scoped
    mapping. ``subsequent_inputs`` are tuples of the remaining (non-state)
    positional arguments for each subsequent call.
    """
    jax_func = jax.jit(jax_func)
    hlo_module = export_hlo_module(jax_func, initial_inputs)
    mil_program, cml_model = _convert_hlo_module(
        hlo_module,
        states=states,
        max_complexity=max_complexity,
        compute_units=compute_units,
    )

    mil_func_name = mil_program.default_function_name
    function_states = states
    if states and all(isinstance(value, Mapping) for value in states.values()):
        function_states = states[mil_func_name]

    # Map argument index -> index of the output holding the updated state
    hlo_func = next(func for func in hlo_module.body if func.name.value == mil_func_name)
    resolved_states = {}
    for in_idx, state_spec in resolve_state_map(hlo_func, function_states).items():
        if state_spec.output is None:
            raise ValueError("run_and_compare_stateful requires updated, not read-only, states")
        resolved_states[in_idx] = state_spec.output

    nonstate_indices = [i for i in range(len(initial_inputs)) if i not in resolved_states]
    state_indices = [i for i in range(len(initial_inputs)) if i in resolved_states]
    state_outputs = set(resolved_states.values())
    state_names = [feature.name for feature in cml_model._spec.description.state]

    cml_state = cml_model.make_state()
    for name, idx in zip(state_names, state_indices):
        cml_state.write_state(name=name, value=_as_numpy(initial_inputs[idx]))

    def run_step(args):
        expected = _as_tuple(jax_func(*args))
        # Outputs that update state are not exposed by the Core ML model
        expected_tensor_outputs = [
            value for i, value in enumerate(expected) if i not in state_outputs
        ]
        _compare_model_outputs(
            cml_model,
            [args[idx] for idx in nonstate_indices],
            expected_tensor_outputs,
            atol=atol,
            rtol=rtol,
            state=cml_state,
        )

        for name, in_idx in zip(state_names, state_indices):
            np.testing.assert_allclose(
                np.asarray(cml_state.read_state(name=name)),
                np.asarray(expected[resolved_states[in_idx]]),
                atol=atol,
                rtol=rtol,
            )
        return expected

    outputs = run_step(initial_inputs)
    for next_nonstate_inputs in subsequent_inputs:
        next_args = [None] * len(initial_inputs)
        nonstate_iter = iter(next_nonstate_inputs)
        for i in range(len(initial_inputs)):
            if i in resolved_states:
                next_args[i] = outputs[resolved_states[i]]
            else:
                next_args[i] = next(nonstate_iter)
        outputs = run_step(next_args)

    return cml_model


def run_and_compare_symbolic(
    jax_func,
    symbolic_input_specs,
    test_shapes,
    *,
    max_complexity: int = 10_000,
    atol=1e-04,
    rtol=1e-05,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    range_dim_max=2048,  # upper bound for RangeDim on symbolic axes
):
    """
    Export ``jax_func`` with symbolic (dynamic) shapes, convert to CoreML,
    and validate at multiple concrete shapes against JAX reference outputs.

    Parameters
    ----------
    jax_func : callable
        The JAX function to export.
    symbolic_input_specs : list of jax.ShapeDtypeStruct
        Input specs with symbolic dimensions (from ``jax.export.symbolic_shape``).
    test_shapes : list of tuples
        Each entry is a tuple of concrete input arrays to test with.
    range_dim_max : int
        Upper bound for RangeDim on symbolic dimensions.
    """
    jax_func = jax.jit(jax_func)
    exported = _jax_export(jax_func)(*symbolic_input_specs)
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)

    def build_ct_inputs(mil_program):
        # Build ct.TensorType inputs with RangeDim for symbolic dimensions
        ct_inputs = []
        for func in mil_program.functions.values():
            for inp_name, inp in func.inputs.items():
                ct_shape = []
                for dim in inp.shape:
                    if isinstance(dim, Symbol):
                        ct_shape.append(ct.RangeDim(1, range_dim_max, default=1))
                    else:
                        ct_shape.append(int(dim))
                ct_inputs.append(ct.TensorType(name=inp_name, shape=ct_shape))
            break  # only first (main) function
        return ct_inputs

    _, cml_model = _convert_hlo_module(
        hlo_module,
        max_complexity=max_complexity,
        compute_units=compute_units,
        ct_inputs=build_ct_inputs,
    )

    for concrete_inputs in test_shapes:
        concrete_inputs = _as_tuple(concrete_inputs)
        _compare_model_outputs(
            cml_model,
            concrete_inputs,
            jax_func(*concrete_inputs),
            atol=atol,
            rtol=rtol,
        )

    return cml_model
