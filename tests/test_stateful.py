import coremltools as ct
import jax
import jax.numpy as jnp
import pytest
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax.export import export, symbolic_shape

from stablehlo_coreml.converter import convert
from tests.utils import export_hlo_module, run_and_compare_stateful


def _instruction_types(mil_program) -> list[str]:
    return [
        op.op_type
        for func in mil_program.functions.values()
        for op in func.operations
    ]


def _accumulator(state, x):
    new_state = state + x
    return new_state * new_state, new_state


def test_accumulator_persists_across_calls():
    initial_state = jnp.zeros((3,), dtype=jnp.float32)
    run_and_compare_stateful(
        _accumulator,
        initial_inputs=(initial_state, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)),
        states={0: 1},
        extra_nonstate_steps=[
            (jnp.array([0.5, -1.0, 2.0], dtype=jnp.float32),),
            (jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),),
        ],
    )


def test_state_inputs_are_omitted_from_model_io():
    inputs = (jnp.zeros((2, 2), dtype=jnp.float32), jnp.ones((2, 2), dtype=jnp.float32))
    hlo_module = export_hlo_module(_accumulator, inputs)
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1})

    mil_func = next(iter(mil_program.functions.values()))
    assert len(mil_func.inputs) == 2
    assert len(mil_func.outputs) == 1
    op_types = _instruction_types(mil_program)
    assert "read_state" in op_types
    assert "coreml_update_state" in op_types

    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
    )
    assert len(list(cml_model.input_description)) == 1
    assert len(list(cml_model.output_description)) == 1


def test_multiple_states():
    def step(k, v, x):
        new_k = k + x
        new_v = v * 2.0 + x
        return new_k + new_v, new_k, new_v

    zeros = jnp.zeros((2, 3), dtype=jnp.float32)
    x0 = jnp.ones((2, 3), dtype=jnp.float32)
    run_and_compare_stateful(
        step,
        initial_inputs=(zeros, zeros + 1.0, x0),
        states={0: 1, 1: 2},
        extra_nonstate_steps=[
            (jnp.full((2, 3), 0.5, dtype=jnp.float32),),
        ],
    )


def test_state_only_output():
    def add_into(state, x):
        return state + x

    inputs = (jnp.zeros((4,), dtype=jnp.float32), jnp.arange(4, dtype=jnp.float32))
    hlo_module = export_hlo_module(add_into, inputs)
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 0})
    mil_func = next(iter(mil_program.functions.values()))
    assert len(mil_func.outputs) == 1
    assert mil_func.outputs[0].dtype == types.fp32

    run_and_compare_stateful(
        add_into,
        initial_inputs=inputs,
        states={0: 0},
        extra_nonstate_steps=[
            (jnp.ones((4,), dtype=jnp.float32),),
        ],
    )


def test_state_by_argument_name():
    def step(state, x):
        new_state = state + x
        return x * new_state, new_state

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(step, inputs)
    mil_by_index = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1})
    # `%arg0` is also accepted without the leading `%`.
    mil_by_name = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={"arg0": 1})
    assert _instruction_types(mil_by_index) == _instruction_types(mil_by_name)


def test_unknown_state_name():
    hlo_module = export_hlo_module(
        _accumulator,
        (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32)),
    )
    with pytest.raises(ValueError, match="Unknown state input"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={"not_an_arg": 1})


def test_invalid_state_index():
    hlo_module = export_hlo_module(
        _accumulator,
        (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32)),
    )
    with pytest.raises(ValueError, match="out of range"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={4: 0})


def test_invalid_state_output_index():
    hlo_module = export_hlo_module(
        _accumulator,
        (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32)),
    )
    with pytest.raises(ValueError, match="out of range"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 5})


def test_duplicate_state_output():
    def step(k, v, x):
        return x, k + x, v + x

    hlo_module = export_hlo_module(
        step,
        (
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.ones((2,), dtype=jnp.float32),
        ),
    )
    with pytest.raises(ValueError, match="cannot update both"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1, 1: 1})


def test_state_type_mismatch():
    def step(state, x):
        return x, state[:1] + x[:1]

    hlo_module = export_hlo_module(
        step,
        (jnp.zeros((4,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32)),
    )
    with pytest.raises(ValueError, match="has type"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1})


def test_symbolic_state_is_rejected():
    def step(state, x):
        return x + state, state + 1.0

    (n,) = symbolic_shape("n")
    exported = export(jax.jit(step))(
        jax.ShapeDtypeStruct((n, 2), jnp.float32),
        jax.ShapeDtypeStruct((n, 2), jnp.float32),
    )
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)
    with pytest.raises(ValueError, match="static shape"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1})


def test_multi_function_module_removes_unmodified_state_updates():
    ctx = jax_mlir.make_ir_context()
    mlir_text = """
    module {
      func.func public @update(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        return %arg1, %0 : tensor<2xf32>, tensor<2xf32>
      }
      func.func public @no_update(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        return %0, %arg0 : tensor<2xf32>, tensor<2xf32>
      }
      func.func public @read_only(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        return %0, %arg0 : tensor<2xf32>, tensor<2xf32>
      }
    }
    """
    hlo_module = ir.Module.parse(mlir_text, context=ctx)
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1})

    PASS_REGISTRY["common::remove_noop_state_update"](mil_program)

    update_ops = [op.op_type for op in mil_program.functions["update"].operations]

    assert "coreml_update_state" in update_ops
    assert "read_state" in update_ops

    no_update_ops = [op.op_type for op in mil_program.functions["no_update"].operations]
    assert "coreml_update_state" not in no_update_ops
    assert "read_state" not in no_update_ops

    read_only_ops = [op.op_type for op in mil_program.functions["read_only"].operations]
    assert "coreml_update_state" not in read_only_ops
    assert "read_state" in read_only_ops
