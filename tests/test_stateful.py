import coremltools as ct
import jax
import jax.numpy as jnp
import pytest
from coremltools.converters.mil.mil import types
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax.export import export, symbolic_shape

from stablehlo_coreml import StateSpec
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
        states={"main": {"state": StateSpec(output=1)}},
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
    mil_by_name = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"state": StateSpec(output=1)}},
    )
    assert _instruction_types(mil_by_index) == _instruction_types(mil_by_name)
    assert list(mil_by_name.functions["main"].inputs) == ["state", "x"]


def test_read_only_state():
    def inspect(state, x):
        return state + x

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(inspect, inputs)
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"state": StateSpec(output=None)}},
    )

    op_types = _instruction_types(mil_program)
    assert "read_state" in op_types
    assert "coreml_update_state" not in op_types
    assert len(mil_program.functions["main"].outputs) == 1


def test_state_name_override():
    def step(state, x):
        new_state = state + x
        return new_state, new_state

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(step, inputs)
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"state": StateSpec(output=1, name="accumulator")}},
    )

    assert list(mil_program.functions["main"].inputs) == ["accumulator", "x"]


def test_state_name_override_takes_precedence_over_jax_argument_name():
    def inspect(cache, state):
        return cache + state

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(inspect, inputs)
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"state": StateSpec(output=None, name="cache")}},
    )

    assert list(mil_program.functions["main"].inputs) == ["arg0", "cache"]


@pytest.mark.parametrize("output", ["cache", "result['cache']"])
def test_state_output_by_jax_result_name(output):
    def step(state, x):
        new_state = state + x
        return {"prediction": new_state * x, "cache": new_state}

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    named_program = convert(
        export_hlo_module(step, inputs),
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"state": StateSpec(output=output)}},
    )
    indexed_program = convert(
        export_hlo_module(step, inputs),
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"state": StateSpec(output=0)}},
    )

    assert _instruction_types(named_program) == _instruction_types(indexed_program)
    assert len(named_program.functions["main"].outputs) == 1


def test_function_scoped_states_export_as_multifunction():
    ctx = jax_mlir.make_ir_context()
    mlir_text = """
    module {
      func.func public @update(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        return %0, %0 : tensor<2xf32>, tensor<2xf32>
      }
      func.func public @inspect(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    }
    """
    hlo_module = ir.Module.parse(mlir_text, context=ctx)
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={
            "update": {"arg0": StateSpec(output=1, name="cache")},
            "inspect": {"arg1": StateSpec(output=None, name="cache")},
        },
    )

    assert mil_program.export_as_multifunction
    assert mil_program.default_function_name == "update"
    assert "coreml_update_state" in [
        op.op_type for op in mil_program.functions["update"].operations
    ]
    assert "coreml_update_state" not in [
        op.op_type for op in mil_program.functions["inspect"].operations
    ]

    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )
    descriptions = {desc.name: desc for desc in cml_model.get_spec().description.functions}
    assert set(descriptions) == {"update", "inspect"}
    assert [state.name for state in descriptions["update"].state] == ["cache"]
    assert [state.name for state in descriptions["inspect"].state] == ["cache"]


def test_flat_state_mapping_rejected_for_multiple_functions():
    ctx = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(
        """
        module {
          func.func public @first(%arg0: tensor<2xf32>) -> tensor<2xf32> {
            return %arg0 : tensor<2xf32>
          }
          func.func public @second(%arg0: tensor<2xf32>) -> tensor<2xf32> {
            return %arg0 : tensor<2xf32>
          }
        }
        """,
        context=ctx,
    )

    with pytest.raises(ValueError, match="flat state mapping"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 0})


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
