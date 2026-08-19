import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import types
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax.export import export, symbolic_shape

from stablehlo_coreml import StateSpec
from stablehlo_coreml.converter import convert
from tests.utils import (
    export_hlo_module,
    model_for_function,
    model_io_names,
    run_and_compare_stateful,
)


def _instruction_types(mil_program) -> list[str]:
    return [
        op.op_type
        for func in mil_program.functions.values()
        for op in func.operations
    ]


def _accumulator(cache, x):
    new_cache = cache + x
    return new_cache * new_cache, new_cache


def test_accumulator_persists_across_calls():
    initial_state = jnp.zeros((3,), dtype=jnp.float32)
    run_and_compare_stateful(
        _accumulator,
        initial_inputs=(initial_state, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)),
        states={"main": {"cache": StateSpec(output=1)}},
        subsequent_inputs=[
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
        subsequent_inputs=[
            (jnp.full((2, 3), 0.5, dtype=jnp.float32),),
        ],
    )


def test_state_only_output():
    def add_into(cache, x):
        return cache + x

    inputs = (jnp.zeros((4,), dtype=jnp.float32), jnp.arange(4, dtype=jnp.float32))
    hlo_module = export_hlo_module(add_into, inputs)
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 0})
    mil_func = next(iter(mil_program.functions.values()))
    assert len(mil_func.outputs) == 1
    assert mil_func.outputs[0].name == "state_update_token"
    assert mil_func.outputs[0].shape == (1,)
    assert mil_func.outputs[0].dtype == types.fp16

    run_and_compare_stateful(
        add_into,
        initial_inputs=inputs,
        states={0: 0},
        subsequent_inputs=[
            (jnp.ones((4,), dtype=jnp.float32),),
        ],
    )


def test_state_by_argument_name():
    def step(cache, x):
        new_cache = cache + x
        return x * new_cache, new_cache

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(step, inputs)
    mil_by_index = convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states={0: 1})
    mil_by_name = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"cache": StateSpec(output=1)}},
    )
    assert _instruction_types(mil_by_index) == _instruction_types(mil_by_name)
    assert list(mil_by_name.functions["main"].inputs) == ["cache", "x"]


def test_read_only_state():
    def inspect(cache, x):
        return cache + x

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(inspect, inputs)
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"cache": StateSpec(output=None)}},
    )

    op_types = _instruction_types(mil_program)
    assert "read_state" in op_types
    assert "coreml_update_state" not in op_types
    assert len(mil_program.functions["main"].outputs) == 1


def test_state_name_override():
    def step(cache, x):
        new_cache = cache + x
        return new_cache, new_cache

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(step, inputs)
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"cache": StateSpec(output=1, name="accumulator")}},
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
    def step(carry, x):
        new_carry = carry + x
        return {"prediction": new_carry * x, "cache": new_carry}

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    named_program = convert(
        export_hlo_module(step, inputs),
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"carry": StateSpec(output=output)}},
    )
    indexed_program = convert(
        export_hlo_module(step, inputs),
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"carry": StateSpec(output=0)}},
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
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    descriptions = {desc.name: desc for desc in cml_model.get_spec().description.functions}
    assert set(descriptions) == {"update", "inspect"}
    assert [state.name for state in descriptions["update"].state] == ["cache"]
    assert [state.name for state in descriptions["inspect"].state] == ["cache"]

    # Both functions must actually load and run with their own state
    update_model = model_for_function(cml_model, "update")
    update_inputs, update_outputs, update_states = model_io_names(update_model)
    assert (update_inputs, update_states) == (["arg1"], ["cache"])
    update_state = update_model.make_state()
    update_state.write_state(name="cache", value=np.array([1.0, 2.0], dtype=np.float32))
    update_out = update_model.predict(
        {"arg1": np.array([3.0, 4.0], dtype=np.float32)}, state=update_state
    )
    np.testing.assert_allclose(update_out[update_outputs[0]], [4.0, 6.0], atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(update_state.read_state(name="cache")), [4.0, 6.0], atol=1e-3
    )

    inspect_model = model_for_function(cml_model, "inspect")
    inspect_inputs, inspect_outputs, inspect_states = model_io_names(inspect_model)
    assert (inspect_inputs, inspect_states) == (["arg0"], ["cache"])
    inspect_state = inspect_model.make_state()
    inspect_state.write_state(name="cache", value=np.array([1.0, 2.0], dtype=np.float32))
    inspect_out = inspect_model.predict(
        {"arg0": np.array([3.0, 4.0], dtype=np.float32)}, state=inspect_state
    )
    np.testing.assert_allclose(inspect_out[inspect_outputs[0]], [4.0, 6.0], atol=1e-3)
    # The read-only state is unchanged
    np.testing.assert_allclose(
        np.asarray(inspect_state.read_state(name="cache")), [1.0, 2.0], atol=1e-3
    )


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


def _symbolic_step_module():
    def step(x, cache):
        return x + cache, cache + 1.0

    (n,) = symbolic_shape("n")
    exported = export(jax.jit(step))(
        jax.ShapeDtypeStruct((n, 2), jnp.float32),
        jax.ShapeDtypeStruct((n, 2), jnp.float32),
    )
    context = jax_mlir.make_ir_context()
    return ir.Module.parse(exported.mlir_module(), context=context)


def test_symbolic_state_is_rejected():
    with pytest.raises(ValueError, match="static shape"):
        convert(
            _symbolic_step_module(),
            minimum_deployment_target=ct.target.iOS18,
            states={1: 1},
        )


def test_symbolic_conversion_works_after_rejected_symbolic_state():
    # Rejecting a state must not leave MIL Symbols behind for the earlier,
    # non-state arguments; they are registered process wide and a later
    # conversion would fail with "Symbol `dim_0` is used already".
    with pytest.raises(ValueError, match="static shape"):
        convert(
            _symbolic_step_module(),
            minimum_deployment_target=ct.target.iOS18,
            states={1: 1},
        )

    mil_program = convert(_symbolic_step_module(), minimum_deployment_target=ct.target.iOS18)
    assert len(mil_program.functions["main"].inputs) == 2


def test_constant_state_update_loads_and_predicts():
    # A state update that const-folds (here: resetting a cache to zeros) used
    # to produce a model that segfaults Core ML at load time.
    def reset(cache, x):
        return jnp.zeros_like(cache), x + cache

    inputs = (jnp.zeros((4,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32))
    mil_program = convert(
        export_hlo_module(reset, inputs),
        minimum_deployment_target=ct.target.iOS18,
        states={0: 0},
    )
    update_value = next(
        op for op in mil_program.functions["main"].operations
        if op.op_type == "coreml_update_state"
    ).value
    assert update_value.val is None, "the state update value must not be a constant"

    run_and_compare_stateful(
        reset,
        initial_inputs=inputs,
        states={0: 0},
        subsequent_inputs=[
            (jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),),
        ],
    )


def test_nonzero_constant_state_update_loads_and_predicts():
    def reset(cache, x):
        return jnp.full_like(cache, 2.5), x * cache

    inputs = (jnp.ones((3,), dtype=jnp.float32), jnp.full((3,), 4.0, dtype=jnp.float32))
    run_and_compare_stateful(
        reset,
        initial_inputs=inputs,
        states={0: 0},
        subsequent_inputs=[
            (jnp.full((3,), 3.0, dtype=jnp.float32),),
        ],
    )


def test_reserved_explicit_state_name_is_rejected():
    def step(cache, x):
        new_cache = cache + x
        return new_cache, new_cache

    hlo_module = export_hlo_module(
        step,
        (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32)),
    )
    with pytest.raises(ValueError, match="state_workaround"):
        convert(
            hlo_module,
            minimum_deployment_target=ct.target.iOS18,
            states={"main": {"cache": StateSpec(output=1, name="state")}},
        )


def test_pytree_state_name_is_sanitized():
    def step(params, x):
        return x * params["w"], params["w"] + x

    spec = (
        {"w": jax.ShapeDtypeStruct((2,), jnp.float32)},
        jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    exported = export(jax.jit(step))(*spec)
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)

    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"main": {"params['w']": StateSpec(output=1)}},
    )
    assert list(mil_program.functions["main"].inputs) == ["params__w__", "x"]

    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    _, _, state_names = model_io_names(cml_model)
    assert state_names == ["params__w__"]

    cml_state = cml_model.make_state()
    cml_state.write_state(name="params__w__", value=np.array([1.0, 2.0], dtype=np.float32))
    cml_model.predict({"x": np.array([3.0, 4.0], dtype=np.float32)}, state=cml_state)
    np.testing.assert_allclose(
        np.asarray(cml_state.read_state(name="params__w__")), [4.0, 6.0], atol=1e-3
    )


def test_single_public_function_is_exported_as_main():
    ctx = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(
        """
        module {
          func.func public @forward(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """,
        context=ctx,
    )
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)

    assert list(mil_program.functions) == ["main"]
    assert mil_program.default_function_name == "main"
    assert not mil_program.export_as_multifunction

    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    input_names, output_names, _ = model_io_names(cml_model)
    assert input_names == ["arg0", "arg1"]
    outputs = cml_model.predict({
        "arg0": np.array([1.0, 2.0], dtype=np.float32),
        "arg1": np.array([3.0, 4.0], dtype=np.float32),
    })
    np.testing.assert_allclose(outputs[output_names[0]], [4.0, 6.0], atol=1e-3)


def test_single_public_function_with_state_is_exported_as_main():
    ctx = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(
        """
        module {
          func.func public @forward(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
            return %0, %0 : tensor<2xf32>, tensor<2xf32>
          }
        }
        """,
        context=ctx,
    )
    # The state mapping stays keyed by the HLO function name
    mil_program = convert(
        hlo_module,
        minimum_deployment_target=ct.target.iOS18,
        states={"forward": {"arg0": StateSpec(output=1, name="cache")}},
    )

    assert list(mil_program.functions) == ["main"]
    cml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    input_names, output_names, state_names = model_io_names(cml_model)
    assert input_names == ["arg1"]
    assert state_names == ["cache"]

    cml_state = cml_model.make_state()
    cml_state.write_state(name="cache", value=np.array([1.0, 2.0], dtype=np.float32))
    outputs = cml_model.predict(
        {"arg1": np.array([3.0, 4.0], dtype=np.float32)}, state=cml_state
    )
    np.testing.assert_allclose(outputs[output_names[0]], [4.0, 6.0], atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(cml_state.read_state(name="cache")), [4.0, 6.0], atol=1e-3
    )
