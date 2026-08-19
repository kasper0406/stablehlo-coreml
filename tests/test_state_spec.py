"""Unit tests for `stablehlo_coreml.state` helpers.

These cover the cases that only show up for modules that were *not* produced by
`jax.export` (hand-written MLIR, torchax exports, symbolic-shape exports), which
carry different argument locations and no `jax.result_info` attributes.
"""

import coremltools as ct
import jax.numpy as jnp
import pytest
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

from stablehlo_coreml import StateSpec
from stablehlo_coreml.converter import convert
from stablehlo_coreml.state import (
    _NAME_LOCATION_RE,
    preferred_argument_name,
    resolve_state_map,
)
from tests.utils import export_hlo_module

# Argument locations here mimic what `ir.Module.parse` and non-JAX frontends
# produce: file/line/column locations rather than the `NameLoc`s that
# `jax.export` emits.
_PARSED_MODULE = """
module {
  func.func public @main(%arg0: tensor<2xf32> loc("-":1:33), %arg1: tensor<2xf32>)
      -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0, %0 : tensor<2xf32>, tensor<2xf32>
  }
}
"""


def _main(hlo_module):
    """The ``@main`` function of ``hlo_module``.

    The caller must keep ``hlo_module`` alive for as long as the returned
    ``FuncOp`` is used, otherwise the underlying MLIR storage is freed.
    """
    return next(func for func in hlo_module.body if func.name.value == "main")


@pytest.fixture
def parse():
    """Parse MLIR text into a module, keeping every parsed module alive."""
    modules = []

    def parse_module(mlir_text: str):
        module = ir.Module.parse(mlir_text, context=jax_mlir.make_ir_context())
        modules.append(module)
        return module

    return parse_module


def _instruction_types(mil_program) -> list[str]:
    return [
        op.op_type
        for func in mil_program.functions.values()
        for op in func.operations
    ]


def _stateful_module(states):
    def step(state, x):
        new_state = state + x
        return new_state * x, new_state

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(step, inputs)
    return convert(hlo_module, minimum_deployment_target=ct.target.iOS18, states=states)


def test_non_name_locations_fall_back_to_ssa_names(parse):
    """Reading `name_str` off a non-`NameLoc` segfaults on jaxlib 0.9.x."""
    hlo_func = _main(parse(_PARSED_MODULE))
    assert [preferred_argument_name(arg) for arg in hlo_func.arguments] == ["arg0", "arg1"]


@pytest.mark.parametrize(
    "location",
    ['loc("-":1:33)', 'loc(fused["a", "b"])', "loc(unknown)", "loc(callsite(\"a\" at \"b\"))"],
)
def test_argument_names_for_every_location_kind(parse, location):
    hlo_func = _main(parse(f"""
    module {{
      func.func public @main(%arg0: tensor<2xf32> {location}) -> tensor<2xf32> {{
        return %arg0 : tensor<2xf32>
      }}
    }}
    """))
    assert preferred_argument_name(hlo_func.arguments[0]) == "arg0"


def test_jax_exported_arguments_keep_their_python_names():
    def step(cache, x):
        return cache + x

    inputs = (jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32))
    hlo_module = export_hlo_module(step, inputs)
    hlo_func = _main(hlo_module)
    assert [preferred_argument_name(arg) for arg in hlo_func.arguments] == ["cache", "x"]


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ('loc("cache")', "cache"),
        ('loc("cache"(unknown))', "cache"),
        ('loc("-":1:33)', None),
        ('loc(fused["a", "b"])', None),
        ("loc(unknown)", None),
    ],
)
def test_name_location_regex(location, expected):
    """The textual fallback used when no location-kind predicate is available."""
    match = _NAME_LOCATION_RE.match(location)
    assert (match.group(1) if match else None) == expected


def test_named_output_on_module_without_result_attrs_raises_value_error(parse):
    """`FuncOp.result_attrs` raises `KeyError` when there are no `res_attrs`."""
    hlo_func = _main(parse(_PARSED_MODULE))
    with pytest.raises(ValueError, match="Unknown state output 'cache'"):
        resolve_state_map(hlo_func, {0: StateSpec(output="cache")})


def test_none_is_accepted_as_a_read_only_state_spec(parse):
    hlo_func = _main(parse(_PARSED_MODULE))
    assert resolve_state_map(hlo_func, {0: None}) == resolve_state_map(
        hlo_func, {0: StateSpec(output=None)}
    )
    assert resolve_state_map(hlo_func, {0: None}) == {0: StateSpec(output=None)}


def test_none_state_converts_to_a_read_only_state():
    from_none = _stateful_module({"main": {"state": None}})
    from_spec = _stateful_module({"main": {"state": StateSpec(output=None)}})

    op_types = _instruction_types(from_none)
    assert "read_state" in op_types
    assert "coreml_update_state" not in op_types
    assert op_types == _instruction_types(from_spec)
