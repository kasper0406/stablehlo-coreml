from dataclasses import dataclass

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Placeholder, Symbol, Var, types
from jaxlib.mlir.dialects.func import FuncOp

from .state import StateSpec, preferred_argument_name
from .translation_context import DYNAMIC_DIM_SENTINEL, TranslationContext
from .utils import dtype_str, get_mil_type_from_ir

# Core ML can only serialize state tensors as fp16. fp32 HLO state is
# stored as fp16 and cast to/from the computation dtype around read/write.
_STATE_VALUE_DTYPES = frozenset({types.fp16, types.fp32})


@dataclass
class _FunctionInterface:
    inputs: dict[str, Placeholder]
    argument_names: dict[int, str]
    state_specs: dict[str, StateSpec]
    state_compute_dtypes: dict[str, object]

    @classmethod
    def from_hlo(
        cls,
        hlo_func: FuncOp,
        state_map: dict[int, StateSpec],
    ) -> "_FunctionInterface":
        inputs = {}
        state_specs: dict[str, StateSpec] = {}
        state_compute_dtypes: dict[str, object] = {}
        argument_names: dict[int, str] = {}
        used_input_names: set[str] = set()
        explicit_state_names = {
            spec.name: in_idx
            for in_idx, spec in state_map.items()
            if spec.name is not None
        }
        sym_counter = 0

        for in_idx, arg in enumerate(hlo_func.arguments):
            shape = arg.type.shape
            context_name = arg.get_name()
            state_spec = state_map.get(in_idx)
            explicit_name = state_spec.name if state_spec is not None else None
            name = explicit_name or preferred_argument_name(arg)
            if explicit_name is not None:
                if name in used_input_names:
                    raise ValueError(
                        f"Core ML state name {name!r} conflicts with another function input"
                    )
            else:
                if name in used_input_names or (
                    name in explicit_state_names and explicit_state_names[name] != in_idx
                ):
                    name = context_name.lstrip("%")
                while name in used_input_names or (
                    name in explicit_state_names and explicit_state_names[name] != in_idx
                ):
                    name = f"{name}_input"
            used_input_names.add(name)
            argument_names[in_idx] = name

            # Reject dynamic state before constructing MIL Symbols — Symbol
            # names are process-global and would leak if we raise afterwards.
            if state_spec is not None and any(d == DYNAMIC_DIM_SENTINEL for d in shape):
                raise ValueError(f"State input {name} must have a static shape, got {shape}")
            if shape == []:
                shape = [1]
            else:
                new_shape = []
                for dim in shape:
                    if dim == DYNAMIC_DIM_SENTINEL:
                        new_shape.append(Symbol(f"dim_{sym_counter}"))
                        sym_counter += 1
                    else:
                        new_shape.append(dim)
                shape = new_shape

            dtype = get_mil_type_from_ir(arg.type.element_type)
            if state_spec is None:
                inputs[name] = mb.placeholder(shape=shape, dtype=dtype)
                continue
            if dtype not in _STATE_VALUE_DTYPES:
                raise ValueError(
                    f"State input {name} has dtype {dtype_str(dtype)}, "
                    "but Core ML states must be floating point (stored as fp16)"
                )
            inputs[name] = mb.state_tensor_placeholder(shape=shape, dtype=types.fp16)
            state_specs[name] = state_spec
            state_compute_dtypes[name] = dtype

        return cls(
            inputs=inputs,
            argument_names=argument_names,
            state_specs=state_specs,
            state_compute_dtypes=state_compute_dtypes,
        )

    def bind_arguments(
        self,
        context: TranslationContext,
        hlo_func: FuncOp,
        ssa_func: Function,
    ) -> dict[str, Var]:
        state_vars = {}
        for in_idx, arg in enumerate(hlo_func.arguments):
            name = self.argument_names[in_idx]
            var = ssa_func.inputs[name]
            if name in self.state_specs:
                state_vars[name] = var
                var = mb.read_state(input=var)
                compute_dtype = self.state_compute_dtypes[name]
                if compute_dtype != types.fp16:
                    var = mb.cast(x=var, dtype=dtype_str(compute_dtype))
            context.add_variable(arg.get_name(), var)
        return state_vars

    def finalize_outputs(
        self,
        outputs: list[Var],
        state_vars: dict[str, Var],
    ) -> list[Var]:
        consumed = set()
        for name, spec in self.state_specs.items():
            if spec.output is None:
                continue
            out_idx = spec.output
            if not 0 <= out_idx < len(outputs):
                raise ValueError(
                    f"State output index {out_idx} is out of range for a function "
                    f"with {len(outputs)} outputs"
                )
            value = outputs[out_idx]
            if self.state_compute_dtypes[name] != types.fp16:
                value = mb.cast(x=value, dtype="fp16")
            mb.coreml_update_state(state=state_vars[name], value=value)
            consumed.add(out_idx)

        final_outputs = [out for i, out in enumerate(outputs) if i not in consumed]
        if final_outputs or not consumed:
            return final_outputs

        # Core ML requires at least one tensor output. Keep state values private
        # and expose a fixed token rather than duplicating every updated state.
        token = mb.const(
            val=np.zeros((1,), dtype=np.float16),
            name="state_update_token",
        )
        return [token]
