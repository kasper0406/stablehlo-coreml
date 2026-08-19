from dataclasses import dataclass

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Placeholder, Symbol, Var, types
from coremltools.converters.mil.mil.passes.defs.preprocess import NameSanitizer
from jaxlib.mlir.dialects.func import FuncOp

from .state import StateSpec, preferred_argument_name
from .translation_context import DYNAMIC_DIM_SENTINEL, TranslationContext
from .utils import dtype_str, get_mil_type_from_ir

# Core ML can only serialize state tensors as fp16. fp32 HLO state is
# stored as fp16 and cast to/from the computation dtype around read/write.
_STATE_VALUE_DTYPES = frozenset({types.fp16, types.fp32})


def sanitize_name(name: str) -> str:
    """Return ``name`` the way Core ML would name it.

    Core ML only accepts names matching ``[a-zA-Z_][a-zA-Z0-9_]*`` and reserves
    a handful of words (``state``, ``tensor``, ...). coremltools rewrites
    offending names, but only for the ``main`` function and without telling the
    caller. We apply the same rewrite up front, so the names we hand to Core ML
    are already final and identical across all functions.
    """
    # A fresh sanitizer keeps this a pure function; uniqueness of the resulting
    # names is handled by the caller.
    return NameSanitizer(prefix="var_").sanitize_name(name)


@dataclass
class _StateBinding:
    """The MIL vars backing one Core ML state input."""

    state: Var  # the state placeholder itself
    read: Var  # the fp16 tensor read out of the state


@dataclass
class _FunctionInterface:
    inputs: dict[str, Placeholder]
    argument_names: dict[int, str]
    state_specs: dict[str, StateSpec]
    state_compute_dtypes: dict[str, object]

    @staticmethod
    def _assign_argument_names(
        hlo_func: FuncOp,
        state_map: dict[int, StateSpec],
    ) -> dict[int, str]:
        explicit_state_names: dict[str, int] = {}
        for in_idx, spec in state_map.items():
            if spec.name is None:
                continue
            sanitized = sanitize_name(spec.name)
            if sanitized != spec.name:
                raise ValueError(
                    f"Core ML state name {spec.name!r} is not a valid Core ML name "
                    f"(Core ML would rename it to {sanitized!r}); "
                    f"use name={sanitized!r} instead"
                )
            explicit_state_names[spec.name] = in_idx

        argument_names: dict[int, str] = {}
        used_input_names: set[str] = set()

        def is_taken(name: str, in_idx: int) -> bool:
            return name in used_input_names or explicit_state_names.get(name, in_idx) != in_idx

        for in_idx, arg in enumerate(hlo_func.arguments):
            state_spec = state_map.get(in_idx)
            explicit_name = state_spec.name if state_spec is not None else None
            if explicit_name is not None:
                name = explicit_name
                if name in used_input_names:
                    raise ValueError(
                        f"Core ML state name {name!r} conflicts with another function input"
                    )
            else:
                name = sanitize_name(preferred_argument_name(arg))
                if is_taken(name, in_idx):
                    name = sanitize_name(arg.get_name().lstrip("%"))
                while is_taken(name, in_idx):
                    name = f"{name}_input"
            used_input_names.add(name)
            argument_names[in_idx] = name

        return argument_names

    @classmethod
    def from_hlo(
        cls,
        hlo_func: FuncOp,
        state_map: dict[int, StateSpec],
    ) -> "_FunctionInterface":
        argument_names = cls._assign_argument_names(hlo_func, state_map)

        # Validate every state argument before constructing any MIL Symbol —
        # Symbol names are process global, so symbols created for earlier
        # arguments would leak if we raised part-way through the build below.
        for in_idx in sorted(state_map):
            arg = hlo_func.arguments[in_idx]
            name = argument_names[in_idx]
            shape = arg.type.shape
            if any(d == DYNAMIC_DIM_SENTINEL for d in shape):
                raise ValueError(f"State input {name} must have a static shape, got {shape}")
            dtype = get_mil_type_from_ir(arg.type.element_type)
            if dtype not in _STATE_VALUE_DTYPES:
                raise ValueError(
                    f"State input {name} has dtype {dtype_str(dtype)}, "
                    "but Core ML states must be floating point (stored as fp16)"
                )

        inputs = {}
        state_specs: dict[str, StateSpec] = {}
        state_compute_dtypes: dict[str, object] = {}
        sym_counter = 0

        for in_idx, arg in enumerate(hlo_func.arguments):
            name = argument_names[in_idx]
            state_spec = state_map.get(in_idx)
            shape = arg.type.shape
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
    ) -> dict[str, _StateBinding]:
        state_bindings = {}
        for in_idx, arg in enumerate(hlo_func.arguments):
            name = self.argument_names[in_idx]
            var = ssa_func.inputs[name]
            if name in self.state_specs:
                var = mb.read_state(input=var)
                state_bindings[name] = _StateBinding(state=ssa_func.inputs[name], read=var)
                compute_dtype = self.state_compute_dtypes[name]
                if compute_dtype != types.fp16:
                    var = mb.cast(x=var, dtype=dtype_str(compute_dtype))
            context.add_variable(arg.get_name(), var)
        return state_bindings

    def finalize_outputs(
        self,
        outputs: list[Var],
        state_bindings: dict[str, _StateBinding],
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
            binding = state_bindings[name]
            value = outputs[out_idx]
            if self.state_compute_dtypes[name] != types.fp16:
                value = mb.cast(x=value, dtype="fp16")
            if value.val is not None:
                # Core ML segfaults while loading a model that writes a compile
                # time constant into a state. Make the written value depend on
                # the state itself, so it survives constant folding. Adding a
                # zero tensor (rather than multiplying the state by zero) keeps
                # NaN/inf already stored in the state out of the update.
                zeros = mb.fill_like(ref_tensor=binding.read, value=np.float16(0))
                value = mb.add(x=zeros, y=value)
            mb.coreml_update_state(state=binding.state, value=value)
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
