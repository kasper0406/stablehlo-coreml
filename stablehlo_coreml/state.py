import re
from collections.abc import Mapping
from dataclasses import dataclass

from jaxlib.mlir import ir
from jaxlib.mlir.dialects.func import FuncOp


@dataclass(frozen=True)
class StateSpec:
    """Describe how a StableHLO function argument is exposed as Core ML state."""

    output: int | str | None
    name: str | None = None


StateMapping = Mapping[int | str, int | StateSpec | None]
FunctionStateMapping = Mapping[str, StateMapping]

# A `NameLoc` renders as `loc("name")`, or `loc("name"(<child location>))` when it
# wraps a child location. Other kinds render differently: `loc("file":line:col)`
# for a file location and `loc(fused[...])` / `loc(callsite(...))` for the rest.
_NAME_LOCATION_RE = re.compile(r'^loc\("([^"\\]*)"(?:\(.*\))?\)$', re.DOTALL)


def _argument_name_aliases(arg) -> list[str]:
    names: list[str] = []
    raw = arg.get_name()
    if raw:
        names.append(raw)
        if raw.startswith("%"):
            names.append(raw[1:])

    loc_name = _argument_location_name(arg)
    if loc_name:
        names.append(loc_name)
    return names


def _is_name_location(loc) -> bool | None:
    """Whether ``loc`` is a ``NameLoc``, or ``None`` if it cannot be determined.

    The location kind *must* be checked before reading ``name_str``: on
    jaxlib 0.9.x (our minimum supported version) reading ``name_str`` off a
    non-``NameLoc`` segfaults the interpreter rather than raising.
    """
    # jaxlib 0.9.x exposes a single `Location` type with `is_a_*` predicates.
    is_a_name = getattr(loc, "is_a_name", None)
    if is_a_name is not None:
        return bool(is_a_name())

    # jaxlib 0.10+ dropped the predicates and instead returns concrete
    # `Location` subclasses such as `ir.NameLoc` and `ir.FileLineColLoc`.
    name_loc_cls = getattr(ir, "NameLoc", None)
    if name_loc_cls is not None:
        return isinstance(loc, name_loc_cls)

    return None


def _argument_location_name(arg) -> str | None:
    loc = getattr(arg, "location", None)
    if loc is None:
        return None

    is_name = _is_name_location(loc)
    if is_name is False:
        return None
    if is_name is None:
        # Unknown binding: fall back to the (stable) textual form rather than
        # risk reading `name_str` off a location that does not have one.
        match = _NAME_LOCATION_RE.match(str(loc))
        name = match.group(1) if match else None
    else:
        name = getattr(loc, "name_str", None)
    return name if isinstance(name, str) and name else None


def preferred_argument_name(arg) -> str:
    return _argument_location_name(arg) or arg.get_name().lstrip("%")


def _result_info_aliases(result_info: str) -> set[str]:
    aliases = {result_info}
    _, separator, suffix = result_info.rpartition("[")
    if not separator or not suffix.endswith("]"):
        return aliases

    quoted_leaf = suffix[:-1]
    if (
        len(quoted_leaf) >= 2
        and quoted_leaf[0] in {"'", '"'}
        and quoted_leaf[-1] == quoted_leaf[0]
    ):
        leaf = quoted_leaf[1:-1]
        if leaf and "'" not in leaf and '"' not in leaf:
            aliases.add(leaf)
    return aliases


def _resolve_state_output(hlo_func: FuncOp, output: int | str | None) -> int | None:
    if output is None or isinstance(output, int):
        return output

    matches: dict[str, list[int]] = {}
    try:
        # `FuncOp.result_attrs` raises `KeyError` when the function carries no
        # `res_attrs` at all, which is the case for hand-written modules.
        result_attrs = hlo_func.result_attrs
    except KeyError:
        result_attrs = ()
    for index, attrs in enumerate(result_attrs):
        try:
            result_info = attrs["jax.result_info"].value
        except KeyError:
            continue
        for alias in _result_info_aliases(result_info):
            matches.setdefault(alias, []).append(index)

    indices = matches.get(output, [])
    if len(indices) > 1:
        raise ValueError(f"State output name {output!r} is ambiguous")
    if not indices:
        known = ", ".join(repr(name) for name in sorted(matches)) or "<none>"
        raise ValueError(f"Unknown state output {output!r}. Known output names: {known}")
    return indices[0]


def _coerce_state_spec(value: int | StateSpec | None) -> StateSpec:
    match value:
        case StateSpec():
            spec = value
        case None:
            # Read-only state: the argument is exposed as state, but no output
            # writes back to it.
            spec = StateSpec(output=None)
        case bool():
            raise TypeError(
                f"State specification must be a StateSpec, output index, or None, got {value!r}"
            )
        case int():
            spec = StateSpec(output=value)
        case _:
            raise TypeError(
                f"State specification must be a StateSpec, output index, or None, got {value!r}"
            )

    if isinstance(spec.output, bool) or not isinstance(spec.output, (int, str, type(None))):
        raise TypeError(f"State output must be an int, str, or None, got {spec.output!r}")
    if spec.name is not None and (not isinstance(spec.name, str) or not spec.name):
        raise TypeError(f"State name must be a non-empty str or None, got {spec.name!r}")
    return spec


def resolve_state_map(hlo_func: FuncOp, states: StateMapping) -> dict[int, StateSpec]:
    args = list(hlo_func.arguments)
    name_to_idx: dict[str, int] = {}
    for i, arg in enumerate(args):
        for alias in _argument_name_aliases(arg):
            name_to_idx.setdefault(alias, i)

    results = hlo_func.type.results
    num_outputs = len(results)

    resolved: dict[int, StateSpec] = {}
    output_owners: dict[int, int] = {}
    state_names: set[str] = set()
    for key, value in states.items():
        spec = _coerce_state_spec(value)
        out_idx = _resolve_state_output(hlo_func, spec.output)
        spec = StateSpec(output=out_idx, name=spec.name)
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise TypeError(f"State input must be an int index or str name, got {key!r}")

        if isinstance(key, int):
            in_idx = key
        elif key not in name_to_idx:
            known = ", ".join(sorted(name_to_idx)) or "<none>"
            raise ValueError(f"Unknown state input {key!r}. Known argument names: {known}")
        else:
            in_idx = name_to_idx[key]

        if not 0 <= in_idx < len(args):
            raise ValueError(
                f"State input index {in_idx} is out of range for a function with {len(args)} arguments"
            )
        if in_idx in resolved:
            raise ValueError(f"Function argument {in_idx} is mapped as state more than once")
        if spec.name is not None and spec.name in state_names:
            raise ValueError(f"Core ML state name {spec.name!r} is used more than once")
        if spec.output is not None and spec.output < 0:
            raise ValueError(f"State output index {spec.output} is invalid")
        if spec.output is not None and spec.output >= num_outputs:
            raise ValueError(
                f"State output index {spec.output} is out of range for a function with {num_outputs} outputs"
            )
        if spec.output is not None and spec.output in output_owners:
            raise ValueError(
                f"Output {spec.output} cannot update both argument "
                f"{output_owners[spec.output]} and {in_idx}"
            )
        if spec.output is not None and args[in_idx].type != results[spec.output]:
            raise ValueError(
                f"State input {in_idx} has type {args[in_idx].type}, but output "
                f"{spec.output} has type {results[spec.output]}"
            )

        resolved[in_idx] = spec
        if spec.name is not None:
            state_names.add(spec.name)
        if spec.output is not None:
            output_owners[spec.output] = in_idx

    return resolved


def normalize_function_state_maps(
    states: StateMapping | FunctionStateMapping | None,
    public_function_names: list[str],
) -> dict[str, StateMapping]:
    if not states:
        return {}

    values = list(states.values())
    has_nested_values = any(isinstance(value, Mapping) for value in values)
    if has_nested_values:
        if not all(isinstance(value, Mapping) for value in values):
            raise TypeError("State mappings cannot mix function mappings with state specifications")
        invalid_names = [name for name in states if not isinstance(name, str)]
        if invalid_names:
            raise TypeError(
                f"Function names in a state mapping must be strings, got {invalid_names[0]!r}"
            )
        unknown = set(states) - set(public_function_names)
        if unknown:
            known = ", ".join(public_function_names) or "<none>"
            raise ValueError(
                f"Unknown public function(s) in state mapping: {', '.join(sorted(unknown))}. "
                f"Known public functions: {known}"
            )
        return {name: mapping for name, mapping in states.items()}

    if len(public_function_names) != 1:
        raise ValueError(
            "A flat state mapping is only supported for single-function modules; "
            "map each public function name to its states"
        )
    return {public_function_names[0]: states}
