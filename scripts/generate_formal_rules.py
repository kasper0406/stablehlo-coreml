#!/usr/bin/env python3
"""Generate executable Python and Lean rule predicates from declarative JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_PATH = ROOT / "formal/rules/remove_noop_slice_update.json"
FIELD_TYPES = {"bool", "dim", "id", "int"}


class SchemaError(ValueError):
    """The declarative rule does not conform to the tiny supported DSL."""


def _fields_by_scope(rule):
    result = {}
    for scope, key in (("match", "match_fields"), ("axis", "axis_fields")):
        fields = {}
        for field in rule[key]:
            if set(field) != {"name", "lean_name", "type"} or field["type"] not in FIELD_TYPES:
                raise SchemaError(f"invalid {scope} field: {field!r}")
            if field["name"] in fields:
                raise SchemaError(f"duplicate {scope} field {field['name']!r}")
            fields[field["name"]] = field
        result[scope] = fields
    return result


def _typecheck(node, fields, *, scopes=frozenset({"match"}), allow_all_axes=True):
    if not isinstance(node, dict) or not isinstance(node.get("op"), str):
        raise SchemaError(f"expression must be an object with an op: {node!r}")
    op = node["op"]
    if op == "field":
        scope, name = node.get("scope"), node.get("name")
        if scope not in scopes:
            raise SchemaError(f"field scope {scope!r} is not available here")
        if scope not in fields or name not in fields[scope]:
            raise SchemaError(f"unknown field {scope}.{name}")
        return fields[scope][name]["type"]
    if op == "const":
        node_type, value = node.get("type"), node.get("value")
        if node_type == "bool" and type(value) is bool:
            return "bool"
        if node_type == "int" and type(value) is int:
            return "int"
        raise SchemaError(f"invalid constant: {node!r}")
    if op == "eq":
        left = _typecheck(node.get("left"), fields, scopes=scopes, allow_all_axes=False)
        right = _typecheck(node.get("right"), fields, scopes=scopes, allow_all_axes=False)
        if left != right:
            raise SchemaError(f"equality type mismatch: {left} and {right}")
        return "bool"
    if op in {"and", "or"}:
        args = node.get("args")
        if not isinstance(args, list) or len(args) < 2:
            raise SchemaError(f"{op} requires at least two arguments")
        if any(
            _typecheck(arg, fields, scopes=scopes, allow_all_axes=allow_all_axes) != "bool" for arg in args
        ):
            raise SchemaError(f"{op} arguments must be boolean")
        return "bool"
    if op == "not":
        if _typecheck(node.get("arg"), fields, scopes=scopes, allow_all_axes=False) != "bool":
            raise SchemaError("not argument must be boolean")
        return "bool"
    if op == "dim_eq_int":
        if _typecheck(node.get("dim"), fields, scopes=scopes, allow_all_axes=False) != "dim":
            raise SchemaError("dim_eq_int dim argument must be a dimension")
        if _typecheck(node.get("value"), fields, scopes=scopes, allow_all_axes=False) != "int":
            raise SchemaError("dim_eq_int value argument must be an integer")
        return "bool"
    if op == "all_axes" and allow_all_axes:
        if _typecheck(node.get("body"), fields, scopes=frozenset({"axis"}), allow_all_axes=False) != "bool":
            raise SchemaError("all_axes body must be boolean")
        return "bool"
    raise SchemaError(f"unsupported expression op {op!r}")


def _python_expr(node, indent=0):
    op = node["op"]
    if op == "field":
        owner = "candidate" if node["scope"] == "match" else "axis"
        return f"{owner}.{node['name']}"
    if op == "const":
        return repr(node["value"])
    if op == "eq":
        return f"({_python_expr(node['left'], indent)} == {_python_expr(node['right'], indent)})"
    if op in {"and", "or"}:
        child_indent = indent + 4
        separator = f"\n{' ' * child_indent}{op} "
        children = separator.join(_python_expr(arg, child_indent) for arg in node["args"])
        return f"(\n{' ' * child_indent}{children}\n{' ' * indent})"
    if op == "not":
        return f"(not {_python_expr(node['arg'], indent)})"
    if op == "dim_eq_int":
        return f"_dim_eq_int({_python_expr(node['dim'], indent)}, {_python_expr(node['value'], indent)})"
    if op == "all_axes":
        return "all(axis_matches(axis) for axis in candidate.axes)"
    raise AssertionError(op)


def _lean_expr(node, fields):
    op = node["op"]
    if op == "field":
        owner = "candidate" if node["scope"] == "match" else "axis"
        return f"{owner}.{fields[node['scope']][node['name']]['lean_name']}"
    if op == "const":
        if node["type"] == "bool":
            return "true" if node["value"] else "false"
        value = str(node["value"])
        return f"({value})" if node["value"] < 0 else value
    if op == "eq":
        return f"({_lean_expr(node['left'], fields)} == {_lean_expr(node['right'], fields)})"
    if op in {"and", "or"}:
        operator = " && " if op == "and" else " || "
        return "(" + operator.join(_lean_expr(arg, fields) for arg in node["args"]) + ")"
    if op == "not":
        return f"(!{_lean_expr(node['arg'], fields)})"
    if op == "dim_eq_int":
        return f"dimEqInt {_lean_expr(node['dim'], fields)} {_lean_expr(node['value'], fields)}"
    if op == "all_axes":
        return "candidate.axes.all axisMatches"
    raise AssertionError(op)


def _axis_body(predicate):
    found = []

    def walk(node):
        if node["op"] == "all_axes":
            found.append(node["body"])
            return
        for value in node.values():
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        walk(item)

    walk(predicate)
    if len(found) != 1:
        raise SchemaError("predicate must contain exactly one all_axes expression")
    return found[0]


def _python_type(field_type):
    return {"bool": "bool", "dim": "Dim", "id": "str", "int": "int"}[field_type]


def _lean_type(field_type):
    return {"bool": "Bool", "dim": "Dim", "id": "String", "int": "Int"}[field_type]


def _generate_python(rule, fields, digest):
    axis_lines = "\n".join(f"    {field['name']}: {_python_type(field['type'])}" for field in rule["axis_fields"])
    match_lines = "\n".join(
        f"    {field['name']}: {_python_type(field['type'])}" for field in rule["match_fields"]
    )
    axis_expr = _python_expr(_axis_body(rule["predicate"]), 4)
    predicate_expr = _python_expr(rule["predicate"], 4)
    return f'''# Generated by scripts/generate_formal_rules.py from
# formal/rules/remove_noop_slice_update.json (sha256: {digest}).
# Do not edit by hand.

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Dim:
    kind: Literal["fixed", "symbol"]
    value: int | str

    def __post_init__(self):
        fixed = self.kind == "fixed" and type(self.value) is int and self.value >= 0
        symbol = self.kind == "symbol" and isinstance(self.value, str) and bool(self.value)
        if not (fixed or symbol):
            raise ValueError("invalid normalized dimension")


@dataclass(frozen=True)
class Axis:
{axis_lines}


@dataclass(frozen=True)
class Match:
{match_lines}
    axes: tuple[Axis, ...]


def _dim_eq_int(dim: Dim, value: int) -> bool:
    return dim.kind == "fixed" and value >= 0 and dim.value == value


def axis_matches(axis: Axis) -> bool:
    return {axis_expr}


def matches(candidate: Match) -> bool:
    return {predicate_expr}
'''


def _generate_lean(rule, fields, digest):
    axis_lines = "\n".join(
        f"  {field['lean_name']} : {_lean_type(field['type'])}" for field in rule["axis_fields"]
    )
    match_lines = "\n".join(
        f"  {field['lean_name']} : {_lean_type(field['type'])}" for field in rule["match_fields"]
    )
    axis_expr = _lean_expr(_axis_body(rule["predicate"]), fields)
    predicate_expr = _lean_expr(rule["predicate"], fields)
    return f'''-- Generated by scripts/generate_formal_rules.py from
-- formal/rules/remove_noop_slice_update.json (sha256: {digest}).
-- Do not edit by hand.

namespace {rule["lean_namespace"]}

inductive Dim where
  | fixed (value : Nat)
  | symbol (id : String)
deriving DecidableEq, Repr

structure Axis where
{axis_lines}
deriving DecidableEq, Repr

structure Match where
{match_lines}
  axes : List Axis
deriving DecidableEq, Repr

def dimEqInt (dim : Dim) (value : Int) : Bool :=
  match dim with
  | .fixed n => if value < 0 then false else n == value.toNat
  | .symbol _ => false

def axisMatches (axis : Axis) : Bool :=
  {axis_expr}

def ruleMatches (candidate : Match) : Bool :=
  {predicate_expr}

end {rule["lean_namespace"]}
'''


def _load_rule(path=RULE_PATH):
    raw = path.read_bytes()
    rule = json.loads(raw)
    required = {
        "schema_version",
        "rule",
        "python_output",
        "lean_output",
        "lean_namespace",
        "match_fields",
        "axis_fields",
        "predicate",
    }
    if set(rule) != required or rule["schema_version"] != 1 or rule["rule"] != "remove_noop_slice_update":
        raise SchemaError("unexpected top-level rule schema")
    fields = _fields_by_scope(rule)
    if _typecheck(rule["predicate"], fields) != "bool":
        raise SchemaError("rule predicate must be boolean")
    _axis_body(rule["predicate"])
    digest = hashlib.sha256(raw).hexdigest()
    return rule, fields, digest


def _write_or_check(path: Path, content: str, *, check: bool) -> bool:
    if check:
        if not path.exists() or path.read_text() != content:
            try:
                display_path = path.relative_to(ROOT)
            except ValueError:
                display_path = path
            print(f"generated file is stale: {display_path}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting stale generated files")
    args = parser.parse_args(argv)
    rule, fields, digest = _load_rule()
    outputs = {
        ROOT / rule["python_output"]: _generate_python(rule, fields, digest),
        ROOT / rule["lean_output"]: _generate_lean(rule, fields, digest),
    }
    results = [_write_or_check(path, content, check=args.check) for path, content in outputs.items()]
    valid = all(results)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
