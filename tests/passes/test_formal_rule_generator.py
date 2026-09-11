import json

import pytest

from scripts import generate_formal_rules


def test_generated_formal_rules_are_current():
    assert generate_formal_rules.main(["--check"]) == 0


def test_rule_generator_rejects_unknown_expression(tmp_path):
    rule = json.loads(generate_formal_rules.RULE_PATH.read_text())
    rule["predicate"] = {"op": "unknown"}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(rule))

    try:
        generate_formal_rules._load_rule(path)
    except generate_formal_rules.SchemaError as error:
        assert "unsupported expression op" in str(error)
    else:
        raise AssertionError("unknown expression was accepted")


@pytest.mark.parametrize(
    "predicate, message",
    [
        (
            {
                "op": "field",
                "scope": "axis",
                "name": "begin",
            },
            "field scope 'axis' is not available here",
        ),
        (
            {
                "op": "all_axes",
                "body": {
                    "op": "eq",
                    "left": {"op": "field", "scope": "match", "name": "x_dtype"},
                    "right": {"op": "field", "scope": "match", "name": "update_dtype"},
                },
            },
            "field scope 'match' is not available here",
        ),
    ],
)
def test_rule_generator_rejects_fields_outside_their_scope(tmp_path, predicate, message):
    rule = json.loads(generate_formal_rules.RULE_PATH.read_text())
    rule["predicate"] = predicate
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(rule))

    with pytest.raises(generate_formal_rules.SchemaError, match=message):
        generate_formal_rules._load_rule(path)


def test_rule_generator_reports_every_stale_artifact(tmp_path, capsys):
    first = tmp_path / "first.py"
    second = tmp_path / "second.lean"

    results = [
        generate_formal_rules._write_or_check(path, "generated\n", check=True)
        for path in (first, second)
    ]

    assert results == [False, False]
    errors = capsys.readouterr().err
    assert "first.py" in errors
    assert "second.lean" in errors
