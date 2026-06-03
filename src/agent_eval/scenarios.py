"""Load and validate YAML evaluation scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_eval.models import Scenario


class ScenarioError(ValueError):
    """Raised when a scenario is missing required or well-formed fields."""


def _string_list(value: Any, field_name: str, source: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ScenarioError(f"{source}: '{field_name}' must be a list of strings")
    return tuple(value)


def load_scenario(path: Path) -> Scenario:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioError(f"{path}: could not read YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise ScenarioError(f"{path}: scenario must be a YAML mapping")

    name = document.get("scenario")
    user_input = document.get("input")
    expected = document.get("expected", {})
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{path}: 'scenario' must be a non-empty string")
    if not isinstance(user_input, dict) or not user_input:
        raise ScenarioError(f"{path}: 'input' must be a non-empty mapping")
    if not isinstance(expected, dict) or not isinstance(expected.get("intent"), str):
        raise ScenarioError(f"{path}: 'expected.intent' must be a string")

    expected_tools = expected.get("tools")
    if expected_tools is not None:
        expected_tools = _string_list(expected_tools, "expected.tools", path)

    return Scenario(
        name=name,
        input=user_input,
        intent=expected["intent"],
        must_ask=_string_list(expected.get("must_ask"), "expected.must_ask", path),
        expected_tools=expected_tools,
        must_not=_string_list(document.get("must_not"), "must_not", path),
        allowed_questions=_string_list(
            document.get("allowed_questions"), "allowed_questions", path
        ),
        source=str(path),
    )


def load_scenarios(path: Path) -> list[Scenario]:
    """Load one scenario file or all YAML files under a directory."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}
        )
    else:
        raise ScenarioError(f"Scenario path does not exist: {path}")

    if not files:
        raise ScenarioError(f"No .yaml or .yml scenario files found in {path}")

    scenarios = [load_scenario(file) for file in files]
    names = [scenario.name for scenario in scenarios]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ScenarioError(f"Duplicate scenario name(s): {', '.join(duplicates)}")
    return scenarios

