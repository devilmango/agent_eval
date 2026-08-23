"""Deterministic, deny-by-default tool fixtures for isolated agent runs."""

from __future__ import annotations

from typing import Any

from agent_eval.models import Scenario


class ToolFixtureError(ValueError):
    """Raised when a runner attempts an unconfigured tool call."""


class ToolSandbox:
    def __init__(self, scenario: Scenario):
        self._fixtures = list(scenario.tool_fixtures)
        self.calls: list[dict[str, Any]] = []

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        arguments = arguments or {}
        self.calls.append({"name": name, "arguments": dict(arguments)})
        for fixture in self._fixtures:
            if fixture["name"] == name and all(arguments.get(key) == value for key, value in fixture.get("arguments", {}).items()):
                if "error" in fixture:
                    raise ToolFixtureError(str(fixture["error"]))
                return fixture.get("response")
        raise ToolFixtureError(f"tool call {name!r} has no matching fixture; external execution is disabled")

