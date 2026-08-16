"""Optional agent-runner plugin interface and scenario execution helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from typing import Any, Protocol

from agent_eval.models import AgentResult, AgentTurnResult, Scenario, ScenarioTurn, ToolCall, TraceEvent


class RunnerError(ValueError):
    """Raised when an agent runner plugin cannot execute a scenario."""


class AgentRunner(Protocol):
    """Run one user turn and return structured, observed agent behavior."""

    def run_turn(
        self,
        scenario: Scenario,
        user_input: Mapping[str, Any],
        history: Sequence[AgentTurnResult],
    ) -> AgentTurnResult:
        """Run a turn with prior captured turns and return its trace summary."""


def load_runner(name: str) -> AgentRunner:
    points = entry_points()
    plugins = points.select(group="agent_eval.runners") if hasattr(points, "select") else points.get("agent_eval.runners", [])
    matches = [point for point in plugins if point.name == name]
    if not matches:
        available = ", ".join(sorted(point.name for point in plugins)) or "none installed"
        raise RunnerError(f"unknown runner '{name}' (available: {available})")
    if len(matches) > 1:
        raise RunnerError(f"multiple installed runners are named '{name}'")
    try:
        runner = matches[0].load()
        if isinstance(runner, type):
            runner = runner()
        elif callable(runner) and not hasattr(runner, "run_turn"):
            runner = runner()
    except Exception as exc:
        raise RunnerError(f"could not load runner '{name}': {exc}") from exc
    if not hasattr(runner, "run_turn"):
        raise RunnerError(f"runner '{name}' must provide run_turn(scenario, user_input, history)")
    return runner


def _coerce_turn(value: Any, scenario_name: str, index: int) -> AgentTurnResult:
    if isinstance(value, AgentTurnResult):
        turn = value
    elif isinstance(value, AgentResult):
        if value.scenario != scenario_name:
            raise RunnerError(
                f"runner returned scenario '{value.scenario}' while executing '{scenario_name}'"
            )
        turn = AgentTurnResult(
            intent=value.intent,
            asked=value.asked,
            tools=value.tools,
            actions=value.actions,
            tool_calls=value.tool_calls,
            events=value.events,
            answer=value.answer,
            notes=value.notes,
        )
    else:
        raise RunnerError(
            f"runner returned {type(value).__name__} for turn {index}; expected AgentTurnResult"
        )
    if not isinstance(turn.intent, str) or not turn.intent:
        raise RunnerError(f"runner returned an empty intent for turn {index} in '{scenario_name}'")
    for field_name in ("asked", "tools", "actions"):
        field_value = getattr(turn, field_name)
        if not isinstance(field_value, tuple) or any(not isinstance(item, str) for item in field_value):
            raise RunnerError(f"runner returned invalid '{field_name}' for turn {index} in '{scenario_name}'")
    if any(not isinstance(call, ToolCall) for call in turn.tool_calls):
        raise RunnerError(f"runner returned invalid tool_calls for turn {index} in '{scenario_name}'")
    if any(not isinstance(event, TraceEvent) for event in turn.events):
        raise RunnerError(f"runner returned invalid events for turn {index} in '{scenario_name}'")
    if turn.answer is not None and not isinstance(turn.answer, str):
        raise RunnerError(f"runner returned invalid answer for turn {index} in '{scenario_name}'")
    if not isinstance(turn.notes, str):
        raise RunnerError(f"runner returned invalid notes for turn {index} in '{scenario_name}'")
    if not turn.tools and turn.tool_calls:
        turn = AgentTurnResult(
            intent=turn.intent, asked=turn.asked,
            tools=tuple(call.name for call in turn.tool_calls),
            actions=turn.actions, tool_calls=turn.tool_calls, events=turn.events,
            answer=turn.answer, notes=turn.notes,
        )
    return turn


def execute_scenarios(scenarios: list[Scenario], runner: AgentRunner) -> list[AgentResult]:
    """Execute all turns sequentially and aggregate each scenario's trace."""
    results: list[AgentResult] = []
    for scenario in scenarios:
        prompts = scenario.turns or (ScenarioTurn(input=scenario.input),)
        history: list[AgentTurnResult] = []
        for index, prompt in enumerate(prompts, start=1):
            try:
                value = runner.run_turn(scenario, prompt.input, tuple(history))
            except Exception as exc:
                if isinstance(exc, RunnerError):
                    raise
                raise RunnerError(
                    f"runner failed for scenario '{scenario.name}', turn {index}: {exc}"
                ) from exc
            history.append(_coerce_turn(value, scenario.name, index))
        results.append(AgentResult(
            scenario=scenario.name,
            intent=history[-1].intent,
            asked=tuple(value for turn in history for value in turn.asked),
            tools=tuple(value for turn in history for value in turn.tools),
            actions=tuple(value for turn in history for value in turn.actions),
            tool_calls=tuple(value for turn in history for value in turn.tool_calls),
            events=tuple(value for turn in history for value in turn.events),
            answer=next((turn.answer for turn in reversed(history) if turn.answer is not None), None),
            notes="\n".join(turn.notes for turn in history if turn.notes),
            turns=tuple(history),
        ))
    return results
