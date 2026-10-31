"""Reusable contract checks for Agent Eval runner and grader plugins."""

from __future__ import annotations

from dataclasses import asdict
from inspect import Parameter, signature
import json
from collections.abc import Mapping
from threading import Event
from typing import Any

from agent_eval.graders import AnswerGrader, GraderError, grade_answer
from agent_eval.models import AgentResult, AgentTurnResult, GradeResult, Scenario
from agent_eval.runner import AgentRunner, ExecutionControl, RunCancelled, RunnerError, execute_scenarios


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _report(plugin_type: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    return {"plugin_type": plugin_type, "passed": all(row["status"] == "passed" for row in checks),
            "checks": checks}


def _probe_scenario() -> Scenario:
    return Scenario(name="agent_eval_conformance_probe", input={"user": "Conformance probe"},
                    intent="CONFORMANCE")


def _turn_signature(method: Any) -> tuple[bool, str]:
    try:
        params = signature(method).parameters
        signature(method).bind(_probe_scenario(), {"user": "Conformance probe"}, ())
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    supports_control = any(
        item.name == "execution_control" or item.kind == Parameter.VAR_KEYWORD
        for item in params.values()
    )
    return supports_control, "run_turn accepts scenario, user_input, and history"


def check_runner_conformance(runner: AgentRunner) -> dict[str, Any]:
    """Exercise a runner's callable, canonical trace, repeatability, cancellation, and errors."""
    checks: list[dict[str, str]] = []
    method = getattr(runner, "run_turn", None)
    if not callable(method):
        return _report("runner", [_check("interface", "failed", "runner must provide callable run_turn")])
    supports_control, detail = _turn_signature(method)
    checks.append(_check("interface", "passed", detail))
    scenario = _probe_scenario()
    observed: list[AgentTurnResult] = []
    try:
        for _ in range(2):
            [result] = execute_scenarios([scenario], runner)
            if not result.turns:
                raise RunnerError("runner result did not contain a structured turn trace")
            turn = result.turns[0]
            if any(not isinstance(item.event_type, str) or item.event_type not in {"question", "action"}
                   or not isinstance(item.name, str) or not item.name.strip() for item in turn.events):
                raise RunnerError("trace events need type 'question' or 'action' and a non-empty name")
            if any(not isinstance(call.name, str) or not call.name.strip()
                   or not isinstance(call.arguments, dict) for call in turn.tool_calls):
                raise RunnerError("tool calls need a non-empty name and an arguments mapping")
            observed.append(turn)
        checks.append(_check("trace_structure", "passed", "runner returned a valid canonical turn with structured events and tool calls"))
        stable_fields = ("intent", "asked", "tools", "actions", "tool_calls", "events", "answer", "notes")
        first = {field: asdict(observed[0])[field] for field in stable_fields}
        second = {field: asdict(observed[1])[field] for field in stable_fields}
        checks.append(_check("determinism", "passed" if first == second else "failed",
                             "repeated probe outputs match" if first == second else "repeated probe outputs differ"))
    except Exception as exc:
        checks.append(_check("trace_structure", "failed", f"probe failed validation: {exc}"))
        checks.append(_check("determinism", "failed", "could not compare repeated valid probe outputs"))

    if supports_control:
        cancelled = Event()
        cancelled.set()
        control = ExecutionControl(scenario.name, cancelled, None)
        try:
            method(scenario, scenario.input, (), execution_control=control)
        except RunCancelled:
            checks.append(_check("cancellation", "passed", "runner observes a pre-cancelled execution_control"))
        except Exception as exc:
            checks.append(_check("cancellation", "failed", f"unexpected cancellation error: {exc}"))
        else:
            checks.append(_check("cancellation", "failed", "runner returned without honoring a pre-cancelled execution_control"))
    else:
        checks.append(_check("cancellation", "failed", "run_turn must accept execution_control to support cooperative cancellation"))

    class _FailingProbe:
        def run_turn(self, probe, user_input, history):
            raise RuntimeError("conformance error probe")

    try:
        execute_scenarios([scenario], _FailingProbe())
    except RunnerError:
        checks.append(_check("error_handling", "passed", "runner exceptions are surfaced as actionable RunnerError values"))
    except Exception as exc:
        checks.append(_check("error_handling", "failed", f"runner exception escaped without normalization: {exc}"))
    else:
        checks.append(_check("error_handling", "failed", "intentional runner error probe did not fail"))
    return _report("runner", checks)


def _grade_shape(value: Any) -> tuple[bool, str]:
    if not isinstance(value, GradeResult):
        return False, f"grade returned {type(value).__name__}; expected GradeResult"
    if (isinstance(value.score, bool) or not isinstance(value.score, (int, float))
            or not 0 <= value.score <= 1):
        return False, "GradeResult.score must be a number in [0, 1]"
    if not isinstance(value.passed, bool) or not isinstance(value.feedback, str):
        return False, "GradeResult.passed must be bool and feedback must be str"
    if not isinstance(value.metadata, Mapping) or not isinstance(value.rubric, Mapping):
        return False, "GradeResult metadata and rubric must be mappings"
    return True, "grader returned a valid normalized GradeResult"


def check_grader_conformance(
    grader: AnswerGrader, *, config: Mapping[str, Any] | None = None,
    answer: str = "Conformance probe",
) -> dict[str, Any]:
    """Check grader interface, normalized output, deterministic behavior, and errors."""
    checks: list[dict[str, str]] = []
    method = getattr(grader, "grade", None)
    if not callable(method):
        return _report("grader", [_check("interface", "failed", "grader must provide callable grade(answer, config)")])
    try:
        signature(method).bind(answer, config or {})
    except (TypeError, ValueError) as exc:
        return _report("grader", [_check("interface", "failed", f"grade must accept answer and config: {exc}")])
    checks.append(_check("interface", "passed", "grade accepts answer and config"))
    selected_config = dict(config or {})
    first = second = None
    try:
        first = method(answer, selected_config)
        valid, detail = _grade_shape(first)
        checks.append(_check("grade_structure", "passed" if valid else "failed", detail))
        second = method(answer, selected_config)
        first_wire = asdict(first) if isinstance(first, GradeResult) else first
        second_wire = asdict(second) if isinstance(second, GradeResult) else second
        stable = json.dumps(first_wire, sort_keys=True, default=str) == json.dumps(second_wire, sort_keys=True, default=str)
        checks.append(_check("determinism", "passed" if stable else "failed",
                             "repeated grade outputs match" if stable else "repeated grade outputs differ"))
    except Exception as exc:
        checks.append(_check("grade_structure", "failed", f"valid-input probe raised: {exc}"))
        checks.append(_check("determinism", "failed", "could not compare repeated valid grades"))

    class _FailingGrader:
        def grade(self, probe_answer, probe_config):
            raise RuntimeError("conformance error probe")

    scenario = Scenario(name="agent_eval_grader_probe", input={"user": "Conformance probe"},
                        intent="CONFORMANCE", answer_graders=({"type": "probe"},))
    try:
        grade_answer(scenario, AgentResult(scenario.name, "CONFORMANCE", answer=answer), {"probe": _FailingGrader()})
    except GraderError:
        checks.append(_check("error_handling", "passed", "grader exceptions are surfaced as actionable GraderError values"))
    except Exception as exc:
        checks.append(_check("error_handling", "failed", f"grader exception escaped without normalization: {exc}"))
    else:
        checks.append(_check("error_handling", "failed", "intentional grader error probe did not fail"))
    return _report("grader", checks)

