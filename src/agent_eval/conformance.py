"""Reusable contract checks for Agent Eval runner and grader plugins."""

from __future__ import annotations

from dataclasses import asdict
from inspect import Parameter, signature
import json
from collections.abc import Mapping
from pathlib import Path
import subprocess
import tempfile
from threading import Event, Thread
from time import sleep
from typing import Any

import yaml

from agent_eval.adapters import result_as_dict
from agent_eval.graders import AnswerGrader, GraderError, grade_answer
from agent_eval.models import AgentResult, AgentTurnResult, GradeResult, Scenario
from agent_eval.runner import AgentRunner, ExecutionControl, RunCancelled, RunnerError, execute_scenarios
from agent_eval.scenarios import load_scenario


class ConformanceError(ValueError):
    """Raised for malformed plugin conformance suites or version matrices."""


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _report(plugin_type: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    return {"plugin_type": plugin_type, "passed": all(row["status"] == "passed" for row in checks),
            "checks": checks}


def _load_suite(suite: Path | Mapping[str, Any], plugin_type: str) -> list[dict[str, Any]]:
    if isinstance(suite, Path):
        try:
            document = yaml.safe_load(suite.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConformanceError(f"could not read conformance suite {suite}: {exc}") from exc
    else:
        document = dict(suite)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ConformanceError("conformance suite must be a mapping with schema_version: 1")
    declared_type = document.get("plugin_type")
    if declared_type not in (None, plugin_type):
        raise ConformanceError(f"conformance suite is for {declared_type!r}, not {plugin_type!r}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ConformanceError("conformance suite needs a non-empty cases list")
    names = [case.get("name") for case in cases if isinstance(case, dict)]
    if len(names) != len(cases) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ConformanceError("every conformance case needs a non-empty name")
    if len(names) != len(set(names)):
        raise ConformanceError("conformance case names must be unique")
    return cases


def _matches(actual: Any, expected: Any) -> bool:
    """Recursively match expected mapping fields; sequences and scalars match exactly."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _matches(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _load_fixture_scenario(document: Any) -> Scenario:
    if not isinstance(document, dict):
        raise ConformanceError("runner fixture case needs a scenario mapping")
    with tempfile.TemporaryDirectory(prefix="agent-eval-conformance-") as directory:
        path = Path(directory) / "scenario.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        try:
            return load_scenario(path)
        except Exception as exc:
            raise ConformanceError(f"invalid fixture scenario: {exc}") from exc


def _expected_error(case: Mapping[str, Any], error: Exception | None) -> tuple[bool, str]:
    expected = case.get("expect_error")
    if not isinstance(expected, dict) or not isinstance(expected.get("type"), str):
        return False, "expect_error must name the expected exception type"
    if error is None:
        return False, f"expected {expected['type']} but the operation succeeded"
    expected_type = expected["type"]
    if expected_type not in {type(error).__name__, *(cls.__name__ for cls in type(error).__mro__)}:
        return False, f"expected {expected_type}, got {type(error).__name__}: {error}"
    expected_text = expected.get("contains")
    if expected_text is not None and (not isinstance(expected_text, str) or expected_text not in str(error)):
        return False, f"error did not contain {expected_text!r}: {error}"
    return True, f"raised expected {expected_type}"


def _probe_scenario() -> Scenario:
    return Scenario(name="agent_eval_conformance_probe", input={"user": "Conformance probe"},
                    intent="CONFORMANCE")


def _turn_signature(method: Any) -> tuple[bool, bool, str]:
    try:
        params = signature(method).parameters
        signature(method).bind(_probe_scenario(), {"user": "Conformance probe"}, ())
    except (TypeError, ValueError) as exc:
        return False, False, str(exc)
    supports_control = any(
        item.name == "execution_control" or item.kind == Parameter.VAR_KEYWORD
        for item in params.values()
    )
    return True, supports_control, "run_turn accepts scenario, user_input, and history"


def check_runner_conformance(
    runner: AgentRunner, *, suite: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise a runner's callable, canonical trace, repeatability, cancellation, and errors."""
    checks: list[dict[str, str]] = []
    method = getattr(runner, "run_turn", None)
    if not callable(method):
        return _report("runner", [_check("interface", "failed", "runner must provide callable run_turn")])
    interface_valid, supports_control, detail = _turn_signature(method)
    checks.append(_check("interface", "passed" if interface_valid else "failed", detail))

    class _FailingProbe:
        def run_turn(self, probe, user_input, history):
            raise RuntimeError("conformance error probe")

    try:
        execute_scenarios([_probe_scenario()], _FailingProbe())
    except RunnerError:
        checks.append(_check("error_handling", "passed", "runner exceptions are surfaced as actionable RunnerError values"))
    except Exception as exc:
        checks.append(_check("error_handling", "failed", f"runner exception escaped without normalization: {exc}"))
    else:
        checks.append(_check("error_handling", "failed", "intentional runner error probe did not fail"))
    report = _report("runner", checks)
    if suite is None:
        _run_runner_probe(runner, checks)
        report["passed"] = all(item["status"] == "passed" for item in checks)
    else:
        suite_cases = _load_suite(suite, "runner")
        fixture_cases = _run_runner_suite(runner, suite_cases)
        report["fixture_cases"] = fixture_cases
        fixtures_passed = all(item["status"] == "passed" for item in fixture_cases)
        checks.append(_check("fixture_suite", "passed" if fixtures_passed else "failed",
                             f"{sum(item['status'] == 'passed' for item in fixture_cases)}/{len(fixture_cases)} fixture cases passed"))
        trace_cases = [case for case in suite_cases if isinstance(case.get("expected_trace"), dict)]
        checks.append(_check("trace_structure", "passed" if trace_cases else "failed",
                             "fixture traces passed canonical validation" if trace_cases
                             else "suite must include at least one expected_trace case"))
        deterministic, detail = _check_fixture_determinism(runner, trace_cases)
        checks.append(_check("determinism", "passed" if deterministic else "failed", detail))
        has_cancellation = any(case.get("cancellation") is not None for case in suite_cases)
        cancellation_passed = supports_control and any(
            case.get("cancellation") is not None and fixture.get("status") == "passed"
            for case, fixture in zip(suite_cases, fixture_cases)
        )
        checks.append(_check("cancellation", "passed" if cancellation_passed else "failed",
                             "in-flight cancellation fixture passed" if cancellation_passed
                             else "suite needs a passing in-flight cancellation case and execution_control support"))
        report["passed"] = (report["passed"] and fixtures_passed and bool(trace_cases)
                            and deterministic and cancellation_passed)
    return report


def _run_runner_probe(runner: AgentRunner, checks: list[dict[str, str]]) -> None:
    scenario = _probe_scenario()
    observed: list[AgentTurnResult] = []
    try:
        for _ in range(2):
            [result] = execute_scenarios([scenario], runner)
            if not result.turns:
                raise RunnerError("runner result did not contain a structured turn trace")
            _validate_trace_turn(result.turns[0])
            observed.append(result.turns[0])
        checks.append(_check("trace_structure", "passed", "runner returned a valid canonical turn with structured events and tool calls"))
        stable = _stable_turn(observed[0]) == _stable_turn(observed[1])
        checks.append(_check("determinism", "passed" if stable else "failed",
                             "repeated probe outputs match" if stable else "repeated probe outputs differ"))
    except Exception as exc:
        checks.append(_check("trace_structure", "failed", f"probe failed validation: {exc}"))
        checks.append(_check("determinism", "failed", "could not compare repeated valid probe outputs"))
    cancellation_passed = _pre_cancelled_probe(runner)
    checks.append(_check("cancellation", "passed" if cancellation_passed else "failed",
                         "runner observes a pre-cancelled execution_control" if cancellation_passed
                         else "runner must accept and honor a pre-cancelled execution_control"))


def _validate_trace_turn(turn: AgentTurnResult) -> None:
    if any(not isinstance(item.event_type, str) or item.event_type not in {"question", "action"}
           or not isinstance(item.name, str) or not item.name.strip() for item in turn.events):
        raise RunnerError("trace events need type 'question' or 'action' and a non-empty name")
    if any(not isinstance(call.name, str) or not call.name.strip()
           or not isinstance(call.arguments, dict) for call in turn.tool_calls):
        raise RunnerError("tool calls need a non-empty name and an arguments mapping")


def _stable_turn(turn: AgentTurnResult) -> dict[str, Any]:
    fields = ("intent", "asked", "tools", "actions", "tool_calls", "events", "answer", "notes")
    value = asdict(turn)
    return {field: value[field] for field in fields}


def _pre_cancelled_probe(runner: AgentRunner) -> bool:
    method = runner.run_turn
    params = signature(method).parameters
    if not any(item.name == "execution_control" or item.kind == Parameter.VAR_KEYWORD
               for item in params.values()):
        return False
    stop = Event()
    stop.set()
    try:
        method(_probe_scenario(), {"user": "Conformance probe"}, (),
               execution_control=ExecutionControl("agent_eval_conformance_probe", stop, None))
    except RunCancelled:
        return True
    except Exception:
        return False
    return False


def _check_fixture_determinism(runner: AgentRunner, cases: list[dict[str, Any]]) -> tuple[bool, str]:
    if not cases:
        return False, "suite needs at least one expected_trace case for repeatability checks"
    try:
        for case in cases:
            scenario = _load_fixture_scenario(case["scenario"])
            outputs = []
            for _ in range(2):
                [result] = execute_scenarios([scenario], runner)
                if not result.turns:
                    raise ConformanceError("runner result did not include a structured turn trace")
                for turn in result.turns:
                    _validate_trace_turn(turn)
                outputs.append(result_as_dict(result))
            for output in outputs:
                output.pop("metadata", None)
                for turn in output.get("turns", []):
                    turn.pop("metadata", None)
            if outputs[0] != outputs[1]:
                return False, f"fixture case {case['name']!r} produced different traces on repeat"
        return True, f"{len(cases)} expected-trace fixture case(s) repeat deterministically"
    except Exception as exc:
        return False, f"could not repeat fixture trace: {exc}"


def _run_runner_suite(runner: AgentRunner, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {"name": case["name"], "status": "failed"}
        try:
            scenario = _load_fixture_scenario(case.get("scenario"))
            expected_error = case.get("expect_error")
            cancellation = case.get("cancellation")
            expected_trace = case.get("expected_trace")
            modes = sum((expected_error is not None, cancellation is not None, expected_trace is not None))
            if modes != 1:
                raise ConformanceError("runner case must define exactly one of expected_trace, expect_error, or cancellation")
            if cancellation is not None:
                if not isinstance(cancellation, dict):
                    raise ConformanceError("cancellation must be a mapping")
                row.update(_run_cancellation_case(runner, scenario, cancellation))
            elif expected_error is not None:
                try:
                    execute_scenarios([scenario], runner)
                    error = None
                except Exception as exc:
                    error = exc
                passed, detail = _expected_error(case, error)
                row.update({"status": "passed" if passed else "failed", "detail": detail})
            else:
                if not isinstance(expected_trace, dict) or not expected_trace:
                    raise ConformanceError("expected_trace must be a non-empty mapping")
                [result] = execute_scenarios([scenario], runner)
                if not result.turns:
                    raise ConformanceError("runner result did not include a structured turn trace")
                for turn in result.turns:
                    _validate_trace_turn(turn)
                actual = result_as_dict(result)
                matched = _matches(actual, expected_trace)
                row.update({"status": "passed" if matched else "failed",
                            "detail": "expected trace matched" if matched else "expected trace differs",
                            "expected": expected_trace, "observed": actual})
        except Exception as exc:
            row["detail"] = str(exc)
        rows.append(row)
    return rows


def _run_cancellation_case(
    runner: AgentRunner, scenario: Scenario, config: Mapping[str, Any],
) -> dict[str, Any]:
    delay_ms = config.get("after_start_ms", 20)
    timeout = config.get("timeout_seconds", 2.0)
    if (isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or not 0 <= delay_ms <= 5000
            or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not 0.05 <= timeout <= 30):
        raise ConformanceError("cancellation after_start_ms must be 0..5000 and timeout_seconds 0.05..30")
    cancelled = Event()
    started = Event()
    finished = Event()
    outcome: dict[str, Any] = {"results": None, "stats": {}, "error": None}

    class _StartedRunner:
        def run_turn(self, fixture, user_input, history, tool_sandbox=None, execution_control=None):
            started.set()
            method = runner.run_turn
            params = signature(method).parameters
            kwargs = {}
            if "tool_sandbox" in params or any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
                kwargs["tool_sandbox"] = tool_sandbox
            if "execution_control" in params or any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
                kwargs["execution_control"] = execution_control
            return method(fixture, user_input, history, **kwargs)

    def execute() -> None:
        try:
            outcome["results"] = execute_scenarios(
                [scenario], _StartedRunner(), cancel_event=cancelled, stats=outcome["stats"],
            )
        except Exception as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    worker = Thread(target=execute, name="agent-eval-conformance-cancel", daemon=True)
    worker.start()
    if not started.wait(timeout=float(timeout)):
        return {"status": "failed", "detail": "runner did not start before cancellation timeout"}
    if delay_ms:
        sleep(delay_ms / 1000)
    if finished.is_set():
        return {"status": "failed", "detail": "runner completed before cancellation was requested"}
    cancelled.set()
    worker.join(timeout=float(timeout))
    if worker.is_alive():
        return {"status": "failed", "detail": "runner did not stop after cancellation"}
    if outcome["error"] is not None:
        return {"status": "failed", "detail": f"runner raised instead of honoring cancellation: {outcome['error']}"}
    if outcome["results"] or outcome["stats"].get("cancelled", 0) != 1:
        return {"status": "failed", "detail": "scenario completed instead of being cancelled"}
    return {"status": "passed", "detail": "in-flight scenario stopped after cancellation"}


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
    answer: str = "Conformance probe", suite: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check grader interface, normalized output, deterministic behavior, and errors."""
    suite_cases = _load_suite(suite, "grader") if suite is not None else None
    if suite_cases and (config is None or not config):
        first_valid = next((case for case in suite_cases if isinstance(case.get("expected_grade"), dict)), None)
        if first_valid is not None:
            config = first_valid.get("config", {})
            answer = first_valid.get("answer", answer)
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
    report = _report("grader", checks)
    if suite_cases is not None:
        fixture_cases = _run_grader_suite(grader, suite_cases)
        report["fixture_cases"] = fixture_cases
        passed = all(item["status"] == "passed" for item in fixture_cases)
        report["passed"] = report["passed"] and passed
        checks.append(_check("fixture_suite", "passed" if passed else "failed",
                             f"{sum(item['status'] == 'passed' for item in fixture_cases)}/{len(fixture_cases)} fixture cases passed"))
    return report


def _run_grader_suite(grader: AnswerGrader, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {"name": case["name"], "status": "failed"}
        try:
            answer = case.get("answer", "")
            config = case.get("config", {})
            if not isinstance(answer, str) or not isinstance(config, dict):
                raise ConformanceError("grader fixture answer must be a string and config a mapping")
            expect_error = case.get("expect_error")
            expected_grade = case.get("expected_grade")
            if (expect_error is None) == (expected_grade is None):
                raise ConformanceError("grader case must define exactly one of expected_grade or expect_error")
            if expect_error is not None:
                try:
                    grader.grade(answer, config)
                    error = None
                except Exception as exc:
                    error = exc
                passed, detail = _expected_error(case, error)
                row.update({"status": "passed" if passed else "failed", "detail": detail})
                rows.append(row)
                continue
            if not isinstance(expected_grade, dict) or not expected_grade:
                raise ConformanceError("expected_grade must be a non-empty mapping")
            output = grader.grade(answer, config)
            valid, detail = _grade_shape(output)
            if not valid:
                raise ConformanceError(detail)
            actual = asdict(output)
            matched = _matches(actual, expected_grade)
            row.update({"status": "passed" if matched else "failed",
                        "detail": "expected grade matched" if matched else "expected grade differs",
                        "expected": expected_grade, "observed": actual})
        except Exception as exc:
            row["detail"] = str(exc)
        rows.append(row)
    return rows


def run_conformance_matrix(
    plugin_type: str, plugin_name: str, suite_path: Path, targets_path: Path,
    *, timeout_seconds: float = 120, grader_config: Mapping[str, Any] | None = None,
    grader_answer: str | None = None,
) -> dict[str, Any]:
    """Run one fixture suite in each configured Python/Agent Eval environment.

    Each target names a Python executable whose environment has the desired
    Agent Eval version and plugin installed. The child process reports the
    versions it actually imported; optional expected versions make drift fail.
    """
    if plugin_type not in {"runner", "grader"}:
        raise ConformanceError("plugin type must be runner or grader")
    try:
        matrix = yaml.safe_load(targets_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConformanceError(f"could not read conformance targets {targets_path}: {exc}") from exc
    if not isinstance(matrix, dict) or matrix.get("schema_version", 1) != 1:
        raise ConformanceError("conformance targets must be a mapping with schema_version: 1")
    targets = matrix.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ConformanceError("conformance targets need a non-empty targets list")
    names = [target.get("name") for target in targets if isinstance(target, dict)]
    if len(names) != len(targets) or any(not isinstance(item, str) or not item.strip() for item in names):
        raise ConformanceError("every conformance target needs a non-empty name")
    if len(names) != len(set(names)):
        raise ConformanceError("conformance target names must be unique")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ConformanceError("matrix timeout must be a positive number")

    suite_absolute = suite_path.resolve()
    rows: list[dict[str, Any]] = []
    capability_rows: dict[str, dict[str, list[str]]] = {}
    for target in targets:
        name = target["name"]
        python = target.get("python", "python")
        row: dict[str, Any] = {
            "name": name, "python_executable": python,
            "status": "failed", "passed": False, "capabilities": {},
        }
        if not isinstance(python, str) or not python.strip():
            row["error"] = "target python must be a non-empty executable path"
            rows.append(row)
            continue
        version_code = (
            "import json, platform, sys, agent_eval; "
            "print(json.dumps({'python': platform.python_version(), 'agent_eval': agent_eval.__version__}))"
        )
        try:
            version_result = subprocess.run(
                [python, "-c", version_code], capture_output=True, text=True,
                timeout=float(timeout_seconds), check=False,
            )
            if version_result.returncode != 0:
                raise ConformanceError(version_result.stderr.strip() or "could not import agent_eval")
            versions = json.loads(version_result.stdout)
            row["python_version"] = versions.get("python")
            row["agent_eval_version"] = versions.get("agent_eval")
            expected_python = target.get("python_version")
            expected_agent_eval = target.get("agent_eval_version")
            version_errors = []
            if expected_python is not None and not str(row["python_version"]).startswith(str(expected_python)):
                version_errors.append(f"expected Python {expected_python}, got {row['python_version']}")
            if expected_agent_eval is not None and row["agent_eval_version"] != str(expected_agent_eval):
                version_errors.append(f"expected Agent Eval {expected_agent_eval}, got {row['agent_eval_version']}")
            selected_config = dict(grader_config if grader_config is not None else matrix.get("grader_config", {}))
            selected_answer = grader_answer if grader_answer is not None else matrix.get("grader_answer", "Conformance probe")
            command = [python, "-m", "agent_eval", "conformance", plugin_type, plugin_name,
                       "--suite", str(suite_absolute), "--format", "json",
                       "--config-json", json.dumps(selected_config, ensure_ascii=False),
                       "--answer", str(selected_answer)]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=float(timeout_seconds), check=False)
            try:
                conformance = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise ConformanceError(
                    completed.stderr.strip() or f"target did not return a conformance JSON report: {exc}"
                ) from exc
            row["checks"] = conformance.get("checks", [])
            row["fixture_cases"] = conformance.get("fixture_cases", [])
            checks = {check.get("name"): check.get("status") for check in row["checks"]}
            checks.update({f"fixture:{case.get('name')}": case.get("status")
                           for case in row["fixture_cases"]})
            row["capabilities"] = checks
            row["passed"] = completed.returncode == 0 and conformance.get("passed") is True and not version_errors
            row["status"] = "passed" if row["passed"] else "failed"
            if version_errors:
                row["version_errors"] = version_errors
            if completed.returncode not in (0, 1):
                row["error"] = completed.stderr.strip() or f"conformance process exited {completed.returncode}"
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ConformanceError) as exc:
            row["error"] = str(exc)
        for capability, status in row["capabilities"].items():
            bucket = capability_rows.setdefault(capability, {"passed": [], "failed": []})
            bucket["passed" if status == "passed" else "failed"].append(name)
        rows.append(row)
    return {
        "schema_version": 1,
        "plugin_type": plugin_type,
        "plugin": plugin_name,
        "suite": str(suite_absolute),
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "target_count": len(rows),
        "targets": rows,
        "capability_matrix": capability_rows,
    }

