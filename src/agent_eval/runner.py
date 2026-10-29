"""Optional agent-runner plugin interface and scenario execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from importlib.metadata import entry_points
from inspect import Parameter, signature
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import BoundedSemaphore, Event, local
from time import monotonic, perf_counter
from typing import Any, Protocol

from agent_eval.models import AgentResult, AgentTurnResult, Scenario, ScenarioTurn, ToolCall, TraceEvent
from agent_eval.tools import ToolSandbox


class RunnerError(ValueError):
    """Raised when an agent runner plugin cannot execute a scenario."""


class ScenarioTimeoutError(RunnerError):
    """Raised when a scenario exceeds its configured wall-clock budget."""


class RunCancelled(RunnerError):
    """Internal signal used to stop scheduling work after cancellation."""


class ExecutionControl:
    """Cooperative cancellation/deadline handle injected into supporting runners."""

    def __init__(self, scenario: str, cancel_event: Event, timeout: float | None):
        self.scenario = scenario
        self.cancel_event = cancel_event
        self.deadline = monotonic() + timeout if timeout is not None else None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set() or self.timed_out

    @property
    def timed_out(self) -> bool:
        return self.deadline is not None and monotonic() >= self.deadline

    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - monotonic())

    def raise_if_cancelled(self) -> None:
        if self.timed_out:
            raise ScenarioTimeoutError(f"scenario '{self.scenario}' timed out")
        if self.cancel_event.is_set():
            raise RunCancelled(f"run cancelled during scenario '{self.scenario}'")

    def raise_if_timed_out(self) -> None:
        if self.timed_out:
            raise ScenarioTimeoutError(f"scenario '{self.scenario}' timed out")


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
            metadata=value.metadata,
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
    if not isinstance(turn.metadata, dict):
        raise RunnerError(f"runner returned invalid metadata for turn {index} in '{scenario_name}'")
    if not turn.tools and turn.tool_calls:
        turn = AgentTurnResult(
            intent=turn.intent, asked=turn.asked,
            tools=tuple(call.name for call in turn.tool_calls),
            actions=turn.actions, tool_calls=turn.tool_calls, events=turn.events,
            answer=turn.answer, notes=turn.notes,
            metadata=turn.metadata,
        )
    return turn


def execute_scenarios(
    scenarios: list[Scenario], runner: AgentRunner, *, max_workers: int = 1,
    checkpoint: Path | None = None, cache: Path | None = None,
    runner_key: str | None = None, runner_factory: Callable[[], AgentRunner] | None = None,
    stats: dict[str, int] | None = None, provider_key: str | None = None,
    provider_limits: Mapping[str, int] | None = None, scenario_timeout: float | None = None,
    cancel_event: Event | None = None,
) -> list[AgentResult]:
    """Execute scenarios concurrently with limits, deadlines, resume, and cache support.

    Results are always returned in input order. Checkpoint/cache files are atomically
    updated after each completed scenario. Cancellation stops new submissions and
    waits for already-started scenarios to finish or cooperate with the stop signal.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise RunnerError("max_workers must be a positive integer")
    if scenario_timeout is not None and (isinstance(scenario_timeout, bool)
                                         or not isinstance(scenario_timeout, (int, float))
                                         or scenario_timeout <= 0):
        raise RunnerError("scenario_timeout must be a positive number")
    limits = dict(provider_limits or {})
    for name, limit in limits.items():
        if not isinstance(name, str) or not name:
            raise RunnerError("provider limit names must be non-empty strings")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise RunnerError(f"provider limit for '{name}' must be a positive integer")
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise RunnerError("scenario names must be unique for execution checkpoints")
    identity = runner_key or _runner_key(runner)
    if checkpoint and cache and checkpoint.resolve() == cache.resolve():
        raise RunnerError("checkpoint and cache paths must be different")
    checkpoint_state = _read_state(checkpoint, identity) if checkpoint else _empty_state(identity)
    cache_state = _read_state(cache, identity) if cache else _empty_state(identity)
    completed: dict[str, AgentResult] = {}
    fingerprints = {scenario.name: _scenario_fingerprint(scenario) for scenario in scenarios}
    requested = 0
    reused = 0
    for scenario in scenarios:
        entry = _matching_entry(checkpoint_state, scenario.name, fingerprints[scenario.name])
        if entry is None:
            entry = _matching_entry(cache_state, scenario.name, fingerprints[scenario.name])
        if entry is not None:
            completed[scenario.name] = _result_from_state(entry.get("result"), scenario.name)
            reused += 1
        else:
            requested += 1

    missing = [scenario for scenario in scenarios if scenario.name not in completed]
    failures: list[tuple[str, Exception]] = []
    worker_local = local()
    stop = cancel_event or Event()
    limiters = {name: BoundedSemaphore(limit) for name, limit in limits.items()}

    def run_one(scenario: Scenario) -> AgentResult:
        if stop.is_set():
            raise RunCancelled(f"run cancelled before scenario '{scenario.name}'")
        active_runner = runner
        if runner_factory is not None:
            active_runner = getattr(worker_local, "runner", None)
            if active_runner is None:
                active_runner = runner_factory()
                worker_local.runner = active_runner
        provider = str(getattr(active_runner, "provider", None)
                       or provider_key or _runner_key(active_runner))
        limiter = limiters.get(provider)
        control = ExecutionControl(scenario.name, stop, scenario_timeout)
        if limiter is not None:
            while not limiter.acquire(timeout=0.05):
                control.raise_if_cancelled()
        try:
            control.raise_if_cancelled()
            return _execute_scenario(scenario, active_runner, control)
        finally:
            if limiter is not None:
                limiter.release()

    active: dict[Any, Scenario] = {}
    next_index = 0
    cancelled = 0
    if max_workers == 1:
        for scenario in missing:
            if stop.is_set():
                cancelled += 1
                continue
            try:
                result = run_one(scenario)
            except RunCancelled:
                cancelled += 1
                continue
            except Exception as exc:
                failures.append((scenario.name, exc))
                continue
            completed[scenario.name] = result
            _save_completed(scenario, result, fingerprints[scenario.name], identity,
                            checkpoint_state, cache_state, checkpoint, cache)
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-eval") as pool:
            while next_index < len(missing) or active:
                while not stop.is_set() and next_index < len(missing) and len(active) < max_workers:
                    scenario = missing[next_index]
                    next_index += 1
                    active[pool.submit(run_one, scenario)] = scenario
                if not active:
                    break
                done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    scenario = active.pop(future)
                    try:
                        result = future.result()
                    except RunCancelled:
                        cancelled += 1
                        continue
                    except Exception as exc:
                        failures.append((scenario.name, exc))
                        continue
                    completed[scenario.name] = result
                    _save_completed(scenario, result, fingerprints[scenario.name], identity,
                                    checkpoint_state, cache_state, checkpoint, cache)
            cancelled += len(missing) - next_index

    if stats is not None:
        stats.update({"requested": len(scenarios), "executed": requested - len(failures) - cancelled,
                      "reused": reused, "failed": len(failures), "cancelled": cancelled,
                      "workers": max_workers})
    if failures:
        partial_results = [completed[scenario.name] for scenario in scenarios
                           if scenario.name in completed]
        if len(failures) == 1 and isinstance(failures[0][1], ScenarioTimeoutError):
            error = failures[0][1]
            error.partial_results = partial_results
            error.execution_stats = dict(stats or {})
            raise error
        details = "; ".join(f"{name}: {error}" for name, error in failures)
        error = RunnerError(f"{len(failures)} scenario execution(s) failed: {details}")
        error.partial_results = partial_results
        error.execution_stats = dict(stats or {})
        raise error from failures[0][1]
    return [completed[scenario.name] for scenario in scenarios if scenario.name in completed]


def _execute_scenario(
    scenario: Scenario, runner: AgentRunner, control: ExecutionControl | None = None,
) -> AgentResult:
    started = perf_counter()
    sandbox = ToolSandbox(scenario)
    prompts = scenario.turns or (ScenarioTurn(input=scenario.input),)
    history: list[AgentTurnResult] = []
    for index, prompt in enumerate(prompts, start=1):
        if control is not None:
            control.raise_if_cancelled()
        try:
            method = runner.run_turn
            parameters = signature(method).parameters.values()
            accepts_tools = any(
                parameter.name == "tool_sandbox" or parameter.kind == Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            accepts_control = any(
                parameter.name == "execution_control" or parameter.kind == Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            kwargs = {"tool_sandbox": sandbox} if accepts_tools else {}
            if accepts_control and control is not None:
                kwargs["execution_control"] = control
            value = method(scenario, prompt.input, tuple(history), **kwargs)
        except Exception as exc:
            if control is not None:
                control.raise_if_timed_out()
                if control.cancel_event.is_set():
                    raise RunCancelled(f"run cancelled during scenario '{scenario.name}'") from exc
            if isinstance(exc, RunnerError):
                raise
            raise RunnerError(
                f"runner failed for scenario '{scenario.name}', turn {index}: {exc}"
            ) from exc
        if control is not None:
            control.raise_if_timed_out()
        history.append(_coerce_turn(value, scenario.name, index))
    metadata: dict[str, Any] = {"latency_ms": round((perf_counter() - started) * 1000, 3),
                                "tool_fixture_calls": len(sandbox.calls)}
    for field_name in ("input_tokens", "output_tokens", "cost_usd"):
        values = [turn.metadata[field_name] for turn in history
                  if isinstance(turn.metadata.get(field_name), (int, float))]
        if values:
            metadata[field_name] = sum(values)
    for field_name in ("model", "prompt_version"):
        values = [turn.metadata[field_name] for turn in history if turn.metadata.get(field_name) is not None]
        if values:
            metadata[field_name] = values[-1]
    return AgentResult(
        scenario=scenario.name, intent=history[-1].intent,
        asked=tuple(value for turn in history for value in turn.asked),
        tools=tuple(value for turn in history for value in turn.tools),
        actions=tuple(value for turn in history for value in turn.actions),
        tool_calls=tuple(value for turn in history for value in turn.tool_calls),
        events=tuple(value for turn in history for value in turn.events),
        answer=next((turn.answer for turn in reversed(history) if turn.answer is not None), None),
        notes="\n".join(turn.notes for turn in history if turn.notes),
        turns=tuple(history), metadata=metadata,
    )


def _runner_key(runner: AgentRunner) -> str:
    cls = type(runner)
    return f"{cls.__module__}.{cls.__qualname__}@{getattr(runner, 'version', 'unspecified')}"


def _scenario_fingerprint(scenario: Scenario) -> str:
    value = asdict(scenario)
    value.pop("source", None)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _empty_state(runner_key: str) -> dict[str, Any]:
    return {"format": "agent-eval-execution-state-v1", "runner_key": runner_key, "results": {}}


def _read_state(path: Path | None, runner_key: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return _empty_state(runner_key)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"could not read execution state {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != "agent-eval-execution-state-v1":
        raise RunnerError(f"execution state {path} has an unsupported format")
    if value.get("runner_key") != runner_key:
        return _empty_state(runner_key)
    if not isinstance(value.get("results"), dict):
        raise RunnerError(f"execution state {path} results must be an object")
    return value


def _matching_entry(state: dict[str, Any], name: str, fingerprint: str) -> dict[str, Any] | None:
    entry = state["results"].get(name)
    if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint:
        return entry
    return None


def _result_from_state(value: Any, scenario_name: str) -> AgentResult:
    if not isinstance(value, dict) or value.get("scenario") != scenario_name:
        raise RunnerError(f"cached result for '{scenario_name}' is malformed")
    def calls(rows):
        return tuple(ToolCall(row["name"], row.get("arguments", {})) for row in rows)
    def events(rows):
        return tuple(TraceEvent(row["type"], row["name"]) for row in rows)
    try:
        turns = tuple(AgentTurnResult(
            intent=row["intent"], asked=tuple(row.get("asked", [])),
            tools=tuple(row.get("tools", [])), actions=tuple(row.get("actions", [])),
            tool_calls=calls(row.get("tool_calls", [])), events=events(row.get("events", [])),
            answer=row.get("answer"), notes=row.get("notes", ""),
            metadata=row.get("metadata", {}),
        ) for row in value.get("turns", []))
        return AgentResult(
            scenario=scenario_name, intent=value["intent"],
            asked=tuple(value.get("asked", [])), tools=tuple(value.get("tools", [])),
            actions=tuple(value.get("actions", [])), tool_calls=calls(value.get("tool_calls", [])),
            events=events(value.get("events", [])), answer=value.get("answer"),
            notes=value.get("notes", ""), turns=turns, metadata=value.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError(f"cached result for '{scenario_name}' is malformed: {exc}") from exc


def _save_completed(
    scenario: Scenario, result: AgentResult, fingerprint: str, runner_key: str,
    checkpoint_state: dict[str, Any], cache_state: dict[str, Any],
    checkpoint: Path | None, cache: Path | None,
) -> None:
    from agent_eval.adapters import result_as_dict

    entry = {"fingerprint": fingerprint, "result": result_as_dict(result)}
    for state in (checkpoint_state, cache_state):
        state["results"][scenario.name] = entry
    for path, state in ((checkpoint, checkpoint_state), (cache, cache_state)):
        if path is not None:
            state["runner_key"] = runner_key
            _write_state(path, state)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        raise RunnerError(f"could not write execution state {path}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
