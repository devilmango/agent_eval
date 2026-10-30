"""Repeat previously failing scenarios and classify failure reproducibility."""

from __future__ import annotations

from inspect import Parameter, signature
from collections.abc import Callable
from typing import Any

from agent_eval.models import AgentResult, Scenario
from agent_eval.runner import AgentRunner, execute_scenarios
from agent_eval.scoring import score_scenario


def _failure_reasons(score: Any) -> list[str]:
    if score.passed:
        return []
    return sorted({diagnostic.category for diagnostic in score.diagnostics}) or ["scenario"]


def investigate_failed_scenarios(
    scenarios: list[Scenario], baseline_results: dict[str, AgentResult], runner: AgentRunner, *,
    repetitions: int, seed: int = 0, scenario_timeout: float | None = None,
    runner_factory: Callable[[], AgentRunner] | None = None,
) -> dict[str, Any]:
    """Rerun only baseline failures with sequential, reproducible seeds."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("flake investigation requires at least two repetitions")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if scenario_timeout is not None and (
        isinstance(scenario_timeout, bool) or not isinstance(scenario_timeout, (int, float))
        or scenario_timeout <= 0
    ):
        raise ValueError("scenario_timeout must be a positive number")
    try:
        parameters = signature(runner.run_turn).parameters.values()
        seed_supported = any(parameter.name == "execution_control"
                             or parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters)
    except (TypeError, ValueError):
        seed_supported = False

    rows = []
    by_name = {scenario.name: scenario for scenario in scenarios}
    for name, baseline in baseline_results.items():
        scenario = by_name.get(name)
        if scenario is None:
            continue
        baseline_score = score_scenario(scenario, baseline, skip_semantic_judges=True)
        if baseline_score.passed:
            continue
        attempts = []
        for index in range(repetitions):
            attempt_seed = seed + index
            try:
                active_runner = runner_factory() if runner_factory is not None else runner
                [result] = execute_scenarios(
                    [scenario], active_runner, scenario_timeout=scenario_timeout, seed=attempt_seed,
                )
                score = score_scenario(scenario, result, skip_semantic_judges=True)
                attempts.append({
                    "run": index + 1, "seed": attempt_seed, "passed": score.passed,
                    "failure_reasons": _failure_reasons(score),
                    "actual_intent": result.intent,
                })
            except Exception as exc:
                attempts.append({"run": index + 1, "seed": attempt_seed, "passed": False,
                                 "failure_reasons": ["execution_error"], "error": str(exc)})
        passed_runs = sum(attempt["passed"] for attempt in attempts)
        failed_runs = len(attempts) - passed_runs
        classification = (
            "consistent" if failed_runs == len(attempts)
            else "unreproduced" if passed_runs == len(attempts)
            else "intermittent"
        )
        rows.append({
            "scenario": name, "classification": classification,
            "baseline_failure_reasons": _failure_reasons(baseline_score),
            "passed_runs": passed_runs, "failed_runs": failed_runs,
            "seeds": [attempt["seed"] for attempt in attempts], "attempts": attempts,
        })
    return {
        "repetitions": repetitions, "base_seed": seed,
        "seed_delivery": "execution_control.seed",
        "runner_accepts_execution_control": seed_supported,
        "scoring_mode": "deterministic; semantic judge graders skipped",
        "scenario_count": len(rows), "scenarios": rows,
        "classification_counts": {
            classification: sum(row["classification"] == classification for row in rows)
            for classification in ("consistent", "intermittent", "unreproduced")
        },
    }
