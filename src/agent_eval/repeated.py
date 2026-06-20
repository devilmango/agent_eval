"""Aggregate repeated evaluations to show run-to-run variability."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any

from agent_eval.models import EvaluationReport


METRICS = (
    "intent_accuracy",
    "tool_accuracy",
    "policy_compliance",
    "unnecessary_question_rate",
)


def _statistics(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "sample_stddev": stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def analyze_repeated_runs(reports: list[EvaluationReport]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("Repeated-run analysis requires at least two evaluation runs")

    scenario_names = [score.scenario for score in reports[0].scores]
    expected_names = set(scenario_names)
    for index, report in enumerate(reports[1:], start=2):
        names = {score.scenario for score in report.scores}
        if names != expected_names:
            raise ValueError(f"Run {index} has a different scenario set")

    pass_rates = [report.passed / report.total if report.total else 0.0 for report in reports]
    metrics: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        values = [getattr(report, metric) for report in reports]
        observed = [value for value in values if value is not None]
        metrics[metric] = _statistics(observed) if observed else {
            "mean": None, "sample_stddev": None, "min": None, "max": None
        }

    scenario_rows: list[dict[str, Any]] = []
    for name in scenario_names:
        outcomes = [
            next(score.passed for score in report.scores if score.scenario == name)
            for report in reports
        ]
        passed = sum(outcomes)
        scenario_rows.append(
            {
                "scenario": name,
                "passed_runs": passed,
                "total_runs": len(outcomes),
                "pass_rate": passed / len(outcomes),
                "failed_run_numbers": [index for index, outcome in enumerate(outcomes, start=1) if not outcome],
            }
        )

    return {
        "runs": len(reports),
        "scenario_count": len(scenario_names),
        "pass_rate": _statistics(pass_rates),
        "run_pass_rates": pass_rates,
        "metrics": metrics,
        "scenarios": scenario_rows,
    }


def text_repeated_report(report: dict[str, Any]) -> str:
    pass_stats = report["pass_rate"]
    lines = [
        f"Repeated evaluation: {report['runs']} runs, {report['scenario_count']} scenarios",
        "",
        f"Pass rate mean: {pass_stats['mean'] * 100:.1f}%",
        f"Pass rate sample stddev: {_percent_or_na(pass_stats['sample_stddev'])}",
        f"Pass rate range: {pass_stats['min'] * 100:.1f}%–{pass_stats['max'] * 100:.1f}%",
        "",
        "Metric variability (mean ± sample stddev):",
    ]
    for metric, values in report["metrics"].items():
        label = metric.replace("_", " ").capitalize()
        if values["mean"] is None:
            value = "n/a"
        else:
            value = f"{values['mean'] * 100:.1f}% ± {_percent_or_na(values['sample_stddev'])}"
        lines.append(f"  {label}: {value}")
    lines.extend(["", "Scenario pass rates:"])
    for scenario in report["scenarios"]:
        lines.append(
            f"  {scenario['scenario']}: {scenario['passed_runs']}/{scenario['total_runs']} "
            f"({scenario['pass_rate'] * 100:.1f}%)"
        )
    return "\n".join(lines)


def _percent_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"

