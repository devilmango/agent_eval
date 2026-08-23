"""Aggregate repeated evaluations to show run-to-run variability."""

from __future__ import annotations

from statistics import NormalDist, mean, stdev
from typing import Any

from agent_eval.models import EvaluationReport


METRICS = (
    "intent_accuracy",
    "tool_accuracy",
    "policy_compliance",
    "unnecessary_question_rate",
    "answer_quality_score",
)


def _statistics(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "sample_stddev": stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials >= 1")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be greater than 0.5 and less than 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    variance = proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    margin = z * variance**0.5 / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def analyze_repeated_runs(
    reports: list[EvaluationReport], *, confidence: float = 0.95
) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("Repeated-run analysis requires at least two evaluation runs")

    scenario_names = [score.scenario for score in reports[0].scores]
    expected_names = set(scenario_names)
    for index, report in enumerate(reports[1:], start=2):
        names = {score.scenario for score in report.scores}
        if names != expected_names:
            raise ValueError(f"Run {index} has a different scenario set")

    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be greater than 0.5 and less than 1")
    pass_rates = [report.passed / report.total if report.total else 0.0 for report in reports]
    successes = sum(report.passed for report in reports)
    trials = sum(report.total for report in reports)
    lower, upper = wilson_interval(successes, trials, confidence) if trials else (0.0, 1.0)
    metrics: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        values = [getattr(report, metric) for report in reports]
        observed = [value for value in values if value is not None]
        metrics[metric] = _statistics(observed) if observed else {
            "mean": None, "sample_stddev": None, "min": None, "max": None
        }
    execution: dict[str, Any] = {}
    for metric in ("latency_ms", "input_tokens", "output_tokens", "cost_usd"):
        values = [report.execution.get(metric, {}).get("total") for report in reports]
        observed = [value for value in values if isinstance(value, (int, float))]
        execution[metric] = _statistics(observed) if observed else {"mean": None, "sample_stddev": None, "min": None, "max": None}

    scenario_rows: list[dict[str, Any]] = []
    for name in scenario_names:
        outcomes = [
            next(score.passed for score in report.scores if score.scenario == name)
            for report in reports
        ]
        passed = sum(outcomes)
        scenario_lower, scenario_upper = wilson_interval(passed, len(outcomes), confidence)
        scenario_rows.append(
            {
                "scenario": name,
                "passed_runs": passed,
                "total_runs": len(outcomes),
                "pass_rate": passed / len(outcomes),
                "confidence_interval": {
                    "lower": scenario_lower,
                    "upper": scenario_upper,
                },
                "failed_run_numbers": [index for index, outcome in enumerate(outcomes, start=1) if not outcome],
            }
        )

    return {
        "runs": len(reports),
        "scenario_count": len(scenario_names),
        "confidence": confidence,
        "pass_rate": {**_statistics(pass_rates), "confidence_interval": {"lower": lower, "upper": upper}},
        "run_pass_rates": pass_rates,
        "metrics": metrics,
        "execution": execution,
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
        f"Pass rate {report['confidence'] * 100:.0f}% Wilson interval: "
        f"{pass_stats['confidence_interval']['lower'] * 100:.1f}%–"
        f"{pass_stats['confidence_interval']['upper'] * 100:.1f}%",
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
            f"({scenario['pass_rate'] * 100:.1f}%; Wilson "
            f"{scenario['confidence_interval']['lower'] * 100:.1f}–"
            f"{scenario['confidence_interval']['upper'] * 100:.1f}%)"
        )
    return "\n".join(lines)


def _percent_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
