"""Aggregate repeated evaluations to show run-to-run variability."""

from __future__ import annotations

from collections import Counter
from statistics import NormalDist, mean, stdev
from typing import Any

from agent_eval.models import EvaluationReport
from agent_eval.models import Scenario


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
    reports: list[EvaluationReport], *, confidence: float = 0.95,
    scenarios: list[Scenario] | None = None,
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
    tag_lookup = {scenario.name: scenario.tags for scenario in scenarios or []}
    failure_totals: Counter[str] = Counter()
    for name in scenario_names:
        matching_scores = [
            next(score for score in report.scores if score.scenario == name)
            for report in reports
        ]
        outcomes = [score.passed for score in matching_scores]
        reasons: Counter[str] = Counter(
            reason for score in matching_scores for reason in _failure_reasons(score)
        )
        failure_totals.update(reasons)
        passed = sum(outcomes)
        scenario_lower, scenario_upper = wilson_interval(passed, len(outcomes), confidence)
        scenario_rows.append(
            {
                "scenario": name,
                "passed_runs": passed,
                "total_runs": len(outcomes),
                "pass_rate": passed / len(outcomes),
                "flaky": 0 < passed < len(outcomes),
                "tags": list(tag_lookup.get(name, ())),
                "failure_reasons": dict(sorted(reasons.items())),
                "confidence_interval": {
                    "lower": scenario_lower,
                    "upper": scenario_upper,
                },
                "failed_run_numbers": [index for index, outcome in enumerate(outcomes, start=1) if not outcome],
            }
        )

    by_tag = _scenario_slices(scenario_rows, lambda row: row["tags"])
    by_model = _run_slices(reports, scenario_names, "models")
    by_prompt = _run_slices(reports, scenario_names, "prompt_versions")

    return {
        "runs": len(reports),
        "scenario_count": len(scenario_names),
        "confidence": confidence,
        "pass_rate": {**_statistics(pass_rates), "confidence_interval": {"lower": lower, "upper": upper}},
        "run_pass_rates": pass_rates,
        "metrics": metrics,
        "execution": execution,
        "scenarios": scenario_rows,
        "flaky_scenarios": [row for row in scenario_rows if row["flaky"]],
        "failure_reasons": dict(sorted(failure_totals.items())),
        "by_tag": by_tag,
        "by_model": by_model,
        "by_prompt_version": by_prompt,
    }


def _failure_reasons(score: Any) -> list[str]:
    if score.passed:
        return []
    reasons = []
    if score.actual_intent == "(missing result)":
        reasons.append("missing_result")
    else:
        if not score.intent_correct:
            reasons.append("wrong_intent")
        if score.missing_questions:
            reasons.append("missing_required_question")
        if score.unnecessary_questions:
            reasons.append("unnecessary_question")
        if score.tool_correct is False:
            reasons.append("tool_mismatch")
        if score.policy_violations:
            reasons.append("policy_violation")
        if score.turn_violations:
            reasons.append("turn_violation")
        if any(not grade.passed for grade in score.answer_grades):
            reasons.append("answer_grader")
        if not reasons:
            reasons.append("other")
    return reasons


def _scenario_slices(
    scenario_rows: list[dict[str, Any]], selector,
) -> list[dict[str, Any]]:
    members: dict[str, list[dict[str, Any]]] = {}
    for row in scenario_rows:
        for key in selector(row):
            members.setdefault(key, []).append(row)
    return [
        {
            "slice": key,
            "scenario_count": len(rows),
            "passed_runs": sum(row["passed_runs"] for row in rows),
            "total_runs": sum(row["total_runs"] for row in rows),
            "pass_rate": sum(row["passed_runs"] for row in rows) / sum(row["total_runs"] for row in rows),
            "flaky_scenarios": [row["scenario"] for row in rows if row["flaky"]],
        }
        for key, rows in sorted(members.items())
    ]


def _run_slices(
    reports: list[EvaluationReport], scenario_names: list[str], field: str,
) -> list[dict[str, Any]]:
    run_groups: dict[str, list[EvaluationReport]] = {}
    for report in reports:
        for value in report.execution.get(field, []):
            run_groups.setdefault(str(value), []).append(report)
    slices = []
    for value, selected in sorted(run_groups.items()):
        outcomes_by_scenario: dict[str, list[bool]] = {name: [] for name in scenario_names}
        passed = 0
        total = 0
        for report in selected:
            for score in report.scores:
                outcomes_by_scenario[score.scenario].append(score.passed)
                passed += int(score.passed)
                total += 1
        slices.append({
            "slice": value,
            "run_count": len(selected),
            "scenario_count": len(scenario_names),
            "passed_runs": passed,
            "total_runs": total,
            "pass_rate": passed / total if total else 0.0,
            "flaky_scenarios": [
                name for name, outcomes in outcomes_by_scenario.items()
                if any(outcomes) and not all(outcomes)
            ],
        })
    return slices


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
    flaky = report["flaky_scenarios"]
    lines.extend(["", f"Flaky scenarios: {len(flaky)}"])
    for scenario in flaky:
        reasons = ", ".join(f"{name}={count}" for name, count in scenario["failure_reasons"].items()) or "no classified reason"
        lines.append(f"  {scenario['scenario']}: {reasons}")
    for heading, key in (("By tag", "by_tag"), ("By model", "by_model"), ("By prompt version", "by_prompt_version")):
        if report[key]:
            lines.extend(["", heading + ":"])
            for item in report[key]:
                lines.append(
                    f"  {item['slice']}: {item['pass_rate'] * 100:.1f}% pass rate; "
                    f"{len(item['flaky_scenarios'])} flaky scenario(s)"
                )
    return "\n".join(lines)


def _percent_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
