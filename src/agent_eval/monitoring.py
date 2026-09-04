"""Comparison helpers for CI annotations, drift checks, and variant experiments."""

from __future__ import annotations

from math import ceil, isfinite
from typing import Any

from agent_eval.comparison import compare_reports
from agent_eval.models import AgentResult, EvaluationReport, Scenario
from agent_eval.scoring import evaluate

METRICS = (
    "intent_accuracy",
    "tool_accuracy",
    "policy_compliance",
    "unnecessary_question_rate",
    "answer_quality_score",
)


def github_annotation_report(
    comparison: dict[str, Any], candidate: EvaluationReport,
) -> tuple[str, str]:
    """Return GitHub workflow commands and a Markdown step summary."""
    scores = {score.scenario: score for score in candidate.scores}
    annotations = []
    for scenario in comparison["regressed_scenarios"]:
        score = scores.get(scenario)
        reasons = []
        if score is not None:
            if not score.intent_correct:
                reasons.append(f"intent {score.actual_intent!r}, expected {score.expected_intent!r}")
            if score.missing_questions:
                reasons.append("missing questions: " + ", ".join(score.missing_questions))
            if score.unnecessary_questions:
                reasons.append("unnecessary questions: " + ", ".join(score.unnecessary_questions))
            if score.tool_correct is False:
                reasons.extend(score.tool_call_mismatches or ("tool check failed",))
            if score.policy_violations:
                reasons.append("policy violations: " + ", ".join(score.policy_violations))
            reasons.extend(score.turn_violations)
        detail = "; ".join(reasons) or "scenario regressed or has no candidate result"
        annotations.append(_workflow_command("error", f"Scenario regression: {scenario}", detail))

    baseline = comparison["baseline"]
    current = comparison["candidate"]
    rows = [
        "## Agent Eval comparison",
        "",
        f"Baseline: **{baseline['passed']}/{baseline['total']} passed**; candidate: **{current['passed']}/{current['total']} passed**.",
        "",
        f"Regressed scenarios: **{len(comparison['regressed_scenarios'])}**; improved: **{len(comparison['improved_scenarios'])}**.",
        "",
        "| Metric | Change (candidate − baseline) |",
        "| --- | ---: |",
    ]
    for metric, delta in comparison["metric_deltas"].items():
        value = "n/a" if delta is None else f"{delta * 100:+.1f} pp"
        rows.append(f"| {metric.replace('_', ' ').capitalize()} | {value} |")
    if comparison["regressed_scenarios"]:
        rows.extend(["", "Regressed: " + ", ".join(comparison["regressed_scenarios"])])
    return "\n".join(annotations), "\n".join(rows) + "\n"


def _workflow_command(level: str, title: str, message: str) -> str:
    def escape_data(value: str) -> str:
        return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    escaped_title = escape_data(title).replace(":", "%3A").replace(",", "%2C")
    return f"::{level} title={escaped_title}::{escape_data(message)}"


def drift_report(
    baseline: EvaluationReport, current: EvaluationReport, *, max_absolute_drift: float,
) -> dict[str, Any]:
    """Compare current sampled behavior with a baseline and apply drift gates."""
    comparison = compare_reports(baseline, current)
    metrics = {}
    passed = not comparison["regressed_scenarios"]
    for metric, delta in comparison["metric_deltas"].items():
        drift = None if delta is None else abs(delta)
        gate_passed = (drift is None and item_is_unavailable(baseline, current, metric)) or (
            drift is not None and drift <= max_absolute_drift
        )
        metrics[metric] = {
            "baseline": getattr(baseline, metric),
            "current": getattr(current, metric),
            "delta": delta,
            "absolute_drift": drift,
            "limit": max_absolute_drift,
            "passed": gate_passed,
        }
        passed = passed and gate_passed
    return {
        "passed": passed,
        "max_absolute_drift": max_absolute_drift,
        "baseline": comparison["baseline"],
        "current": comparison["candidate"],
        "regressed_scenarios": comparison["regressed_scenarios"],
        "improved_scenarios": comparison["improved_scenarios"],
        "metrics": metrics,
    }


def experiment_matrix(
    scenarios: list[Scenario], variants: list[tuple[str, dict[str, AgentResult]]],
) -> dict[str, Any]:
    """Evaluate named result sets and summarize quality and execution metadata."""
    rows = []
    for name, results in variants:
        report = evaluate(scenarios, results, strict=False)
        costs = _metadata_values(results, "cost_usd")
        input_tokens = _metadata_values(results, "input_tokens")
        output_tokens = _metadata_values(results, "output_tokens")
        latencies = _metadata_values(results, "latency_ms")
        rows.append({
            "name": name,
            "models": sorted({str(result.metadata["model"]) for result in results.values()
                               if result.metadata.get("model") is not None}),
            "prompt_versions": sorted({str(result.metadata["prompt_version"]) for result in results.values()
                                        if result.metadata.get("prompt_version") is not None}),
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": report.passed / report.total if report.total else None,
            "metrics": {metric: getattr(report, metric) for metric in METRICS},
            "execution": {
                "cost_usd": sum(costs) if costs else None,
                "input_tokens": sum(input_tokens) if input_tokens else None,
                "output_tokens": sum(output_tokens) if output_tokens else None,
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                "p95_latency_ms": _nearest_rank(latencies, 0.95),
                "scenarios_with_metadata": sum(
                    1 for result in results.values() if any(key in result.metadata for key in (
                        "cost_usd", "input_tokens", "output_tokens", "latency_ms", "model", "prompt_version"
                    ))
                ),
            },
            "failed_scenarios": [score.scenario for score in report.scores if not score.passed],
        })
    return {"variant_count": len(rows), "scenario_count": len(scenarios), "variants": rows}


def _metadata_values(results: dict[str, AgentResult], key: str) -> list[float]:
    values = []
    for result in results.values():
        value = result.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            values.append(float(value))
    return values


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, ceil(quantile * len(values)) - 1)]


def item_is_unavailable(baseline: EvaluationReport, current: EvaluationReport, metric: str) -> bool:
    """An optional metric absent from both runs has no drift to gate."""
    return getattr(baseline, metric) is None and getattr(current, metric) is None


def read_variant_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError("variants must use NAME=RESULTS.jsonl syntax")
    return name.strip(), Path(path.strip())
