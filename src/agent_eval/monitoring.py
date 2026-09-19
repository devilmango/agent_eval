"""Comparison helpers for CI annotations, drift checks, and variant experiments."""

from __future__ import annotations

from math import ceil, isfinite
from pathlib import Path
from typing import Any

from agent_eval.comparison import compare_reports, paired_bootstrap_intervals
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
    intervals = comparison.get("paired_bootstrap", {}).get("metrics", {})
    if intervals:
        rows.extend(["", "Paired bootstrap intervals:", "", "| Metric | Estimate | Confidence interval | Paired scenarios |", "| --- | ---: | ---: | ---: |"])
        for metric, interval in intervals.items():
            estimate = interval["estimate"]
            lower, upper = interval["lower"], interval["upper"]
            estimate_text = "n/a" if estimate is None else f"{estimate * 100:+.1f} pp"
            interval_text = "n/a" if lower is None else f"{lower * 100:+.1f} to {upper * 100:+.1f} pp"
            rows.append(f"| {metric.replace('_', ' ').capitalize()} | {estimate_text} | {interval_text} | {interval['paired_scenarios']} |")
    if comparison["regressed_scenarios"]:
        rows.extend(["", "Regressed: " + ", ".join(comparison["regressed_scenarios"])])
    return "\n".join(annotations), "\n".join(rows) + "\n"


def github_scenario_impact_report(report: dict[str, Any]) -> tuple[str, str]:
    """Render scenario change impact as GitHub workflow annotations and Markdown."""
    annotations = []
    changed_names = {item["id"] for item in report.get("details", [])}
    for item in report.get("details", []):
        name = item["id"]
        dimensions = item.get("changed_dimensions", [])
        reason = item.get("change", "changed")
        if dimensions:
            reason += " behavior: " + ", ".join(dimensions)
        matching_gates = [gate["target"] for gate in report.get("gates", [])
                          if gate.get("affected") and (
                              gate.get("type") == "overall_pass_rate"
                              or name in gate.get("scenarios", [])
                              or (gate.get("type") == "critical_scenario_pass_rate" and gate.get("target") == name)
                          )]
        if matching_gates:
            reason += "; may affect release gates: " + ", ".join(matching_gates)
        level = "warning" if matching_gates else "notice"
        annotations.append(_workflow_command(level, f"Scenario impact: {name}", reason))

    lines = ["## Agent Eval scenario change impact", "",
             f"Scenario set: **{_markdown_cell(str(report['scenario_set']))}** "
             f"({report['baseline_version']} → {report['current_version']})",
             f"Changes: **{len(report['added'])} added**, **{len(report['removed'])} removed**, **{len(report['changed'])} changed**; "
             f"affected release gates: **{report['affected_gate_count']}**.",
             f"Recommended version: **{report['recommended_version']}** ({report['recommended_bump']}).", ""]
    if changed_names:
        lines.extend(["| Scenario | Change | Behavior dimensions |", "| --- | --- | --- |"])
        for item in report["details"]:
            dimensions = ", ".join(item.get("changed_dimensions", [])) or "—"
            lines.append(f"| {_markdown_cell(item['id'])} | {_markdown_cell(item['change'])} | {_markdown_cell(dimensions)} |")
    else:
        lines.append("No scenario contracts changed.")
    gates = report.get("gates", [])
    lines.extend(["", "### Release gates", ""])
    if gates:
        lines.extend(["| Gate | Status | Minimum | Scenarios |", "| --- | --- | ---: | --- |"])
        for gate in gates:
            names = ", ".join(gate.get("scenarios", [])) or gate["target"]
            status = "affected" if gate.get("affected") else "unchanged"
            lines.append(f"| {_markdown_cell(str(gate['type']) + ': ' + str(gate['target']))} | {status} | "
                         f"{_markdown_cell(str(gate.get('minimum', '—')))} | {_markdown_cell(names)} |")
    else:
        lines.append("No release gates are configured.")
    return "\n".join(annotations), "\n".join(lines) + "\n"


def _workflow_command(level: str, title: str, message: str) -> str:
    def escape_data(value: str) -> str:
        return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    escaped_title = escape_data(title).replace(":", "%3A").replace(",", "%2C")
    return f"::{level} title={escaped_title}::{escape_data(message)}"


def _markdown_cell(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("|", "\\|").replace("\r", " ").replace("\n", " "))


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
    *, confidence: float = 0.95, samples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Evaluate named result sets and summarize quality and execution metadata."""
    rows = []
    evaluations = []
    for name, results in variants:
        report = evaluate(scenarios, results, strict=False)
        evaluations.append(report)
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
    baseline_name = variants[0][0] if variants else None
    comparisons = []
    if evaluations:
        for index in range(1, len(evaluations)):
            comparison = compare_reports(evaluations[0], evaluations[index])
            comparisons.append({
                "baseline": baseline_name,
                "candidate": variants[index][0],
                "metric_deltas": comparison["metric_deltas"],
                "regressed_scenarios": comparison["regressed_scenarios"],
                "improved_scenarios": comparison["improved_scenarios"],
                "paired_bootstrap": paired_bootstrap_intervals(
                    evaluations[0], evaluations[index], confidence=confidence,
                    samples=samples, seed=seed,
                ),
            })
    return {"variant_count": len(rows), "scenario_count": len(scenarios),
            "baseline_variant": baseline_name, "variants": rows, "comparisons": comparisons}


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
