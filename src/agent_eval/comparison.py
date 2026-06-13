"""Compare evaluation reports from two agent versions."""

from __future__ import annotations

from typing import Any

from agent_eval.models import EvaluationReport


def compare_reports(baseline: EvaluationReport, candidate: EvaluationReport) -> dict[str, Any]:
    baseline_scores = {score.scenario: score for score in baseline.scores}
    candidate_scores = {score.scenario: score for score in candidate.scores}
    names = sorted(baseline_scores.keys() | candidate_scores.keys())

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    for name in names:
        before = baseline_scores.get(name)
        after = candidate_scores.get(name)
        if before is None or after is None:
            regressed.append(name)
        elif before.passed and not after.passed:
            regressed.append(name)
        elif not before.passed and after.passed:
            improved.append(name)
        else:
            unchanged.append(name)

    metrics = (
        "intent_accuracy",
        "tool_accuracy",
        "policy_compliance",
        "unnecessary_question_rate",
    )
    deltas: dict[str, float | None] = {}
    for metric in metrics:
        old = getattr(baseline, metric)
        new = getattr(candidate, metric)
        deltas[metric] = None if old is None or new is None else new - old

    return {
        "baseline": {"passed": baseline.passed, "failed": baseline.failed, "total": baseline.total},
        "candidate": {"passed": candidate.passed, "failed": candidate.failed, "total": candidate.total},
        "pass_delta": candidate.passed - baseline.passed,
        "metric_deltas": deltas,
        "improved_scenarios": improved,
        "regressed_scenarios": regressed,
        "unchanged_scenarios": unchanged,
    }


def text_comparison(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines = [
        "Evaluation comparison",
        "",
        f"Baseline:  {baseline['passed']}/{baseline['total']} passed ({baseline['failed']} failed)",
        f"Candidate: {candidate['passed']}/{candidate['total']} passed ({candidate['failed']} failed)",
        f"Pass delta: {comparison['pass_delta']:+d}",
        "",
        "Metric deltas (candidate − baseline):",
    ]
    for metric, delta in comparison["metric_deltas"].items():
        label = metric.replace("_", " ").capitalize()
        value = "n/a" if delta is None else f"{delta * 100:+.1f} percentage points"
        lines.append(f"  {label}: {value}")
    lines.extend(["", f"Improved: {len(comparison['improved_scenarios'])}", f"Regressed: {len(comparison['regressed_scenarios'])}"])
    if comparison["improved_scenarios"]:
        lines.append("  + " + ", ".join(comparison["improved_scenarios"]))
    if comparison["regressed_scenarios"]:
        lines.append("  - " + ", ".join(comparison["regressed_scenarios"]))
    return "\n".join(lines)
