"""Compare evaluation reports from two agent versions."""

from __future__ import annotations

import random
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
        "answer_quality_score",
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


def paired_bootstrap_intervals(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate uncertainty in paired scenario-level metric changes by bootstrap."""
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be greater than 0.5 and less than 1")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    before = {score.scenario: score for score in baseline.scores}
    after = {score.scenario: score for score in candidate.scores}
    common = sorted(before.keys() & after.keys())

    def answer_score(score: Any) -> float | None:
        values = [grade.score for grade in score.answer_grades]
        return sum(values) / len(values) if values else None

    pairs: dict[str, list[tuple[float, float]]] = {name: [] for name in (
        "pass_rate", "intent_accuracy", "tool_accuracy", "answer_quality_score",
    )}
    for name in common:
        left, right = before[name], after[name]
        pairs["pass_rate"].append((float(left.passed), float(right.passed)))
        pairs["intent_accuracy"].append((float(left.intent_correct), float(right.intent_correct)))
        if left.tool_correct is not None and right.tool_correct is not None:
            pairs["tool_accuracy"].append((float(left.tool_correct), float(right.tool_correct)))
        left_answer, right_answer = answer_score(left), answer_score(right)
        if left_answer is not None and right_answer is not None:
            pairs["answer_quality_score"].append((left_answer, right_answer))

    rng = random.Random(seed)
    alpha = (1 - confidence) / 2
    output = {}
    for metric, observations in pairs.items():
        count = len(observations)
        estimate = (sum(right - left for left, right in observations) / count) if count else None
        low = high = None
        if count >= 2:
            estimates = []
            for _ in range(samples):
                draw = [observations[rng.randrange(count)] for _ in range(count)]
                estimates.append(sum(right - left for left, right in draw) / count)
            estimates.sort()
            low = _quantile(estimates, alpha)
            high = _quantile(estimates, 1 - alpha)
        output[metric] = {
            "estimate": estimate,
            "lower": low,
            "upper": high,
            "confidence": confidence,
            "paired_scenarios": count,
        }
    return {"method": "paired scenario bootstrap", "samples": samples, "seed": seed,
            "matched_scenarios": len(common), "metrics": output}


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


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
    intervals = comparison.get("paired_bootstrap", {}).get("metrics", {})
    if intervals:
        lines.extend(["", "Paired bootstrap confidence intervals (candidate − baseline):"])
        for metric, interval in intervals.items():
            lower, upper = interval["lower"], interval["upper"]
            if lower is None or upper is None:
                value = f"n/a ({interval['paired_scenarios']} paired scenario(s))"
            else:
                value = f"{lower * 100:+.1f} to {upper * 100:+.1f} pp"
            lines.append(f"  {metric.replace('_', ' ').capitalize()}: {value}")
    lines.extend(["", f"Improved: {len(comparison['improved_scenarios'])}", f"Regressed: {len(comparison['regressed_scenarios'])}"])
    if comparison["improved_scenarios"]:
        lines.append("  + " + ", ".join(comparison["improved_scenarios"]))
    if comparison["regressed_scenarios"]:
        lines.append("  - " + ", ".join(comparison["regressed_scenarios"]))
    return "\n".join(lines)
