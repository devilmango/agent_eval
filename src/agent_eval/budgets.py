"""Execution-metric gates for CI and local evaluation runs."""

from __future__ import annotations

from math import ceil
from math import isfinite
from typing import Any

from agent_eval.models import AgentResult, EvaluationReport


def apply_budgets(
    report: EvaluationReport,
    results: dict[str, AgentResult],
    scenario_count: int,
    *, max_cost_usd: float | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_p95_latency_ms: float | None = None,
) -> tuple[EvaluationReport, bool]:
    limits = {
        "cost_usd": max_cost_usd,
        "input_tokens": max_input_tokens,
        "output_tokens": max_output_tokens,
        "p95_latency_ms": max_p95_latency_ms,
    }
    if all(value is None for value in limits.values()):
        return report, True
    gates: dict[str, Any] = {}
    passed = True
    for metric, limit in limits.items():
        if limit is None:
            continue
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not isfinite(limit) or limit < 0:
            raise ValueError(f"budget for {metric} must be non-negative")
        source_field = "latency_ms" if metric == "p95_latency_ms" else metric
        values = [result.metadata[source_field] for result in results.values()
                  if isinstance(result.metadata.get(source_field), (int, float))
                  and not isinstance(result.metadata.get(source_field), bool)
                  and isfinite(result.metadata.get(source_field))]
        missing = max(0, scenario_count - len(values))
        if metric == "p95_latency_ms" and values:
            ordered = sorted(values)
            observed = ordered[max(0, ceil(0.95 * len(ordered)) - 1)]
        else:
            observed = sum(values) if metric in {"cost_usd", "input_tokens", "output_tokens"} and values else (max(values) if values else None)
        gate_passed = observed is not None and missing == 0 and observed <= limit
        gates[metric] = {"limit": limit, "observed": observed, "missing_scenarios": missing,
                         "passed": gate_passed}
        passed = passed and gate_passed
    return EvaluationReport(
        total=report.total, passed=report.passed, failed=report.failed,
        intent_accuracy=report.intent_accuracy, tool_accuracy=report.tool_accuracy,
        policy_compliance=report.policy_compliance,
        unnecessary_question_rate=report.unnecessary_question_rate,
        answer_quality_score=report.answer_quality_score, scores=report.scores,
        execution={**report.execution, "budget_gates": gates},
    ), passed
