"""Human-readable and machine-readable report formatting."""

from __future__ import annotations

import json

from agent_eval.models import EvaluationReport


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def text_report(report: EvaluationReport, *, details: bool = True) -> str:
    lines = [
        f"{report.total} scenarios",
        "",
        f"Passed: {report.passed}",
        f"Failed: {report.failed}",
        "",
        f"Intent accuracy:       {format_percent(report.intent_accuracy)}",
        f"Tool accuracy:         {format_percent(report.tool_accuracy)}",
        f"Policy compliance:     {format_percent(report.policy_compliance)}",
        f"Unnecessary questions: {format_percent(report.unnecessary_question_rate)}",
    ]
    if details and report.failed:
        lines.extend(["", "Failures:"])
        for score in report.scores:
            if score.passed:
                continue
            reasons: list[str] = []
            if not score.intent_correct:
                reasons.append(f"intent {score.actual_intent!r}, expected {score.expected_intent!r}")
            if score.missing_questions:
                reasons.append("missing questions: " + ", ".join(score.missing_questions))
            if score.unnecessary_questions:
                reasons.append("unnecessary questions: " + ", ".join(score.unnecessary_questions))
            if score.tool_correct is False:
                reasons.append("tool calls differ from expected")
            if score.policy_violations:
                reasons.append("policy violations: " + ", ".join(score.policy_violations))
            if score.notes:
                reasons.append(score.notes)
            lines.append(f"  - {score.scenario}: {'; '.join(reasons) or 'missing result'}")
    return "\n".join(lines)


def json_report(report: EvaluationReport) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False)

