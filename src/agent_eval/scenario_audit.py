"""Governance checks for critical scenarios."""

from __future__ import annotations

from datetime import date
from typing import Any

from agent_eval.models import Scenario


def audit_scenarios(
    scenarios: list[Scenario], *, max_review_age_days: int = 90, as_of: date | None = None,
) -> dict[str, Any]:
    if max_review_age_days < 0:
        raise ValueError("maximum review age must be zero or greater")
    current_date = as_of or date.today()
    checked: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.severity != "critical":
            continue
        findings = []
        if not scenario.owner:
            findings.append("missing_owner")
        if not scenario.reviewed_at:
            findings.append("missing_review_date")
            age = None
        else:
            reviewed = date.fromisoformat(scenario.reviewed_at)
            age = (current_date - reviewed).days
            if age < 0:
                findings.append("review_date_in_future")
            elif age > max_review_age_days:
                findings.append("review_stale")
        checked.append({
            "scenario": scenario.name,
            "owner": scenario.owner,
            "severity": scenario.severity,
            "reviewed_at": scenario.reviewed_at,
            "review_age_days": age,
            "findings": findings,
            "passed": not findings,
        })
    return {
        "as_of": current_date.isoformat(),
        "max_review_age_days": max_review_age_days,
        "critical_scenarios_checked": len(checked),
        "passed": all(item["passed"] for item in checked),
        "scenarios": checked,
    }


def text_scenario_audit(report: dict[str, Any]) -> str:
    lines = [
        f"Critical scenario metadata audit: {'PASS' if report['passed'] else 'FAIL'}",
        f"As of: {report['as_of']} (maximum review age: {report['max_review_age_days']} days)",
        f"Critical scenarios checked: {report['critical_scenarios_checked']}",
    ]
    if not report["scenarios"]:
        lines.extend(["", "No scenarios marked critical."])
    for item in report["scenarios"]:
        status = "PASS" if item["passed"] else "FAIL"
        details = ", ".join(item["findings"]) or "review current"
        lines.append(
            f"  {status} {item['scenario']}: owner={item['owner'] or 'missing'}, "
            f"reviewed_at={item['reviewed_at'] or 'missing'} ({details})"
        )
    return "\n".join(lines)
