"""Risk-based release gate profiles for repeated scenario evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_eval.models import EvaluationReport, Scenario


class GateProfileError(ValueError):
    """Raised when a release gate profile is invalid or does not match scenarios."""


def load_gate_profile(path: Path) -> dict[str, Any]:
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GateProfileError(f"could not read release gate profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise GateProfileError(f"{path}: release gate profile must be a YAML mapping")
    unknown = set(profile) - {"min_overall_pass_rate", "critical_tags", "critical_scenarios"}
    if unknown:
        raise GateProfileError(f"{path}: unsupported profile field(s): {', '.join(sorted(unknown))}")
    if not profile:
        raise GateProfileError(f"{path}: profile must define at least one release gate")
    if "min_overall_pass_rate" in profile:
        _threshold(profile["min_overall_pass_rate"], f"{path}: min_overall_pass_rate")
    for field in ("critical_tags", "critical_scenarios"):
        values = profile.get(field, {})
        if not isinstance(values, dict) or any(not isinstance(key, str) or not key.strip() for key in values):
            raise GateProfileError(f"{path}: {field} must be a mapping from IDs to minimum pass rates")
        for name, threshold in values.items():
            _threshold(threshold, f"{path}: {field}.{name}")
    if profile.get("min_overall_pass_rate") is None and not profile.get("critical_tags") and not profile.get("critical_scenarios"):
        raise GateProfileError(f"{path}: profile must define at least one threshold")
    return profile


def evaluate_release_gates(
    scenarios: list[Scenario], reports: list[EvaluationReport], profile: dict[str, Any],
) -> dict[str, Any]:
    if not reports:
        raise GateProfileError("release gating requires at least one evaluation run")
    expected = {scenario.name for scenario in scenarios}
    for index, report in enumerate(reports, start=1):
        names = {score.scenario for score in report.scores}
        if names != expected:
            raise GateProfileError(f"evaluation run {index} does not cover the selected scenario set")
    total_outcomes = sum(report.total for report in reports)
    passed_outcomes = sum(report.passed for report in reports)
    overall_rate = passed_outcomes / total_outcomes if total_outcomes else 0.0
    gates = []

    if "min_overall_pass_rate" in profile:
        minimum = profile["min_overall_pass_rate"]
        gates.append({"type": "overall_pass_rate", "target": "all scenarios",
                      "observed": overall_rate, "minimum": minimum,
                      "scenario_count": len(scenarios), "run_count": len(reports),
                      "passed": overall_rate >= minimum})

    for tag, minimum in profile.get("critical_tags", {}).items():
        tagged = [scenario for scenario in scenarios
                  if any(value.casefold() == tag.casefold() for value in scenario.tags)]
        if not tagged:
            raise GateProfileError(f"critical tag '{tag}' does not match any selected scenario")
        tagged_names = {scenario.name for scenario in tagged}
        outcomes = [score.passed for report in reports for score in report.scores
                    if score.scenario in tagged_names]
        observed = sum(outcomes) / len(outcomes) if outcomes else 0.0
        gates.append({"type": "critical_tag_pass_rate", "target": tag,
                      "observed": observed, "minimum": minimum,
                      "scenario_count": len(tagged), "run_count": len(reports),
                      "passed": observed >= minimum})

    score_names = {score.scenario for report in reports for score in report.scores}
    for name, minimum in profile.get("critical_scenarios", {}).items():
        if name not in score_names:
            raise GateProfileError(f"critical scenario '{name}' is not in the selected scenario set")
        outcomes = [next(score.passed for score in report.scores if score.scenario == name)
                    for report in reports]
        observed = sum(outcomes) / len(outcomes)
        gates.append({"type": "critical_scenario_pass_rate", "target": name,
                      "observed": observed, "minimum": minimum,
                      "scenario_count": 1, "run_count": len(reports),
                      "passed": observed >= minimum})

    return {"passed": all(gate["passed"] for gate in gates),
            "run_count": len(reports), "scenario_count": len(scenarios),
            "overall_pass_rate": overall_rate, "gates": gates}


def text_release_gate_report(report: dict[str, Any]) -> str:
    lines = [
        f"Release gate: {'PASS' if report['passed'] else 'FAIL'}",
        f"Overall pass rate: {report['overall_pass_rate'] * 100:.1f}% "
        f"across {report['run_count']} run(s) and {report['scenario_count']} scenario(s)",
        "",
        "Configured gates:",
    ]
    for gate in report["gates"]:
        lines.append(
            f"  {'PASS' if gate['passed'] else 'FAIL'} {gate['type']} {gate['target']}: "
            f"{gate['observed'] * 100:.1f}% (minimum {gate['minimum'] * 100:.1f}%)"
        )
    return "\n".join(lines)


def _threshold(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise GateProfileError(f"{label} must be a number between 0 and 1")
