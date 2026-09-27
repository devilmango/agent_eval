"""Release gates evaluated over recorded run history."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from agent_eval.history import HistoryError


_SUPPORTED_FIELDS = {
    "min_runs", "min_average_pass_rate", "min_latest_pass_rate",
    "max_pass_rate_drop", "critical_scenarios", "max_average_cost_usd",
    "max_latest_cost_usd", "max_average_latency_ms", "max_latest_latency_ms",
}
_EXECUTION_LIMITS = {
    "max_average_cost_usd": ("cost_usd", "total", "average_cost_usd"),
    "max_latest_cost_usd": ("cost_usd", "total", "latest_cost_usd"),
    "max_average_latency_ms": ("latency_ms", "mean", "average_latency_ms"),
    "max_latest_latency_ms": ("latency_ms", "mean", "latest_latency_ms"),
}


def load_history_gate_profile(path: Path) -> dict[str, Any]:
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HistoryError(f"could not read history gate profile {path}: {exc}") from exc
    if not isinstance(profile, dict) or not profile:
        raise HistoryError(f"{path}: history gate profile must be a non-empty YAML mapping")
    unknown = set(profile) - _SUPPORTED_FIELDS
    if unknown:
        raise HistoryError(f"{path}: unsupported history gate field(s): {', '.join(sorted(unknown))}")
    min_runs = profile.get("min_runs", 1)
    if isinstance(min_runs, bool) or not isinstance(min_runs, int) or min_runs < 1:
        raise HistoryError(f"{path}: min_runs must be a positive integer")
    for name in ("min_average_pass_rate", "min_latest_pass_rate", "max_pass_rate_drop"):
        if name in profile:
            _rate(profile[name], f"{path}: {name}")
    for name in _EXECUTION_LIMITS:
        if name in profile:
            _nonnegative(profile[name], f"{path}: {name}")
    critical = profile.get("critical_scenarios", {})
    if (not isinstance(critical, dict) or any(not isinstance(name, str) or not name.strip()
                                              for name in critical)):
        raise HistoryError(f"{path}: critical_scenarios must map non-empty scenario IDs to pass rates")
    for name, threshold in critical.items():
        _rate(threshold, f"{path}: critical_scenarios.{name}")
    profile["min_runs"] = min_runs
    profile["critical_scenarios"] = critical
    return profile


def _rate(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise HistoryError(f"{label} must be a number between 0 and 1")


def _nonnegative(value: Any, label: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0):
        raise HistoryError(f"{label} must be a finite non-negative number")


def evaluate_history_gates(records: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    """Check run count, recent quality, trend movement, and named scenario history."""
    rates = [record.get("pass_rate") for record in records]
    observed_rates = [rate for rate in rates if isinstance(rate, (int, float)) and not isinstance(rate, bool)]
    average = sum(observed_rates) / len(observed_rates) if observed_rates else None
    latest = rates[-1] if rates and isinstance(rates[-1], (int, float)) and not isinstance(rates[-1], bool) else None
    drop = (rates[0] - rates[-1] if len(rates) >= 2
            and isinstance(rates[0], (int, float)) and not isinstance(rates[0], bool)
            and isinstance(rates[-1], (int, float)) and not isinstance(rates[-1], bool) else None)
    gates = [{"type": "minimum_runs", "minimum": profile["min_runs"],
              "observed": len(records), "passed": len(records) >= profile["min_runs"]}]
    for configured_name, (metric, statistic, gate_name) in _EXECUTION_LIMITS.items():
        if configured_name not in profile:
            continue
        maximum = profile[configured_name]
        values = [_execution_value(record, metric, statistic) for record in records]
        observations = [value for value in values if value is not None]
        is_average = configured_name.startswith("max_average_")
        observed = (sum(observations) / len(observations) if is_average and observations
                    else values[-1] if not is_average and values else None)
        expected_observations = len(records) if is_average else (1 if records else 0)
        observation_count = len(observations) if is_average else int(observed is not None)
        complete = (bool(records) and len(observations) == len(records) if is_average
                    else bool(records) and observed is not None)
        gates.append({
            "type": gate_name,
            "maximum": maximum,
            "observed": observed,
            "observations": observation_count,
            "expected_observations": expected_observations,
            "passed": complete and observed <= maximum if observed is not None else False,
        })
    if "min_average_pass_rate" in profile:
        minimum = profile["min_average_pass_rate"]
        gates.append({"type": "average_pass_rate", "minimum": minimum, "observed": average,
                      "observations": len(observed_rates), "expected_observations": len(records),
                      "passed": len(observed_rates) == len(records) and average is not None and average >= minimum})
    if "min_latest_pass_rate" in profile:
        minimum = profile["min_latest_pass_rate"]
        gates.append({"type": "latest_pass_rate", "minimum": minimum, "observed": latest,
                      "passed": latest is not None and latest >= minimum})
    if "max_pass_rate_drop" in profile:
        maximum = profile["max_pass_rate_drop"]
        gates.append({"type": "max_pass_rate_drop", "maximum": maximum, "observed": drop,
                      "passed": drop is not None and drop <= maximum})
    for name, minimum in profile["critical_scenarios"].items():
        outcomes = [record.get("scenario_results", {}).get(name) for record in records]
        passed_count = sum(outcome is True for outcome in outcomes)
        complete = len(outcomes) == len(records) and all(isinstance(outcome, bool) for outcome in outcomes)
        observed = passed_count / len(records) if records else None
        gates.append({"type": "critical_scenario_pass_rate", "target": name,
                      "minimum": minimum, "observed": observed,
                      "observations": sum(isinstance(outcome, bool) for outcome in outcomes),
                      "expected_observations": len(records),
                      "passed": complete and observed is not None and observed >= minimum})
    return {"passed": all(gate["passed"] for gate in gates), "run_count": len(records),
            "average_pass_rate": average, "latest_pass_rate": latest,
            "pass_rate_drop": drop, "gates": gates,
            "runs": [{"run_id": record.get("run_id"), "recorded_at": record["recorded_at"],
                      "label": record.get("label"), "pass_rate": record.get("pass_rate")}
                     for record in records]}


def text_history_gate_report(report: dict[str, Any]) -> str:
    lines = [f"History release gate: {'PASS' if report['passed'] else 'FAIL'}",
             f"Runs evaluated: {report['run_count']}",
             f"Average pass rate: {_display(report['average_pass_rate'])}",
             f"Latest pass rate: {_display(report['latest_pass_rate'])}", "", "Configured gates:"]
    for gate in report["gates"]:
        value = gate.get("observed")
        is_execution = gate["type"].endswith(("cost_usd", "latency_ms"))
        shown = (str(value) if gate["type"] == "minimum_runs" else
                 f"{value:.6g}" if is_execution and isinstance(value, (int, float)) else
                 f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "n/a")
        threshold = gate.get("minimum", gate.get("maximum"))
        rule = "maximum" if "maximum" in gate else "minimum"
        expected = (str(threshold) if is_execution else
                    f"{threshold * 100:.1f}%" if gate["type"] != "minimum_runs" else str(threshold))
        target = f" {gate['target']}" if gate.get("target") else ""
        lines.append(f"  {'PASS' if gate['passed'] else 'FAIL'} {gate['type']}{target}: "
                     f"{shown} ({rule} {expected})")
    return "\n".join(lines)


def _display(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value * 100:.1f}%"


def _execution_value(record: dict[str, Any], metric: str, statistic: str) -> float | None:
    execution = record.get("execution")
    summary = execution.get(metric) if isinstance(execution, dict) else None
    value = summary.get(statistic) if isinstance(summary, dict) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0):
        return None
    return float(value)
