"""Append-only plugin conformance history and release trend summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class ConformanceHistoryError(ValueError):
    """Raised for malformed conformance reports or history files."""


def _check_statuses(row: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for collection, prefix in ((row.get("checks", []), ""),
                               (row.get("fixture_cases", []), "fixture:")):
        if not isinstance(collection, list):
            raise ConformanceHistoryError("conformance checks and fixture_cases must be lists")
        for check in collection:
            if not isinstance(check, dict) or not isinstance(check.get("name"), str):
                raise ConformanceHistoryError("every conformance check needs a name")
            name = prefix + check["name"]
            status = check.get("status")
            if status not in {"passed", "failed", "skipped"}:
                status = "failed"
            if name in statuses:
                raise ConformanceHistoryError(f"duplicate conformance check name: {name}")
            statuses[name] = status
    return statuses


def _rows(report: dict[str, Any], plugin_version: str | None) -> list[dict[str, Any]]:
    plugin_type = report.get("plugin_type")
    plugin_name = report.get("plugin") or report.get("plugin_name")
    if plugin_type not in {"runner", "grader"} or not isinstance(plugin_name, str) or not plugin_name:
        raise ConformanceHistoryError("report needs plugin_type and plugin name")
    release = plugin_version or report.get("plugin_version") or "unknown"
    if not isinstance(release, str) or not release.strip():
        raise ConformanceHistoryError("plugin version must be a non-empty string")
    targets = report.get("targets")
    if targets is None:
        targets = [{
            "name": report.get("target", "default"),
            "python_version": report.get("python_version"),
            "agent_eval_version": report.get("agent_eval_version"),
            "passed": report.get("passed") is True,
            "checks": report.get("checks", []),
            "fixture_cases": report.get("fixture_cases", []),
        }]
    if not isinstance(targets, list) or not targets:
        raise ConformanceHistoryError("report needs a non-empty target or targets list")
    rows = []
    for target in targets:
        if not isinstance(target, dict) or not isinstance(target.get("name"), str):
            raise ConformanceHistoryError("every conformance target needs a name")
        rows.append({
            "schema_version": 1,
            "run_id": str(uuid4()),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "plugin_type": plugin_type,
            "plugin": plugin_name,
            "plugin_version": release,
            "target": target["name"],
            "python_version": target.get("python_version"),
            "agent_eval_version": target.get("agent_eval_version"),
            "passed": target.get("passed") is True,
            "checks": _check_statuses(target),
        })
    return rows


def record_conformance_report(
    report_path: Path, store_path: Path, *, plugin_version: str | None = None,
) -> dict[str, Any]:
    """Append sanitized check outcomes from a conformance or matrix report."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConformanceHistoryError(f"could not read conformance report {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ConformanceHistoryError("conformance report must be a JSON object")
    rows = _rows(report, plugin_version)
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        with store_path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise ConformanceHistoryError(f"could not append conformance history {store_path}: {exc}") from exc
    return {"store": str(store_path), "record_count": len(rows),
            "run_ids": [row["run_id"] for row in rows], "plugin_version": rows[0]["plugin_version"]}


def load_conformance_history(path: Path) -> list[dict[str, Any]]:
    """Read and validate append-only conformance history JSONL."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConformanceHistoryError(f"could not read conformance history {path}: {exc}") from exc
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConformanceHistoryError(f"invalid JSONL at {path}:{number}: {exc.msg}") from exc
        required = ("plugin_type", "plugin", "plugin_version", "target", "recorded_at", "checks")
        if not isinstance(row, dict) or any(field not in row for field in required):
            raise ConformanceHistoryError(f"invalid conformance history record at {path}:{number}")
        if not isinstance(row["checks"], dict):
            raise ConformanceHistoryError(f"checks must be a mapping at {path}:{number}")
        rows.append(row)
    return rows


def conformance_trend_report(
    rows: list[dict[str, Any]], *, plugin: str | None = None,
    target: str | None = None, limit: int | None = None,
) -> dict[str, Any]:
    """Summarize check regressions and intermittent outcomes across releases."""
    if limit is not None and limit < 1:
        raise ConformanceHistoryError("trend limit must be positive")
    selected = [row for row in rows
                if (plugin is None or row.get("plugin") == plugin)
                and (target is None or row.get("target") == target)]
    selected.sort(key=lambda row: (str(row.get("recorded_at", "")), str(row.get("run_id", ""))))
    if limit is not None:
        selected = selected[-limit:]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in selected:
        checks = row.get("checks", {})
        if not isinstance(checks, dict):
            raise ConformanceHistoryError("history record checks must be a mapping")
        for check_name, status in checks.items():
            if status not in {"passed", "failed", "skipped"}:
                raise ConformanceHistoryError(f"invalid status for check {check_name!r}: {status!r}")
            key = (str(row.get("plugin_type", "")), str(row.get("plugin", "")),
                   str(row.get("target", "")), str(check_name))
            groups.setdefault(key, []).append(row | {"_status": status})
    checks_report = []
    for (plugin_type, plugin_name, target_name, check_name), history in sorted(groups.items()):
        statuses = [item["_status"] for item in history]
        has_pass = "passed" in statuses
        has_fail = "failed" in statuses
        intermittent = has_pass and has_fail
        first_regression = next((index for index, status in enumerate(statuses)
                                 if status == "failed" and "passed" in statuses[:index]), None)
        first_failed = next((index for index, status in enumerate(statuses) if status == "failed"), None)
        intermittent_since = next((index for index in range(1, len(statuses))
                                   if len(set(statuses[:index + 1]) & {"passed", "failed"}) == 2), None)
        checks_report.append({
            "plugin_type": plugin_type, "plugin": plugin_name, "target": target_name,
            "check": check_name,
            "current_status": statuses[-1],
            "classification": "intermittent" if intermittent else
                              ("failing" if has_fail else ("passing" if has_pass else "skipped")),
            "run_count": len(history),
            "first_failure_version": (history[first_failed].get("plugin_version") if first_failed is not None else None),
            "first_failure_at": (history[first_failed].get("recorded_at") if first_failed is not None else None),
            "first_regression_version": (history[first_regression].get("plugin_version")
                                         if first_regression is not None else None),
            "intermittent_since_version": (history[intermittent_since].get("plugin_version")
                                            if intermittent_since is not None else None),
            "history": [{"plugin_version": item.get("plugin_version"),
                         "recorded_at": item.get("recorded_at"), "status": item["_status"],
                         "run_id": item.get("run_id")} for item in history],
        })
    return {
        "schema_version": 1,
        "record_count": len(selected),
        "plugin_filter": plugin,
        "target_filter": target,
        "check_count": len(checks_report),
        "intermittent_count": sum(item["classification"] == "intermittent" for item in checks_report),
        "regression_count": sum(item["first_regression_version"] is not None for item in checks_report),
        "checks": checks_report,
    }


def text_conformance_trend_report(report: dict[str, Any]) -> str:
    lines = [f"Conformance history: {report['record_count']} run target(s), "
             f"{report['check_count']} check timeline(s)",
             f"Regressions: {report['regression_count']}; intermittent: {report['intermittent_count']}"]
    for item in report["checks"]:
        state = item["classification"].upper()
        label = f"{item['plugin']}@{item['target']} / {item['check']}"
        detail = f"current {item['current_status']} across {item['run_count']} run(s)"
        if item["first_regression_version"]:
            detail += f"; first regression at {item['first_regression_version']}"
        if item["intermittent_since_version"]:
            detail += f"; intermittent since {item['intermittent_since_version']}"
        lines.append(f"  {state:>11}  {label}: {detail}")
    return "\n".join(lines)
