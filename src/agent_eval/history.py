"""Append-only evaluation history and trend summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4


class HistoryError(ValueError):
    """Raised when an evaluation history file is malformed or cannot be written."""


def _validate_record(record: Any, source: str) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise HistoryError(f"{source}: unsupported or malformed history record")
    if not isinstance(record.get("recorded_at"), str) or not isinstance(record.get("metrics"), dict):
        raise HistoryError(f"{source}: history record needs recorded_at and metrics")
    if any(not isinstance(record.get(key), int) for key in ("total", "passed", "failed")):
        raise HistoryError(f"{source}: history record needs integer total, passed, and failed counts")
    if any(isinstance(record.get(key), bool) for key in ("total", "passed", "failed")):
        raise HistoryError(f"{source}: history counts cannot be booleans")
    if any(value is not None and (
        not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value)
    ) for value in record["metrics"].values()):
        raise HistoryError(f"{source}: metric values must be numbers or null")
    return record


def load_history(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HistoryError(f"could not read history {path}: {exc}") from exc
    records = []
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HistoryError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        records.append(_validate_record(item, f"{path}:{number}"))
    return sorted(records, key=lambda item: item["recorded_at"])


def record_history(
    path: Path,
    report: dict[str, Any],
    *,
    label: str | None = None,
    source_type: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Append a compact report summary without storing scenarios or trace contents."""
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise HistoryError("evaluation report is missing its metrics object")
    counts = {key: report.get(key) for key in ("total", "passed", "failed")}
    if any(not isinstance(value, int) for value in counts.values()):
        raise HistoryError("evaluation report is missing valid total/pass/fail counts")
    execution = report.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    record = {
        "schema_version": 1,
        "run_id": run_id or uuid4().hex,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "source": source_type,
        **counts,
        "pass_rate": counts["passed"] / counts["total"] if counts["total"] else None,
        "metrics": {key: metrics.get(key) for key in (
            "intent_accuracy", "tool_accuracy", "policy_compliance",
            "unnecessary_question_rate", "answer_quality_score",
        )},
        "execution": {
            key: execution.get(key)
            for key in ("cost_usd", "input_tokens", "output_tokens", "latency_ms")
            if key in execution
        },
        "models": execution.get("models", []),
        "prompt_versions": execution.get("prompt_versions", []),
    }
    _validate_record(record, str(path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Validate the whole existing file before adding a record. This prevents
        # extending a damaged history with more data that would be hard to recover.
        if path.exists():
            load_history(path)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        raise HistoryError(f"could not append history {path}: {exc}") from exc
    return record


def trend_report(records: list[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise HistoryError("history limit must be at least 1")
    selected = records[-limit:] if limit is not None else list(records)
    if not selected:
        raise HistoryError("history contains no evaluation records")
    first, latest = selected[0], selected[-1]
    deltas = {}
    metric_names = sorted(set(first["metrics"]) | set(latest["metrics"]))
    for metric in metric_names:
        before, after = first["metrics"].get(metric), latest["metrics"].get(metric)
        deltas[metric] = None if before is None or after is None else after - before
    return {
        "run_count": len(selected),
        "first": first,
        "latest": latest,
        "deltas": deltas,
        "runs": selected,
    }


def text_trend_report(report: dict[str, Any]) -> str:
    first, latest = report["first"], report["latest"]
    lines = [
        f"Evaluation history: {report['run_count']} runs",
        "",
        f"Latest: {latest.get('label') or latest['run_id']} at {latest['recorded_at']}",
        f"Pass rate: {format_rate(latest.get('pass_rate'))} ({latest['passed']}/{latest['total']})",
        "",
        f"Trend ({first.get('label') or first['recorded_at']} → {latest.get('label') or latest['recorded_at']}):",
    ]
    pass_delta = _delta(first.get("pass_rate"), latest.get("pass_rate"))
    lines.append(f"  Pass rate: {pass_delta}")
    for metric, delta in report["deltas"].items():
        lines.append(f"  {metric.replace('_', ' ')}: {_delta_text(delta)}")
    lines.extend(["", "Recent runs:"])
    for run in report["runs"]:
        lines.append(
            f"  {run['recorded_at']}  {run.get('label') or run['run_id']}  "
            f"{run['passed']}/{run['total']} passed ({format_rate(run.get('pass_rate'))})"
        )
    return "\n".join(lines)


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _delta(before: float | None, after: float | None) -> str:
    return _delta_text(None if before is None or after is None else after - before)


def _delta_text(delta: float | None) -> str:
    return "n/a" if delta is None else f"{delta * 100:+.1f} pp"
