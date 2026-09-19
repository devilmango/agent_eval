"""Append-only evaluation history and trend summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from html import escape
from math import isfinite
import os
from pathlib import Path
import tempfile
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
    for key in ("critical_scenarios", "scenario_results"):
        values = record.get(key, {})
        if not isinstance(values, dict) or any(not isinstance(name, str) or not isinstance(passed, bool)
                                               for name, passed in values.items()):
            raise HistoryError(f"{source}: {key} must map scenario IDs to boolean outcomes")
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
    critical_scenarios: dict[str, bool] | None = None,
    scenario_results: dict[str, bool] | None = None,
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
        "critical_scenarios": dict(sorted((critical_scenarios or {}).items())),
        "scenario_results": dict(sorted((scenario_results or {}).items())),
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


def merge_history(input_paths: list[Path], output_path: Path) -> dict[str, Any]:
    """Merge history stores by run ID, rejecting duplicate IDs with conflicting results."""
    if not input_paths:
        raise HistoryError("history merge requires at least one input store")
    output_resolved = output_path.resolve()
    if any(path.resolve() == output_resolved for path in input_paths):
        raise HistoryError("history merge output must differ from every input store")
    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for path in input_paths:
        for record in load_history(path):
            run_id = record["run_id"]
            existing = unique.get(run_id)
            if existing is None:
                unique[run_id] = record
                continue
            duplicate_count += 1
            if _run_identity(existing) != _run_identity(record):
                raise HistoryError(f"conflicting history records share run ID {run_id}")
            if record["recorded_at"] > existing["recorded_at"]:
                unique[run_id] = record
    records = sorted(unique.values(), key=lambda item: (item["recorded_at"], item["run_id"]))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                                      dir=output_path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(temporary_name, output_path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise HistoryError(f"could not write merged history {output_path}: {exc}") from exc
    return {"input_count": len(input_paths), "record_count": len(records),
            "duplicate_count": duplicate_count, "records": records}


def _run_identity(record: dict[str, Any]) -> str:
    fields = ("total", "passed", "failed", "pass_rate", "metrics", "execution", "models",
              "prompt_versions", "critical_scenarios", "scenario_results")
    return json.dumps({field: record.get(field) for field in fields}, sort_keys=True, separators=(",", ":"))


def compare_history(records: list[dict[str, Any]], run_ids: list[str] | None = None,
                    labels: list[str] | None = None) -> dict[str, Any]:
    """Compare selected runs in their supplied order (IDs) or history order (labels)."""
    if not run_ids and not labels:
        raise HistoryError("history compare requires at least two --run-id or --label selections")
    by_id = {record["run_id"]: record for record in records}
    selected: list[dict[str, Any]] = []
    for run_id in run_ids or []:
        if run_id not in by_id:
            raise HistoryError(f"history run ID not found: {run_id}")
        selected.append(by_id[run_id])
    if labels:
        selected.extend(record for record in records if record.get("label") in labels)
    unique = {record["run_id"]: record for record in selected}
    if len(unique) < 2:
        raise HistoryError("history compare needs at least two distinct matching runs")
    # Explicit run IDs define their order; label selection uses chronological order.
    ordered_ids = []
    for run_id in run_ids or []:
        if run_id in unique and run_id not in ordered_ids:
            ordered_ids.append(run_id)
    ordered_ids.extend(record["run_id"] for record in records
                       if record["run_id"] in unique and record["run_id"] not in ordered_ids)
    runs = [unique[run_id] for run_id in ordered_ids]
    baseline = runs[0]
    dimensions = ["pass_rate", "cost_usd", "latency_ms"]
    deltas = {}
    for key in dimensions:
        before = _history_value(baseline, key)
        deltas[key] = {record["run_id"]: _subtract(before, _history_value(record, key)) for record in runs[1:]}
    scenario_names = sorted({name for run in runs for name in run.get("scenario_results", {})})
    scenario_comparisons = []
    for name in scenario_names:
        outcomes = {run["run_id"]: run.get("scenario_results", {}).get(name) for run in runs}
        regressions = [run["run_id"] for run in runs[1:]
                       if outcomes[baseline["run_id"]] is True and outcomes[run["run_id"]] is False]
        scenario_comparisons.append({"scenario": name, "outcomes": outcomes, "regressed_in": regressions,
                                     "critical": name in {critical for run in runs
                                                          for critical in run.get("critical_scenarios", {})}})
    return {"run_count": len(runs), "baseline": baseline, "runs": runs,
            "deltas_vs_baseline": deltas,
            "critical_scenarios": sorted({name for run in runs for name in run.get("critical_scenarios", {})}),
            "scenario_comparisons": scenario_comparisons}


def _history_value(record: dict[str, Any], key: str) -> float | None:
    if key == "pass_rate":
        value = record.get("pass_rate")
    else:
        execution = record.get("execution", {})
        value = execution.get(key) if isinstance(execution, dict) else None
        if isinstance(value, dict):
            value = value.get("total") if key == "cost_usd" else value.get("mean")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _subtract(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def text_history_comparison(report: dict[str, Any]) -> str:
    lines = [f"Evaluation comparison: {report['run_count']} runs",
             f"Baseline: {_run_label(report['baseline'])}", "",
             "Run                              Pass rate     Cost (USD)   Mean latency (ms)"]
    for run in report["runs"]:
        execution = run.get("execution", {})
        cost = _history_value(run, "cost_usd")
        latency = _history_value(run, "latency_ms")
        lines.append(f"{_run_label(run):32.32} "
                     f"{format_rate(run.get('pass_rate')):12} "
                     f"{cost if cost is not None else 'n/a':>12} "
                     f"{latency if latency is not None else 'n/a':>18}")
    lines.extend(["", "Change vs baseline:"])
    for key, title in (("pass_rate", "Pass rate"), ("cost_usd", "Cost (USD)"), ("latency_ms", "Mean latency (ms)")):
        for run in report["runs"][1:]:
            delta = report["deltas_vs_baseline"][key][run["run_id"]]
            rendered_delta = "n/a" if delta is None else f"{delta * 100:+.1f} pp" if key == "pass_rate" else f"{delta:+.4g}"
            lines.append(f"  {title} · {_run_label(run)}: {rendered_delta}")
    lines.extend(["", "Critical scenarios:"])
    for name in report["critical_scenarios"]:
        statuses = ["pass" if run.get("critical_scenarios", {}).get(name) is True else
                    "fail" if run.get("critical_scenarios", {}).get(name) is False else "n/a"
                    for run in report["runs"]]
        lines.append(f"  {name}: " + " | ".join(statuses))
    lines.extend(["", "Scenario outcomes:"])
    for item in report["scenario_comparisons"]:
        statuses = ["pass" if item["outcomes"].get(run["run_id"]) is True else
                    "fail" if item["outcomes"].get(run["run_id"]) is False else "n/a"
                    for run in report["runs"]]
        marker = " [critical]" if item["critical"] else ""
        regression = "  REGRESSION" if item["regressed_in"] else ""
        lines.append(f"  {item['scenario']}{marker}: " + " | ".join(statuses) + regression)
    return "\n".join(lines)


def html_history_comparison(report: dict[str, Any]) -> str:
    headers = "".join(f"<th scope=\"col\">{escape(_run_label(run))}</th>"
                      for run in report["runs"])
    rows = []
    metrics = [("Pass rate", lambda run: format_rate(run.get("pass_rate"))),
               ("Cost (USD)", lambda run: _display(_history_value(run, "cost_usd"))),
               ("Mean latency (ms)", lambda run: _display(_history_value(run, "latency_ms")))]
    for title, value in metrics:
        rows.append("<tr><th scope=\"row\">" + escape(title) + "</th>" +
                    "".join("<td>" + escape(str(value(run))) + "</td>" for run in report["runs"]) + "</tr>")
    for key, title in (("pass_rate", "Pass rate Δ"), ("cost_usd", "Cost Δ"), ("latency_ms", "Mean latency Δ")):
        cells = ["<td>baseline</td>"]
        for run in report["runs"][1:]:
            delta = report["deltas_vs_baseline"][key][run["run_id"]]
            rendered_delta = "n/a" if delta is None else f"{delta * 100:+.1f} pp" if key == "pass_rate" else f"{delta:+.4g}"
            cells.append("<td>" + escape(rendered_delta) + "</td>")
        rows.append("<tr><th scope=\"row\">" + escape(title) + "</th>" + "".join(cells) + "</tr>")
    for name in report["critical_scenarios"]:
        statuses = ["pass" if run.get("critical_scenarios", {}).get(name) is True else
                    "fail" if run.get("critical_scenarios", {}).get(name) is False else "n/a"
                    for run in report["runs"]]
        rows.append("<tr><th scope=\"row\">Critical: " + escape(name) + "</th>" +
                    "".join("<td>" + status + "</td>" for status in statuses) + "</tr>")
    for item in report["scenario_comparisons"]:
        cells = []
        for run in report["runs"]:
            outcome = item["outcomes"].get(run["run_id"])
            status = "pass" if outcome is True else "fail" if outcome is False else "n/a"
            if run["run_id"] in item["regressed_in"]:
                status += " · regression"
            cells.append("<td>" + escape(status) + "</td>")
        marker = " (critical)" if item["critical"] else ""
        rows.append("<tr><th scope=\"row\">Scenario: " + escape(item["scenario"] + marker) + "</th>" +
                    "".join(cells) + "</tr>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Agent Eval run comparison</title>"
            "<style>body{font:16px/1.5 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1100px;color:#17202a}"
            "table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.6rem;border-bottom:1px solid #d8dee4}"
            "thead{background:#f3f6f8}</style></head><body><h1>Evaluation run comparison</h1><table><thead><tr><th>Measure</th>"
            + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></body></html>")


def _display(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _run_label(record: dict[str, Any]) -> str:
    label = str(record.get("label") or "run")
    return f"{label} [{record['run_id'][:8]}]"


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


def filter_history(
    records: list[dict[str, Any]], *, models: list[str] | None = None,
    prompt_versions: list[str] | None = None, labels: list[str] | None = None,
    since: str | None = None, until: str | None = None,
) -> list[dict[str, Any]]:
    """Filter history records by metadata and inclusive ISO-8601 date bounds."""
    since_time = _parse_date_bound(since, "--since") if since else None
    until_time = _parse_date_bound(until, "--until", end_of_day=True) if until else None
    if since_time and until_time and since_time > until_time:
        raise HistoryError("--since must be earlier than or equal to --until")
    selected = []
    for record in records:
        try:
            recorded_time = datetime.fromisoformat(record["recorded_at"].replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise HistoryError("history record has an invalid recorded_at timestamp") from exc
        if recorded_time.tzinfo is None:
            recorded_time = recorded_time.replace(tzinfo=timezone.utc)
        recorded_time = recorded_time.astimezone(timezone.utc)
        record_models = record.get("models", [])
        record_prompts = record.get("prompt_versions", [])
        if models and not set(models).intersection(record_models):
            continue
        if prompt_versions and not set(prompt_versions).intersection(record_prompts):
            continue
        if labels and record.get("label") not in labels:
            continue
        if since_time and recorded_time < since_time:
            continue
        if until_time and recorded_time > until_time:
            continue
        selected.append(record)
    return selected


def _parse_date_bound(value: str, flag: str, *, end_of_day: bool = False) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoryError(f"{flag} must be an ISO-8601 date or timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if end_of_day and len(value) == 10:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed.astimezone(timezone.utc)


def html_trend_report(report: dict[str, Any]) -> str:
    """Render a self-contained, escaped HTML history report."""
    runs = report["runs"]
    first, latest = report["first"], report["latest"]
    metric_names = sorted(set(first["metrics"]) | set(latest["metrics"]))
    delta_rows = "".join(
        "<tr>" + f"<th scope=\"row\">{escape(name.replace('_', ' '))}</th>"
        + f"<td>{escape(_delta_text(report['deltas'].get(name)))}</td></tr>"
        for name in metric_names
    )
    rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        model_names = ", ".join(run.get("models", [])) or "—"
        prompt_names = ", ".join(run.get("prompt_versions", [])) or "—"
        rows.append(
            "<tr>"
            f"<td>{escape(run['recorded_at'])}</td>"
            f"<th scope=\"row\">{escape(str(run.get('label') or run['run_id']))}</th>"
            f"<td>{run['passed']}/{run['total']} ({escape(format_rate(run.get('pass_rate')))})</td>"
            f"<td>{escape(model_names)}</td><td>{escape(prompt_names)}</td>"
            + "".join(f"<td>{escape(format_rate(metrics.get(name)))}</td>" for name in metric_names)
            + "</tr>"
        )
    metric_headers = "".join(f"<th scope=\"col\">{escape(name.replace('_', ' '))}</th>" for name in metric_names)
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Eval history</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1200px;color:#17202a;background:#fff}
h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}caption{text-align:left;font-weight:600;margin:.5rem 0}
th,td{text-align:left;padding:.55rem .7rem;border-bottom:1px solid #d8dee4;vertical-align:top}thead{background:#f3f6f8}
.summary{display:flex;gap:2rem;flex-wrap:wrap}.summary p{margin:.3rem 0}
</style></head><body>
""" + f"""<header><p>Agent Eval</p><h1>Evaluation history</h1><p>{report['run_count']} matching runs</p></header>
<section><h2>Latest run</h2><div class="summary"><p><strong>Label:</strong> {escape(str(latest.get('label') or latest['run_id']))}</p>
<p><strong>Recorded:</strong> {escape(latest['recorded_at'])}</p><p><strong>Pass rate:</strong> {escape(format_rate(latest.get('pass_rate')))}</p></div></section>
<section><h2>Change across selected runs</h2><table><caption>Latest minus earliest retained run</caption><tbody>
<tr><th scope="row">Pass rate</th><td>{escape(_delta(first.get('pass_rate'), latest.get('pass_rate')))}</td></tr>{delta_rows}
</tbody></table></section>
<section><h2>Runs</h2><table><caption>Filtered evaluation history</caption><thead><tr><th scope="col">Recorded</th><th scope="col">Label / run ID</th><th scope="col">Pass rate</th><th scope="col">Models</th><th scope="col">Prompt versions</th>{metric_headers}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></section></body></html>"""


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
