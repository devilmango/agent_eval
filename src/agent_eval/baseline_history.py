"""Append-only approval history for scenario snapshot baselines."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class BaselineHistoryError(ValueError):
    """Raised when a baseline approval log is invalid or cannot be written."""


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    """Hash the scenario snapshot while excluding its approval annotation."""
    content = {key: value for key, value in snapshot.items() if key != "approval"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_baseline_history(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BaselineHistoryError(f"could not read baseline history {path}: {exc}") from exc
    records = []
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BaselineHistoryError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if (not isinstance(record, dict) or record.get("schema_version") != 1
                or not isinstance(record.get("approval_id"), str)
                or not isinstance(record.get("recorded_at"), str)
                or record.get("event") not in {"approval", "restore", "import"}
                or not isinstance(record.get("snapshot"), dict)):
            raise BaselineHistoryError(f"{path}:{number}: malformed baseline history record")
        if record.get("snapshot_sha256") != snapshot_digest(record["snapshot"]):
            raise BaselineHistoryError(f"{path}:{number}: approved snapshot digest does not match")
        records.append(record)
    return records


def append_baseline_history(
    path: Path, snapshot: dict[str, Any], *, previous_snapshot: dict[str, Any] | None,
    event: str = "approval", approved_by: str | None = None,
    restored_from: str | None = None, imported_from: str | None = None,
) -> dict[str, Any]:
    """Record an approved snapshot and retain the full, input-free snapshot for recovery."""
    if event not in {"approval", "restore", "import"}:
        raise BaselineHistoryError("baseline history event must be approval, restore, or import")
    previous_rows = {row["id"]: row for row in previous_snapshot.get("scenarios", [])
                     if isinstance(row, dict) and isinstance(row.get("id"), str)} if previous_snapshot else {}
    current_rows = {row["id"]: row for row in snapshot.get("scenarios", [])
                    if isinstance(row, dict) and isinstance(row.get("id"), str)}
    added = sorted(set(current_rows) - set(previous_rows))
    removed = sorted(set(previous_rows) - set(current_rows))
    changed = sorted(name for name in set(current_rows) & set(previous_rows)
                     if current_rows[name].get("contract_sha256") != previous_rows[name].get("contract_sha256"))
    record = {
        "schema_version": 1,
        "approval_id": uuid4().hex,
        "event": event,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": approved_by,
        "scenario_set": snapshot.get("scenario_set"),
        "version": snapshot.get("version"),
        "version_diff": {
            "from_version": previous_snapshot.get("version") if previous_snapshot else None,
            "to_version": snapshot.get("version"),
            "scenario_set_changed": bool(previous_snapshot and previous_snapshot.get("scenario_set") != snapshot.get("scenario_set")),
            "added_scenarios": added,
            "removed_scenarios": removed,
            "changed_scenarios": changed,
        },
        "previous_snapshot_sha256": snapshot_digest(previous_snapshot) if previous_snapshot else None,
        "snapshot_sha256": snapshot_digest(snapshot),
        "restored_from": restored_from,
        "imported_from": imported_from,
        "snapshot": snapshot,
    }
    try:
        if path.exists():
            load_baseline_history(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
        return record
    except OSError as exc:
        raise BaselineHistoryError(f"could not append baseline history {path}: {exc}") from exc


def baseline_history_report(records: list[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise BaselineHistoryError("history limit must be at least 1")
    selected = records[-limit:] if limit is not None else list(records)
    if not selected:
        raise BaselineHistoryError("baseline history contains no approvals")
    return {"approval_count": len(selected), "approvals": [
        {key: record.get(key) for key in (
            "approval_id", "event", "recorded_at", "approved_by", "scenario_set", "version",
            "version_diff", "previous_snapshot_sha256", "snapshot_sha256", "restored_from", "imported_from",
        )}
        for record in selected
    ]}
