"""Portable, integrity-checked scenario baseline envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from agent_eval.baseline_history import snapshot_digest


PORTABLE_FORMAT = "agent-eval-scenario-baseline"


class PortableBaselineError(ValueError):
    """Raised when a portable baseline cannot be read, validated, or written."""


def export_portable_baseline(baseline_path: Path, output_path: Path) -> dict[str, Any]:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableBaselineError(f"could not read baseline {baseline_path}: {exc}") from exc
    validate_snapshot(baseline, str(baseline_path))
    envelope = {"format": PORTABLE_FORMAT, "format_version": 1,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_sha256": snapshot_digest(baseline), "snapshot": baseline}
    _write_json_atomic(output_path, envelope)
    return {"exported": True, "path": str(output_path), "scenario_set": baseline["scenario_set"],
            "version": baseline["version"], "snapshot_sha256": envelope["snapshot_sha256"]}


def import_portable_baseline(path: Path) -> dict[str, Any]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableBaselineError(f"could not read portable baseline {path}: {exc}") from exc
    if (not isinstance(envelope, dict) or envelope.get("format") != PORTABLE_FORMAT
            or isinstance(envelope.get("format_version"), bool) or envelope.get("format_version") != 1
            or not isinstance(envelope.get("snapshot"), dict)):
        raise PortableBaselineError(f"{path}: unsupported or malformed portable baseline")
    snapshot = envelope["snapshot"]
    validate_snapshot(snapshot, str(path))
    if envelope.get("snapshot_sha256") != snapshot_digest(snapshot):
        raise PortableBaselineError(f"{path}: snapshot digest does not match")
    return snapshot


def validate_snapshot(snapshot: Any, source: str) -> None:
    if (not isinstance(snapshot, dict) or not isinstance(snapshot.get("scenario_set"), str)
            or not snapshot.get("scenario_set", "").strip()
            or not isinstance(snapshot.get("version"), str)
            or not re.fullmatch(r"\d+\.\d+\.\d+", snapshot.get("version", ""))
            or not isinstance(snapshot.get("scenarios"), list)):
        raise PortableBaselineError(f"{source}: malformed scenario baseline snapshot")
    ids = set()
    for row in snapshot["scenarios"]:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in ids or not isinstance(row.get("version"), int)
                or isinstance(row.get("version"), bool) or row["version"] < 1
                or not _is_digest(row.get("contract_sha256"))
                or not _is_digest(row.get("input_sha256"))):
            raise PortableBaselineError(f"{source}: malformed or duplicate scenario row")
        ids.add(row["id"])


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise PortableBaselineError(f"could not write portable baseline {path}: {exc}") from exc
