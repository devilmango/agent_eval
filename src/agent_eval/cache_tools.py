"""Inspection, compatibility checks, and pruning for execution result caches."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
import tempfile
from typing import Any

from agent_eval.models import Scenario
from agent_eval.runner import _scenario_fingerprint


class CacheError(ValueError):
    """Raised when an execution cache cannot safely be inspected or changed."""


_FORMAT = "agent-eval-execution-state-v1"


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"format": _FORMAT, "runner_key": None, "results": {}}
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheError(f"could not read cache {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != _FORMAT:
        raise CacheError(f"cache {path} has an unsupported format")
    if not isinstance(value.get("results"), dict):
        raise CacheError(f"cache {path} results must be an object")
    for name, entry in value["results"].items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise CacheError(f"cache {path} contains a malformed result entry")
    return value


def _timestamp(entry: dict[str, Any]) -> datetime | None:
    value = entry.get("updated_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def inspect_cache(path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Return a privacy-conscious inventory without exposing cached responses."""
    state = _read_cache(path)
    moment = now or datetime.now(timezone.utc)
    entries = []
    for name, entry in sorted(state["results"].items()):
        saved_at = _timestamp(entry)
        entries.append({
            "scenario": name,
            "fingerprint": entry.get("fingerprint"),
            "updated_at": saved_at.isoformat() if saved_at else None,
            "age_days": max(0.0, round((moment - saved_at).total_seconds() / 86400, 3)) if saved_at else None,
            "has_result": isinstance(entry.get("result"), dict),
        })
    return {
        "path": str(path), "format": state["format"], "runner_key": state.get("runner_key"),
        "entry_count": len(entries), "dated_entries": sum(item["updated_at"] is not None for item in entries),
        "unknown_age_entries": sum(item["updated_at"] is None for item in entries), "entries": entries,
    }


def explain_cache(path: Path, scenarios: list[Scenario], *, runner_key: str) -> dict[str, Any]:
    """Explain cache hits and misses for a planned scenario set."""
    state = _read_cache(path)
    runner_matches = state.get("runner_key") == runner_key
    per_scenario = []
    fingerprints = {scenario.name: _scenario_fingerprint(scenario) for scenario in scenarios}
    for scenario in scenarios:
        entry = state["results"].get(scenario.name)
        if not runner_matches:
            status, reason = "miss", "runner identity changed"
        elif entry is None:
            status, reason = "miss", "no cached scenario result"
        elif not isinstance(entry.get("fingerprint"), str) or entry.get("fingerprint") != fingerprints[scenario.name]:
            status, reason = "miss", "scenario fingerprint changed"
        elif not isinstance(entry.get("result"), dict):
            status, reason = "miss", "cached result is missing or malformed"
        else:
            status, reason = "hit", "runner identity and scenario fingerprint match"
        per_scenario.append({"scenario": scenario.name, "status": status, "reason": reason})
    return {
        "path": str(path), "compatible": runner_matches, "runner_key_matches": runner_matches,
        "cache_runner_key": state.get("runner_key"), "requested_runner_key": runner_key,
        "hit_count": sum(row["status"] == "hit" for row in per_scenario),
        "miss_count": sum(row["status"] == "miss" for row in per_scenario),
        "unused_entries": sorted(set(state["results"]) - set(fingerprints)),
        "scenarios": per_scenario,
    }


def _write_cache(path: Path, state: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(state, file, indent=2, ensure_ascii=False)
            file.write("\n")
        temporary.replace(path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise CacheError(f"could not write cache {path}: {exc}") from exc


def prune_cache(
    path: Path, *, older_than_days: float, dry_run: bool = False,
    include_unknown_age: bool = False, now: datetime | None = None,
) -> dict[str, Any]:
    """Remove entries older than the cutoff; legacy undated entries are kept by default."""
    if (isinstance(older_than_days, bool) or not isinstance(older_than_days, (int, float))
            or not isfinite(older_than_days) or older_than_days < 0):
        raise CacheError("older_than_days must be a non-negative number")
    state = _read_cache(path)
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(days=older_than_days)
    removed, retained_unknown = [], []
    keep = {}
    for name, entry in state["results"].items():
        saved_at = _timestamp(entry)
        if saved_at is None:
            if include_unknown_age:
                removed.append(name)
            else:
                retained_unknown.append(name)
                keep[name] = entry
        elif saved_at < cutoff:
            removed.append(name)
        else:
            keep[name] = entry
    remaining_count = len(keep)
    if not dry_run and removed:
        state["results"] = keep
        _write_cache(path, state)
    return {
        "path": str(path), "older_than_days": older_than_days,
        "cutoff": cutoff.isoformat(), "dry_run": dry_run,
        "removed": sorted(removed), "removed_count": len(removed),
        "retained_unknown_age": sorted(retained_unknown),
        "remaining_count": remaining_count,
    }
