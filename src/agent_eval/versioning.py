"""Scenario-set snapshots, compatibility checks, and manifest migrations."""

from __future__ import annotations

import hashlib
import json
import re
import re
from pathlib import Path
from typing import Any

import yaml

from agent_eval.models import Scenario


MANIFEST_NAME = "scenario-set.yaml"
CURRENT_SCHEMA_VERSION = 1
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class VersioningError(ValueError):
    """Raised for invalid version manifests or snapshots."""


def _semver(value: Any, source: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        raise VersioningError(f"{source}: version must use MAJOR.MINOR.PATCH semver")
    return tuple(int(part) for part in value.split("."))


def _contract(scenario: Scenario) -> dict[str, Any]:
    contract = {
        "input": scenario.input,
        "intent": scenario.intent,
        "must_ask": scenario.must_ask,
        "expected_tools": scenario.expected_tools,
        "must_not": [{"action": rule.action, "unless_asked": rule.unless_asked} for rule in scenario.must_not],
        "allowed_questions": scenario.allowed_questions,
        "expected_tool_calls": [
            {"name": call.name, "arguments": call.arguments} for call in (scenario.expected_tool_calls or ())
        ] if scenario.expected_tool_calls is not None else None,
        "answer_graders": scenario.answer_graders,
        "turns": [
            {"input": turn.input, "intent": turn.intent, "must_ask": turn.must_ask,
             "expected_tools": turn.expected_tools,
             "must_not": [{"action": rule.action, "unless_asked": rule.unless_asked} for rule in turn.must_not],
             "allowed_questions": turn.allowed_questions}
            for turn in scenario.turns
        ],
        "tags": scenario.tags,
    }
    if scenario.severity is not None:
        contract["severity"] = scenario.severity
    return contract


def snapshot(scenarios: list[Scenario], scenario_dir: Path) -> dict[str, Any]:
    """Create a stable, source-path-independent JSON snapshot for review and diffs."""
    path = scenario_dir / MANIFEST_NAME
    document: dict[str, Any] = {}
    if path.exists():
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise VersioningError(f"could not read {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise VersioningError(f"{path}: expected a YAML mapping")
    schema_version = document.get("schema_version", CURRENT_SCHEMA_VERSION)
    if schema_version not in {0, CURRENT_SCHEMA_VERSION}:
        raise VersioningError(f"{path}: unsupported schema_version {schema_version!r}")
    version = document.get("version", "0.1.0")
    parsed_version = _semver(version, str(path))
    name = document.get("scenario_set", scenario_dir.name)
    if not isinstance(name, str) or not name.strip():
        raise VersioningError(f"{path}: scenario_set must be a non-empty string")
    rows = []
    for scenario in sorted(scenarios, key=lambda item: item.name):
        contract = _contract(scenario)
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=list)
        behavior = {
            "intent": scenario.intent,
            "must_ask": list(scenario.must_ask),
            "expected_tools": list(scenario.expected_tools) if scenario.expected_tools is not None else None,
            "expected_tool_calls": sorted(call.name for call in (scenario.expected_tool_calls or ()))
            if scenario.expected_tool_calls is not None else None,
            "prohibited_actions": sorted(
                [rule.action, sorted(rule.unless_asked)] for rule in scenario.must_not
            ),
            "allowed_questions": list(scenario.allowed_questions),
            "tags": list(scenario.tags),
            "severity": scenario.severity,
            "turns": [{
                "intent": turn.intent,
                "must_ask": list(turn.must_ask),
                "expected_tools": list(turn.expected_tools) if turn.expected_tools is not None else None,
                "prohibited_actions": sorted(
                    [rule.action, sorted(rule.unless_asked)] for rule in turn.must_not
                ),
                "allowed_questions": list(turn.allowed_questions),
            } for turn in scenario.turns],
            "answer_grader_types": sorted({str(grader.get("type", "unknown")) for grader in scenario.answer_graders}),
        }
        input_data = scenario.input if not scenario.turns else [turn.input for turn in scenario.turns]
        input_encoded = json.dumps(input_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=list)
        rows.append({"id": scenario.name, "version": scenario.version,
                     "contract_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                     "input_sha256": hashlib.sha256(input_encoded.encode("utf-8")).hexdigest(),
                     "behavior": behavior})
    return {"scenario_set": name, "version": version, "version_tuple": list(parsed_version),
            "schema_version": CURRENT_SCHEMA_VERSION, "scenarios": rows}


def _next_version(version: tuple[int, int, int], bump: str) -> str:
    major, minor, patch = version
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def compare_snapshots(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Classify scenario additions, removals, and contract changes by release impact."""
    old_rows = {row["id"]: row for row in baseline.get("scenarios", [])}
    new_rows = {row["id"]: row for row in current.get("scenarios", [])}
    added = sorted(new_rows.keys() - old_rows.keys())
    removed = sorted(old_rows.keys() - new_rows.keys())
    changed = sorted(name for name in old_rows.keys() & new_rows.keys()
                     if old_rows[name].get("contract_sha256") != new_rows[name].get("contract_sha256"))
    if removed or changed:
        bump = "major"
    elif added:
        bump = "minor"
    else:
        bump = "patch"
    old_version_value = baseline.get("version_tuple")
    old_version = tuple(old_version_value) if old_version_value is not None else _semver(baseline.get("version"), "baseline")
    current_version = tuple(current["version_tuple"])
    sufficient = (
        current_version[0] > old_version[0]
        if bump == "major"
        else current_version[0] > old_version[0] or current_version[:2] > old_version[:2]
        if bump == "minor"
        else current_version > old_version
    )
    unversioned_changes = sorted(name for name in changed
                                 if int(new_rows[name].get("version", 1)) <= int(old_rows[name].get("version", 1)))
    details = []
    for name in sorted(set(added) | set(removed) | set(changed)):
        old, new = old_rows.get(name), new_rows.get(name)
        if old is None or new is None:
            details.append({"id": name, "change": "added" if old is None else "removed",
                            "behavior": (new or old).get("behavior")})
            continue
        old_behavior, new_behavior = old.get("behavior"), new.get("behavior")
        dimensions = []
        if isinstance(old_behavior, dict) and isinstance(new_behavior, dict):
            dimensions = sorted(key for key in set(old_behavior) | set(new_behavior)
                                if old_behavior.get(key) != new_behavior.get(key))
        if old.get("input_sha256") != new.get("input_sha256"):
            dimensions.append("input")
        if not dimensions:
            dimensions = ["scenario_contract"]
        details.append({"id": name, "change": "changed", "changed_dimensions": dimensions,
                        "before": old_behavior, "after": new_behavior})
    return {
        "scenario_set": current["scenario_set"],
        "baseline_version": baseline.get("version"),
        "current_version": current["version"],
        "added": added,
        "removed": removed,
        "changed": changed,
        "details": details,
        "unversioned_changes": unversioned_changes,
        "recommended_bump": bump,
        "recommended_version": _next_version(old_version, bump),
        "version_sufficient": sufficient and not unversioned_changes,
    }


def change_impact_report(compatibility: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Map scenario contract edits to the configured release gates they can affect."""
    details = compatibility.get("details", [])
    affected_ids = {item["id"] for item in details}
    tags_by_id: dict[str, set[str]] = {}
    for item in details:
        for side in (item.get("behavior"), item.get("before"), item.get("after")):
            if isinstance(side, dict):
                tags_by_id.setdefault(item["id"], set()).update(
                    str(tag).casefold() for tag in side.get("tags", [])
                )
    gates = []
    if "min_overall_pass_rate" in profile and affected_ids:
        gates.append({"type": "overall_pass_rate", "target": "all scenarios",
                      "minimum": profile["min_overall_pass_rate"], "affected": True,
                      "reason": "any scenario contract change can alter the overall pass rate"})
    for tag, minimum in profile.get("critical_tags", {}).items():
        matching = sorted(name for name, tags in tags_by_id.items() if tag.casefold() in tags)
        gates.append({"type": "critical_tag_pass_rate", "target": tag, "minimum": minimum,
                      "affected": bool(matching), "scenarios": matching,
                      "reason": "changed scenario matches this critical tag" if matching else "no changed scenario matches this tag"})
    for name, minimum in profile.get("critical_scenarios", {}).items():
        gates.append({"type": "critical_scenario_pass_rate", "target": name, "minimum": minimum,
                      "affected": name in affected_ids,
                      "reason": "critical scenario contract changed" if name in affected_ids else "critical scenario is unchanged"})
    return {"scenario_set": compatibility["scenario_set"],
            "baseline_version": compatibility["baseline_version"],
            "current_version": compatibility["current_version"],
            "recommended_bump": compatibility["recommended_bump"],
            "recommended_version": compatibility["recommended_version"],
            "version_sufficient": compatibility["version_sufficient"],
            "added": compatibility["added"], "removed": compatibility["removed"],
            "changed": compatibility["changed"], "details": details,
            "affected_gate_count": sum(gate["affected"] for gate in gates), "gates": gates}


def write_manifest(scenario_dir: Path, current: dict[str, Any], version: str | None = None) -> Path:
    """Write or migrate a scenario-set manifest to the current manifest schema."""
    selected = version or current["version"]
    _semver(selected, "scenario set")
    target = scenario_dir / MANIFEST_NAME
    target.write_text(
        f"scenario_set: {json.dumps(current['scenario_set'], ensure_ascii=False)}\n"
        f"version: {selected}\n"
        f"schema_version: {CURRENT_SCHEMA_VERSION}\n",
        encoding="utf-8",
    )
    return target


def migrate_scenario_versions(scenarios: list[Scenario], baseline: dict[str, Any]) -> list[str]:
    """Increment per-scenario versions for changed contracts relative to a snapshot."""
    old_rows = {row["id"]: row for row in baseline.get("scenarios", [])}
    migrated: list[str] = []
    for scenario in scenarios:
        old = old_rows.get(scenario.name)
        if not old or old.get("contract_sha256") == hashlib.sha256(
            json.dumps(_contract(scenario), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, default=list).encode("utf-8")
        ).hexdigest():
            continue
        previous_version = int(old.get("version", 1))
        if scenario.version > previous_version:
            continue
        source = Path(scenario.source)
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise VersioningError(f"could not migrate {source}: {exc}") from exc
        if not isinstance(document, dict):
            raise VersioningError(f"{source}: expected a YAML mapping")
        text = source.read_text(encoding="utf-8")
        version_line = re.compile(r"^version\s*:\s*.*$", re.MULTILINE)
        if version_line.search(text):
            text = version_line.sub(f"version: {previous_version + 1}", text, count=1)
        else:
            scenario_line = re.compile(r"^(scenario\s*:\s*.*)$", re.MULTILINE)
            if not scenario_line.search(text):
                raise VersioningError(f"{source}: cannot locate top-level scenario field for version migration")
            text = scenario_line.sub(rf"\1\nversion: {previous_version + 1}", text, count=1)
        source.write_text(text, encoding="utf-8")
        migrated.append(scenario.name)
    return migrated


def text_version_report(snapshot_data: dict[str, Any], compatibility: dict[str, Any] | None = None) -> str:
    lines = [f"Scenario set: {snapshot_data['scenario_set']} ({snapshot_data['version']})",
             f"Manifest schema: {snapshot_data['schema_version']}",
             f"Scenarios: {len(snapshot_data['scenarios'])}"]
    if compatibility is not None:
        lines.extend(["", f"Baseline: {compatibility['baseline_version']}",
                      f"Recommended bump: {compatibility['recommended_bump']} ({compatibility['recommended_version']})",
                      f"Version sufficient: {'yes' if compatibility['version_sufficient'] else 'no'}"])
        for key in ("added", "removed", "changed", "unversioned_changes"):
            lines.append(f"{key.replace('_', ' ').capitalize()}: " + (", ".join(compatibility[key]) or "none"))
    return "\n".join(lines)
