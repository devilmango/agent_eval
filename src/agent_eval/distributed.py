"""Deterministic scenario sharding and strict result merging for CI workers."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

from agent_eval.models import AgentResult, Scenario


class ShardError(ValueError):
    """Raised when a shard manifest or distributed result set is inconsistent."""


def scenario_fingerprint(scenario: Scenario) -> str:
    value = asdict(scenario)
    value.pop("source", None)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_shard_manifest(scenarios: list[Scenario], *, shard_count: int) -> dict[str, Any]:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ShardError("shard_count must be a positive integer")
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ShardError("scenario names must be unique before sharding")
    fingerprints = {scenario.name: scenario_fingerprint(scenario) for scenario in scenarios}
    shards = [{"index": index, "scenarios": [name for offset, name in enumerate(names)
                                                   if offset % shard_count == index]}
              for index in range(shard_count)]
    suite_digest = hashlib.sha256(json.dumps(fingerprints, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
    return {
        "format": "agent-eval-shards-v1",
        "shard_count": shard_count,
        "suite_fingerprint": suite_digest,
        "scenario_fingerprints": fingerprints,
        "shards": shards,
    }


def validate_shard_manifest(scenarios: list[Scenario], manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict) or manifest.get("format") != "agent-eval-shards-v1":
        raise ShardError("unsupported shard manifest format")
    current = create_shard_manifest(scenarios, shard_count=manifest.get("shard_count"))
    if manifest.get("scenario_fingerprints") != current["scenario_fingerprints"]:
        raise ShardError("scenario set does not match shard manifest")
    if manifest.get("suite_fingerprint") != current["suite_fingerprint"]:
        raise ShardError("suite fingerprint does not match shard manifest")
    if manifest.get("shards") != current["shards"]:
        raise ShardError("shard assignments do not match scenario set")


def select_shard(scenarios: list[Scenario], manifest: dict[str, Any], *,
                 shard_index: int) -> list[Scenario]:
    validate_shard_manifest(scenarios, manifest)
    count = manifest["shard_count"]
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < count:
        raise ShardError(f"shard_index must be between 0 and {count - 1}")
    names = set(manifest["shards"][shard_index]["scenarios"])
    return [scenario for scenario in scenarios if scenario.name in names]


def merge_shard_results(
    scenarios: list[Scenario], manifest: dict[str, Any],
    shard_results: list[tuple[int, list[AgentResult]]],
) -> list[AgentResult]:
    validate_shard_manifest(scenarios, manifest)
    by_index: dict[int, list[AgentResult]] = {}
    for index, results in shard_results:
        if index in by_index:
            raise ShardError(f"duplicate shard result for index {index}")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < manifest["shard_count"]:
            raise ShardError(f"invalid shard index {index!r}")
        by_index[index] = results

    expected_indexes = set(range(manifest["shard_count"]))
    if set(by_index) != expected_indexes:
        absent = sorted(expected_indexes - set(by_index))
        expected_names = [name for shard in manifest["shards"]
                          if shard["index"] in absent for name in shard["scenarios"]]
        if expected_names:
            raise ShardError("missing scenario result(s): " + ", ".join(expected_names))
        raise ShardError("missing shard result(s): " + ", ".join(map(str, absent)))

    shard_names = {shard["index"]: set(shard["scenarios"]) for shard in manifest["shards"]}
    merged: dict[str, AgentResult] = {}
    for index, results in by_index.items():
        seen: set[str] = set()
        for result in results:
            name = result.scenario
            if name in seen or name in merged:
                raise ShardError(f"duplicate scenario result '{name}'")
            seen.add(name)
            if name not in shard_names[index]:
                raise ShardError(f"unexpected scenario '{name}' in shard {index}")
            merged[name] = result
        missing = shard_names[index] - seen
        if missing:
            raise ShardError("missing scenario result(s): " + ", ".join(sorted(missing)))

    return [merged[scenario.name] for scenario in scenarios]
