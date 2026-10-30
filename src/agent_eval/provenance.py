"""Stable run provenance fingerprints attached to captured scenario results."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Any

from agent_eval.models import AgentResult, Scenario
from agent_eval.runner import _scenario_fingerprint


PROVENANCE_VERSION = 1


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def build_result_provenance(
    scenario: Scenario, result: AgentResult, *, runner_key: str,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint inputs and execution identity without copying sensitive values."""
    model = result.metadata.get("model")
    prompt = result.metadata.get("prompt_version")
    fields = {
        "scenario_fingerprint": _scenario_fingerprint(scenario),
        "runner_fingerprint": fingerprint(runner_key),
        "model_fingerprint": fingerprint(model) if model is not None else None,
        "prompt_fingerprint": fingerprint(prompt) if prompt is not None else None,
        "tool_fixture_fingerprint": fingerprint(scenario.tool_fixtures),
        "evaluation_config_fingerprint": fingerprint(evaluation_config),
    }
    fields["run_fingerprint"] = fingerprint(fields)
    return {"format_version": PROVENANCE_VERSION, **fields}


def attach_provenance(
    scenarios: list[Scenario], results: list[AgentResult], *, runner_key: str,
    evaluation_config: dict[str, Any],
) -> list[AgentResult]:
    by_name = {scenario.name: scenario for scenario in scenarios}
    attached = []
    for result in results:
        scenario = by_name.get(result.scenario)
        if scenario is None:
            continue
        metadata = dict(result.metadata)
        metadata["provenance"] = build_result_provenance(
            scenario, result, runner_key=runner_key, evaluation_config=evaluation_config,
        )
        attached.append(replace(result, metadata=metadata))
    return attached


def provenance_manifest(results: list[AgentResult]) -> dict[str, Any]:
    """Create a compact sidecar manifest keyed by scenario name."""
    rows = {
        result.scenario: result.metadata["provenance"]
        for result in results if isinstance(result.metadata.get("provenance"), dict)
    }
    return {"format_version": PROVENANCE_VERSION, "scenario_count": len(rows), "scenarios": rows}


def compare_result_provenance(
    baseline: dict[str, AgentResult], candidate: dict[str, AgentResult],
) -> dict[str, Any]:
    """Report protocol drift separately from intentional runner/model/prompt variants."""
    important_fields = (
        "scenario_fingerprint", "tool_fixture_fingerprint", "evaluation_config_fingerprint",
    )
    variant_fields = ("runner_fingerprint", "model_fingerprint", "prompt_fingerprint")
    changed: dict[str, list[str]] = {field: [] for field in (*important_fields, *variant_fields)}
    missing: list[str] = []
    common = sorted(baseline.keys() & candidate.keys())
    for name in common:
        left = baseline[name].metadata.get("provenance")
        right = candidate[name].metadata.get("provenance")
        if not isinstance(left, dict) or not isinstance(right, dict):
            missing.append(name)
            continue
        for field in changed:
            if left.get(field) != right.get(field):
                changed[field].append(name)
    changed = {field: names for field, names in changed.items() if names}
    incompatible = sorted(field for field in important_fields if changed.get(field))
    return {
        "compatible": not incompatible and not missing,
        "compared_scenarios": len(common),
        "missing_provenance": missing,
        "changed_fields": changed,
        "incompatible_fields": incompatible,
        "variant_fields": sorted(field for field in variant_fields if changed.get(field)),
        "warnings": ([f"provenance missing for {len(missing)} matched scenario(s)"] if missing else [])
        + [f"{field} differs for {len(changed[field])} scenario(s)" for field in changed],
    }
