"""Portable run artifact bundles with integrity verification."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any

from agent_eval import __version__
from agent_eval.adapters import result_as_dict
from agent_eval.models import AgentResult, EvaluationReport, Scenario
from agent_eval.results import load_results
from agent_eval.scenarios import load_scenarios
from agent_eval.scoring import evaluate
from agent_eval.versioning import snapshot as scenario_snapshot


class BundleError(ValueError):
    """Raised when a bundle cannot be created or verified."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(
    target: Path,
    scenario_source: Path,
    scenarios: list[Scenario],
    results: dict[str, AgentResult],
    report: EvaluationReport,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Create a new, self-contained run directory and fail if target exists."""
    if target.exists():
        raise BundleError(f"bundle destination already exists: {target}")
    source_root = scenario_source if scenario_source.is_dir() else scenario_source.parent
    source_root = source_root.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        scenario_dir = temporary / "scenarios"
        scenario_dir.mkdir()
        copied: set[str] = set()
        for scenario in scenarios:
            source = Path(scenario.source).resolve()
            try:
                relative = source.relative_to(source_root)
            except ValueError as exc:
                raise BundleError(f"scenario file {source} is outside source root {source_root}") from exc
            destination = scenario_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.add(str(relative))
        manifest = source_root / "scenario-set.yaml"
        if manifest.is_file() and "scenario-set.yaml" not in copied:
            shutil.copyfile(manifest, scenario_dir / "scenario-set.yaml")

        ordered_results = [results[scenario.name] for scenario in scenarios if scenario.name in results]
        result_text = "".join(json.dumps(result_as_dict(result), ensure_ascii=False) + "\n"
                              for result in ordered_results)
        (temporary / "results.jsonl").write_text(result_text, encoding="utf-8")
        _write_json(temporary / "report.json", report.as_dict())
        snapshot_data = scenario_snapshot(scenarios, source_root)
        _write_json(temporary / "scenario-snapshot.json", snapshot_data)
        _write_json(temporary / "run-config.json", run_config)
        files = {}
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            files[path.relative_to(temporary).as_posix()] = _hash(path)
        run_id = sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest_data = {
            "format_version": 1,
            "agent_eval_version": __version__,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        _write_json(temporary / "bundle-manifest.json", manifest_data)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(target)
        return manifest_data
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_bundle(root: Path) -> dict[str, Any]:
    """Verify every file listed in a bundle manifest and reject unsafe paths."""
    manifest_path = root / "bundle-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"could not read bundle manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise BundleError(f"{manifest_path}: unsupported or malformed bundle manifest")
    errors: list[str] = []
    verified = 0
    for relative, expected in manifest["files"].items():
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe path: {relative}")
            continue
        path = root.joinpath(*rel.parts)
        if path.is_symlink():
            errors.append(f"symlink not allowed: {relative}")
            continue
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"path escapes bundle: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing file: {relative}")
        elif _hash(path) != expected:
            errors.append(f"hash mismatch: {relative}")
        else:
            verified += 1
    declared = set(manifest["files"])
    actual = {item.relative_to(root).as_posix() for item in root.rglob("*")
              if item.is_file() and item != manifest_path}
    for relative in sorted(actual - declared):
        errors.append(f"unlisted file: {relative}")
    computed_run_id = sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("run_id") != computed_run_id:
        errors.append("run id does not match file hash inventory")
    return {"valid": not errors, "run_id": manifest.get("run_id"),
            "verified_files": verified, "errors": errors}


def replay_bundle(root: Path, *, strict: bool = False, judge_cache: Path | None = None,
                  judge_repetitions: int = 1,
                  skip_semantic_judges: bool = False) -> tuple[dict[str, Any], EvaluationReport]:
    """Re-score a verified bundle's recorded scenarios and results without running an agent."""
    verification = verify_bundle(root)
    if not verification["valid"]:
        raise BundleError("cannot replay invalid bundle: " + "; ".join(verification["errors"]))
    scenarios = load_scenarios(root / "scenarios")
    results = load_results(root / "results.jsonl")
    report = evaluate(scenarios, results, strict=strict, judge_cache=judge_cache,
                      judge_repetitions=judge_repetitions,
                      skip_semantic_judges=skip_semantic_judges)
    return verification, report
