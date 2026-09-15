"""Create human-review scenario drafts from recorded agent traces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import yaml

from agent_eval.adapters import PathMappingAdapter, resolve_trace_path
from agent_eval.redaction import load_redaction_rules, redact_document
from agent_eval.scenarios import ScenarioError, load_scenario


class DraftError(ValueError):
    """Raised when traces cannot be safely converted into review drafts."""


def list_scenario_drafts(directory: Path) -> list[dict[str, Any]]:
    """List drafts and mark later drafts with identical normalized input."""
    if not directory.is_dir():
        raise DraftError(f"draft directory does not exist: {directory}")
    seen_inputs: dict[str, str] = {}
    rows = []
    for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
        document = _read_draft(path)
        if document.get("draft") is not True:
            continue
        fingerprint = sha256(json.dumps(
            document.get("input"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        duplicate_of = seen_inputs.get(fingerprint)
        seen_inputs.setdefault(fingerprint, path.name)
        review = document.get("review", {})
        rows.append({
            "file": path.name,
            "scenario": document.get("scenario"),
            "status": review.get("status", "pending") if isinstance(review, dict) else "pending",
            "reviewer": review.get("reviewer") if isinstance(review, dict) else None,
            "duplicate_of": duplicate_of,
        })
    return rows


def review_scenario_draft(
    path: Path, *, status: str, reviewer: str, notes: str | None = None,
) -> dict[str, Any]:
    if status not in {"approved", "rejected"}:
        raise DraftError("draft review status must be approved or rejected")
    if not reviewer.strip():
        raise DraftError("reviewer must be a non-empty name")
    document = _read_draft(path)
    if status == "approved":
        _validate_approved_draft(document, path)
    document["review"] = {
        "status": status,
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    _write_yaml(path, document)
    return {"file": str(path), **document["review"]}


def promote_scenario_draft(
    path: Path, output_dir: Path, *, allow_duplicate: bool = False,
) -> dict[str, Any]:
    document = _read_draft(path)
    review = document.get("review", {})
    if not isinstance(review, dict) or review.get("status") != "approved":
        raise DraftError("only approved drafts can be promoted")
    _validate_approved_draft(document, path)
    siblings = list_scenario_drafts(path.parent)
    duplicate = next((row for row in siblings if row["file"] == path.name and row["duplicate_of"]), None)
    if duplicate and not allow_duplicate:
        raise DraftError(f"draft input duplicates {duplicate['duplicate_of']}; use --allow-duplicate after review")
    scenario_id = document.get("scenario")
    target = output_dir / f"{scenario_id}.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    promoted = {
        key: document[key]
        for key in ("scenario", "version", "input", "turns", "expected", "must_not",
                    "allowed_questions", "tags", "tool_fixtures", "owner", "severity", "reviewed_at")
        if key in document
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", prefix=".agent-eval-draft-",
            dir=output_dir, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(yaml.safe_dump(promoted, sort_keys=False, allow_unicode=True))
        load_scenario(temporary_path)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(
                "# Promoted from a human-approved trace draft.\n"
                + yaml.safe_dump(promoted, sort_keys=False, allow_unicode=True)
            )
    except FileExistsError as exc:
        raise DraftError(f"scenario destination already exists: {target}") from exc
    except ScenarioError as exc:
        raise DraftError(f"approved draft is not a valid executable scenario: {exc}") from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
    review["status"] = "promoted"
    review["promoted_to"] = str(target)
    _write_yaml(path, document)
    return {"scenario": scenario_id, "path": str(target), "duplicate_override": bool(duplicate)}


def _read_draft(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DraftError(f"could not read draft {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("draft") is not True:
        raise DraftError(f"{path}: file is not an Agent Eval review draft")
    return document


def _validate_approved_draft(document: dict[str, Any], path: Path) -> None:
    expected = document.get("expected")
    if not isinstance(document.get("scenario"), str) or not document["scenario"].strip():
        raise DraftError(f"{path}: approved draft needs a non-empty scenario ID")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", document["scenario"]) or ".." in document["scenario"]:
        raise DraftError(f"{path}: scenario ID must use letters, numbers, dot, underscore, or hyphen")
    if not isinstance(document.get("input"), dict) or not document["input"]:
        raise DraftError(f"{path}: approved draft needs a non-empty input mapping")
    if not isinstance(expected, dict) or not isinstance(expected.get("intent"), str) or not expected["intent"].strip():
        raise DraftError(f"{path}: enter a human-approved expected.intent before approval")
    for field in ("must_ask", "tools"):
        values = expected.get(field)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise DraftError(f"{path}: set expected.{field} to a list (use [] when none are expected)")
    must_not = document.get("must_not")
    if not isinstance(must_not, list):
        raise DraftError(f"{path}: set must_not to a list (use [] when no actions are prohibited)")


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DraftError(f"could not update draft {path}: {exc}") from exc


def create_scenario_drafts(
    traces_path: Path,
    mapping_path: Path,
    input_path: str,
    output_dir: Path,
    *,
    redaction_config: Path | None = None,
) -> dict[str, Any]:
    """Write incomplete YAML drafts; observed agent behavior never becomes expected behavior."""
    if output_dir.exists():
        raise DraftError(f"draft output directory already exists: {output_dir}")
    adapter = PathMappingAdapter.from_yaml(mapping_path)
    redaction_rules = load_redaction_rules(redaction_config)
    try:
        lines = traces_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DraftError(f"could not read trace file {traces_path}: {exc}") from exc

    drafts: list[dict[str, Any]] = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            source = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DraftError(f"{traces_path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(source, dict):
            raise DraftError(f"{traces_path}:{line_number}: each trace record must be an object")
        source = redact_document(source, redaction_rules)
        try:
            result = adapter.adapt(source)
            user_input = resolve_trace_path(source, input_path)
        except Exception as exc:
            raise DraftError(f"{traces_path}:{line_number}: {exc}") from exc
        if isinstance(user_input, str):
            user_input = {"user": user_input}
        elif not isinstance(user_input, dict):
            raise DraftError(
                f"{traces_path}:{line_number}: input path {input_path!r} must resolve to a string or object"
            )
        if not user_input:
            raise DraftError(f"{traces_path}:{line_number}: captured input object is empty")
        slug = _slug(result.scenario)
        sequence = len(drafts) + 1
        drafts.append({
            "file_name": f"{sequence:03d}-{slug}.yaml",
            "document": {
                "draft": True,
                "scenario": f"draft_{slug}_{sequence:03d}",
                "input": user_input,
                "observed": {
                    "source_scenario": result.scenario,
                    "intent": result.intent,
                    "asked": list(result.asked),
                    "tools": list(result.tools),
                    "actions": list(result.actions),
                },
                "expected": {
                    "intent": None,
                    "must_ask": None,
                    "tools": None,
                },
                "must_not": None,
                "review": {"status": "pending", "reviewer": None, "reviewed_at": None},
                "review_required": True,
            },
        })
    if not drafts:
        raise DraftError(f"no trace records found in {traces_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.drafts-", dir=output_dir.parent))
    try:
        for item in drafts:
            document = item["document"]
            text = (
                "# Review-only draft generated from observed trace data.\n"
                "# Replace every null expected value with human-approved behavior before use.\n"
                + yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
            )
            (temporary / item["file_name"]).write_text(text, encoding="utf-8")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "draft_count": len(drafts),
        "output_dir": str(output_dir),
        "files": [item["file_name"] for item in drafts],
        "review_required": True,
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_-").lower()
    return slug or "trace"
