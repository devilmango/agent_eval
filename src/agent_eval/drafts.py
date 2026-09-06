"""Create human-review scenario drafts from recorded agent traces."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import yaml

from agent_eval.adapters import PathMappingAdapter, resolve_trace_path


class DraftError(ValueError):
    """Raised when traces cannot be safely converted into review drafts."""


def create_scenario_drafts(
    traces_path: Path,
    mapping_path: Path,
    input_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write incomplete YAML drafts; observed agent behavior never becomes expected behavior."""
    if output_dir.exists():
        raise DraftError(f"draft output directory already exists: {output_dir}")
    adapter = PathMappingAdapter.from_yaml(mapping_path)
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
