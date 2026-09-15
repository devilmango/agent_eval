"""Configurable redaction for raw trace documents before export or drafting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml


class RedactionError(ValueError):
    """Raised when a trace redaction policy is invalid or cannot be read."""


@dataclass(frozen=True)
class RedactionRule:
    path: tuple[str, ...]
    action: str
    replacement: Any = "[REDACTED]"
    required: bool = False


_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[\*\]")


def load_redaction_rules(path: Path | None) -> list[RedactionRule]:
    if path is None:
        return []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RedactionError(f"could not read redaction policy {path}: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"rules"} or not isinstance(document["rules"], list):
        raise RedactionError("redaction policy must contain only a top-level 'rules' list")
    rules = []
    for index, item in enumerate(document["rules"], start=1):
        if not isinstance(item, dict) or set(item) - {"path", "action", "replacement", "required"}:
            raise RedactionError(f"redaction rule {index} must contain path, action, and optional replacement/required")
        path_value, action = item.get("path"), item.get("action")
        if not isinstance(path_value, str) or not path_value.strip():
            raise RedactionError(f"redaction rule {index} needs a non-empty path")
        tokens = tuple("*" if token == "[*]" else token for token in _PATH_TOKEN.findall(path_value))
        normalized = re.sub(r"\[\*\]", ".*", path_value).strip(".")
        if not tokens or ".".join(tokens) != normalized:
            raise RedactionError(f"redaction rule {index} has an invalid path: {path_value}")
        if tokens == ("*",) or action not in {"remove", "mask"}:
            raise RedactionError(f"redaction rule {index} must target a field and use remove or mask")
        required = item.get("required", False)
        if not isinstance(required, bool):
            raise RedactionError(f"redaction rule {index} required must be a boolean")
        replacement = item.get("replacement", "[REDACTED]")
        rules.append(RedactionRule(tokens, action, replacement, required))
    return rules


def redact_document(
    document: Any, rules: list[RedactionRule], *, match_counts: list[int] | None = None,
) -> Any:
    """Return a deep copy with configured fields removed or replaced."""
    output = deepcopy(document)
    for index, rule in enumerate(rules):
        matched = _apply(output, rule.path, rule.action, deepcopy(rule.replacement))
        if match_counts is not None:
            match_counts[index] += matched
    return output


def _apply(node: Any, path: tuple[str, ...], action: str, replacement: Any) -> int:
    if not path:
        return 0
    key, *tail = path
    if key == "*":
        if isinstance(node, list):
            if not tail:
                matched = len(node)
                if action == "remove":
                    node.clear()
                else:
                    node[:] = [deepcopy(replacement) for _ in node]
                return matched
            else:
                return sum(_apply(item, tuple(tail), action, replacement) for item in node)
        return 0
    if isinstance(node, dict) and key in node:
        if not tail:
            if action == "remove":
                del node[key]
            else:
                node[key] = deepcopy(replacement)
            return 1
        else:
            return _apply(node[key], tuple(tail), action, replacement)
    return 0


def audit_redaction_file(path: Path, rules: list[RedactionRule]) -> dict[str, Any]:
    """Check rule coverage over JSONL traces without returning any trace content."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RedactionError(f"could not read traces {path}: {exc}") from exc
    counts = [0 for _ in rules]
    misses = [0 for _ in rules]
    record_count = 0
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RedactionError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise RedactionError(f"{path}:{line_number}: each trace record must be a JSON object")
        record_count += 1
        per_record = [0 for _ in rules]
        redact_document(record, rules, match_counts=per_record)
        for index, count in enumerate(per_record):
            counts[index] += count
            if count == 0:
                misses[index] += 1
    if record_count == 0:
        raise RedactionError(f"no trace records found in {path}")
    rule_rows = [
        {
            "path": ".".join(rule.path).replace(".*", "[*]"),
            "action": rule.action,
            "required": rule.required,
            "matches": counts[index],
            "unmatched_records": misses[index],
            "matched": counts[index] > 0,
            "passed": not rule.required or misses[index] == 0,
        }
        for index, rule in enumerate(rules)
    ]
    return {
        "trace_file": str(path),
        "record_count": record_count,
        "rules_checked": len(rules),
        "passed": all(row["passed"] for row in rule_rows),
        "rules": rule_rows,
    }


def text_redaction_audit(report: dict[str, Any]) -> str:
    lines = [
        f"Redaction policy audit: {'PASS' if report['passed'] else 'FAIL'}",
        f"Trace records: {report['record_count']}",
        f"Rules checked: {report['rules_checked']}",
        "",
        "Rule coverage:",
    ]
    for rule in report["rules"]:
        status = "FAIL" if not rule["passed"] else "WARN" if not rule["matched"] else "PASS"
        gate = "required" if rule["required"] else "optional"
        lines.append(
            f"  {status} {rule['path']} ({rule['action']}, {gate}): "
            f"{rule['matches']} match(es), {rule['unmatched_records']} record(s) unmatched"
        )
    return "\n".join(lines)
