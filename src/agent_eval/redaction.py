"""Configurable redaction for raw trace documents before export or drafting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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
        if not isinstance(item, dict) or set(item) - {"path", "action", "replacement"}:
            raise RedactionError(f"redaction rule {index} must contain path, action, and optional replacement")
        path_value, action = item.get("path"), item.get("action")
        if not isinstance(path_value, str) or not path_value.strip():
            raise RedactionError(f"redaction rule {index} needs a non-empty path")
        tokens = tuple("*" if token == "[*]" else token for token in _PATH_TOKEN.findall(path_value))
        normalized = re.sub(r"\[\*\]", ".*", path_value).strip(".")
        if not tokens or ".".join(tokens) != normalized:
            raise RedactionError(f"redaction rule {index} has an invalid path: {path_value}")
        if tokens == ("*",) or action not in {"remove", "mask"}:
            raise RedactionError(f"redaction rule {index} must target a field and use remove or mask")
        replacement = item.get("replacement", "[REDACTED]")
        rules.append(RedactionRule(tokens, action, replacement))
    return rules


def redact_document(document: Any, rules: list[RedactionRule]) -> Any:
    """Return a deep copy with configured fields removed or replaced."""
    output = deepcopy(document)
    for rule in rules:
        _apply(output, rule.path, rule.action, deepcopy(rule.replacement))
    return output


def _apply(node: Any, path: tuple[str, ...], action: str, replacement: Any) -> None:
    if not path:
        return
    key, *tail = path
    if key == "*":
        if isinstance(node, list):
            if not tail:
                if action == "remove":
                    node.clear()
                else:
                    node[:] = [deepcopy(replacement) for _ in node]
            else:
                for item in node:
                    _apply(item, tuple(tail), action, replacement)
        return
    if isinstance(node, dict) and key in node:
        if not tail:
            if action == "remove":
                del node[key]
            else:
                node[key] = deepcopy(replacement)
        else:
            _apply(node[key], tuple(tail), action, replacement)
