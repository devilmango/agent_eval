"""Portable, integrity-checked scenario baseline envelopes."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from hashlib import sha256
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


def export_portable_baseline(
    baseline_path: Path,
    output_path: Path,
    *,
    private_key_path: Path | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    if key_id is not None and private_key_path is None:
        raise PortableBaselineError("--key-id requires --private-key")
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableBaselineError(f"could not read baseline {baseline_path}: {exc}") from exc
    validate_snapshot(baseline, str(baseline_path))
    envelope = {"format": PORTABLE_FORMAT, "format_version": 1,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_sha256": snapshot_digest(baseline), "snapshot": baseline}
    if private_key_path is not None:
        signing = _cryptography()
        try:
            private_key = signing["serialization"].load_pem_private_key(
                private_key_path.read_bytes(), password=None)
        except Exception as exc:
            raise PortableBaselineError(f"could not load Ed25519 private key {private_key_path}: {exc}") from exc
        if not isinstance(private_key, signing["Ed25519PrivateKey"]):
            raise PortableBaselineError(f"{private_key_path}: private key must be Ed25519")
        public_bytes = private_key.public_key().public_bytes(
            signing["Encoding"].Raw, signing["PublicFormat"].Raw)
        if key_id is not None and (not isinstance(key_id, str) or not key_id.strip()):
            raise PortableBaselineError("key ID must be a non-empty string")
        envelope["signature"] = {
            "algorithm": "Ed25519",
            "key_id": key_id.strip() if key_id is not None else sha256(public_bytes).hexdigest(),
        }
        envelope["signature"]["value"] = base64.b64encode(
            private_key.sign(_canonical_payload(envelope))).decode("ascii")
    _write_json_atomic(output_path, envelope)
    return {"exported": True, "path": str(output_path), "scenario_set": baseline["scenario_set"],
            "version": baseline["version"], "snapshot_sha256": envelope["snapshot_sha256"],
            "signed": private_key_path is not None,
            "key_id": envelope.get("signature", {}).get("key_id")}


def import_portable_baseline(
    path: Path,
    *,
    public_key_path: Path | None = None,
    require_signature: bool = False,
) -> dict[str, Any]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableBaselineError(f"could not read portable baseline {path}: {exc}") from exc
    if (not isinstance(envelope, dict) or envelope.get("format") != PORTABLE_FORMAT
            or isinstance(envelope.get("format_version"), bool) or envelope.get("format_version") != 1
            or not isinstance(envelope.get("snapshot"), dict)):
        raise PortableBaselineError(f"{path}: unsupported or malformed portable baseline")
    signature = envelope.get("signature")
    if signature is None:
        if require_signature:
            raise PortableBaselineError(f"{path}: a signature is required")
        if public_key_path is not None:
            raise PortableBaselineError(f"{path}: a public key was supplied, but the baseline is unsigned")
    else:
        if public_key_path is None:
            raise PortableBaselineError(f"{path}: signed baseline requires --public-key")
        if (not isinstance(signature, dict) or signature.get("algorithm") != "Ed25519"
                or not isinstance(signature.get("key_id"), str) or not signature["key_id"].strip()
                or not isinstance(signature.get("value"), str)):
            raise PortableBaselineError(f"{path}: malformed or unsupported baseline signature")
        signing = _cryptography()
        try:
            public_key = signing["serialization"].load_pem_public_key(public_key_path.read_bytes())
            signature_bytes = base64.b64decode(signature["value"], validate=True)
            if not isinstance(public_key, signing["Ed25519PublicKey"]):
                raise PortableBaselineError(f"{public_key_path}: public key must be Ed25519")
            public_key.verify(signature_bytes, _canonical_payload(envelope))
        except PortableBaselineError:
            raise
        except Exception as exc:
            raise PortableBaselineError(f"{path}: baseline signature verification failed: {exc}") from exc
    snapshot = envelope["snapshot"]
    validate_snapshot(snapshot, str(path))
    if envelope.get("snapshot_sha256") != snapshot_digest(snapshot):
        raise PortableBaselineError(f"{path}: snapshot digest does not match")
    return snapshot


def _canonical_payload(envelope: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in envelope.items() if key != "signature"}
    signature = envelope.get("signature")
    if isinstance(signature, dict):
        payload["signature"] = {key: value for key, value in signature.items() if key != "value"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cryptography() -> dict[str, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    except ImportError as exc:
        raise PortableBaselineError(
            "Ed25519 signing requires the optional dependency; install agent-eval[signing]"
        ) from exc
    return {"serialization": serialization, "Ed25519PrivateKey": Ed25519PrivateKey,
            "Ed25519PublicKey": Ed25519PublicKey, "Encoding": Encoding,
            "PublicFormat": PublicFormat}


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
