"""Policy checks for approving scenario-set baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ApprovalPolicyError(ValueError):
    """Raised when an approval policy is invalid."""


_BUMP_RANK = {"none": 0, "patch": 1, "minor": 2, "major": 3}


def load_approval_policy(path: Path) -> dict[str, Any]:
    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ApprovalPolicyError(f"could not read approval policy {path}: {exc}") from exc
    if not isinstance(policy, dict):
        raise ApprovalPolicyError(f"{path}: approval policy must be a YAML mapping")
    supported = {"require_approved_by", "allowed_reviewers", "minimum_version_bump",
                 "require_release_gates_passed"}
    unknown = set(policy) - supported
    if unknown:
        raise ApprovalPolicyError(f"{path}: unsupported policy field(s): {', '.join(sorted(unknown))}")
    for field in ("require_approved_by", "require_release_gates_passed"):
        if field in policy and not isinstance(policy[field], bool):
            raise ApprovalPolicyError(f"{path}: {field} must be a boolean")
    reviewers = policy.get("allowed_reviewers", [])
    if (not isinstance(reviewers, list) or any(not isinstance(item, str) or not item.strip() for item in reviewers)
            or len(set(reviewers)) != len(reviewers)):
        raise ApprovalPolicyError(f"{path}: allowed_reviewers must be a list of unique non-empty strings")
    bump = policy.get("minimum_version_bump", "none")
    if not isinstance(bump, str) or bump not in _BUMP_RANK:
        raise ApprovalPolicyError(f"{path}: minimum_version_bump must be none, patch, minor, or major")
    if reviewers:
        policy["require_approved_by"] = True
    policy["allowed_reviewers"] = reviewers
    policy["minimum_version_bump"] = bump
    policy.setdefault("require_approved_by", False)
    policy.setdefault("require_release_gates_passed", False)
    return policy


def policy_blockers(
    review: dict[str, Any], policy: dict[str, Any], *, approved_by: str | None,
    release_gates: dict[str, Any] | None,
) -> list[str]:
    blockers = []
    if policy["require_approved_by"] and not approved_by:
        blockers.append("approval policy requires --approved-by")
    allowed = policy["allowed_reviewers"]
    if approved_by and allowed and approved_by not in allowed:
        blockers.append(f"reviewer '{approved_by}' is not in the allowed_reviewers list")
    has_changes = bool(review.get("added") or review.get("removed") or review.get("changed"))
    minimum = policy["minimum_version_bump"]
    recommended = review.get("recommended_bump") or "none"
    if has_changes and _BUMP_RANK.get(recommended, 0) < _BUMP_RANK[minimum]:
        blockers.append(f"approval policy requires at least a {minimum} version bump")
    if policy["require_release_gates_passed"]:
        if release_gates is None:
            blockers.append("approval policy requires --profile and --runs for release-gate validation")
        elif not release_gates["passed"]:
            blockers.append("one or more configured release gates failed")
    return blockers
