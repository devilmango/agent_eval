"""Calibration of semantic judges against human-labeled gold examples."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from agent_eval.graders import GraderError, JudgeRequest, SemanticJudge, load_grader_registry
from agent_eval.models import GradeResult


class CalibrationError(ValueError):
    """Raised when a gold set is invalid or judge output is malformed."""


def calibrate_grounding(path: Path, *, min_accuracy: float | None = None) -> dict[str, Any]:
    """Measure the deterministic evidence grader against human-reviewed examples."""
    if min_accuracy is not None and (
        isinstance(min_accuracy, bool) or not isinstance(min_accuracy, (int, float))
        or not isfinite(min_accuracy) or not 0 <= min_accuracy <= 1
    ):
        raise CalibrationError("minimum accuracy must be between 0 and 1")
    try:
        raw = path.read_bytes()
        document = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CalibrationError(f"could not read grounding gold set {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CalibrationError("grounding gold set must be a YAML mapping")
    reviewer = document.get("reviewed_by")
    reviewed_at = document.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise CalibrationError("grounding gold set requires non-empty 'reviewed_by' human-review metadata")
    try:
        reviewed_date = (reviewed_at if type(reviewed_at) is date else
                         date.fromisoformat(reviewed_at) if isinstance(reviewed_at, str) else None)
    except ValueError as exc:
        raise CalibrationError("grounding gold set 'reviewed_at' must be an ISO date") from exc
    if reviewed_date is None:
        raise CalibrationError("grounding gold set requires 'reviewed_at' in YYYY-MM-DD format")
    grader_config = document.get("grader", {})
    cases = document.get("cases")
    if not isinstance(grader_config, dict):
        raise CalibrationError("grounding gold set 'grader' must be a mapping")
    if not isinstance(cases, list) or not cases:
        raise CalibrationError("grounding gold set requires a non-empty 'cases' list")
    grader = load_grader_registry().get("evidence_grounded")
    if grader is None:
        raise CalibrationError("built-in evidence_grounded grader is not available")

    rows: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        label = f"cases[{index - 1}]"
        if not isinstance(case, dict):
            raise CalibrationError(f"{label} must be a mapping")
        case_id = case.get("id")
        answer = case.get("answer")
        expected_pass = case.get("expected_passed")
        evidence = case.get("evidence")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise CalibrationError(f"{label}.id must be a unique non-empty string")
        seen.add(case_id)
        if not isinstance(answer, str) or not isinstance(expected_pass, bool):
            raise CalibrationError(f"{label} requires an answer string and boolean expected_passed")
        if evidence is None:
            raise CalibrationError(f"{label} requires evidence reviewed for this example")
        config = {**grader_config, "evidence": evidence}
        try:
            output = grader.grade(answer, config)
        except GraderError as exc:
            raise CalibrationError(f"invalid grounding example {case_id!r}: {exc}") from exc
        predicted = output.passed
        if expected_pass and predicted:
            tp += 1
        elif expected_pass and not predicted:
            fn += 1
        elif not expected_pass and predicted:
            fp += 1
        else:
            tn += 1
        rows.append({"id": case_id, "expected_passed": expected_pass,
                     "predicted_passed": predicted, "score": output.score,
                     "feedback": output.feedback, "metadata": output.metadata})

    count = len(rows)
    accuracy = (tp + tn) / count
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_set": document.get("gold_set", path.stem),
        "gold_set_sha256": sha256(raw).hexdigest(),
        "grader": "evidence_grounded",
        "human_reviewed": True,
        "reviewed_by": reviewer.strip(),
        "reviewed_at": reviewed_date.isoformat(),
        "grader_config": grader_config,
        "cases": count,
        "confusion_matrix": {"true_positive": tp, "true_negative": tn,
                             "false_positive": fp, "false_negative": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "minimum_accuracy": min_accuracy,
        "passed_gate": min_accuracy is None or accuracy >= min_accuracy,
        "results": rows,
    }


def text_grounding_calibration_report(report: dict[str, Any]) -> str:
    matrix = report["confusion_matrix"]
    lines = [
        f"Evidence grounding calibration: {report['gold_set']} ({report['cases']} reviewed cases)",
        f"Reviewed by: {report['reviewed_by']} on {report['reviewed_at']}",
        f"Accuracy: {report['accuracy'] * 100:.1f}%",
        f"Precision: {report['precision'] * 100:.1f}%",
        f"Recall: {report['recall'] * 100:.1f}%",
        f"F1: {report['f1'] * 100:.1f}%",
        f"Confusion: TP {matrix['true_positive']}, TN {matrix['true_negative']}, "
        f"FP {matrix['false_positive']}, FN {matrix['false_negative']}",
        "Gate: " + ("PASS" if report["passed_gate"] else "FAIL"),
    ]
    for row in report["results"]:
        if row["expected_passed"] != row["predicted_passed"]:
            lines.append(f"  DISAGREEMENT {row['id']}: expected {row['expected_passed']}, "
                         f"predicted {row['predicted_passed']} (score {row['score']:.3f})")
    return "\n".join(lines)


def calibrate_judge(
    path: Path,
    judge_name: str,
    judges: dict[str, SemanticJudge],
    *, model: str | None = None,
    threshold: float = 0.7,
    min_accuracy: float | None = None,
) -> dict[str, Any]:
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise CalibrationError("threshold must be between 0 and 1")
    if min_accuracy is not None and (not isfinite(min_accuracy) or not 0 <= min_accuracy <= 1):
        raise CalibrationError("minimum accuracy must be between 0 and 1")
    judge = judges.get(judge_name)
    if judge is None:
        raise CalibrationError(f"unknown semantic judge {judge_name!r}")
    try:
        raw = path.read_bytes()
        document = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CalibrationError(f"could not read gold set {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CalibrationError("gold set must be a YAML mapping")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CalibrationError("gold set requires a non-empty 'cases' list")
    default_rubric = document.get("rubric")
    rows: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    score_errors: list[float] = []
    brier_errors: list[float] = []
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        label = f"cases[{index - 1}]"
        if not isinstance(case, dict):
            raise CalibrationError(f"{label} must be a mapping")
        case_id = case.get("id")
        user_input = case.get("input")
        answer = case.get("answer")
        expected_pass = case.get("expected_passed")
        expected_score = case.get("expected_score")
        rubric = case.get("rubric", default_rubric)
        expected_context = case.get("expected", {})
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise CalibrationError(f"{label}.id must be a unique non-empty string")
        seen.add(case_id)
        if isinstance(user_input, str):
            user_input = {"user": user_input}
        if not isinstance(user_input, dict) or not isinstance(answer, str):
            raise CalibrationError(f"{label} requires input (string or mapping) and answer string")
        if not isinstance(expected_context, dict):
            raise CalibrationError(f"{label}.expected must be a mapping")
        if expected_pass is not None and not isinstance(expected_pass, bool):
            raise CalibrationError(f"{label}.expected_passed must be a boolean")
        if not isinstance(expected_pass, bool) and expected_score is None:
            raise CalibrationError(f"{label} requires expected_passed or expected_score")
        if expected_score is not None and (
            isinstance(expected_score, bool) or not isinstance(expected_score, (int, float))
            or not isfinite(expected_score) or not 0 <= expected_score <= 1
        ):
            raise CalibrationError(f"{label}.expected_score must be between 0 and 1")
        if expected_pass is None:
            expected_pass = expected_score >= threshold
        if not isinstance(rubric, (str, dict)) or not rubric:
            raise CalibrationError(f"{label} requires a non-empty rubric")
        request = JudgeRequest(
            scenario=case_id, user_input=user_input, conversation=(user_input,),
            expected=expected_context, answer=answer, rubric=rubric, model=model,
        )
        try:
            output = judge.judge(request)
        except Exception as exc:
            raise CalibrationError(f"judge {judge_name!r} failed on {case_id!r}: {exc}") from exc
        if not isinstance(output, GradeResult):
            raise CalibrationError(f"judge {judge_name!r} returned a non-GradeResult for {case_id!r}")
        if isinstance(output.score, bool) or not isinstance(output.score, (int, float)) or not isfinite(output.score) or not 0 <= output.score <= 1:
            raise CalibrationError(f"judge {judge_name!r} returned a score outside [0, 1] for {case_id!r}")
        if not isinstance(output.feedback, str):
            raise CalibrationError(f"judge {judge_name!r} returned invalid rationale for {case_id!r}")
        predicted = float(output.score) >= threshold
        if expected_pass and predicted:
            tp += 1
        elif expected_pass and not predicted:
            fn += 1
        elif not expected_pass and predicted:
            fp += 1
        else:
            tn += 1
        target_score = float(expected_score) if expected_score is not None else float(expected_pass)
        error = float(output.score) - target_score
        score_errors.append(abs(error))
        brier_errors.append(error * error)
        rows.append({"id": case_id, "expected_passed": expected_pass,
                     "predicted_passed": predicted, "score": float(output.score),
                     "expected_score": expected_score, "feedback": output.feedback})

    total = len(rows)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    passed_gate = min_accuracy is None or accuracy >= min_accuracy
    return {
        "gold_set": document.get("gold_set", path.stem),
        "gold_set_sha256": sha256(raw).hexdigest(),
        "judge": judge_name,
        "judge_version": str(getattr(judge, "version", "unspecified")),
        "model": model,
        "threshold": threshold,
        "cases": total,
        "confusion_matrix": {"true_positive": tp, "true_negative": tn,
                             "false_positive": fp, "false_negative": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_score_error": mean(score_errors) if score_errors else None,
        "brier_score": mean(brier_errors) if brier_errors else None,
        "minimum_accuracy": min_accuracy,
        "passed_gate": passed_gate,
        "results": rows,
    }


def text_calibration_report(report: dict[str, Any]) -> str:
    matrix = report["confusion_matrix"]
    lines = [
        f"Judge calibration: {report['gold_set']} ({report['cases']} cases)",
        f"Judge: {report['judge']} v{report['judge_version']}",
        f"Accuracy: {report['accuracy'] * 100:.1f}%",
        f"Precision: {report['precision'] * 100:.1f}%",
        f"Recall: {report['recall'] * 100:.1f}%",
        f"F1: {report['f1'] * 100:.1f}%",
        f"Confusion: TP {matrix['true_positive']}, TN {matrix['true_negative']}, "
        f"FP {matrix['false_positive']}, FN {matrix['false_negative']}",
    ]
    if report["brier_score"] is not None:
        lines.append(f"Brier score: {report['brier_score']:.4f}")
    if report["mean_absolute_score_error"] is not None:
        lines.append(f"Mean absolute score error: {report['mean_absolute_score_error']:.4f}")
    lines.append("Gate: " + ("PASS" if report["passed_gate"] else "FAIL"))
    for row in report["results"]:
        if row["expected_passed"] != row["predicted_passed"]:
            lines.append(f"  DISAGREEMENT {row['id']}: expected {row['expected_passed']}, "
                         f"predicted {row['predicted_passed']} (score {row['score']:.3f})")
    return "\n".join(lines)
