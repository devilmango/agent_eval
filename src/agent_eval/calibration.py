"""Calibration of semantic judges against human-labeled gold examples."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from agent_eval.graders import JudgeRequest, SemanticJudge
from agent_eval.models import GradeResult


class CalibrationError(ValueError):
    """Raised when a gold set is invalid or judge output is malformed."""


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
