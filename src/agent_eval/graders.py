"""Built-in answer graders and the external grader plugin interface."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any, Protocol

from agent_eval.models import AgentResult, GradeResult, Scenario


class GraderError(ValueError):
    """Raised when a grader is missing, invalid, or fails to return a score."""


class AnswerGrader(Protocol):
    """Plugin contract for evaluating an agent's final answer."""

    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        """Return a normalized score in [0, 1] and a pass/fail decision."""


class _ContainsAny:
    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        values = _string_values(config)
        matches = [value for value in values if _contains(answer, value, config)]
        score = len(matches) / len(values)
        threshold = _threshold(config, default=1 / len(values))
        return _result("contains_any", config, score, score >= threshold,
                       "Matched: " + (", ".join(matches) if matches else "none"))


class _ContainsAll:
    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        values = _string_values(config)
        matches = [value for value in values if _contains(answer, value, config)]
        score = len(matches) / len(values)
        threshold = _threshold(config, default=1.0)
        return _result("contains_all", config, score, score >= threshold,
                       "Matched: " + (", ".join(matches) if matches else "none"))


class _NotContains:
    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        values = _string_values(config)
        found = [value for value in values if _contains(answer, value, config)]
        score = 0.0 if found else 1.0
        return _result("not_contains", config, score, not found,
                       "Prohibited text found: " + (", ".join(found) if found else "none"))


class _MinLength:
    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        minimum = config.get("value")
        if not isinstance(minimum, int) or minimum < 0:
            raise GraderError("min_length grader requires a non-negative integer 'value'")
        score = min(1.0, len(answer) / minimum) if minimum else 1.0
        passed = len(answer) >= minimum
        return _result("min_length", config, score, passed,
                       f"Answer length {len(answer)}; minimum {minimum}")


class _MaxLength:
    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        maximum = config.get("value")
        if not isinstance(maximum, int) or maximum < 0:
            raise GraderError("max_length grader requires a non-negative integer 'value'")
        score = 1.0 if len(answer) <= maximum else maximum / len(answer) if answer else 0.0
        passed = len(answer) <= maximum
        return _result("max_length", config, score, passed,
                       f"Answer length {len(answer)}; maximum {maximum}")


def _string_values(config: Mapping[str, Any]) -> list[str]:
    values = config.get("values")
    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise GraderError("text grader requires a non-empty list of string 'values'")
    return values


def _contains(answer: str, value: str, config: Mapping[str, Any]) -> bool:
    if config.get("case_sensitive", False):
        return value in answer
    return value.casefold() in answer.casefold()


def _threshold(config: Mapping[str, Any], default: float = 1.0) -> float:
    threshold = config.get("threshold", default)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise GraderError("grader threshold must be a number between 0 and 1")
    return float(threshold)


def _result(grader: str, config: Mapping[str, Any], score: float, passed: bool, feedback: str) -> GradeResult:
    return GradeResult(
        grader=grader,
        score=score,
        passed=passed,
        feedback=feedback,
        rubric=dict(config),
    )


def load_grader_registry() -> dict[str, AnswerGrader]:
    registry: dict[str, AnswerGrader] = {
        "contains_any": _ContainsAny(),
        "contains_all": _ContainsAll(),
        "not_contains": _NotContains(),
        "min_length": _MinLength(),
        "max_length": _MaxLength(),
    }
    points = entry_points()
    if hasattr(points, "select"):
        plugins = points.select(group="agent_eval.graders")
    else:  # pragma: no cover - compatibility for older importlib.metadata
        plugins = points.get("agent_eval.graders", [])
    for point in plugins:
        if point.name in registry:
            raise GraderError(f"grader plugin '{point.name}' conflicts with a built-in grader")
        plugin = point.load()
        if isinstance(plugin, type):
            plugin = plugin()
        elif callable(plugin) and not hasattr(plugin, "grade"):
            plugin = plugin()
        if not hasattr(plugin, "grade"):
            raise GraderError(f"grader plugin '{point.name}' must provide grade(answer, config)")
        registry[point.name] = plugin
    return registry


def grade_answer(
    scenario: Scenario,
    result: AgentResult,
    registry: Mapping[str, AnswerGrader],
) -> tuple[GradeResult, ...]:
    grades: list[GradeResult] = []
    for spec in scenario.answer_graders:
        grader_name = spec["type"]
        grader = registry.get(grader_name)
        if grader is None:
            raise GraderError(
                f"unknown answer grader '{grader_name}' in scenario '{scenario.name}'; "
                "install a plugin exposing it in the 'agent_eval.graders' entry-point group"
            )
        try:
            output = grader.grade(result.answer or "", spec)
        except GraderError:
            raise
        except Exception as exc:
            raise GraderError(f"grader '{grader_name}' failed for '{scenario.name}': {exc}") from exc
        if not isinstance(output, GradeResult):
            raise GraderError(f"grader '{grader_name}' must return a GradeResult")
        if isinstance(output.score, bool) or not isinstance(output.score, (int, float)) or not 0 <= output.score <= 1:
            raise GraderError(f"grader '{grader_name}' returned a score outside [0, 1]")
        grades.append(
            GradeResult(
                grader=grader_name,
                score=float(output.score),
                passed=bool(output.passed),
                feedback=output.feedback,
                rubric=dict(spec),
                metadata=dict(output.metadata),
            )
        )
    return tuple(grades)
