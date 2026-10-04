"""Built-in answer graders and the external grader plugin interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import entry_points
import json
from math import isfinite
import re
from typing import Any, Protocol
from pathlib import Path
from statistics import mean, pstdev

from agent_eval.models import AgentResult, GradeResult, Scenario


class GraderError(ValueError):
    """Raised when a grader is missing, invalid, or fails to return a score."""


class AnswerGrader(Protocol):
    """Plugin contract for evaluating an agent's final answer."""

    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        """Return a normalized score in [0, 1] and a pass/fail decision."""


@dataclass(frozen=True)
class JudgeRequest:
    scenario: str
    user_input: Mapping[str, Any]
    conversation: tuple[Mapping[str, Any], ...]
    expected: Mapping[str, Any]
    answer: str
    rubric: str | Mapping[str, Any]
    model: str | None = None


class SemanticJudge(Protocol):
    version: str

    def judge(self, request: JudgeRequest) -> GradeResult:
        """Return a scored judgment and a human-readable rationale."""


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


class _EvidenceGrounded:
    """Check lexical support for each answer sentence and validate source citations."""

    _citation = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
    _stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
        "will", "with",
    }

    def grade(self, answer: str, config: Mapping[str, Any]) -> GradeResult:
        evidence = self._evidence(config.get("evidence"))
        require_citations = config.get("require_citations", True)
        if not isinstance(require_citations, bool):
            raise GraderError("evidence_grounded 'require_citations' must be a boolean")
        minimum = config.get("min_supported_token_ratio", 0.8)
        if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                or not isfinite(minimum) or not 0 <= minimum <= 1):
            raise GraderError("evidence_grounded 'min_supported_token_ratio' must be between 0 and 1")
        if not isinstance(answer, str):
            raise GraderError("evidence_grounded answer must be a string")

        answer = re.sub(r"([.!?])\s*(\[[A-Za-z0-9_.:-]+\])", r"\2\1", answer.strip())
        sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", answer) if sentence.strip()]
        supported_count = 0
        token_count = 0
        unsupported: list[str] = []
        citations: list[str] = []
        unknown: list[str] = []
        missing_citations = 0
        for sentence in sentences:
            sentence_citations = self._citation.findall(sentence)
            citations.extend(sentence_citations)
            unknown.extend(citation for citation in sentence_citations if citation not in evidence)
            if require_citations and not sentence_citations:
                missing_citations += 1
            cited_evidence = ([evidence[item] for item in sentence_citations if item in evidence]
                              if sentence_citations else list(evidence.values()))
            source_tokens = {token for text in cited_evidence for token in self._tokens(text)}
            answer_tokens = self._tokens(self._citation.sub("", sentence))
            token_count += len(answer_tokens)
            for token in answer_tokens:
                if token in source_tokens:
                    supported_count += 1
                else:
                    unsupported.append(token)

        ratio = supported_count / token_count if token_count else 0.0
        citation_valid = not unknown and (not require_citations or missing_citations == 0)
        passed = bool(sentences) and ratio >= float(minimum) and citation_valid
        score = ratio if citation_valid else 0.0
        feedback = f"Supported {supported_count}/{token_count} meaningful answer tokens."
        if unsupported:
            feedback += " Unsupported tokens: " + ", ".join(unsupported) + "."
        if missing_citations:
            feedback += f" {missing_citations} sentence(s) lack citations."
        if unknown:
            feedback += " Unknown citation(s): " + ", ".join(dict.fromkeys(unknown)) + "."
        return GradeResult(
            grader="evidence_grounded", score=score, passed=passed,
            feedback=feedback, rubric=dict(config), metadata={
                "supported_tokens": supported_count,
                "answer_tokens": token_count,
                "supported_token_ratio": ratio,
                "unsupported_tokens": unsupported,
                "citations": citations,
                "unknown_citations": list(dict.fromkeys(unknown)),
                "uncited_sentences": missing_citations,
                "minimum_supported_token_ratio": float(minimum),
            },
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())
        return [token for token in tokens if token not in _EvidenceGrounded._stop_words]

    @staticmethod
    def _evidence(value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            entries = [{"id": key, "text": text} for key, text in value.items()]
        elif isinstance(value, list):
            entries = value
        else:
            raise GraderError("evidence_grounded requires non-empty 'evidence' entries")
        evidence: dict[str, str] = {}
        for index, entry in enumerate(entries):
            if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                    or not entry["id"].strip() or not isinstance(entry.get("text"), str)
                    or not entry["text"].strip()):
                raise GraderError(f"evidence_grounded evidence[{index}] needs non-empty 'id' and 'text'")
            source_id = entry["id"].strip()
            if source_id in evidence:
                raise GraderError(f"evidence_grounded evidence IDs must be unique: {source_id}")
            evidence[source_id] = entry["text"]
        if not evidence:
            raise GraderError("evidence_grounded requires non-empty 'evidence' entries")
        return evidence


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
        "evidence_grounded": _EvidenceGrounded(),
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


def load_judge_registry() -> dict[str, SemanticJudge]:
    points = entry_points()
    plugins = points.select(group="agent_eval.judges") if hasattr(points, "select") else points.get("agent_eval.judges", [])
    registry: dict[str, SemanticJudge] = {}
    for point in plugins:
        if point.name in registry:
            raise GraderError(f"multiple semantic judges are named '{point.name}'")
        try:
            judge = point.load()
            if isinstance(judge, type):
                judge = judge()
            elif callable(judge) and not hasattr(judge, "judge"):
                judge = judge()
        except Exception as exc:
            raise GraderError(f"could not load semantic judge '{point.name}': {exc}") from exc
        if not hasattr(judge, "judge"):
            raise GraderError(f"semantic judge '{point.name}' must provide judge(request)")
        registry[point.name] = judge
    return registry


def grade_answer(
    scenario: Scenario,
    result: AgentResult,
    registry: Mapping[str, AnswerGrader],
    judges: Mapping[str, SemanticJudge] | None = None,
    *, judge_cache: Path | None = None, judge_repetitions: int = 1,
    skip_semantic_judges: bool = False,
) -> tuple[GradeResult, ...]:
    if judge_repetitions < 1:
        raise GraderError("judge repetitions must be at least one")
    grades: list[GradeResult] = []
    for spec in scenario.answer_graders:
        grader_name = spec["type"]
        if grader_name == "llm_judge":
            if skip_semantic_judges:
                grades.append(GradeResult(
                    grader=grader_name, score=0.0, passed=False,
                    feedback="Semantic judge skipped by request.", rubric=dict(spec),
                    metadata={"skipped": True},
                ))
                continue
            judge_name = spec.get("judge")
            judge = (judges or {}).get(judge_name) if isinstance(judge_name, str) else None
            if judge is None:
                raise GraderError(
                    f"unknown semantic judge {judge_name!r} in scenario '{scenario.name}'; "
                    "install a plugin exposing it in the 'agent_eval.judges' entry-point group"
                )
            rubric = spec.get("rubric")
            if not isinstance(rubric, (str, dict)) or not rubric:
                raise GraderError("llm_judge requires a non-empty 'rubric'")
            model = spec.get("model")
            if model is not None and not isinstance(model, str):
                raise GraderError("llm_judge 'model' must be a string")
            request = JudgeRequest(
                scenario=scenario.name,
                user_input=scenario.input,
                conversation=tuple(turn.input for turn in scenario.turns) or (scenario.input,),
                expected={"intent": scenario.intent, "must_ask": scenario.must_ask,
                          "tools": scenario.expected_tools, "must_not": [rule.action for rule in scenario.must_not],
                          "turns": [{"intent": turn.intent, "must_ask": turn.must_ask,
                                     "tools": turn.expected_tools,
                                     "must_not": [rule.action for rule in turn.must_not]}
                                    for turn in scenario.turns]},
                answer=result.answer or "",
                rubric=rubric,
                model=model,
            )
            cache_key = sha256(json.dumps({"request": request.__dict__, "judge": judge_name,
                                           "version": str(getattr(judge, "version", "unspecified"))},
                                          sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
            cache = {}
            if judge_cache and judge_cache.exists():
                try:
                    cache = json.loads(judge_cache.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise GraderError(f"could not read judge cache {judge_cache}: {exc}") from exc
                if not isinstance(cache, dict):
                    raise GraderError(f"judge cache {judge_cache} must contain a JSON object")
            outputs = []
            for repeat_index in range(judge_repetitions):
                key = f"{cache_key}:{repeat_index}"
                cached = cache.get(key)
                if isinstance(cached, dict):
                    output = GradeResult(grader="llm_judge", score=float(cached["score"]),
                                         passed=bool(cached.get("passed", False)),
                                         feedback=str(cached.get("feedback", "")),
                                         metadata=dict(cached.get("metadata", {})))
                else:
                    try:
                        output = judge.judge(request)
                    except Exception as exc:
                        raise GraderError(f"semantic judge '{judge_name}' failed for '{scenario.name}': {exc}") from exc
                if not isinstance(output, GradeResult):
                    raise GraderError(f"semantic judge '{judge_name}' must return a GradeResult")
                if isinstance(output.score, bool) or not isinstance(output.score, (int, float)) or not 0 <= output.score <= 1:
                    raise GraderError(f"semantic judge '{judge_name}' returned a score outside [0, 1]")
                if not isinstance(output.metadata, Mapping) or not isinstance(output.feedback, str):
                    raise GraderError(f"semantic judge '{judge_name}' returned invalid metadata or rationale")
                if judge_cache and key not in cache:
                    cache[key] = {"score": output.score, "passed": output.passed,
                                  "feedback": output.feedback, "metadata": output.metadata}
                    judge_cache.parent.mkdir(parents=True, exist_ok=True)
                    judge_cache.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
                outputs.append(output)
            output = outputs[0]
            threshold = spec.get("threshold", 0.7)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise GraderError("llm_judge threshold must be a number between 0 and 1")
            fingerprint = sha256(json.dumps(rubric, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            metadata = {
                **output.metadata,
                "judge": judge_name,
                "configured_model": model,
                "model": output.metadata.get("model", model),
                "grader_version": str(getattr(judge, "version", "unspecified")),
                "rubric_sha256": fingerprint,
                "repeated_scores": [float(item.score) for item in outputs],
                "score_stddev": pstdev(float(item.score) for item in outputs) if len(outputs) > 1 else 0.0,
                "judge_agreement": (sum(item.passed == outputs[0].passed for item in outputs) / len(outputs)),
                "cache": "enabled" if judge_cache else "disabled",
            }
            score = mean(float(item.score) for item in outputs)
            grades.append(GradeResult(
                grader=grader_name,
                score=score,
                passed=score >= float(threshold),
                feedback="\n".join(dict.fromkeys(item.feedback for item in outputs)),
                rubric=dict(spec),
                metadata=metadata,
            ))
            continue

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
