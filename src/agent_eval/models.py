"""Data structures shared by scenario loading, scoring, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallExpectation:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    name: str


@dataclass(frozen=True)
class GradeResult:
    grader: str
    score: float
    passed: bool
    feedback: str = ""
    rubric: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioTurn:
    input: dict[str, Any]
    intent: str | None = None
    must_ask: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] | None = None
    must_not: tuple[PolicyRule, ...] = ()
    allowed_questions: tuple[str, ...] = ()
    expected_tool_calls: tuple[ToolCallExpectation, ...] | None = None


@dataclass(frozen=True)
class AgentTurnResult:
    intent: str
    asked: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    events: tuple[TraceEvent, ...] = ()
    answer: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRule:
    action: str
    unless_asked: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    input: dict[str, Any]
    intent: str
    must_ask: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] | None = None
    must_not: tuple[PolicyRule, ...] = ()
    allowed_questions: tuple[str, ...] = ()
    expected_tool_calls: tuple[ToolCallExpectation, ...] | None = None
    answer_graders: tuple[dict[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    source: str = ""
    turns: tuple[ScenarioTurn, ...] = ()
    tool_fixtures: tuple[dict[str, Any], ...] = ()
    version: int = 1
    owner: str | None = None
    severity: str | None = None
    reviewed_at: str | None = None


@dataclass(frozen=True)
class FailureDiagnostic:
    category: str
    message: str
    turn_index: int | None = None
    tool_call_index: int | None = None
    event_index: int | None = None
    citation_ids: tuple[str, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    expected: Any = None
    actual: Any = None


@dataclass(frozen=True)
class AgentResult:
    scenario: str
    intent: str
    asked: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    events: tuple[TraceEvent, ...] = ()
    answer: str | None = None
    notes: str = ""
    turns: tuple[AgentTurnResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioScore:
    scenario: str
    passed: bool
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    missing_questions: tuple[str, ...] = ()
    unnecessary_questions: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] | None = None
    actual_tools: tuple[str, ...] = ()
    tool_call_mismatches: tuple[str, ...] = ()
    tool_correct: bool | None = None
    policy_violations: tuple[str, ...] = ()
    answer_grades: tuple[GradeResult, ...] = ()
    turn_violations: tuple[str, ...] = ()
    notes: str = ""
    diagnostics: tuple[FailureDiagnostic, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["first_divergence"] = asdict(self.diagnostics[0]) if self.diagnostics else None
        return value


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    passed: int
    failed: int
    intent_accuracy: float
    tool_accuracy: float | None
    policy_compliance: float
    unnecessary_question_rate: float
    answer_quality_score: float | None = None
    scores: tuple[ScenarioScore, ...] = field(default_factory=tuple)
    execution: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "metrics": {
                "intent_accuracy": self.intent_accuracy,
                "tool_accuracy": self.tool_accuracy,
                "policy_compliance": self.policy_compliance,
                "unnecessary_question_rate": self.unnecessary_question_rate,
                "answer_quality_score": self.answer_quality_score,
            },
            "scenarios": [score.as_dict() for score in self.scores],
            "execution": self.execution,
        }
