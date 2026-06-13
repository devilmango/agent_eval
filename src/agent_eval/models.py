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
    source: str = ""


@dataclass(frozen=True)
class AgentResult:
    scenario: str
    intent: str
    asked: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    events: tuple[TraceEvent, ...] = ()
    notes: str = ""


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
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    passed: int
    failed: int
    intent_accuracy: float
    tool_accuracy: float | None
    policy_compliance: float
    unnecessary_question_rate: float
    scores: tuple[ScenarioScore, ...] = field(default_factory=tuple)

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
            },
            "scenarios": [score.as_dict() for score in self.scores],
        }
