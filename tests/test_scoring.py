import unittest

from agent_eval.models import (
    AgentResult,
    PolicyRule,
    Scenario,
    ToolCall,
    ToolCallExpectation,
    TraceEvent,
)
from agent_eval.scoring import score_scenario


class ScenarioScoringTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            name="reschedule",
            input={"user": "Move my appointment"},
            intent="RESCHEDULE",
            must_ask=("patient_identity", "confirmation"),
            expected_tools=("lookup", "reserve"),
            expected_tool_calls=(
                ToolCallExpectation(
                    "lookup", {"patient.id": {"exists": True}, "date": {"one_of": ["tomorrow", "Friday"]}}
                ),
                ToolCallExpectation("reserve", {"slot": {"contains": "morning"}}),
            ),
            must_not=(PolicyRule("cancel", ("confirmation",)),),
        )

    def test_accepts_expected_tool_order_arguments_and_policy_prerequisite(self):
        result = AgentResult(
            scenario="reschedule",
            intent="reschedule",
            asked=("patient_identity", "confirmation"),
            tools=("lookup", "reserve"),
            actions=("cancel",),
            events=(
                TraceEvent("question", "confirmation"),
                TraceEvent("action", "cancel"),
            ),
            tool_calls=(
                ToolCall("lookup", {"patient": {"id": "p-1"}, "date": "tomorrow"}),
                ToolCall("reserve", {"slot": "morning-10"}),
            ),
        )
        score = score_scenario(self.scenario, result)
        self.assertTrue(score.passed)
        self.assertTrue(score.tool_correct)
        self.assertEqual(score.policy_violations, ())

    def test_detects_wrong_order_and_missing_policy_prerequisite(self):
        result = AgentResult(
            scenario="reschedule",
            intent="RESCHEDULE",
            asked=("patient_identity",),
            tools=("reserve", "lookup"),
            actions=("cancel",),
            events=(
                TraceEvent("action", "cancel"),
                TraceEvent("question", "confirmation"),
            ),
            tool_calls=(
                ToolCall("reserve", {"slot": "morning-10"}),
                ToolCall("lookup", {"patient": {"id": "p-1"}, "date": "tomorrow"}),
            ),
        )
        score = score_scenario(self.scenario, result)
        self.assertFalse(score.passed)
        self.assertFalse(score.tool_correct)
        self.assertTrue(any("expected 'lookup'" in message for message in score.tool_call_mismatches))
        self.assertEqual(len(score.policy_violations), 1)

    def test_unconditional_must_not_rule_blocks_action(self):
        scenario = Scenario(
            name="no-cancel",
            input={"user": "Move it"},
            intent="RESCHEDULE",
            must_not=(PolicyRule("cancel"),),
        )
        score = score_scenario(
            scenario,
            AgentResult("no-cancel", "RESCHEDULE", actions=("cancel",)),
        )
        self.assertEqual(score.policy_violations, ("cancel",))

    def test_conditional_policy_requires_ordered_event_evidence(self):
        result = AgentResult(
            scenario="reschedule",
            intent="RESCHEDULE",
            asked=("patient_identity", "confirmation"),
            actions=("cancel",),
        )
        score = score_scenario(self.scenario, result)
        self.assertTrue(score.policy_violations)
