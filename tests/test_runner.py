import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from agent_eval.models import AgentResult, AgentTurnResult, PolicyRule, Scenario, ScenarioTurn, TraceEvent
from agent_eval.runner import execute_scenarios, load_runner
from agent_eval.results import load_results
from agent_eval.scoring import evaluate


class ConversationalRunner:
    def __init__(self):
        self.seen_history = []

    def run_turn(self, scenario, user_input, history):
        self.seen_history.append(len(history))
        if user_input["user"].startswith("I need"):
            return AgentTurnResult(intent="BOOK", asked=("date",), events=())
        return AgentTurnResult(intent="BOOK", asked=("time",), answer="Booked")


class RunnerTests(unittest.TestCase):
    def test_executes_turns_in_order_with_history_and_scores_each_turn(self):
        scenario = Scenario(
            name="booking",
            input={"user": "I need an appointment"},
            intent="BOOK",
            turns=(
                ScenarioTurn(input={"user": "I need an appointment"}, intent="BOOK", must_ask=("date",)),
                ScenarioTurn(input={"user": "At 10 AM"}, intent="BOOK", must_ask=("time",)),
            ),
        )
        runner = ConversationalRunner()
        [result] = execute_scenarios([scenario], runner)
        report = evaluate([scenario], {"booking": result})
        self.assertEqual(runner.seen_history, [0, 1])
        self.assertEqual(len(result.turns), 2)
        self.assertEqual(result.asked, ("date", "time"))
        self.assertEqual(report.passed, 1)

    def test_single_turn_scenario_still_records_a_turn_result(self):
        scenario = Scenario(
            name="booking", input={"user": "book"}, intent="BOOK",
            turns=(ScenarioTurn(input={"user": "book"}, intent="BOOK"),),
        )
        class NoQuestionsRunner:
            def run_turn(self, scenario, user_input, history):
                return AgentTurnResult(intent="BOOK")

        report = evaluate([scenario], {"booking": execute_scenarios([scenario], NoQuestionsRunner())[0]})
        self.assertEqual(report.passed, 1)

    def test_turn_policy_requires_prerequisite_before_prohibited_action(self):
        scenario = Scenario(
            name="cancel", input={"user": "cancel"}, intent="CANCEL",
            turns=(ScenarioTurn(
                input={"user": "cancel"}, intent="CANCEL",
                must_not=(PolicyRule("cancel_appointment", ("confirmation",)),),
            ),),
        )
        turn = AgentTurnResult(
            intent="CANCEL", actions=("cancel_appointment",),
            events=(TraceEvent("action", "cancel_appointment"), TraceEvent("question", "confirmation")),
        )
        result = AgentResult(
            scenario="cancel", intent="CANCEL", actions=turn.actions, events=turn.events, turns=(turn,),
        )
        report = evaluate([scenario], {"cancel": result})
        self.assertEqual(report.failed, 1)
        self.assertTrue(any("prohibited action" in item for item in report.scores[0].turn_violations))

    def test_loads_runner_from_optional_entry_point(self):
        point = SimpleNamespace(name="fixture", load=ConversationalRunner)
        metadata = SimpleNamespace(select=lambda **kwargs: [point])
        with patch("agent_eval.runner.entry_points", return_value=metadata):
            runner = load_runner("fixture")
        self.assertIsInstance(runner, ConversationalRunner)

    def test_nested_turn_results_load_as_an_aggregate_trace(self):
        record = {
            "scenario": "booking",
            "turns": [
                {"intent": "BOOK", "asked": ["date"]},
                {"intent": "BOOK", "asked": ["time"], "answer": "Booked"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = load_results(path)["booking"]
        self.assertEqual(result.intent, "BOOK")
        self.assertEqual(result.asked, ("date", "time"))
        self.assertEqual(result.answer, "Booked")
        self.assertEqual(len(result.turns), 2)
