import tempfile
import unittest
from pathlib import Path

from agent_eval.scenarios import load_scenario


class ScenarioTurnLoadingTests(unittest.TestCase):
    def test_loads_turn_inputs_and_turn_expectations(self):
        contents = """scenario: booking
expected:
  intent: BOOK
turns:
  - input:
      user: I need an appointment
    expected:
      intent: BOOK
      must_ask: [date]
  - input:
      user: Tomorrow morning
    expected:
      intent: BOOK
      must_ask: [time]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "booking.yaml"
            path.write_text(contents, encoding="utf-8")
            scenario = load_scenario(path)
        self.assertEqual(len(scenario.turns), 2)
        self.assertEqual(scenario.input["user"], "I need an appointment")
        self.assertEqual(scenario.turns[1].must_ask, ("time",))
