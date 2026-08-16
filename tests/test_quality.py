import unittest

from agent_eval.models import AgentResult, PolicyRule, Scenario
from agent_eval.quality import mutation_report, scenario_coverage


class ScenarioQualityTests(unittest.TestCase):
    def test_coverage_summarizes_asserted_behavior_dimensions(self):
        scenario = Scenario(
            name="booking", input={"user": "book"}, intent="BOOK",
            must_ask=("date",), expected_tools=("reserve",),
            must_not=(PolicyRule("cancel"),), tags=("safety",),
        )
        coverage = scenario_coverage([scenario])
        self.assertEqual(coverage["intent_labels"], {"BOOK": 1})
        self.assertEqual(coverage["checks"]["policy_scenarios"], 1)
        self.assertEqual(coverage["tags"], {"safety": 1})

    def test_mutation_audit_catches_expected_behavior_regressions(self):
        scenario = Scenario(
            name="booking", input={"user": "book"}, intent="BOOK",
            must_ask=("date",), expected_tools=("reserve",),
            must_not=(PolicyRule("cancel"),),
        )
        result = AgentResult(
            scenario="booking", intent="BOOK", asked=("date",), tools=("reserve",),
        )
        report = mutation_report([scenario], {"booking": result})
        self.assertEqual(report["mutation_count"], 5)
        self.assertEqual(report["killed"], 5)
        self.assertEqual(report["survived"], 0)
