import unittest

from agent_eval.models import Scenario
from agent_eval.scenarios import ScenarioError, filter_scenarios


def scenario(name, *tags):
    return Scenario(name=name, input={"user": "hello"}, intent="HELLO", tags=tags)


class ScenarioFilterTests(unittest.TestCase):
    def test_includes_any_requested_tag_and_excludes_matching_tags(self):
        scenarios = [scenario("schedule", "scheduling", "safety"), scenario("faq", "faq"), scenario("rx", "medication")]
        selected = filter_scenarios(scenarios, include_tags=("FAQ", "scheduling"), exclude_tags=("safety",))
        self.assertEqual([item.name for item in selected], ["faq"])

    def test_empty_filter_result_is_reported(self):
        with self.assertRaisesRegex(ScenarioError, "No scenarios match"):
            filter_scenarios([scenario("faq", "faq")], include_tags=("missing",))
