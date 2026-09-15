from datetime import date
import tempfile
import unittest
from pathlib import Path

from agent_eval.cli import main
from agent_eval.scenario_audit import audit_scenarios
from agent_eval.scenarios import ScenarioError, load_scenarios


class ScenarioGovernanceTests(unittest.TestCase):
    def test_audits_critical_owner_and_review_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good.yaml").write_text(
                "scenario: critical-good\nseverity: critical\nowner: operations\n"
                "reviewed_at: '2026-10-01'\ninput: {user: hi}\nexpected: {intent: HI}\n",
                encoding="utf-8",
            )
            (root / "stale.yaml").write_text(
                "scenario: critical-stale\nseverity: critical\n"
                "reviewed_at: '2026-01-01'\ninput: {user: hi}\nexpected: {intent: HI}\n",
                encoding="utf-8",
            )
            (root / "routine.yaml").write_text(
                "scenario: routine\nseverity: low\ninput: {user: hi}\nexpected: {intent: HI}\n",
                encoding="utf-8",
            )
            report = audit_scenarios(
                load_scenarios(root), max_review_age_days=30, as_of=date(2026, 10, 14),
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["critical_scenarios_checked"], 2)
        findings = {item["scenario"]: item["findings"] for item in report["scenarios"]}
        self.assertEqual(findings["critical-good"], [])
        self.assertIn("missing_owner", findings["critical-stale"])
        self.assertIn("review_stale", findings["critical-stale"])

    def test_cli_gates_stale_critical_scenario_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            scenarios = Path(directory)
            (scenarios / "case.yaml").write_text(
                "scenario: safety_case\nseverity: critical\nowner: safety-team\n"
                "reviewed_at: '2025-01-01'\ninput: {user: hi}\nexpected: {intent: HI}\n",
                encoding="utf-8",
            )
            status = main([
                "scenario-audit", str(scenarios), "--max-review-age-days", "90",
                "--as-of", "2026-10-14", "--format", "json",
            ])
        self.assertEqual(status, 1)

    def test_scenario_metadata_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "scenario: bad\nseverity: extreme\ninput: {user: hi}\nexpected: {intent: HI}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ScenarioError):
                load_scenarios(path)


if __name__ == "__main__":
    unittest.main()
