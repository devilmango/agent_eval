import tempfile
import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from agent_eval.budgets import apply_budgets
from agent_eval.cli import main
from agent_eval.models import AgentResult, Scenario
from agent_eval.quality import failure_analysis
from agent_eval.scenarios import load_scenarios
from agent_eval.scoring import evaluate
from agent_eval.versioning import change_impact_report, compare_snapshots, migrate_scenario_versions, snapshot, write_manifest


class AnalysisAndBudgetTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_failure_clusters_and_slices(self):
        scenarios = [
            Scenario(name="a", input={"user": "book"}, intent="BOOK", must_ask=("date",), tags=("booking",)),
            Scenario(name="b", input={"user": "hours"}, intent="HOURS", tags=("faq",)),
        ]
        results = {"a": AgentResult("a", "OTHER", metadata={"model": "model-x"}),
                   "b": AgentResult("b", "HOURS", metadata={"model": "model-y"})}
        report = evaluate(scenarios, results)
        analysis = failure_analysis(scenarios, report, results)
        reasons = {item["reason"] for item in analysis["failure_clusters"]}
        self.assertIn("wrong_intent", reasons)
        self.assertIn("missing_question:date", reasons)
        self.assertIn("model", analysis["slices"])
        self.assertEqual(analysis["slices"]["tag"][0]["failed"], 1)

    def test_budgets_use_totals_and_fail_when_measurements_are_missing(self):
        scenario = Scenario(name="a", input={"user": "hi"}, intent="GREET")
        report = evaluate([scenario], {"a": AgentResult("a", "GREET")})
        enriched, passed = apply_budgets(
            report, {"a": AgentResult("a", "GREET", metadata={"cost_usd": .02, "input_tokens": 30,
                                                                      "output_tokens": 10, "latency_ms": 100})},
            1, max_cost_usd=.03, max_input_tokens=40, max_output_tokens=10,
            max_p95_latency_ms=120,
        )
        self.assertTrue(passed)
        self.assertEqual(enriched.execution["budget_gates"]["cost_usd"]["observed"], .02)
        _, missing_passed = apply_budgets(report, {"a": AgentResult("a", "GREET")}, 1,
                                          max_cost_usd=1)
        self.assertFalse(missing_passed)

    def test_snapshot_detects_breaking_changes_and_migrates_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenario-set.yaml").write_text(
                "scenario_set: sample\nversion: 1.0.0\nschema_version: 0\n", encoding="utf-8")
            scenario_path = root / "greeting.yaml"
            scenario_path.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: GREET}\n", encoding="utf-8")
            baseline = snapshot(load_scenarios(root), root)
            scenario_path.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: WELCOME}\n", encoding="utf-8")
            current = snapshot(load_scenarios(root), root)
            diff = compare_snapshots(current, baseline)
            self.assertEqual(diff["recommended_bump"], "major")
            self.assertIn("intent", diff["details"][0]["changed_dimensions"])
            self.assertFalse(diff["version_sufficient"])
            self.assertEqual(migrate_scenario_versions(load_scenarios(root), baseline), ["greet"])
            migrated = load_scenarios(root)
            write_manifest(root, current, diff["recommended_version"])
            final_diff = compare_snapshots(snapshot(migrated, root), baseline)
            self.assertEqual(final_diff["unversioned_changes"], [])
            self.assertTrue(final_diff["version_sufficient"])
            self.assertEqual(migrated[0].version, 2)

    def test_impact_report_maps_changes_to_overall_and_critical_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scenario.yaml"
            source.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: GREET}\ntags: [safety]\nseverity: critical\n", encoding="utf-8")
            old = snapshot(load_scenarios(root), root)
            self.assertEqual(compare_snapshots(old, json.loads(json.dumps(old)))["changed"], [])
            source.write_text("scenario: greet\ninput: {user: hello}\nexpected: {intent: GREET}\ntags: [safety]\nseverity: critical\n", encoding="utf-8")
            current = snapshot(load_scenarios(root), root)
            compatibility = compare_snapshots(current, old)
            impact = change_impact_report(compatibility, {
                "min_overall_pass_rate": .9, "critical_tags": {"safety": 1.0},
                "critical_scenarios": {"greet": 1.0},
            })
        self.assertIn("input", impact["details"][0]["changed_dimensions"])
        self.assertEqual(impact["affected_gate_count"], 3)
        self.assertTrue(all(gate["affected"] for gate in impact["gates"]))

    def test_cli_scenario_impact_emits_github_annotations_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios"
            scenarios.mkdir()
            (scenarios / "scenario.yaml").write_text(
                "scenario: safety_check\ninput: {user: hi}\nexpected: {intent: HELP}\n"
                "tags: [safety]\nseverity: critical\n", encoding="utf-8")
            (root / "profile.yaml").write_text(
                "min_overall_pass_rate: 0.9\ncritical_tags:\n  safety: 1.0\n"
                "critical_scenarios:\n  safety_check: 1.0\n", encoding="utf-8")
            baseline = root / "baseline.json"
            summary = root / "summary.md"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["scenario-version", str(scenarios), "--format", "json",
                                       "--output", str(baseline)]), 0)
            (scenarios / "scenario.yaml").write_text(
                "scenario: safety_check\ninput: {user: hi}\nexpected: {intent: BOOK}\n"
                "tags: [safety]\nseverity: critical\n", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                status = main(["scenario-impact", str(scenarios), "--baseline", str(baseline),
                               "--profile", str(root / "profile.yaml"), "--github-annotations",
                               "--summary-file", str(summary)])
            summary_content = summary.read_text(encoding="utf-8")
        self.assertEqual(status, 0)
        self.assertIn("::warning title=Scenario impact%3A safety_check::", output.getvalue())
        self.assertIn("AFFECTED overall_pass_rate", output.getvalue())
        self.assertIn("Agent Eval scenario change impact", summary_content)

    def test_scenario_baseline_requires_explicit_approval_and_version_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios"
            scenarios.mkdir()
            scenario = scenarios / "greeting.yaml"
            scenario.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: GREET}\n", encoding="utf-8")
            manifest = scenarios / "scenario-set.yaml"
            manifest.write_text("scenario_set: sample\nversion: 0.1.0\nschema_version: 1\n", encoding="utf-8")
            approved_path = root / "approved.json"
            preview = StringIO()
            with redirect_stdout(preview):
                self.assertEqual(main(["scenario-baseline", str(scenarios), "--baseline", str(approved_path)]), 0)
            self.assertFalse(approved_path.exists())
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["scenario-baseline", str(scenarios), "--baseline", str(approved_path),
                                       "--approve", "--approved-by", "reviewer-a"]), 0)
            initial_baseline = json.loads(approved_path.read_text(encoding="utf-8"))
            scenario.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: WELCOME}\n", encoding="utf-8")
            blocked = StringIO()
            with redirect_stdout(blocked):
                self.assertEqual(main(["scenario-baseline", str(scenarios), "--baseline", str(approved_path),
                                       "--approve", "--format", "json"]), 1)
            self.assertEqual(json.loads(approved_path.read_text(encoding="utf-8")), initial_baseline)
            manifest.write_text("scenario_set: sample\nversion: 1.0.0\nschema_version: 1\n", encoding="utf-8")
            scenario.write_text("scenario: greet\nversion: 2\ninput: {user: hi}\nexpected: {intent: WELCOME}\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["scenario-baseline", str(scenarios), "--baseline", str(approved_path),
                                       "--approve", "--approved-by", "reviewer-b", "--format", "json"]), 0)
            approved = json.loads(approved_path.read_text(encoding="utf-8"))
        self.assertIn("Preview only", preview.getvalue())
        self.assertIn("per-scenario version increment", blocked.getvalue())
        self.assertEqual(approved["approval"]["approved_by"], "reviewer-b")
        self.assertEqual(approved["version"], "1.0.0")

    def test_cli_scenario_migration_and_triage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenario-set.yaml").write_text(
                "scenario_set: sample\nversion: 1.0.0\nschema_version: 1\n", encoding="utf-8")
            path = root / "greeting.yaml"
            path.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: GREET}\n", encoding="utf-8")
            baseline = root / "baseline.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["scenario-version", str(root), "--format", "json", "--output", str(baseline)]), 0)
            path.write_text("scenario: greet\ninput: {user: hi}\nexpected: {intent: WELCOME}\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["scenario-version", str(root), "--baseline", str(baseline)]), 1)
                self.assertEqual(main(["scenario-version", str(root), "--baseline", str(baseline), "--migrate"]), 0)
            self.assertIn("version: 2", path.read_text(encoding="utf-8"))

        output = StringIO()
        with redirect_stdout(output):
            status = main(["triage", str(self.ROOT / "scenarios"), "--responses",
                           str(self.ROOT / "examples/responses-before.jsonl"), "--format", "json"])
        self.assertEqual(status, 0)
        self.assertIn("failure_clusters", json.loads(output.getvalue()))

    def test_cli_execution_budget_gate(self):
        arguments = ["run", str(self.ROOT / "scenarios/clinic_hours.yaml"), "--responses",
                     str(self.ROOT / "examples/budget-responses.jsonl"), "--strict"]
        with redirect_stdout(StringIO()):
            passing = main(arguments + ["--max-cost-usd", "0.01", "--max-input-tokens", "100",
                                        "--max-output-tokens", "50", "--max-p95-latency-ms", "200"])
            failing = main(arguments + ["--max-cost-usd", "0.0001"])
        self.assertEqual(passing, 0)
        self.assertEqual(failing, 1)


if __name__ == "__main__":
    unittest.main()
