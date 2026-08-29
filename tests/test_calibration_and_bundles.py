import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from agent_eval.calibration import calibrate_judge
from agent_eval.cli import main
from agent_eval.graders import JudgeRequest
from agent_eval.models import GradeResult
from agent_eval.bundles import verify_bundle


ROOT = Path(__file__).resolve().parents[1]


class FixtureJudge:
    version = "fixture-1"

    def judge(self, request: JudgeRequest) -> GradeResult:
        score = 0.1 if "diagnosis" in request.answer else 0.9
        return GradeResult("fixture", score, score >= 0.7, "fixture rationale")


class CalibrationAndBundleTests(unittest.TestCase):
    def test_gold_set_calibration_reports_classification_and_score_metrics(self):
        report = calibrate_judge(ROOT / "examples/judge-gold-set.yaml", "fixture",
                                 {"fixture": FixtureJudge()}, min_accuracy=1.0)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["confusion_matrix"]["true_positive"], 1)
        self.assertEqual(report["confusion_matrix"]["true_negative"], 1)
        self.assertIsNotNone(report["brier_score"])
        self.assertTrue(report["passed_gate"])

    def test_calibration_cli_uses_registered_judge_and_can_gate_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            with patch("agent_eval.cli.load_judge_registry", return_value={"fixture": FixtureJudge()}):
                with redirect_stdout(StringIO()):
                    status = main(["calibrate-judge", str(ROOT / "examples/judge-gold-set.yaml"),
                                   "--judge", "fixture", "--min-accuracy", "1.0",
                                   "--format", "json", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(report["judge_version"], "fixture-1")

    def test_cli_creates_and_verifies_reproducible_run_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "run"
            with redirect_stdout(StringIO()):
                status = main(["run", str(ROOT / "scenarios"), "--responses",
                               str(ROOT / "examples/responses.jsonl"), "--strict",
                               "--format", "html", "--bundle-dir", str(bundle)])
                verify_status = main(["verify-bundle", str(bundle)])
            self.assertEqual(status, 0)
            self.assertEqual(verify_status, 0)
            for item in ("scenarios", "results.jsonl", "report.json", "scenario-snapshot.json",
                         "run-config.json", "bundle-manifest.json"):
                self.assertTrue((bundle / item).exists())
            (bundle / "results.jsonl").write_text("corrupted\n", encoding="utf-8")
            self.assertFalse(verify_bundle(bundle)["valid"])


if __name__ == "__main__":
    unittest.main()
