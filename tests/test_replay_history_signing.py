import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from agent_eval.cli import main
from agent_eval.history import HistoryError
from agent_eval.history_gates import evaluate_history_gates, load_history_gate_profile
from agent_eval.portable_baseline import (
    PortableBaselineError,
    export_portable_baseline,
    import_portable_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


class ReplayHistorySigningTests(unittest.TestCase):
    def test_replay_bundle_rescores_without_running_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            report = Path(directory) / "replayed.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "run", str(ROOT / "scenarios"), "--responses",
                    str(ROOT / "examples/responses.jsonl"), "--strict",
                    "--bundle-dir", str(bundle),
                ]), 0)
                with patch("agent_eval.cli.execute_scenarios", side_effect=AssertionError("agent called")):
                    status = main(["replay-bundle", str(bundle), "--format", "json", "--output", str(report)])
            self.assertEqual(status, 0)
            replayed = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(replayed["total"], 3)
            self.assertEqual(replayed["passed"], 3)

    def test_replay_refuses_tampered_bundle_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            with redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "run", str(ROOT / "scenarios"), "--responses",
                    str(ROOT / "examples/responses.jsonl"), "--strict",
                    "--bundle-dir", str(bundle),
                ]), 0)
            (bundle / "results.jsonl").write_text("{}\n", encoding="utf-8")
            with patch("agent_eval.cli.evaluate", side_effect=AssertionError("scored invalid bundle")):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    status = main(["replay-bundle", str(bundle)])
            self.assertEqual(status, 2)

    def test_replay_refuses_unlisted_scenario_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            with redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "run", str(ROOT / "scenarios"), "--responses",
                    str(ROOT / "examples/responses.jsonl"), "--strict",
                    "--bundle-dir", str(bundle),
                ]), 0)
            (bundle / "scenarios" / "unexpected.yaml").write_text("scenario: extra\n", encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                status = main(["replay-bundle", str(bundle)])
            self.assertEqual(status, 2)

    def test_history_gates_support_average_and_latest_execution_budgets(self):
        records = [
            {"run_id": "old", "recorded_at": "2026-01-01", "pass_rate": 1.0,
             "execution": {"cost_usd": {"total": 0.10}, "latency_ms": {"mean": 100}}},
            {"run_id": "new", "recorded_at": "2026-01-02", "pass_rate": 1.0,
             "execution": {"cost_usd": {"total": 0.30}, "latency_ms": {"mean": 300}}},
        ]
        report = evaluate_history_gates(records, {
            "min_runs": 1, "critical_scenarios": {},
            "max_average_cost_usd": 0.19,
            "max_latest_cost_usd": 0.31,
            "max_average_latency_ms": 200,
            "max_latest_latency_ms": 250,
        })
        self.assertFalse(report["passed"])
        by_type = {gate["type"]: gate for gate in report["gates"]}
        self.assertAlmostEqual(by_type["average_cost_usd"]["observed"], 0.20)
        self.assertTrue(by_type["latest_cost_usd"]["passed"])
        self.assertTrue(by_type["average_latency_ms"]["passed"])
        self.assertFalse(by_type["latest_latency_ms"]["passed"])

    def test_history_execution_gate_fails_when_a_selected_metric_is_missing(self):
        report = evaluate_history_gates([
            {"recorded_at": "2026-01-01", "execution": {"cost_usd": {"total": 0.1}}},
            {"recorded_at": "2026-01-02", "execution": {}},
        ], {"min_runs": 1, "critical_scenarios": {}, "max_average_cost_usd": 1.0,
            "max_latest_cost_usd": 1.0})
        gate = next(gate for gate in report["gates"] if gate["type"] == "average_cost_usd")
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["observations"], 1)
        latest = next(gate for gate in report["gates"] if gate["type"] == "latest_cost_usd")
        self.assertFalse(latest["passed"])
        self.assertIsNone(latest["observed"])

    def test_history_gate_profile_accepts_and_validates_execution_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.yaml"
            profile.write_text("max_average_cost_usd: 0.5\nmax_latest_latency_ms: 120\n", encoding="utf-8")
            loaded = load_history_gate_profile(profile)
            self.assertEqual(loaded["max_average_cost_usd"], 0.5)
            profile.write_text("max_average_cost_usd: -1\n", encoding="utf-8")
            with self.assertRaises(HistoryError):
                load_history_gate_profile(profile)

    def test_history_gate_cli_flags_override_execution_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, profile, output = root / "history.jsonl", root / "profile.yaml", root / "gate.json"
            record = {"schema_version": 1, "run_id": "one", "recorded_at": "2026-01-01T00:00:00+00:00",
                      "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0, "metrics": {},
                      "execution": {"cost_usd": {"total": 0.02}, "latency_ms": {"mean": 100}},
                      "critical_scenarios": {}, "scenario_results": {}}
            store.write_text(json.dumps(record) + "\n", encoding="utf-8")
            profile.write_text("min_runs: 1\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                status = main(["history", "gate", str(store), "--profile", str(profile),
                               "--max-latest-cost-usd", "0.01", "--format", "json",
                               "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(status, 1)
            self.assertFalse(report["passed"])
            self.assertAlmostEqual(report["gates"][1]["observed"], 0.02)

    @unittest.skipUnless(__import__("importlib").util.find_spec("cryptography"), "signing extra is not installed")
    def test_signed_portable_baseline_round_trip_and_tamper_detection(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        snapshot = {"scenario_set": "clinic", "version": "1.0.0", "scenarios": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, private_path, public_path, portable = (root / name for name in
                ("baseline.json", "private.pem", "public.pem", "portable.json"))
            source.write_text(json.dumps(snapshot), encoding="utf-8")
            private = Ed25519PrivateKey.generate()
            private_path.write_bytes(private.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()))
            public_path.write_bytes(private.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
            export_portable_baseline(source, portable, private_key_path=private_path)
            self.assertEqual(import_portable_baseline(portable, public_key_path=public_path), snapshot)
            envelope = json.loads(portable.read_text(encoding="utf-8"))
            envelope["snapshot"]["version"] = "2.0.0"
            portable.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaises(PortableBaselineError):
                import_portable_baseline(portable, public_key_path=public_path)

    def test_unsigned_baseline_requires_explicit_signature_policy(self):
        snapshot = {"scenario_set": "clinic", "version": "1.0.0", "scenarios": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, portable = root / "baseline.json", root / "portable.json"
            source.write_text(json.dumps(snapshot), encoding="utf-8")
            export_portable_baseline(source, portable)
            self.assertEqual(import_portable_baseline(portable), snapshot)
            with self.assertRaises(PortableBaselineError):
                import_portable_baseline(portable, require_signature=True)


if __name__ == "__main__":
    unittest.main()
