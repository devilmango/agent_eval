"""Command-line interface for the agent evaluation harness."""

from __future__ import annotations

import argparse
from datetime import date
from math import isfinite
import json
import os
import signal
import sys
from pathlib import Path
from threading import Event
from typing import Any

from agent_eval import __version__
from agent_eval.bundles import BundleError, create_bundle, replay_bundle, verify_bundle
from agent_eval.cache_tools import CacheError, explain_cache, inspect_cache, prune_cache
from agent_eval.distributed import (ShardError, create_shard_manifest, merge_shard_results,
                                    select_shard)
from agent_eval.calibration import (CalibrationError, calibrate_grounding, calibrate_judge,
                                   text_calibration_report, text_grounding_calibration_report)
from agent_eval.adapters import AdapterError, PathMappingAdapter, load_adapter_plugin, result_as_dict
from agent_eval.approval_policies import ApprovalPolicyError, load_approval_policy, policy_blockers
from agent_eval.baseline_history import (BaselineHistoryError, append_baseline_history,
                                         baseline_history_report, load_baseline_history)
from agent_eval.budgets import apply_budgets
from agent_eval.comparison import compare_reports, paired_bootstrap_intervals, text_comparison
from agent_eval.drafts import (
    DraftError, create_scenario_drafts, list_scenario_drafts,
    promote_scenario_draft, review_scenario_draft,
)
from agent_eval.gate_profiles import GateProfileError, evaluate_release_gates, load_gate_profile, text_release_gate_report
from agent_eval.graders import GraderError, load_judge_registry
from agent_eval.history import (HistoryError, compare_history, filter_history, html_history_comparison,
                                html_trend_report, load_history, merge_history, record_history,
                                text_history_comparison, text_trend_report, trend_report)
from agent_eval.history_gates import (evaluate_history_gates, load_history_gate_profile,
                                      text_history_gate_report)
from agent_eval.models import AgentResult
from agent_eval.monitoring import (drift_report, experiment_matrix, github_annotation_report,
                                   github_scenario_impact_report, read_variant_spec)
from agent_eval.minimize import MinimizeError, minimize_failure
from agent_eval.otel import OpenTelemetryAdapter, load_otel_document
from agent_eval.portable_baseline import (PortableBaselineError, export_portable_baseline,
                                          import_portable_baseline)
from agent_eval.quality import failure_analysis, mutation_report, scenario_coverage, scenario_suggestions, judge_agreement, text_coverage, text_mutation_report, text_failure_analysis
from agent_eval.report import (html_report, json_report, junit_report, repeated_html_report,
                               sarif_report, text_report)
from agent_eval.redaction import (
    RedactionError, audit_redaction_file, load_redaction_rules,
    redact_document, text_redaction_audit,
)
from agent_eval.robustness import VARIANT_TYPES, run_robustness
from agent_eval.repeated import analyze_repeated_runs, text_repeated_report
from agent_eval.results import ResultError, load_results
from agent_eval.runner import RunnerError, execute_scenarios, load_runner
from agent_eval.scenarios import ScenarioError, filter_scenarios, load_scenarios
from agent_eval.scoring import evaluate
from agent_eval.scenario_audit import audit_scenarios, text_scenario_audit
from agent_eval.versioning import (VersioningError, change_impact_report, compare_snapshots,
                                   migrate_scenario_versions, snapshot as scenario_snapshot,
                                   approved_scenario_snapshot, scenario_baseline_review,
                                   write_scenario_baseline,
                                   text_version_report, write_manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Evaluate agent behavior against YAML scenarios and recorded JSONL results.",
    )
    parser.add_argument("--version", action="version", version=f"agent-eval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score recorded agent results against scenarios")
    run.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    run.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="JSON Lines file with one agent result per scenario",
    )
    run.add_argument("--format", choices=("text", "json", "html", "junit", "sarif"), default="text")
    run.add_argument("--output", type=Path, help="write the report to a file instead of stdout")
    run.add_argument("--strict", action="store_true", help="fail when any scenario result is missing")
    run.add_argument("--quiet-details", action="store_true", help="hide per-scenario failure details")
    run.add_argument("--judge-cache", type=Path, help="cache semantic judge results in this JSON file")
    run.add_argument("--judge-repetitions", type=int, default=1, help="repeat semantic judgments for stability (default: 1)")
    run.add_argument("--bundle-dir", type=Path, help="write a self-contained, hash-verified run bundle to a new directory")
    _add_budget_arguments(run)
    _add_tag_filters(run)

    adapt = commands.add_parser("adapt", help="convert framework JSONL traces into Agent Eval results")
    adapt.add_argument("traces", type=Path, help="raw JSON Lines trace file")
    adapt.add_argument("--mapping", required=True, type=Path, help="YAML field-path mapping")
    adapt.add_argument("--output", type=Path, help="output JSONL path (defaults to stdout)")
    adapt.add_argument("--redaction-config", type=Path, help="YAML rules applied to raw traces before conversion")

    framework = commands.add_parser("adapt-framework", help="convert traces with an installed framework adapter")
    framework.add_argument("adapter", help="adapter entry-point name")
    framework.add_argument("traces", type=Path, help="framework trace JSON Lines file")
    framework.add_argument("--output", type=Path, help="output JSONL path (defaults to stdout)")
    framework.add_argument("--redaction-config", type=Path, help="YAML rules applied to raw traces before conversion")

    otel = commands.add_parser("adapt-otel", help="convert an OTLP/JSON trace export into Agent Eval results")
    otel.add_argument("traces", type=Path, help="OTLP/JSON trace export containing resourceSpans")
    otel.add_argument("--output", type=Path, help="output JSONL path (defaults to stdout)")
    otel.add_argument("--redaction-config", type=Path, help="YAML rules applied to the OTLP document before conversion")
    otel.add_argument("--scenario-attribute", default="agent_eval.scenario")
    otel.add_argument("--intent-attribute", default="agent_eval.intent")
    otel.add_argument(
        "--include-sensitive-attributes",
        action="store_true",
        help="include tool arguments and agent output text from trace attributes",
    )

    compare = commands.add_parser("compare", help="compare candidate results with a baseline run")
    compare.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    compare.add_argument("--baseline", required=True, type=Path, help="baseline JSONL results")
    compare.add_argument("--candidate", required=True, type=Path, help="candidate JSONL results")
    compare.add_argument("--format", choices=("text", "json"), default="text")
    compare.add_argument("--strict", action="store_true", help="fail when either run is missing results")
    compare.add_argument("--confidence", type=float, default=0.95, help="paired bootstrap confidence level (default: 0.95)")
    compare.add_argument("--bootstrap-samples", type=int, default=2000, help="paired bootstrap resamples (minimum: 100)")
    compare.add_argument("--seed", type=int, default=0, help="bootstrap random seed for reproducible intervals")
    compare.add_argument("--github-annotations", action="store_true", help="emit GitHub Actions error annotations for regressed scenarios")
    compare.add_argument("--summary-file", type=Path, help="append a Markdown summary (defaults to GITHUB_STEP_SUMMARY when set)")
    _add_tag_filters(compare)

    monitor = commands.add_parser("monitor", help="compare sampled current behavior with a recorded baseline")
    monitor.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    monitor.add_argument("--baseline", required=True, type=Path, help="baseline canonical result JSONL")
    monitor.add_argument("--current", required=True, type=Path, help="current sampled canonical result JSONL")
    monitor.add_argument("--max-absolute-drift", type=float, default=0.05, help="maximum absolute change allowed for each metric (0–1; default: 0.05)")
    monitor.add_argument("--format", choices=("text", "json"), default="text")
    monitor.add_argument("--output", type=Path)
    _add_tag_filters(monitor)

    matrix = commands.add_parser("experiment", help="compare named model or prompt result sets")
    matrix.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    matrix.add_argument("--variant", action="append", required=True, metavar="NAME=RESULTS.jsonl", help="named result set; repeat at least twice")
    matrix.add_argument("--format", choices=("text", "json"), default="text")
    matrix.add_argument("--output", type=Path)
    matrix.add_argument("--confidence", type=float, default=0.95, help="paired bootstrap confidence level (default: 0.95)")
    matrix.add_argument("--bootstrap-samples", type=int, default=2000, help="paired bootstrap resamples (minimum: 100)")
    matrix.add_argument("--seed", type=int, default=0, help="bootstrap random seed for reproducible intervals")
    _add_tag_filters(matrix)

    history = commands.add_parser("history", help="store evaluation summaries and inspect quality trends")
    history_commands = history.add_subparsers(dest="history_command", required=True)
    history_record = history_commands.add_parser("record", help="append an evaluation or verified bundle summary")
    history_record.add_argument("--store", required=True, type=Path, help="append-only JSONL history path")
    history_sources = history_record.add_mutually_exclusive_group(required=True)
    history_sources.add_argument("--responses", type=Path, help="canonical result JSONL to evaluate")
    history_sources.add_argument("--bundle", type=Path, help="verified run bundle to summarize")
    history_record.add_argument("--scenarios", type=Path, help="scenario directory or file (required with --responses)")
    history_record.add_argument("--label", help="human-readable release, model, or prompt label")
    history_record.add_argument("--strict", action="store_true", help="require a result for every scenario")
    history_record.add_argument("--format", choices=("text", "json"), default="text")
    history_trend = history_commands.add_parser("trend", help="summarize metric trends from a history file")
    history_trend.add_argument("store", type=Path, help="evaluation history JSONL")
    history_trend.add_argument("--limit", type=int, help="show only the latest N records")
    history_trend.add_argument("--format", choices=("text", "json", "html"), default="text")
    history_trend.add_argument("--output", type=Path)
    history_trend.add_argument("--model", action="append", help="include runs matching this model (repeatable)")
    history_trend.add_argument("--prompt-version", action="append", help="include runs matching this prompt version (repeatable)")
    history_trend.add_argument("--label", action="append", help="include runs with this exact label (repeatable)")
    history_trend.add_argument("--since", help="inclusive ISO-8601 date or timestamp")
    history_trend.add_argument("--until", help="inclusive ISO-8601 date or timestamp")
    history_compare = history_commands.add_parser("compare", help="compare selected runs or releases side by side")
    history_compare.add_argument("store", type=Path, help="evaluation history JSONL")
    history_compare.add_argument("--run-id", action="append", help="select an exact run ID (repeatable; order sets baseline)")
    history_compare.add_argument("--label", action="append", help="select all runs with this exact label (repeatable)")
    history_compare.add_argument("--format", choices=("text", "json", "html"), default="text")
    history_compare.add_argument("--output", type=Path)
    history_merge = history_commands.add_parser("merge", help="merge history stores and deduplicate identical run IDs")
    history_merge.add_argument("stores", nargs="+", type=Path, help="input history JSONL stores")
    history_merge.add_argument("--output", required=True, type=Path, help="new merged history JSONL path")
    history_merge.add_argument("--format", choices=("text", "json"), default="text")
    history_gate = history_commands.add_parser("gate", help="gate recent evaluation quality against recorded history")
    history_gate.add_argument("store", type=Path, help="evaluation history JSONL")
    history_gate.add_argument("--profile", required=True, type=Path, help="YAML history release gate profile")
    history_gate.add_argument("--limit", type=int, help="evaluate only the most recent N matching runs")
    history_gate.add_argument("--label", action="append", help="include records with this label (repeatable)")
    history_gate.add_argument("--model", action="append", help="include records matching this model (repeatable)")
    history_gate.add_argument("--prompt-version", action="append", help="include records matching this prompt version (repeatable)")
    history_gate.add_argument("--since", help="inclusive ISO-8601 date or timestamp")
    history_gate.add_argument("--until", help="inclusive ISO-8601 date or timestamp")
    history_gate.add_argument("--max-average-cost-usd", type=float,
                              help="fail if average recorded total cost exceeds this amount")
    history_gate.add_argument("--max-latest-cost-usd", type=float,
                              help="fail if the latest run's recorded total cost exceeds this amount")
    history_gate.add_argument("--max-average-latency-ms", type=float,
                              help="fail if average recorded mean latency exceeds this value")
    history_gate.add_argument("--max-latest-latency-ms", type=float,
                              help="fail if the latest run's recorded mean latency exceeds this value")
    history_gate.add_argument("--format", choices=("text", "json"), default="text")
    history_gate.add_argument("--output", type=Path)

    drafts = commands.add_parser("draft-scenarios", help="create human-review YAML drafts from trace JSONL")
    drafts.add_argument("traces", type=Path, help="raw trace records as JSON Lines")
    drafts.add_argument("--mapping", required=True, type=Path, help="trace-to-result YAML field mapping")
    drafts.add_argument("--input-path", required=True, help="dot path to the captured user input in each trace")
    drafts.add_argument("--output-dir", required=True, type=Path, help="new directory for review-only YAML drafts")
    drafts.add_argument("--redaction-config", type=Path, help="YAML rules applied to each raw trace before draft creation")
    drafts.add_argument("--format", choices=("text", "json"), default="text")
    drafts.add_argument("--report", type=Path, help="write the draft creation summary to a file")

    redaction_audit = commands.add_parser("redaction-audit", help="check configured redaction rule coverage over JSONL traces")
    redaction_audit.add_argument("traces", type=Path, help="raw trace records as JSON Lines")
    redaction_audit.add_argument("--config", required=True, type=Path, help="redaction policy YAML")
    redaction_audit.add_argument("--strict", action="store_true", help="fail when any required rule misses a trace record")
    redaction_audit.add_argument("--format", choices=("text", "json"), default="text")
    redaction_audit.add_argument("--output", type=Path)

    scenario_audit = commands.add_parser("scenario-audit", help="check owners and review freshness for critical scenarios")
    scenario_audit.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    scenario_audit.add_argument("--max-review-age-days", type=int, default=90)
    scenario_audit.add_argument("--as-of", help="date used for review-age calculation (YYYY-MM-DD)")
    scenario_audit.add_argument("--format", choices=("text", "json"), default="text")
    scenario_audit.add_argument("--output", type=Path)

    draft_admin = commands.add_parser("drafts", help="list, review, and promote trace-derived scenario drafts")
    draft_commands = draft_admin.add_subparsers(dest="draft_command", required=True)
    draft_list = draft_commands.add_parser("list", help="show review status and exact-input duplicates")
    draft_list.add_argument("directory", type=Path)
    draft_list.add_argument("--format", choices=("text", "json"), default="text")
    draft_list.add_argument("--output", type=Path)
    draft_review = draft_commands.add_parser("review", help="approve or reject a draft after editing expected behavior")
    draft_review.add_argument("draft", type=Path)
    draft_review.add_argument("--status", required=True, choices=("approved", "rejected"))
    draft_review.add_argument("--reviewer", required=True)
    draft_review.add_argument("--notes")
    draft_promote = draft_commands.add_parser("promote", help="copy an approved draft into an executable scenario set")
    draft_promote.add_argument("draft", type=Path)
    draft_promote.add_argument("--output-dir", required=True, type=Path)
    draft_promote.add_argument("--allow-duplicate", action="store_true", help="allow promotion after reviewing an exact-input duplicate")

    gate = commands.add_parser("gate", help="apply risk-based release thresholds to one or more runs")
    gate.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    gate.add_argument("--runs", nargs="+", required=True, type=Path, help="one or more canonical result JSONL files")
    gate.add_argument("--profile", required=True, type=Path, help="YAML release gate profile")
    gate.add_argument("--format", choices=("text", "json"), default="text")
    gate.add_argument("--output", type=Path)

    repeat = commands.add_parser("repeat", help="analyze variability across repeated result files")
    repeat.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    repeat.add_argument("--runs", nargs="+", required=True, type=Path, help="two or more JSONL result files")
    repeat.add_argument("--format", choices=("text", "json", "html"), default="text")
    repeat.add_argument("--output", type=Path, help="write the report to a file instead of stdout")
    repeat.add_argument("--strict", action="store_true", help="fail when any run is missing scenario results")
    repeat.add_argument("--min-runs", type=int, default=2, help="minimum number of run files required (default: 2)")
    repeat.add_argument("--confidence", type=float, default=0.95, help="Wilson interval confidence, between 0.5 and 1 (default: 0.95)")
    repeat.add_argument(
        "--min-pass-rate", type=float,
        help="fail unless the overall Wilson lower bound meets this pass-rate threshold (0–1)",
    )
    _add_tag_filters(repeat)

    run_agent = commands.add_parser("run-agent", help="execute scenarios through an installed agent runner")
    run_agent.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    run_agent.add_argument("--runner", required=True, help="runner entry-point name")
    run_agent.add_argument("--results", required=True, type=Path, help="write captured canonical JSONL results here")
    run_agent.add_argument("--workers", type=int, default=1,
                           help="maximum concurrent scenario executions (default: 1)")
    run_agent.add_argument("--checkpoint", type=Path,
                           help="atomically save completed scenarios and resume matching work")
    run_agent.add_argument("--cache", type=Path,
                           help="reuse matching results across runs from this incremental cache")
    run_agent.add_argument("--check-cache", action="store_true",
                           help="check cache format and runner identity before executing")
    run_agent.add_argument("--provider-limit", action="append", default=[], metavar="PROVIDER=N",
                           help="cap simultaneous calls for a provider (repeatable)")
    run_agent.add_argument("--scenario-timeout", type=float,
                           help="per-scenario timeout in seconds (runner can poll execution_control)")
    run_agent.add_argument("--shard-manifest", type=Path,
                           help="run one partition described by an agent-eval shard manifest")
    run_agent.add_argument("--shard-index", type=int,
                           help="zero-based shard index; requires --shard-manifest")
    run_agent.add_argument("--report", type=Path, help="write evaluation report here (defaults to stdout)")
    run_agent.add_argument("--format", choices=("text", "json", "html", "junit", "sarif"), default="text")
    run_agent.add_argument("--judge-cache", type=Path, help="cache semantic judge results in this JSON file")
    run_agent.add_argument("--judge-repetitions", type=int, default=1, help="repeat semantic judgments for stability (default: 1)")
    run_agent.add_argument("--bundle-dir", type=Path, help="write a self-contained, hash-verified run bundle to a new directory")
    _add_budget_arguments(run_agent)
    _add_tag_filters(run_agent)

    cache = commands.add_parser("cache", help="inspect, explain, check, or prune an execution cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_inspect = cache_commands.add_parser("inspect", help="show cache metadata without exposing results")
    cache_inspect.add_argument("cache_file", type=Path)
    cache_inspect.add_argument("--format", choices=("text", "json"), default="text")
    cache_inspect.add_argument("--output", type=Path)
    for cache_command, help_text in (("check", "check runner and scenario compatibility before a run"),
                                     ("explain", "explain cache hits and misses for scenarios")):
        cache_detail = cache_commands.add_parser(cache_command, help=help_text)
        cache_detail.add_argument("cache_file", type=Path)
        cache_detail.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
        cache_detail.add_argument("--runner-key", required=True, help="same runner identity used by run-agent")
        cache_detail.add_argument("--format", choices=("text", "json"), default="text")
        cache_detail.add_argument("--output", type=Path)
    cache_prune = cache_commands.add_parser("prune", help="remove entries older than a specified age")
    cache_prune.add_argument("cache_file", type=Path)
    cache_prune.add_argument("--older-than-days", type=float, required=True)
    cache_prune.add_argument("--dry-run", action="store_true", help="report eligible entries without deleting them")
    cache_prune.add_argument("--include-unknown-age", action="store_true",
                             help="also remove legacy entries that have no completion timestamp")
    cache_prune.add_argument("--format", choices=("text", "json"), default="text")
    cache_prune.add_argument("--output", type=Path)

    minimize = commands.add_parser("minimize", help="reduce a recorded failure to a smaller reproducible trace")
    minimize.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    minimize.add_argument("--responses", required=True, type=Path, help="canonical JSONL results from the failing run")
    minimize.add_argument("--scenario", required=True, help="scenario name to minimize")
    minimize.add_argument("--output", type=Path, help="write the minimized reproduction report to a JSON file")

    shard = commands.add_parser("shard", help="write a deterministic manifest for distributed scenario execution")
    shard.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    shard.add_argument("--count", type=int, required=True, help="number of CI workers/shards")
    shard.add_argument("--output", type=Path, required=True, help="write the shard manifest JSON")

    merge = commands.add_parser("merge-results", help="validate and merge distributed shard result files")
    merge.add_argument("scenarios", type=Path, help="scenario YAML file or directory used to create the manifest")
    merge.add_argument("--manifest", required=True, type=Path, help="manifest from the shard command")
    merge.add_argument("--shard-result", action="append", default=[], metavar="INDEX=PATH",
                       help="result JSONL file for a shard (repeat once per shard)")
    merge.add_argument("--output", required=True, type=Path, help="merged canonical results JSONL")

    coverage = commands.add_parser("coverage", help="summarize behavior dimensions declared by scenarios")
    coverage.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    coverage.add_argument("--format", choices=("text", "json"), default="text")
    coverage.add_argument("--output", type=Path, help="write the report to a file instead of stdout")

    suggest = commands.add_parser("suggest", help="generate review-only candidates from scenario coverage gaps")
    suggest.add_argument("scenarios", type=Path)
    suggest.add_argument("--limit", type=int, default=20)
    suggest.add_argument("--output", type=Path)

    audit = commands.add_parser("judge-agreement", help="compare semantic judge decisions across JSON evaluation reports")
    audit.add_argument("reports", nargs="+", type=Path)
    audit.add_argument("--output", type=Path)

    triage = commands.add_parser("triage", help="cluster evaluation failures and report performance slices")
    triage.add_argument("scenarios", type=Path)
    triage.add_argument("--responses", required=True, type=Path)
    triage.add_argument("--format", choices=("text", "json"), default="text")
    triage.add_argument("--output", type=Path)
    _add_tag_filters(triage)

    version = commands.add_parser("scenario-version", help="snapshot or compare a versioned scenario set")
    version.add_argument("scenarios", type=Path)
    version.add_argument("--baseline", type=Path, help="previous scenario-version JSON snapshot")
    version.add_argument("--format", choices=("text", "json"), default="text")
    version.add_argument("--output", type=Path)
    version.add_argument("--write-manifest", action="store_true", help="migrate/write scenario-set.yaml")
    version.add_argument("--migrate", action="store_true", help="bump changed scenario versions and write the recommended set version")

    baseline = commands.add_parser("scenario-baseline", help="review and explicitly approve a scenario snapshot baseline")
    baseline.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    baseline.add_argument("--baseline", required=True, type=Path, help="approved snapshot path to review or update")
    baseline.add_argument("--approve", action="store_true", help="write the reviewed snapshot after version checks pass")
    baseline.add_argument("--approved-by", help="optional reviewer identifier recorded in approval metadata")
    baseline.add_argument("--history-log", type=Path, help="append-only approval log (default: <baseline>.history.jsonl)")
    baseline.add_argument("--policy", type=Path, help="optional YAML approval policy")
    baseline.add_argument("--profile", type=Path, help="release gate profile used by approval policies")
    baseline.add_argument("--runs", type=Path, help="candidate result JSONL used by release gate approval policies")
    baseline.add_argument("--format", choices=("text", "json"), default="text")
    baseline.add_argument("--output", type=Path, help="write the review report to a file instead of stdout")

    baseline_history = commands.add_parser("scenario-baseline-history", help="inspect append-only scenario baseline approvals")
    baseline_history.add_argument("history_log", type=Path)
    baseline_history.add_argument("--limit", type=int, help="show only the most recent approvals")
    baseline_history.add_argument("--format", choices=("text", "json"), default="text")
    baseline_history.add_argument("--output", type=Path)

    baseline_restore = commands.add_parser("scenario-baseline-restore", help="restore a previously approved scenario snapshot")
    baseline_restore.add_argument("--baseline", required=True, type=Path)
    baseline_restore.add_argument("--history-log", required=True, type=Path)
    baseline_restore.add_argument("--approval-id", required=True, help="approval ID from scenario-baseline-history")
    baseline_restore.add_argument("--approve", action="store_true", help="confirm the restore and update the baseline")
    baseline_restore.add_argument("--approved-by", help="reviewer recorded for the restore")
    baseline_restore.add_argument("--format", choices=("text", "json"), default="text")
    baseline_restore.add_argument("--output", type=Path)

    baseline_export = commands.add_parser("scenario-baseline-export", help="export a baseline as a portable integrity-checked JSON file")
    baseline_export.add_argument("baseline", type=Path, help="approved scenario snapshot JSON")
    baseline_export.add_argument("--output", required=True, type=Path, help="portable baseline destination")
    baseline_export.add_argument("--private-key", type=Path, help="optional Ed25519 private PEM key used to sign the export")
    baseline_export.add_argument("--key-id", help="optional identifier recorded with the signature")
    baseline_export.add_argument("--format", choices=("text", "json"), default="text")

    baseline_import = commands.add_parser("scenario-baseline-import", help="import a portable baseline into a scenario set")
    baseline_import.add_argument("portable", type=Path, help="portable baseline JSON export")
    baseline_import.add_argument("--scenarios", required=True, type=Path, help="target scenario file or directory")
    baseline_import.add_argument("--baseline", required=True, type=Path, help="destination approved snapshot JSON")
    baseline_import.add_argument("--history-log", type=Path, help="append-only approval log (default: <baseline>.history.jsonl)")
    baseline_import.add_argument("--approve", action="store_true", help="confirm importing and activating this baseline")
    baseline_import.add_argument("--approved-by", help="reviewer recorded for the import")
    baseline_import.add_argument("--public-key", type=Path, help="trusted Ed25519 public PEM key for signed imports")
    baseline_import.add_argument("--require-signature", action="store_true", help="reject unsigned portable baselines")
    baseline_import.add_argument("--format", choices=("text", "json"), default="text")
    baseline_import.add_argument("--output", type=Path)

    impact = commands.add_parser("scenario-impact", help="report scenario edits and the release gates they can affect")
    impact.add_argument("scenarios", type=Path)
    impact.add_argument("--baseline", required=True, type=Path, help="approved scenario-version JSON snapshot")
    impact.add_argument("--profile", required=True, type=Path, help="release gate profile YAML")
    impact.add_argument("--format", choices=("text", "json"), default="text")
    impact.add_argument("--output", type=Path)
    impact.add_argument("--github-annotations", action="store_true", help="emit GitHub Actions annotations and step summary")
    impact.add_argument("--summary-file", type=Path, help="append a Markdown summary (defaults to GITHUB_STEP_SUMMARY when set)")

    calibrate = commands.add_parser("calibrate-judge", help="measure a semantic judge against human-labeled gold examples")
    calibrate.add_argument("gold_set", type=Path, help="YAML gold set")
    calibrate.add_argument("--judge", required=True, help="semantic judge plugin entry-point name")
    calibrate.add_argument("--model", help="judge model identifier recorded in each request")
    calibrate.add_argument("--threshold", type=float, default=0.7, help="score threshold for pass/fail classification")
    calibrate.add_argument("--min-accuracy", type=float, help="fail unless gold-set accuracy meets this threshold")
    calibrate.add_argument("--format", choices=("text", "json"), default="text")
    calibrate.add_argument("--output", type=Path)

    calibrate_ground = commands.add_parser(
        "calibrate-grounding", help="measure evidence-grounding checks against human-reviewed examples",
    )
    calibrate_ground.add_argument("gold_set", type=Path, help="human-reviewed evidence-grounding YAML gold set")
    calibrate_ground.add_argument("--min-accuracy", type=float,
                                  help="fail unless accuracy over reviewed examples reaches this threshold")
    calibrate_ground.add_argument("--judge", help="optional semantic judge plugin; compares entailment with lexical results")
    calibrate_ground.add_argument("--format", choices=("text", "json"), default="text")
    calibrate_ground.add_argument("--output", type=Path)

    robustness = commands.add_parser(
        "robustness", help="run controlled input and tool-result perturbations against an agent",
    )
    robustness.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    robustness.add_argument("--runner", required=True, help="installed agent runner plugin")
    robustness.add_argument("--type", action="append", choices=(
        "paraphrase", "typo", "ambiguous", "prompt_injection", "conflicting_tool_results",
    ), help="variant type to run (repeatable; defaults to all types)")
    robustness.add_argument("--format", choices=("text", "json"), default="text")
    robustness.add_argument("--output", type=Path)

    verify = commands.add_parser("verify-bundle", help="check hashes and integrity of a run artifact bundle")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--format", choices=("text", "json"), default="text")
    verify.add_argument("--output", type=Path)

    replay = commands.add_parser(
        "replay-bundle",
        help="re-score a verified bundle without running its agent (semantic judge plugins may make provider calls)",
    )
    replay.add_argument("bundle", type=Path, help="self-contained evaluation bundle")
    replay.add_argument("--format", choices=("text", "json", "html", "junit", "sarif"), default="text")
    replay.add_argument("--output", type=Path, help="write the fresh report instead of stdout")
    replay.add_argument("--strict", action="store_true", help="fail when a bundle is missing scenario results")
    replay.add_argument("--judge-cache", type=Path, help="cache semantic judge results in this JSON file")
    replay.add_argument("--judge-repetitions", type=int, default=1,
                        help="repeat semantic judgments for stability (default: 1)")
    replay.add_argument("--no-judges", action="store_true",
                        help="skip semantic judge graders; deterministic local graders still run")

    mutate = commands.add_parser("mutate", help="check whether scenarios catch representative trace regressions")
    mutate.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    mutate.add_argument("--responses", required=True, type=Path, help="canonical JSONL agent results")
    mutate.add_argument("--format", choices=("text", "json"), default="text")
    mutate.add_argument("--output", type=Path, help="write the report to a file instead of stdout")
    _add_tag_filters(mutate)
    return parser


def _add_tag_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", action="append", default=[], help="include scenarios matching any tag; repeatable")
    parser.add_argument("--exclude-tag", action="append", default=[], help="exclude scenarios with this tag; repeatable")


def _add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-cost-usd", type=float, help="fail if total recorded run cost exceeds this amount")
    parser.add_argument("--max-input-tokens", type=int, help="fail if total input tokens exceed this count")
    parser.add_argument("--max-output-tokens", type=int, help="fail if total output tokens exceed this count")
    parser.add_argument("--max-p95-latency-ms", type=float, help="fail if nearest-rank p95 scenario latency exceeds this value")


def _adapt_jsonl(input_path: Path, adapter: PathMappingAdapter, redaction_rules=None) -> list[str]:
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AdapterError(f"Could not read trace file {input_path}: {exc}") from exc
    output: list[str] = []
    seen_scenarios: set[str] = set()
    for number, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            record: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise AdapterError(f"line {number}: each trace record must be a JSON object")
        record = redact_document(record, redaction_rules or [])
        try:
            result = adapter.adapt(record)
        except Exception as exc:
            raise AdapterError(f"line {number}: {exc}") from exc
        if not isinstance(result, AgentResult):
            raise AdapterError(f"line {number}: adapter must return an AgentResult")
        if result.scenario in seen_scenarios:
            raise AdapterError(f"line {number}: duplicate trace for '{result.scenario}'")
        seen_scenarios.add(result.scenario)
        output.append(json.dumps(result_as_dict(result), ensure_ascii=False))
    return output


def _selected_results(path: Path, all_scenarios: list, selected_scenarios: list) -> dict:
    results = load_results(path)
    all_names = {scenario.name for scenario in all_scenarios}
    selected_names = {scenario.name for scenario in selected_scenarios}
    return {
        name: result
        for name, result in results.items()
        if name not in all_names or name in selected_names
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cache":
            if args.cache_command == "inspect":
                report = inspect_cache(args.cache_file)
                if args.format == "json":
                    rendered = json.dumps(report, indent=2, ensure_ascii=False)
                else:
                    lines = [f"Execution cache: {report['path']}",
                             f"Runner: {report['runner_key'] or '(unknown)' }",
                             f"Entries: {report['entry_count']} ({report['dated_entries']} dated, "
                             f"{report['unknown_age_entries']} age unknown)"]
                    lines.extend(f"  {entry['scenario']}: {entry['age_days']} days old" if entry['age_days'] is not None
                                 else f"  {entry['scenario']}: age unknown" for entry in report["entries"])
                    rendered = "\n".join(lines)
                _write_output(args.output, rendered)
                return 0
            if args.cache_command in {"check", "explain"}:
                requested_scenarios = load_scenarios(args.scenarios)
                report = explain_cache(args.cache_file, requested_scenarios, runner_key=args.runner_key)
                if args.format == "json":
                    rendered = json.dumps(report, indent=2, ensure_ascii=False)
                else:
                    lines = [f"Cache compatibility: {'compatible' if report['compatible'] else 'incompatible'}",
                             f"Runner: {'matches' if report['runner_key_matches'] else 'does not match'}",
                             f"Scenario cache hits: {report['hit_count']}; misses: {report['miss_count']}"]
                    lines.extend(f"  {item['status'].upper()} {item['scenario']}: {item['reason']}"
                                 for item in report["scenarios"])
                    if report["unused_entries"]:
                        lines.append("Unused cache entries: " + ", ".join(report["unused_entries"]))
                    rendered = "\n".join(lines)
                _write_output(args.output, rendered)
                return 0 if report["compatible"] else 1
            report = prune_cache(args.cache_file, older_than_days=args.older_than_days,
                                 dry_run=args.dry_run, include_unknown_age=args.include_unknown_age)
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else (
                f"Cache prune {'preview' if args.dry_run else 'complete'}: "
                f"{report['removed_count']} removed; {report['remaining_count']} remain"
                + ("\nRemoved: " + ", ".join(report["removed"]) if report["removed"] else "")
                + ("\nKept age-unknown entries: " + ", ".join(report["retained_unknown_age"])
                   if report["retained_unknown_age"] else "")
            )
            _write_output(args.output, rendered)
            return 0

        if args.command == "minimize":
            scenario_rows = load_scenarios(args.scenarios)
            scenario = next((item for item in scenario_rows if item.name == args.scenario), None)
            if scenario is None:
                raise MinimizeError(f"scenario '{args.scenario}' was not found")
            results = load_results(args.responses)
            result = results.get(args.scenario)
            if result is None:
                raise MinimizeError(f"no recorded result for scenario '{args.scenario}'")
            report = minimize_failure(scenario, result)
            _write_output(args.output, json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        if args.command == "verify-bundle":
            verification = verify_bundle(args.bundle)
            rendered = json.dumps(verification, indent=2, ensure_ascii=False) if args.format == "json" else (
                f"Bundle {verification['run_id']}: "
                f"{'VALID' if verification['valid'] else 'INVALID'} "
                f"({verification['verified_files']} files checked)"
                + ("\n" + "\n".join(verification["errors"]) if verification["errors"] else "")
            )
            _write_output(args.output, rendered)
            return 0 if verification["valid"] else 1

        if args.command == "replay-bundle":
            if args.judge_repetitions < 1:
                raise GraderError("--judge-repetitions must be at least one")
            verification, report = replay_bundle(
                args.bundle, strict=args.strict, judge_cache=args.judge_cache,
                judge_repetitions=args.judge_repetitions,
                skip_semantic_judges=args.no_judges,
            )
            if args.format == "json":
                rendered = json_report(report)
            elif args.format == "html":
                rendered = html_report(report)
            elif args.format == "junit":
                rendered = junit_report(report)
            elif args.format == "sarif":
                rendered = sarif_report(report)
            else:
                rendered = f"Replayed bundle {verification['run_id']}\n\n{text_report(report)}"
            _write_output(args.output, rendered)
            return 0 if report.failed == 0 else 1

        if args.command == "redaction-audit":
            rules = load_redaction_rules(args.config)
            if not rules:
                raise RedactionError("redaction policy must define at least one rule")
            report = audit_redaction_file(args.traces, rules)
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else text_redaction_audit(report)
            _write_output(args.output, rendered)
            return 1 if args.strict and not report["passed"] else 0

        if args.command == "calibrate-judge":
            report = calibrate_judge(
                args.gold_set, args.judge, load_judge_registry(), model=args.model,
                threshold=args.threshold, min_accuracy=args.min_accuracy,
            )
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else text_calibration_report(report)
            _write_output(args.output, rendered)
            return 0 if report["passed_gate"] else 1

        if args.command == "calibrate-grounding":
            report = calibrate_grounding(
                args.gold_set, min_accuracy=args.min_accuracy,
                judge_name=args.judge,
                judges=load_judge_registry() if args.judge else None,
            )
            rendered = (json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json"
                        else text_grounding_calibration_report(report))
            _write_output(args.output, rendered)
            return 0 if report["passed_gate"] else 1

        if args.command == "adapt":
            adapter = PathMappingAdapter.from_yaml(args.mapping)
            output = _adapt_jsonl(args.traces, adapter, load_redaction_rules(args.redaction_config))
            rendered = "\n".join(output) + ("\n" if output else "")
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
            else:
                sys.stdout.write(rendered)
            return 0

        if args.command == "adapt-framework":
            adapter = load_adapter_plugin(args.adapter)
            output = _adapt_jsonl(args.traces, adapter, load_redaction_rules(args.redaction_config))
            _write_output(args.output, "\n".join(output))
            return 0

        if args.command == "adapt-otel":
            adapter = OpenTelemetryAdapter(
                scenario_attribute=args.scenario_attribute,
                intent_attribute=args.intent_attribute,
                include_sensitive_attributes=args.include_sensitive_attributes,
            )
            trace_document = redact_document(
                load_otel_document(args.traces), load_redaction_rules(args.redaction_config)
            )
            results = adapter.adapt_document(trace_document)
            seen: set[str] = set()
            lines = []
            for result in results:
                if result.scenario in seen:
                    raise AdapterError(f"duplicate OpenTelemetry run for '{result.scenario}'")
                seen.add(result.scenario)
                lines.append(json.dumps(result_as_dict(result), ensure_ascii=False))
            _write_output(args.output, "\n".join(lines))
            return 0

        if args.command == "draft-scenarios":
            report = create_scenario_drafts(
                args.traces, args.mapping, args.input_path, args.output_dir,
                redaction_config=args.redaction_config,
            )
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else (
                f"Created {report['draft_count']} review-only scenario draft(s) in {report['output_dir']}\n"
                "Expected behavior fields are blank and require human review."
            )
            _write_output(args.report, rendered)
            return 0

        if args.command == "drafts":
            if args.draft_command == "list":
                drafts_report = list_scenario_drafts(args.directory)
                if args.format == "json":
                    rendered = json.dumps({"draft_count": len(drafts_report), "drafts": drafts_report}, indent=2, ensure_ascii=False)
                else:
                    lines = [f"Scenario drafts: {len(drafts_report)}", ""]
                    for item in drafts_report:
                        duplicate = f" (duplicate input of {item['duplicate_of']})" if item["duplicate_of"] else ""
                        lines.append(f"  {item['status']:>9}  {item['file']}{duplicate}")
                    rendered = "\n".join(lines)
                _write_output(args.output, rendered)
                return 0
            if args.draft_command == "review":
                reviewed = review_scenario_draft(
                    args.draft, status=args.status, reviewer=args.reviewer, notes=args.notes,
                )
                print(json.dumps(reviewed, indent=2, ensure_ascii=False))
                return 0
            promoted = promote_scenario_draft(
                args.draft, args.output_dir, allow_duplicate=args.allow_duplicate,
            )
            print(json.dumps(promoted, indent=2, ensure_ascii=False))
            return 0

        if args.command == "history":
            if args.history_command == "trend":
                records = filter_history(
                    load_history(args.store), models=args.model, prompt_versions=args.prompt_version,
                    labels=args.label, since=args.since, until=args.until,
                )
                report = trend_report(records, limit=args.limit)
                if args.format == "json":
                    rendered = json.dumps(report, indent=2, ensure_ascii=False)
                elif args.format == "html":
                    rendered = html_trend_report(report)
                else:
                    rendered = text_trend_report(report)
                _write_output(args.output, rendered)
                return 0
            if args.history_command == "compare":
                report = compare_history(load_history(args.store), args.run_id, args.label)
                if args.format == "json":
                    rendered = json.dumps(report, indent=2, ensure_ascii=False)
                elif args.format == "html":
                    rendered = html_history_comparison(report)
                else:
                    rendered = text_history_comparison(report)
                _write_output(args.output, rendered)
                return 0
            if args.history_command == "merge":
                report = merge_history(args.stores, args.output)
                if args.format == "json":
                    rendered = json.dumps({key: value for key, value in report.items() if key != "records"},
                                          indent=2, ensure_ascii=False)
                else:
                    rendered = (f"Merged {report['record_count']} unique runs from {report['input_count']} stores; "
                                f"removed {report['duplicate_count']} duplicate run ID(s) into {args.output}")
                print(rendered)
                return 0
            if args.history_command == "gate":
                if args.limit is not None and args.limit < 1:
                    raise HistoryError("history gate --limit must be at least 1")
                records = filter_history(
                    load_history(args.store), models=args.model, prompt_versions=args.prompt_version,
                    labels=args.label, since=args.since, until=args.until,
                )
                if args.limit is not None:
                    records = records[-args.limit:]
                profile = load_history_gate_profile(args.profile)
                for name in ("max_average_cost_usd", "max_latest_cost_usd",
                             "max_average_latency_ms", "max_latest_latency_ms"):
                    value = getattr(args, name)
                    if value is not None:
                        if not isfinite(value) or value < 0:
                            raise HistoryError(f"history gate --{name.replace('_', '-')} must be a finite non-negative number")
                        profile[name] = value
                report = evaluate_history_gates(records, profile)
                rendered = (json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json"
                            else text_history_gate_report(report))
                _write_output(args.output, rendered)
                return 0 if report["passed"] else 1
            if args.bundle:
                verification = verify_bundle(args.bundle)
                if not verification["valid"]:
                    raise BundleError("cannot record invalid bundle: " + "; ".join(verification["errors"]))
                try:
                    report_data = json.loads((args.bundle / "report.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise BundleError(f"could not read bundle report: {exc}") from exc
                try:
                    scenario_snapshot_data = json.loads((args.bundle / "scenario-snapshot.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise BundleError(f"could not read bundle scenario snapshot: {exc}") from exc
                score_rows = report_data.get("scenarios", [])
                scenario_rows = scenario_snapshot_data.get("scenarios", [])
                if (not isinstance(score_rows, list) or not all(isinstance(score, dict) for score in score_rows)
                        or not isinstance(scenario_rows, list) or not all(isinstance(row, dict) for row in scenario_rows)):
                    raise BundleError("bundle report or scenario snapshot has malformed scenario rows")
                critical = {row["id"]: next((score.get("passed") for score in score_rows
                                             if score.get("scenario") == row["id"]), None)
                            for row in scenario_rows if isinstance(row.get("id"), str)
                            if isinstance(row.get("behavior"), dict)
                            and row["behavior"].get("severity") == "critical"}
                critical = {name: passed for name, passed in critical.items() if isinstance(passed, bool)}
                scenario_results = {score["scenario"]: score["passed"] for score in score_rows
                                    if isinstance(score.get("scenario"), str) and isinstance(score.get("passed"), bool)}
                recorded = record_history(args.store, report_data, label=args.label,
                                          source_type="bundle", run_id=verification["run_id"],
                                          critical_scenarios=critical, scenario_results=scenario_results)
            else:
                if args.scenarios is None:
                    raise HistoryError("--scenarios is required when recording --responses")
                history_scenarios = load_scenarios(args.scenarios)
                history_results = _selected_results(args.responses, history_scenarios, history_scenarios)
                history_report = evaluate(history_scenarios, history_results, strict=args.strict)
                score_by_name = {score.scenario: score.passed for score in history_report.scores}
                critical = {scenario.name: score_by_name[scenario.name] for scenario in history_scenarios
                            if scenario.severity == "critical" and scenario.name in score_by_name}
                recorded = record_history(args.store, history_report.as_dict(), label=args.label,
                                          source_type="results", critical_scenarios=critical,
                                          scenario_results=score_by_name)
            rendered = json.dumps(recorded, indent=2, ensure_ascii=False) if args.format == "json" else (
                f"Recorded evaluation {recorded['run_id']} ({recorded['passed']}/{recorded['total']} passed) "
                f"in {args.store}"
            )
            print(rendered)
            return 0

        all_scenarios = load_scenarios(args.scenarios) if hasattr(args, "scenarios") else []
        if args.command == "shard":
            manifest = create_shard_manifest(all_scenarios, shard_count=args.count)
            _write_output(args.output, json.dumps(manifest, indent=2, ensure_ascii=False))
            return 0
        if args.command == "merge-results":
            try:
                manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ShardError(f"could not read shard manifest {args.manifest}: {exc}") from exc
            shard_results = []
            for value in args.shard_result:
                if "=" not in value:
                    raise ShardError("--shard-result must use INDEX=PATH")
                index_text, path_text = value.split("=", 1)
                try:
                    index = int(index_text)
                except ValueError as exc:
                    raise ShardError(f"invalid shard index {index_text!r}") from exc
                shard_results.append((index, list(load_results(Path(path_text)).values())))
            merged = merge_shard_results(all_scenarios, manifest, shard_results)
            _write_output(args.output, "\n".join(
                json.dumps(result_as_dict(result), ensure_ascii=False) for result in merged
            ))
            return 0
        if args.command == "robustness":
            report = run_robustness(
                all_scenarios, load_runner(args.runner), types=args.type or VARIANT_TYPES,
            )
            if args.format == "json":
                rendered = json.dumps(report, indent=2, ensure_ascii=False)
            else:
                lines = [f"Robustness suite: {report['scenario_count']} scenarios, "
                         f"{report['variant_count']} variants",
                         f"Unexpected changes: {report['unexpected_changes']}",
                         f"Skipped variants: {len(report['skipped_variants'])}"]
                for row in report["variants"]:
                    lines.append(f"  {'FAIL' if row['unexpected'] else 'PASS'} "
                                 f"{row['scenario']} [{row['variant']}]: "
                                 f"{', '.join(row['changed_fields']) or 'behavior unchanged'}")
                    lines.extend("    " + reason for reason in row["reasons"])
                rendered = "\n".join(lines)
            _write_output(args.output, rendered)
            return 0 if report["passed"] else 1
        if args.command == "scenario-audit":
            audit_date = date.fromisoformat(args.as_of) if args.as_of else None
            report = audit_scenarios(
                all_scenarios, max_review_age_days=args.max_review_age_days, as_of=audit_date,
            )
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else text_scenario_audit(report)
            _write_output(args.output, rendered)
            return 0 if report["passed"] else 1
        if args.command == "gate":
            profile = load_gate_profile(args.profile)
            reports = [
                evaluate(all_scenarios, _selected_results(run_path, all_scenarios, all_scenarios))
                for run_path in args.runs
            ]
            gate_report = evaluate_release_gates(all_scenarios, reports, profile)
            rendered = json.dumps(gate_report, indent=2, ensure_ascii=False) if args.format == "json" else text_release_gate_report(gate_report)
            _write_output(args.output, rendered)
            return 0 if gate_report["passed"] else 1
        if args.command == "scenario-version":
            scenario_dir = args.scenarios if args.scenarios.is_dir() else args.scenarios.parent
            current = scenario_snapshot(all_scenarios, scenario_dir)
            compatibility = None
            if args.baseline:
                try:
                    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise VersioningError(f"could not read baseline snapshot {args.baseline}: {exc}") from exc
                if not isinstance(baseline, dict) or not isinstance(baseline.get("scenarios"), list):
                    raise VersioningError(f"{args.baseline}: not an Agent Eval scenario snapshot")
                compatibility = compare_snapshots(current, baseline)
            if args.migrate:
                if not args.baseline:
                    raise VersioningError("--migrate requires --baseline")
                migrate_scenario_versions(all_scenarios, baseline)
                selected_version = compatibility["recommended_version"]
                write_manifest(scenario_dir, current, selected_version)
                all_scenarios = load_scenarios(args.scenarios)
                current = scenario_snapshot(all_scenarios, scenario_dir)
                compatibility = compare_snapshots(current, baseline)
            elif args.write_manifest:
                selected_version = compatibility["recommended_version"] if compatibility else current["version"]
                write_manifest(scenario_dir, current, selected_version)
                current = scenario_snapshot(all_scenarios, scenario_dir)
            payload = {**current, "compatibility": compatibility}
            rendered = json.dumps(payload, indent=2, ensure_ascii=False) if args.format == "json" else text_version_report(current, compatibility)
            _write_output(args.output, rendered)
            return 1 if compatibility and (not compatibility["version_sufficient"] or compatibility["unversioned_changes"]) else 0
        if args.command == "scenario-baseline-history":
            report = baseline_history_report(load_baseline_history(args.history_log), limit=args.limit)
            if args.format == "json":
                rendered = json.dumps(report, indent=2, ensure_ascii=False)
            else:
                lines = [f"Scenario baseline history: {args.history_log}",
                         f"Approvals shown: {report['approval_count']}"]
                for record in report["approvals"]:
                    reviewer = record["approved_by"] or "unspecified"
                    lines.append(f"  {record['approval_id']} {record['event']} {record['scenario_set']} "
                                 f"{record['version']} by {reviewer} at {record['recorded_at']}")
                rendered = "\n".join(lines)
            _write_output(args.output, rendered)
            return 0
        if args.command == "scenario-baseline-export":
            if args.baseline.resolve() == args.output.resolve():
                raise PortableBaselineError("portable export destination must differ from the source baseline")
            report = export_portable_baseline(args.baseline, args.output,
                                              private_key_path=args.private_key, key_id=args.key_id)
            rendered = (json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json"
                        else f"Exported {report['scenario_set']} {report['version']} to {args.output} "
                             f"(SHA-256 {report['snapshot_sha256']}; "
                             f"{'signed with ' + report['key_id'] if report['signed'] else 'unsigned'})")
            print(rendered)
            return 0
        if args.command == "scenario-baseline-import":
            if not args.approve:
                raise PortableBaselineError("importing a baseline requires explicit --approve")
            history_path = args.history_log or args.baseline.with_suffix(args.baseline.suffix + ".history.jsonl")
            if len({args.baseline.resolve(), args.portable.resolve(), history_path.resolve()}) != 3:
                raise PortableBaselineError("portable source, --baseline, and --history-log must be different paths")
            imported = import_portable_baseline(
                args.portable, public_key_path=args.public_key,
                require_signature=args.require_signature,
            )
            scenario_dir = args.scenarios if args.scenarios.is_dir() else args.scenarios.parent
            target_snapshot = scenario_snapshot(all_scenarios, scenario_dir)
            if imported["scenario_set"] != target_snapshot["scenario_set"]:
                raise PortableBaselineError(
                    f"portable baseline is for scenario set '{imported['scenario_set']}', "
                    f"but target is '{target_snapshot['scenario_set']}'"
                )
            previous = None
            if args.baseline.exists():
                try:
                    previous = json.loads(args.baseline.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise VersioningError(f"could not read baseline snapshot {args.baseline}: {exc}") from exc
                if not isinstance(previous, dict) or not isinstance(previous.get("scenarios"), list):
                    raise VersioningError(f"{args.baseline}: not an Agent Eval scenario snapshot")
            approved = approved_scenario_snapshot(imported, approved_by=args.approved_by,
                                                  imported_from=str(args.portable))
            history_record = append_baseline_history(
                history_path, approved, previous_snapshot=previous, event="import",
                approved_by=args.approved_by, imported_from=str(args.portable),
            )
            write_scenario_baseline(args.baseline, approved)
            report = {"imported": True, "baseline_path": str(args.baseline),
                      "history_log": str(history_path), "scenario_set": approved["scenario_set"],
                      "version": approved["version"], "approval_id": history_record["approval_id"],
                      "imported_from": str(args.portable), "approved_by": args.approved_by}
            rendered = (json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json"
                        else f"Imported {report['scenario_set']} {report['version']} into {args.baseline}; "
                             f"approval {report['approval_id']} recorded.")
            _write_output(args.output, rendered)
            return 0
        if args.command == "scenario-baseline-restore":
            if not args.approve:
                raise VersioningError("restoring an approval requires explicit --approve")
            if args.baseline.resolve() == args.history_log.resolve():
                raise VersioningError("--baseline and --history-log must be different paths")
            records = load_baseline_history(args.history_log)
            source = next((record for record in records if record["approval_id"] == args.approval_id), None)
            if source is None:
                raise BaselineHistoryError(f"approval ID not found in history: {args.approval_id}")
            previous = None
            if args.baseline.exists():
                try:
                    previous = json.loads(args.baseline.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise VersioningError(f"could not read baseline snapshot {args.baseline}: {exc}") from exc
                if not isinstance(previous, dict) or not isinstance(previous.get("scenarios"), list):
                    raise VersioningError(f"{args.baseline}: not an Agent Eval scenario snapshot")
            restored = approved_scenario_snapshot(source["snapshot"], approved_by=args.approved_by,
                                                  restored_from=args.approval_id)
            record = append_baseline_history(args.history_log, restored, previous_snapshot=previous,
                                             event="restore", approved_by=args.approved_by,
                                             restored_from=args.approval_id)
            write_scenario_baseline(args.baseline, restored)
            report = {"restored": True, "baseline_path": str(args.baseline),
                      "approval_id": record["approval_id"], "restored_from": args.approval_id,
                      "scenario_set": restored.get("scenario_set"), "version": restored.get("version"),
                      "approved_by": args.approved_by}
            rendered = (json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json"
                        else f"Restored {report['scenario_set']} {report['version']} from approval "
                             f"{args.approval_id}; restore approval {record['approval_id']} recorded.")
            _write_output(args.output, rendered)
            return 0
        if args.command == "scenario-baseline":
            scenario_dir = args.scenarios if args.scenarios.is_dir() else args.scenarios.parent
            history_path = args.history_log or args.baseline.with_suffix(args.baseline.suffix + ".history.jsonl")
            if args.baseline.resolve() == history_path.resolve():
                raise VersioningError("--baseline and --history-log must be different paths")
            try:
                baseline = (json.loads(args.baseline.read_text(encoding="utf-8"))
                            if args.baseline.exists() else None)
            except (OSError, json.JSONDecodeError) as exc:
                raise VersioningError(f"could not read baseline snapshot {args.baseline}: {exc}") from exc
            if baseline is not None and (not isinstance(baseline, dict) or not isinstance(baseline.get("scenarios"), list)):
                raise VersioningError(f"{args.baseline}: not an Agent Eval scenario snapshot")
            current = scenario_snapshot(all_scenarios, scenario_dir)
            report = scenario_baseline_review(current, baseline)
            approval_policy = load_approval_policy(args.policy) if args.policy else {
                "require_approved_by": False, "allowed_reviewers": [],
                "minimum_version_bump": "none", "require_release_gates_passed": False,
            }
            release_gate_report = None
            if args.profile or args.runs:
                if not args.profile or not args.runs:
                    raise ApprovalPolicyError("--profile and --runs must be supplied together")
                release_gate_report = evaluate_release_gates(
                    all_scenarios, [evaluate(all_scenarios, load_results(args.runs), strict=True)],
                    load_gate_profile(args.profile),
                )
            policy_errors = policy_blockers(report, approval_policy, approved_by=args.approved_by,
                                            release_gates=release_gate_report)
            report["policy_blockers"] = policy_errors
            report["release_gates"] = release_gate_report
            report["approval_ready"] = report["approval_ready"] and not policy_errors
            report["approval_blockers"].extend(policy_errors)
            report["baseline_path"] = str(args.baseline)
            report["history_log"] = str(history_path)
            report["approval_requested"] = args.approve
            report["approved"] = False
            if args.approve and report["approval_ready"]:
                approved = approved_scenario_snapshot(current, approved_by=args.approved_by)
                record = append_baseline_history(history_path, approved, previous_snapshot=baseline,
                                                 approved_by=args.approved_by)
                write_scenario_baseline(args.baseline, approved)
                report["approved"] = True
                report["approval"] = approved["approval"]
                report["approval_id"] = record["approval_id"]
            if args.format == "json":
                rendered = json.dumps(report, indent=2, ensure_ascii=False)
            else:
                baseline_label = report["baseline_version"] or "not initialized"
                action = "approved and updated" if report["approved"] else "review only"
                lines = [f"Scenario baseline review: {report['scenario_set']}",
                         f"Baseline version: {baseline_label}",
                         f"Current version: {report['current_version']}",
                         f"Changes: {len(report['added'])} added, {len(report['removed'])} removed, "
                         f"{len(report['changed'])} changed",
                         f"Approval check: {'READY' if report['approval_ready'] else 'BLOCKED'}"]
                for item in report["details"]:
                    dimensions = ", ".join(item.get("changed_dimensions", []))
                    lines.append(f"  {item['id']}: {item['change']}" + (f" ({dimensions})" if dimensions else ""))
                if report["approval_blockers"]:
                    lines.extend(["", "Approval blockers:"])
                    lines.extend(f"  - {blocker}" for blocker in report["approval_blockers"])
                if report["approved"]:
                    lines.extend(["", f"Baseline {action}: {args.baseline}",
                                  f"Approval ID: {report['approval_id']}",
                                  f"History log: {history_path}"])
                elif args.approve:
                    lines.extend(["", f"Approval blocked; baseline was not changed: {args.baseline}"])
                else:
                    lines.extend(["", f"Preview only. Pass --approve to update {args.baseline}."])
                rendered = "\n".join(lines)
            _write_output(args.output, rendered)
            return 1 if args.approve and not report["approval_ready"] else 0
        if args.command == "scenario-impact":
            scenario_dir = args.scenarios if args.scenarios.is_dir() else args.scenarios.parent
            try:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise VersioningError(f"could not read baseline snapshot {args.baseline}: {exc}") from exc
            if not isinstance(baseline, dict) or not isinstance(baseline.get("scenarios"), list):
                raise VersioningError(f"{args.baseline}: not an Agent Eval scenario snapshot")
            current = scenario_snapshot(all_scenarios, scenario_dir)
            compatibility = compare_snapshots(current, baseline)
            report = change_impact_report(compatibility, load_gate_profile(args.profile), current)
            if args.format == "json":
                rendered = json.dumps(report, indent=2, ensure_ascii=False)
            else:
                lines = [f"Scenario change impact: {report['scenario_set']} {report['baseline_version']} → {report['current_version']}",
                         f"Changes: {len(report['added'])} added, {len(report['removed'])} removed, {len(report['changed'])} changed",
                         f"Recommended release: {report['recommended_bump']} ({report['recommended_version']})",
                         f"Affected release gates: {report['affected_gate_count']}", "", "Scenario details:"]
                for item in report["details"]:
                    dimensions = ", ".join(item.get("changed_dimensions", []))
                    lines.append(f"  {item['id']}: {item['change']}" + (f" ({dimensions})" if dimensions else ""))
                lines.extend(["", "Release gates:"])
                for gate in report["gates"]:
                    lines.append(f"  {'AFFECTED' if gate['affected'] else 'unchanged'} {gate['type']} {gate['target']}: {gate['reason']}")
                plan = report["evaluation_plan"]
                lines.extend(["", "Suggested evaluation plan:",
                              "  Scenarios: " + (", ".join(plan["scenarios_to_run"]) or "none"),
                              "  Gates: " + (", ".join(plan["affected_gates"]) or "none")])
                lines.extend(f"  $ {command}" for command in plan["commands"])
                rendered = "\n".join(lines)
            _write_output(args.output, rendered)
            if args.github_annotations or args.summary_file:
                annotations, summary = github_scenario_impact_report(report)
                if args.github_annotations and annotations:
                    print(annotations)
                summary_path = args.summary_file or (
                    Path(os.environ["GITHUB_STEP_SUMMARY"])
                    if args.github_annotations and os.environ.get("GITHUB_STEP_SUMMARY") else None
                )
                if summary_path:
                    with summary_path.open("a", encoding="utf-8") as handle:
                        handle.write(summary)
                elif args.github_annotations:
                    print(summary)
            return 0
        if args.command == "coverage":
            report = scenario_coverage(all_scenarios)
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else text_coverage(report)
            _write_output(args.output, rendered)
            return 0
        if args.command == "suggest":
            if args.limit < 1:
                raise ValueError("--limit must be at least 1")
            _write_output(args.output, json.dumps(scenario_suggestions(all_scenarios, limit=args.limit), indent=2, ensure_ascii=False))
            return 0
        if args.command == "judge-agreement":
            _write_output(args.output, json.dumps(judge_agreement(args.reports), indent=2, ensure_ascii=False))
            return 0
        scenarios = filter_scenarios(
            all_scenarios,
            include_tags=tuple(args.tag),
            exclude_tags=tuple(args.exclude_tag),
        )
        shard_manifest = None
        if args.command == "run-agent":
            if args.shard_manifest is not None:
                if args.shard_index is None:
                    raise ShardError("--shard-manifest requires --shard-index")
                if args.tag or args.exclude_tag:
                    raise ShardError("tag filters cannot be combined with a complete shard manifest")
                try:
                    shard_manifest = json.loads(args.shard_manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ShardError(f"could not read shard manifest {args.shard_manifest}: {exc}") from exc
                scenarios = select_shard(all_scenarios, shard_manifest, shard_index=args.shard_index)
            elif args.shard_index is not None:
                raise ShardError("--shard-index requires --shard-manifest")
        if args.command == "compare":
            baseline = evaluate(
                scenarios, _selected_results(args.baseline, all_scenarios, scenarios), strict=args.strict
            )
            candidate = evaluate(
                scenarios, _selected_results(args.candidate, all_scenarios, scenarios), strict=args.strict
            )
            comparison = compare_reports(baseline, candidate)
            comparison["paired_bootstrap"] = paired_bootstrap_intervals(
                baseline, candidate, confidence=args.confidence,
                samples=args.bootstrap_samples, seed=args.seed,
            )
            if args.github_annotations:
                annotations, summary = github_annotation_report(comparison, candidate)
                if annotations:
                    print(annotations)
                summary_path = args.summary_file or (Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None)
                if summary_path:
                    with summary_path.open("a", encoding="utf-8") as handle:
                        handle.write(summary)
                elif not annotations:
                    print(summary)
                return 0 if not comparison["regressed_scenarios"] else 1
            if args.format == "json":
                print(json.dumps(comparison, indent=2, ensure_ascii=False))
            else:
                print(text_comparison(comparison))
            return 0 if not comparison["regressed_scenarios"] else 1

        if args.command == "monitor":
            if not 0 <= args.max_absolute_drift <= 1:
                raise ValueError("--max-absolute-drift must be between 0 and 1")
            baseline = evaluate(scenarios, _selected_results(args.baseline, all_scenarios, scenarios))
            current = evaluate(scenarios, _selected_results(args.current, all_scenarios, scenarios))
            drift = drift_report(baseline, current, max_absolute_drift=args.max_absolute_drift)
            if args.format == "json":
                rendered = json.dumps(drift, indent=2, ensure_ascii=False)
            else:
                lines = ["Behavior drift monitor", "", f"Status: {'PASS' if drift['passed'] else 'FAIL'}",
                         f"Maximum absolute metric drift: {args.max_absolute_drift:.1%}", "", "Metric changes:"]
                for metric, item in drift["metrics"].items():
                    delta = "n/a" if item["delta"] is None else f"{item['delta'] * 100:+.1f} pp"
                    lines.append(f"  {'PASS' if item['passed'] else 'FAIL'} {metric}: {delta}")
                lines.extend(["", "Regressed scenarios: " + (", ".join(drift["regressed_scenarios"]) or "none")])
                rendered = "\n".join(lines)
            _write_output(args.output, rendered)
            return 0 if drift["passed"] else 1

        if args.command == "experiment":
            specs = [read_variant_spec(value) for value in args.variant]
            names = [name.casefold() for name, _ in specs]
            if len(specs) < 2:
                raise ValueError("experiment requires at least two --variant values")
            if len(names) != len(set(names)):
                raise ValueError("experiment variant names must be unique")
            report = experiment_matrix(
                scenarios,
                [(name, _selected_results(path, all_scenarios, scenarios)) for name, path in specs],
                confidence=args.confidence, samples=args.bootstrap_samples, seed=args.seed,
            )
            if args.format == "json":
                rendered = json.dumps(report, indent=2, ensure_ascii=False)
            else:
                lines = [f"Experiment matrix: {report['variant_count']} variants across {report['scenario_count']} scenarios", ""]
                for variant in report["variants"]:
                    lines.extend([f"{variant['name']}: {variant['passed']}/{variant['total']} passed ({_percent(variant['pass_rate'])})",
                                  f"  models={', '.join(variant['models']) or 'n/a'}, prompt_versions={', '.join(variant['prompt_versions']) or 'n/a'}",
                                  f"  intent={_percent(variant['metrics']['intent_accuracy'])}, tool={_percent(variant['metrics']['tool_accuracy'])}, policy={_percent(variant['metrics']['policy_compliance'])}, unnecessary_questions={_percent(variant['metrics']['unnecessary_question_rate'])}",
                                  f"  cost=${variant['execution']['cost_usd'] if variant['execution']['cost_usd'] is not None else 'n/a'}, input_tokens={variant['execution']['input_tokens'] if variant['execution']['input_tokens'] is not None else 'n/a'}, output_tokens={variant['execution']['output_tokens'] if variant['execution']['output_tokens'] is not None else 'n/a'}, p95_latency_ms={variant['execution']['p95_latency_ms'] if variant['execution']['p95_latency_ms'] is not None else 'n/a'}"])
                    if variant["failed_scenarios"]:
                        lines.append("  failing scenarios: " + ", ".join(variant["failed_scenarios"]))
                    lines.append("")
                for comparison in report["comparisons"]:
                    lines.append(
                        f"Paired comparison {comparison['baseline']} → {comparison['candidate']}: "
                        f"pass-rate delta {comparison['paired_bootstrap']['metrics']['pass_rate']['estimate']:+.1%}"
                        if comparison["paired_bootstrap"]["metrics"]["pass_rate"]["estimate"] is not None
                        else f"Paired comparison {comparison['baseline']} → {comparison['candidate']}: no matched scenarios"
                    )
                    interval = comparison["paired_bootstrap"]["metrics"]["pass_rate"]
                    if interval["lower"] is not None:
                        lines.append(f"  {interval['confidence']:.0%} interval: {interval['lower']:+.1%} to {interval['upper']:+.1%}")
                rendered = "\n".join(lines).rstrip()
            _write_output(args.output, rendered)
            return 0

        if args.command == "triage":
            results = _selected_results(args.responses, all_scenarios, scenarios)
            report = evaluate(scenarios, results, strict=False)
            analysis = failure_analysis(scenarios, report, results)
            rendered = json.dumps(analysis, indent=2, ensure_ascii=False) if args.format == "json" else text_failure_analysis(analysis)
            _write_output(args.output, rendered)
            return 0

        if args.command == "mutate":
            report = mutation_report(scenarios, _selected_results(args.responses, all_scenarios, scenarios))
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else text_mutation_report(report)
            _write_output(args.output, rendered)
            return 2 if report["mutation_count"] == 0 else 1 if report["survived"] else 0

        if args.command == "repeat":
            if args.min_runs < 2:
                raise ValueError("--min-runs must be at least 2")
            if len(args.runs) < args.min_runs:
                raise ValueError(f"Repeated-run analysis requires at least {args.min_runs} result files")
            if args.min_pass_rate is not None and not 0 <= args.min_pass_rate <= 1:
                raise ValueError("--min-pass-rate must be between 0 and 1")
            reports = [
                evaluate(
                    scenarios,
                    _selected_results(run_path, all_scenarios, scenarios),
                    strict=args.strict,
                )
                for run_path in args.runs
            ]
            repeated = analyze_repeated_runs(reports, confidence=args.confidence, scenarios=scenarios)
            if args.format == "json":
                rendered = json.dumps(repeated, indent=2, ensure_ascii=False)
            elif args.format == "html":
                rendered = repeated_html_report(repeated)
            else:
                rendered = text_repeated_report(repeated)
            _write_output(args.output, rendered)
            gate_passed = all(report.failed == 0 for report in reports)
            if args.min_pass_rate is not None:
                gate_passed = gate_passed and (
                    repeated["pass_rate"]["confidence_interval"]["lower"] >= args.min_pass_rate
                )
            return 0 if gate_passed else 1

        if args.command == "run-agent":
            runner = load_runner(args.runner)
            execution_stats: dict[str, int] = {}
            runner_version = getattr(runner, "version", "unspecified")
            if args.check_cache:
                if args.cache is None:
                    raise RunnerError("--check-cache requires --cache")
                preflight = explain_cache(args.cache, scenarios,
                                          runner_key=f"{args.runner}@{runner_version}")
                print(json.dumps(preflight, indent=2, ensure_ascii=False), file=sys.stderr)
                if not preflight["compatible"]:
                    raise RunnerError("cache preflight failed: runner identity does not match")
            cancel_event = Event()
            previous_handlers = {}

            def request_cancel(signum, frame):
                cancel_event.set()

            for signum in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
                if signum not in previous_handlers:
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, request_cancel)
            execution_failed = False
            try:
                results = execute_scenarios(
                    scenarios, runner, max_workers=args.workers,
                    checkpoint=args.checkpoint, cache=args.cache,
                    runner_key=f"{args.runner}@{runner_version}",
                    runner_factory=(lambda: load_runner(args.runner)) if args.workers > 1 else None,
                    stats=execution_stats, provider_key=args.runner,
                    provider_limits=_parse_provider_limits(args.provider_limit),
                    scenario_timeout=args.scenario_timeout, cancel_event=cancel_event,
                )
            except RunnerError as exc:
                if not hasattr(exc, "partial_results"):
                    raise
                results = exc.partial_results
                execution_stats = exc.execution_stats
                execution_failed = True
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
            args.results.write_text(
                "".join(json.dumps(result_as_dict(result), ensure_ascii=False) + "\n" for result in results),
                encoding="utf-8",
            )
            report = evaluate(scenarios, {result.scenario: result for result in results}, strict=not execution_stats.get("cancelled", 0) and not execution_failed,
                              judge_cache=args.judge_cache, judge_repetitions=args.judge_repetitions)
            report.execution["agent_execution"] = execution_stats
            report, budgets_passed = apply_budgets(
                report, {result.scenario: result for result in results}, len(scenarios),
                max_cost_usd=args.max_cost_usd, max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens, max_p95_latency_ms=args.max_p95_latency_ms)
            if args.bundle_dir:
                create_bundle(args.bundle_dir, args.scenarios, scenarios,
                              {result.scenario: result for result in results}, report,
                              {"command": "run-agent", "runner": args.runner, "format": args.format,
                               "strict": not execution_failed and not execution_stats.get("cancelled", 0),
                               "selected_scenarios": [item.name for item in scenarios],
                               "execution": execution_stats,
                               "workers": args.workers,
                               "scenario_timeout": args.scenario_timeout,
                               "provider_limits": _parse_provider_limits(args.provider_limit),
                               "shard_index": args.shard_index,
                               "shard_count": shard_manifest.get("shard_count") if shard_manifest else None,
                               "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                               "cache": str(args.cache) if args.cache else None,
                               "judge_repetitions": args.judge_repetitions,
                               "budgets": {"max_cost_usd": args.max_cost_usd,
                                           "max_input_tokens": args.max_input_tokens,
                                           "max_output_tokens": args.max_output_tokens,
                                           "max_p95_latency_ms": args.max_p95_latency_ms}})
            if args.format == "json":
                rendered = json_report(report)
            elif args.format == "html":
                rendered = html_report(report)
            elif args.format == "junit":
                rendered = junit_report(report)
            elif args.format == "sarif":
                rendered = sarif_report(report)
            else:
                rendered = text_report(report)
            _write_output(args.report, rendered)
            if execution_stats.get("cancelled", 0):
                return 130
            return 0 if report.failed == 0 and budgets_passed and not execution_failed else 1

        results = _selected_results(args.responses, all_scenarios, scenarios)
        report = evaluate(scenarios, results, strict=args.strict,
                          judge_cache=args.judge_cache, judge_repetitions=args.judge_repetitions)
        report, budgets_passed = apply_budgets(
            report, results, len(scenarios), max_cost_usd=args.max_cost_usd,
            max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens,
            max_p95_latency_ms=args.max_p95_latency_ms)
        if args.bundle_dir:
            create_bundle(args.bundle_dir, args.scenarios, scenarios, results, report,
                          {"command": "run", "format": args.format, "strict": args.strict,
                           "selected_scenarios": [item.name for item in scenarios],
                           "judge_repetitions": args.judge_repetitions,
                           "judge_cache_enabled": args.judge_cache is not None,
                           "budgets": {"max_cost_usd": args.max_cost_usd,
                                       "max_input_tokens": args.max_input_tokens,
                                       "max_output_tokens": args.max_output_tokens,
                                       "max_p95_latency_ms": args.max_p95_latency_ms}})
    except (AdapterError, BundleError, CacheError, DraftError, GateProfileError, HistoryError,
            GraderError, MinimizeError, RedactionError, RunnerError, ScenarioError, ResultError,
            OSError, ValueError) as exc:
        print(f"agent-eval: error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = json_report(report)
    elif args.format == "html":
        rendered = html_report(report)
    elif args.format == "junit":
        rendered = junit_report(report)
    elif args.format == "sarif":
        rendered = sarif_report(report)
    else:
        rendered = text_report(report, details=not args.quiet_details)
    _write_output(args.output, rendered)
    return 0 if report.failed == 0 and budgets_passed else 1


def _write_output(path: Path | None, content: str) -> None:
    if path is None:
        print(content)
    else:
        path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _parse_provider_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        provider, separator, limit_text = value.partition("=")
        provider = provider.strip()
        if not separator or not provider:
            raise RunnerError("--provider-limit must use PROVIDER=N")
        try:
            limit = int(limit_text)
        except ValueError as exc:
            raise RunnerError(f"invalid provider limit {value!r}: N must be a positive integer") from exc
        if provider in limits:
            raise RunnerError(f"provider limit for '{provider}' was specified more than once")
        if limit < 1:
            raise RunnerError(f"provider limit for '{provider}' must be a positive integer")
        limits[provider] = limit
    return limits


if __name__ == "__main__":
    raise SystemExit(main())
