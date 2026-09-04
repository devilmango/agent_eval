"""Command-line interface for the agent evaluation harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from agent_eval import __version__
from agent_eval.bundles import BundleError, create_bundle, verify_bundle
from agent_eval.calibration import CalibrationError, calibrate_judge, text_calibration_report
from agent_eval.adapters import AdapterError, PathMappingAdapter, load_adapter_plugin, result_as_dict
from agent_eval.budgets import apply_budgets
from agent_eval.comparison import compare_reports, text_comparison
from agent_eval.graders import GraderError, load_judge_registry
from agent_eval.models import AgentResult
from agent_eval.monitoring import drift_report, experiment_matrix, github_annotation_report, read_variant_spec
from agent_eval.otel import OpenTelemetryAdapter, load_otel_document
from agent_eval.quality import failure_analysis, mutation_report, scenario_coverage, scenario_suggestions, judge_agreement, text_coverage, text_mutation_report, text_failure_analysis
from agent_eval.report import html_report, json_report, repeated_html_report, text_report
from agent_eval.repeated import analyze_repeated_runs, text_repeated_report
from agent_eval.results import ResultError, load_results
from agent_eval.runner import RunnerError, execute_scenarios, load_runner
from agent_eval.scenarios import ScenarioError, filter_scenarios, load_scenarios
from agent_eval.scoring import evaluate
from agent_eval.versioning import VersioningError, compare_snapshots, migrate_scenario_versions, snapshot as scenario_snapshot, text_version_report, write_manifest


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
    run.add_argument("--format", choices=("text", "json", "html"), default="text")
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

    framework = commands.add_parser("adapt-framework", help="convert traces with an installed framework adapter")
    framework.add_argument("adapter", help="adapter entry-point name")
    framework.add_argument("traces", type=Path, help="framework trace JSON Lines file")
    framework.add_argument("--output", type=Path, help="output JSONL path (defaults to stdout)")

    otel = commands.add_parser("adapt-otel", help="convert an OTLP/JSON trace export into Agent Eval results")
    otel.add_argument("traces", type=Path, help="OTLP/JSON trace export containing resourceSpans")
    otel.add_argument("--output", type=Path, help="output JSONL path (defaults to stdout)")
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
    _add_tag_filters(matrix)

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
    run_agent.add_argument("--report", type=Path, help="write evaluation report here (defaults to stdout)")
    run_agent.add_argument("--format", choices=("text", "json", "html"), default="text")
    run_agent.add_argument("--judge-cache", type=Path, help="cache semantic judge results in this JSON file")
    run_agent.add_argument("--judge-repetitions", type=int, default=1, help="repeat semantic judgments for stability (default: 1)")
    run_agent.add_argument("--bundle-dir", type=Path, help="write a self-contained, hash-verified run bundle to a new directory")
    _add_budget_arguments(run_agent)
    _add_tag_filters(run_agent)

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

    calibrate = commands.add_parser("calibrate-judge", help="measure a semantic judge against human-labeled gold examples")
    calibrate.add_argument("gold_set", type=Path, help="YAML gold set")
    calibrate.add_argument("--judge", required=True, help="semantic judge plugin entry-point name")
    calibrate.add_argument("--model", help="judge model identifier recorded in each request")
    calibrate.add_argument("--threshold", type=float, default=0.7, help="score threshold for pass/fail classification")
    calibrate.add_argument("--min-accuracy", type=float, help="fail unless gold-set accuracy meets this threshold")
    calibrate.add_argument("--format", choices=("text", "json"), default="text")
    calibrate.add_argument("--output", type=Path)

    verify = commands.add_parser("verify-bundle", help="check hashes and integrity of a run artifact bundle")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--format", choices=("text", "json"), default="text")
    verify.add_argument("--output", type=Path)

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


def _adapt_jsonl(input_path: Path, adapter: PathMappingAdapter) -> list[str]:
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

        if args.command == "calibrate-judge":
            report = calibrate_judge(
                args.gold_set, args.judge, load_judge_registry(), model=args.model,
                threshold=args.threshold, min_accuracy=args.min_accuracy,
            )
            rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.format == "json" else text_calibration_report(report)
            _write_output(args.output, rendered)
            return 0 if report["passed_gate"] else 1

        if args.command == "adapt":
            adapter = PathMappingAdapter.from_yaml(args.mapping)
            output = _adapt_jsonl(args.traces, adapter)
            rendered = "\n".join(output) + ("\n" if output else "")
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
            else:
                sys.stdout.write(rendered)
            return 0

        if args.command == "adapt-framework":
            adapter = load_adapter_plugin(args.adapter)
            output = _adapt_jsonl(args.traces, adapter)
            _write_output(args.output, "\n".join(output))
            return 0

        if args.command == "adapt-otel":
            adapter = OpenTelemetryAdapter(
                scenario_attribute=args.scenario_attribute,
                intent_attribute=args.intent_attribute,
                include_sensitive_attributes=args.include_sensitive_attributes,
            )
            results = adapter.adapt_document(load_otel_document(args.traces))
            seen: set[str] = set()
            lines = []
            for result in results:
                if result.scenario in seen:
                    raise AdapterError(f"duplicate OpenTelemetry run for '{result.scenario}'")
                seen.add(result.scenario)
                lines.append(json.dumps(result_as_dict(result), ensure_ascii=False))
            _write_output(args.output, "\n".join(lines))
            return 0

        all_scenarios = load_scenarios(args.scenarios)
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
        if args.command == "compare":
            baseline = evaluate(
                scenarios, _selected_results(args.baseline, all_scenarios, scenarios), strict=args.strict
            )
            candidate = evaluate(
                scenarios, _selected_results(args.candidate, all_scenarios, scenarios), strict=args.strict
            )
            comparison = compare_reports(baseline, candidate)
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
            )
            if args.format == "json":
                rendered = json.dumps(report, indent=2, ensure_ascii=False)
            else:
                lines = [f"Experiment matrix: {report['variant_count']} variants across {report['scenario_count']} scenarios", ""]
                for variant in report["variants"]:
                    lines.extend([f"{variant['name']}: {variant['passed']}/{variant['total']} passed ({variant['pass_rate']:.1%})",
                                  f"  models={', '.join(variant['models']) or 'n/a'}, prompt_versions={', '.join(variant['prompt_versions']) or 'n/a'}",
                                  f"  intent={_percent(variant['metrics']['intent_accuracy'])}, tool={_percent(variant['metrics']['tool_accuracy'])}, policy={_percent(variant['metrics']['policy_compliance'])}, unnecessary_questions={_percent(variant['metrics']['unnecessary_question_rate'])}",
                                  f"  cost=${variant['execution']['cost_usd'] if variant['execution']['cost_usd'] is not None else 'n/a'}, input_tokens={variant['execution']['input_tokens'] if variant['execution']['input_tokens'] is not None else 'n/a'}, output_tokens={variant['execution']['output_tokens'] if variant['execution']['output_tokens'] is not None else 'n/a'}, p95_latency_ms={variant['execution']['p95_latency_ms'] if variant['execution']['p95_latency_ms'] is not None else 'n/a'}"])
                    if variant["failed_scenarios"]:
                        lines.append("  failing scenarios: " + ", ".join(variant["failed_scenarios"]))
                    lines.append("")
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
            repeated = analyze_repeated_runs(reports, confidence=args.confidence)
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
            results = execute_scenarios(scenarios, runner)
            args.results.write_text(
                "".join(json.dumps(result_as_dict(result), ensure_ascii=False) + "\n" for result in results),
                encoding="utf-8",
            )
            report = evaluate(scenarios, {result.scenario: result for result in results}, strict=True,
                              judge_cache=args.judge_cache, judge_repetitions=args.judge_repetitions)
            report, budgets_passed = apply_budgets(
                report, {result.scenario: result for result in results}, len(scenarios),
                max_cost_usd=args.max_cost_usd, max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens, max_p95_latency_ms=args.max_p95_latency_ms)
            if args.bundle_dir:
                create_bundle(args.bundle_dir, args.scenarios, scenarios,
                              {result.scenario: result for result in results}, report,
                              {"command": "run-agent", "runner": args.runner, "format": args.format,
                               "strict": True, "selected_scenarios": [item.name for item in scenarios],
                               "judge_repetitions": args.judge_repetitions,
                               "budgets": {"max_cost_usd": args.max_cost_usd,
                                           "max_input_tokens": args.max_input_tokens,
                                           "max_output_tokens": args.max_output_tokens,
                                           "max_p95_latency_ms": args.max_p95_latency_ms}})
            if args.format == "json":
                rendered = json_report(report)
            elif args.format == "html":
                rendered = html_report(report)
            else:
                rendered = text_report(report)
            _write_output(args.report, rendered)
            return 0 if report.failed == 0 and budgets_passed else 1

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
    except (AdapterError, GraderError, RunnerError, ScenarioError, ResultError, OSError, ValueError) as exc:
        print(f"agent-eval: error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = json_report(report)
    elif args.format == "html":
        rendered = html_report(report)
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


if __name__ == "__main__":
    raise SystemExit(main())
