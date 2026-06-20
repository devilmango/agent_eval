"""Command-line interface for the agent evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent_eval import __version__
from agent_eval.adapters import AdapterError, PathMappingAdapter, result_as_dict
from agent_eval.comparison import compare_reports, text_comparison
from agent_eval.report import html_report, json_report, repeated_html_report, text_report
from agent_eval.repeated import analyze_repeated_runs, text_repeated_report
from agent_eval.results import ResultError, load_results
from agent_eval.scenarios import ScenarioError, load_scenarios
from agent_eval.scoring import evaluate


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

    adapt = commands.add_parser("adapt", help="convert framework JSONL traces into Agent Eval results")
    adapt.add_argument("traces", type=Path, help="raw JSON Lines trace file")
    adapt.add_argument("--mapping", required=True, type=Path, help="YAML field-path mapping")
    adapt.add_argument("--output", type=Path, help="output JSONL path (defaults to stdout)")

    compare = commands.add_parser("compare", help="compare candidate results with a baseline run")
    compare.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    compare.add_argument("--baseline", required=True, type=Path, help="baseline JSONL results")
    compare.add_argument("--candidate", required=True, type=Path, help="candidate JSONL results")
    compare.add_argument("--format", choices=("text", "json"), default="text")
    compare.add_argument("--strict", action="store_true", help="fail when either run is missing results")

    repeat = commands.add_parser("repeat", help="analyze variability across repeated result files")
    repeat.add_argument("scenarios", type=Path, help="scenario YAML file or directory")
    repeat.add_argument("--runs", nargs="+", required=True, type=Path, help="two or more JSONL result files")
    repeat.add_argument("--format", choices=("text", "json", "html"), default="text")
    repeat.add_argument("--output", type=Path, help="write the report to a file instead of stdout")
    repeat.add_argument("--strict", action="store_true", help="fail when any run is missing scenario results")
    return parser


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
        except AdapterError as exc:
            raise AdapterError(f"line {number}: {exc}") from exc
        if result.scenario in seen_scenarios:
            raise AdapterError(f"line {number}: duplicate trace for '{result.scenario}'")
        seen_scenarios.add(result.scenario)
        output.append(json.dumps(result_as_dict(result), ensure_ascii=False))
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "adapt":
            adapter = PathMappingAdapter.from_yaml(args.mapping)
            output = _adapt_jsonl(args.traces, adapter)
            rendered = "\n".join(output) + ("\n" if output else "")
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
            else:
                sys.stdout.write(rendered)
            return 0

        scenarios = load_scenarios(args.scenarios)
        if args.command == "compare":
            baseline = evaluate(scenarios, load_results(args.baseline), strict=args.strict)
            candidate = evaluate(scenarios, load_results(args.candidate), strict=args.strict)
            comparison = compare_reports(baseline, candidate)
            if args.format == "json":
                print(json.dumps(comparison, indent=2, ensure_ascii=False))
            else:
                print(text_comparison(comparison))
            return 0 if not comparison["regressed_scenarios"] else 1

        if args.command == "repeat":
            if len(args.runs) < 2:
                raise ValueError("Repeated-run analysis requires at least two result files")
            reports = [
                evaluate(scenarios, load_results(run_path), strict=args.strict)
                for run_path in args.runs
            ]
            repeated = analyze_repeated_runs(reports)
            if args.format == "json":
                rendered = json.dumps(repeated, indent=2, ensure_ascii=False)
            elif args.format == "html":
                rendered = repeated_html_report(repeated)
            else:
                rendered = text_repeated_report(repeated)
            _write_output(args.output, rendered)
            return 0 if all(report.failed == 0 for report in reports) else 1

        results = load_results(args.responses)
        report = evaluate(scenarios, results, strict=args.strict)
    except (AdapterError, ScenarioError, ResultError, OSError, ValueError) as exc:
        print(f"agent-eval: error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = json_report(report)
    elif args.format == "html":
        rendered = html_report(report)
    else:
        rendered = text_report(report, details=not args.quiet_details)
    _write_output(args.output, rendered)
    return 0 if report.failed == 0 else 1


def _write_output(path: Path | None, content: str) -> None:
    if path is None:
        print(content)
    else:
        path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
