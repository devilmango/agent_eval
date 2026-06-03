"""Command-line interface for the agent evaluation harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_eval import __version__
from agent_eval.report import json_report, text_report
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
    run.add_argument("--format", choices=("text", "json"), default="text")
    run.add_argument("--strict", action="store_true", help="fail when any scenario result is missing")
    run.add_argument("--quiet-details", action="store_true", help="hide per-scenario failure details")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = load_scenarios(args.scenarios)
        results = load_results(args.responses)
        report = evaluate(scenarios, results, strict=args.strict)
    except (ScenarioError, ResultError) as exc:
        print(f"agent-eval: error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json_report(report))
    else:
        print(text_report(report, details=not args.quiet_details))
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

