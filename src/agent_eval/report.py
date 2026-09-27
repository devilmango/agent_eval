"""Human-readable and machine-readable report formatting."""

from __future__ import annotations

import json
from html import escape

from agent_eval.models import EvaluationReport
from agent_eval.repeated import METRICS


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def text_report(report: EvaluationReport, *, details: bool = True) -> str:
    lines = [
        f"{report.total} scenarios",
        "",
        f"Passed: {report.passed}",
        f"Failed: {report.failed}",
        "",
        f"Intent accuracy:       {format_percent(report.intent_accuracy)}",
        f"Tool accuracy:         {format_percent(report.tool_accuracy)}",
        f"Policy compliance:     {format_percent(report.policy_compliance)}",
        f"Unnecessary questions: {format_percent(report.unnecessary_question_rate)}",
        f"Answer quality:        {format_percent(report.answer_quality_score)}",
    ]
    budget_gates = report.execution.get("budget_gates", {})
    skipped_grades = sum(
        grade.metadata.get("skipped") is True
        for score in report.scores for grade in score.answer_grades
    )
    if skipped_grades:
        lines.extend(["", f"Semantic judge grades skipped: {skipped_grades}"])
    if budget_gates:
        lines.extend(["", "Execution budgets:"])
        for name, gate in budget_gates.items():
            observed = "missing" if gate["observed"] is None else str(gate["observed"])
            status = "PASS" if gate["passed"] else "FAIL"
            lines.append(f"  {status} {name}: {observed} (limit {gate['limit']}; missing scenarios {gate['missing_scenarios']})")
    if details and report.failed:
        lines.extend(["", "Failures:"])
        for score in report.scores:
            if score.passed:
                continue
            reasons: list[str] = []
            if not score.intent_correct:
                reasons.append(f"intent {score.actual_intent!r}, expected {score.expected_intent!r}")
            if score.missing_questions:
                reasons.append("missing questions: " + ", ".join(score.missing_questions))
            if score.unnecessary_questions:
                reasons.append("unnecessary questions: " + ", ".join(score.unnecessary_questions))
            if score.tool_correct is False:
                reasons.extend(score.tool_call_mismatches or ("tool set differs from expected",))
            if score.policy_violations:
                reasons.append("policy violations: " + ", ".join(score.policy_violations))
            reasons.extend(score.turn_violations)
            if score.notes:
                reasons.append(score.notes)
            for grade in score.answer_grades:
                if grade.metadata.get("skipped") is True:
                    continue
                if not grade.passed:
                    reasons.append(f"{grade.grader} grader: {grade.feedback or 'failed'}")
            lines.append(f"  - {score.scenario}: {'; '.join(reasons) or 'missing result'}")
    return "\n".join(lines)


def json_report(report: EvaluationReport) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False)


def html_report(report: EvaluationReport) -> str:
    metric_rows = "\n".join(
        f"<tr><th scope=\"row\">{escape(label)}</th><td>{escape(format_percent(value))}</td></tr>"
        for label, value in (
            ("Intent accuracy", report.intent_accuracy),
            ("Tool accuracy", report.tool_accuracy),
            ("Policy compliance", report.policy_compliance),
            ("Unnecessary questions", report.unnecessary_question_rate),
            ("Answer quality", report.answer_quality_score),
        )
    )
    budget_gates = report.execution.get("budget_gates", {})
    budget_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{escape(name)}</th>"
        f"<td>{escape('missing' if gate['observed'] is None else str(gate['observed']))}</td>"
        f"<td>{escape(str(gate['limit']))}</td>"
        f"<td>{'Passed' if gate['passed'] else 'Failed'} ({gate['missing_scenarios']} missing)</td>"
        "</tr>"
        for name, gate in budget_gates.items()
    )
    scenario_rows = []
    for score in report.scores:
        failures = []
        if not score.intent_correct:
            failures.append(f"Intent: expected {score.expected_intent}, got {score.actual_intent}")
        if score.missing_questions:
            failures.append("Missing questions: " + ", ".join(score.missing_questions))
        if score.unnecessary_questions:
            failures.append("Unnecessary questions: " + ", ".join(score.unnecessary_questions))
        if score.tool_correct is False:
            failures.extend(score.tool_call_mismatches or ("Tool set differs from expected",))
        if score.policy_violations:
            failures.append("Policy violations: " + ", ".join(score.policy_violations))
        failures.extend(score.turn_violations)
        if score.notes:
            failures.append(score.notes)
        skipped_graders = []
        for grade in score.answer_grades:
            if grade.metadata.get("skipped") is True:
                skipped_graders.append(grade.grader)
                continue
            if not grade.passed:
                failures.append(f"{grade.grader} grader: {grade.feedback or 'failed'}")
        if skipped_graders:
            failures.append("Skipped graders: " + ", ".join(skipped_graders))
        outcome = "Passed" if score.passed else "Failed"
        reason = "; ".join(failures) if failures else "—"
        scenario_rows.append(
            "<tr>"
            f"<th scope=\"row\">{escape(score.scenario)}</th>"
            f"<td><span class=\"status {'pass' if score.passed else 'fail'}\">{outcome}</span></td>"
            f"<td>{escape(reason)}</td>"
            "</tr>"
        )
    summary = f"{report.passed} of {report.total} scenarios passed"
    return _document(
        "Agent Eval report",
        f"""
        <header class="page-header">
          <p class="eyebrow">Agent Eval</p>
          <h1>Evaluation report</h1>
          <p>{escape(summary)}</p>
        </header>
        <section aria-labelledby="summary-heading">
          <h2 id="summary-heading">Summary</h2>
          <div class="summary-grid">
            <div class="metric"><span>Scenarios</span><strong>{report.total}</strong></div>
            <div class="metric"><span>Passed</span><strong>{report.passed}</strong></div>
            <div class="metric"><span>Failed</span><strong>{report.failed}</strong></div>
          </div>
        </section>
        <section aria-labelledby="metrics-heading">
          <h2 id="metrics-heading">Metrics</h2>
          <table><caption>Evaluation metrics</caption><tbody>{metric_rows}</tbody></table>
        </section>
        <section aria-labelledby="scenarios-heading">
          <h2 id="scenarios-heading">Scenarios</h2>
          <table>
            <caption>Scenario results and failure details</caption>
            <thead><tr><th scope="col">Scenario</th><th scope="col">Result</th><th scope="col">Details</th></tr></thead>
            <tbody>{''.join(scenario_rows)}</tbody>
          </table>
        </section>
        {f'<section aria-labelledby="budgets-heading"><h2 id="budgets-heading">Execution budgets</h2><table><caption>Measured values compared with configured limits</caption><thead><tr><th scope="col">Metric</th><th scope="col">Observed</th><th scope="col">Limit</th><th scope="col">Gate</th></tr></thead><tbody>{budget_rows}</tbody></table></section>' if budget_gates else ''}
        """,
    )


def repeated_html_report(report: dict) -> str:
    pass_stats = report["pass_rate"]
    metric_rows = []
    for metric in METRICS:
        stats = report["metrics"][metric]
        mean_value = format_percent(stats["mean"])
        deviation = format_percent(stats["sample_stddev"])
        metric_rows.append(
            f"<tr><th scope=\"row\">{escape(metric.replace('_', ' ').capitalize())}</th>"
            f"<td>{escape(mean_value)}</td><td>{escape(deviation)}</td>"
            f"<td>{escape(format_percent(stats['min']))}–{escape(format_percent(stats['max']))}</td></tr>"
        )
    scenario_rows = []
    for scenario in report["scenarios"]:
        failed_runs = ", ".join(str(number) for number in scenario["failed_run_numbers"]) or "—"
        scenario_rows.append(
            f"<tr><th scope=\"row\">{escape(scenario['scenario'])}</th>"
            f"<td>{scenario['passed_runs']}/{scenario['total_runs']}</td>"
            f"<td>{scenario['pass_rate'] * 100:.1f}% (Wilson {scenario['confidence_interval']['lower'] * 100:.1f}–{scenario['confidence_interval']['upper'] * 100:.1f}%)</td>"
            f"<td>{escape(failed_runs)}</td></tr>"
        )
    flaky_rows = []
    for scenario in report.get("flaky_scenarios", []):
        reasons = ", ".join(
            f"{name}: {count}" for name, count in scenario.get("failure_reasons", {}).items()
        ) or "No classified failure reason"
        flaky_rows.append(
            f"<tr><th scope=\"row\">{escape(scenario['scenario'])}</th>"
            f"<td>{escape(', '.join(scenario.get('tags', [])) or '—')}</td>"
            f"<td>{escape(reasons)}</td></tr>"
        )
    slice_sections = []
    for label, key in (("Tag", "by_tag"), ("Model", "by_model"), ("Prompt version", "by_prompt_version")):
        rows = "".join(
            f"<tr><th scope=\"row\">{escape(item['slice'])}</th>"
            f"<td>{item['pass_rate'] * 100:.1f}%</td>"
            f"<td>{escape(', '.join(item['flaky_scenarios']) or '—')}</td></tr>"
            for item in report.get(key, [])
        )
        if rows:
            slice_sections.append(
                f"<h3>By {escape(label.lower())}</h3><table><thead><tr>"
                f"<th scope=\"col\">{escape(label)}</th><th scope=\"col\">Pass rate</th>"
                f"<th scope=\"col\">Flaky scenarios</th></tr></thead><tbody>{rows}</tbody></table>"
            )
    return _document(
        "Agent Eval repeated-run report",
        f"""
        <header class="page-header">
          <p class="eyebrow">Agent Eval</p>
          <h1>Repeated-run report</h1>
          <p>{report['runs']} runs across {report['scenario_count']} scenarios</p>
        </header>
        <section aria-labelledby="stability-heading">
          <h2 id="stability-heading">Pass-rate stability</h2>
          <div class="summary-grid">
            <div class="metric"><span>Mean pass rate</span><strong>{pass_stats['mean'] * 100:.1f}%</strong></div>
            <div class="metric"><span>Sample standard deviation</span><strong>{escape(format_percent(pass_stats['sample_stddev']))}</strong></div>
            <div class="metric"><span>Observed range</span><strong>{pass_stats['min'] * 100:.1f}%–{pass_stats['max'] * 100:.1f}%</strong></div>
            <div class="metric"><span>{report['confidence'] * 100:.0f}% Wilson interval</span><strong>{pass_stats['confidence_interval']['lower'] * 100:.1f}%–{pass_stats['confidence_interval']['upper'] * 100:.1f}%</strong></div>
          </div>
        </section>
        <section aria-labelledby="metrics-heading">
          <h2 id="metrics-heading">Metric variability</h2>
          <table><caption>Mean, sample standard deviation, and observed range across runs</caption>
            <thead><tr><th scope="col">Metric</th><th scope="col">Mean</th><th scope="col">Sample stddev</th><th scope="col">Range</th></tr></thead>
            <tbody>{''.join(metric_rows)}</tbody>
          </table>
        </section>
        <section aria-labelledby="scenarios-heading">
          <h2 id="scenarios-heading">Scenario stability</h2>
          <table><caption>Pass frequency by scenario</caption>
            <thead><tr><th scope="col">Scenario</th><th scope="col">Passed runs</th><th scope="col">Pass rate</th><th scope="col">Failed run numbers</th></tr></thead>
            <tbody>{''.join(scenario_rows)}</tbody>
          </table>
        </section>
        <section aria-labelledby="flakiness-heading">
          <h2 id="flakiness-heading">Flakiness analysis</h2>
          <p>{len(report.get('flaky_scenarios', []))} scenarios passed in some runs and failed in others.</p>
          <table><caption>Intermittent failures and their reason counts</caption>
            <thead><tr><th scope="col">Scenario</th><th scope="col">Tags</th><th scope="col">Failure reasons</th></tr></thead>
            <tbody>{''.join(flaky_rows) or '<tr><td colspan="3">No flaky scenarios</td></tr>'}</tbody>
          </table>
          {''.join(slice_sections)}
        </section>
        """,
    )


def _document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f5f7fa; }}
    body {{ margin: 0; }}
    main {{ max-width: 1000px; margin: 0 auto; padding: 40px 24px 64px; }}
    .page-header {{ border-bottom: 1px solid #dbe2ea; padding-bottom: 24px; margin-bottom: 32px; }}
    .eyebrow {{ color: #486581; font-size: 0.8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
    h1 {{ font-size: clamp(2rem, 4vw, 3rem); margin: 8px 0; }}
    h2 {{ margin: 32px 0 14px; font-size: 1.25rem; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .metric {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 10px; padding: 16px; display: grid; gap: 8px; }}
    .metric span {{ color: #52606d; font-size: .9rem; }}
    .metric strong {{ font-size: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe2ea; border-radius: 10px; overflow: hidden; }}
    caption {{ text-align: left; color: #52606d; padding: 0 0 8px; }}
    th, td {{ text-align: left; vertical-align: top; padding: 12px 14px; border-bottom: 1px solid #e6ebf0; }}
    thead {{ background: #eef2f6; }}
    tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
    .status {{ border-radius: 999px; display: inline-block; font-size: .85rem; font-weight: 700; padding: 3px 10px; }}
    .pass {{ color: #17663a; background: #e5f5eb; }}
    .fail {{ color: #9b1c1c; background: #fde8e7; }}
    @media (max-width: 600px) {{ main {{ padding: 24px 12px 48px; }} th, td {{ padding: 9px; }} }}
  </style>
</head>
<body><main>{body}</main></body>
</html>
"""
