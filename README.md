# Agent Eval

**A lightweight, provider-independent harness for evaluating AI agent behavior against repeatable scenarios.**

Describe expected behavior in YAML, execute scenarios through an optional runner plugin or score saved JSON Lines, and catch changes in intent selection, tool use, required questions, and policy-sensitive actions before they reach users.

The core package has no model-provider or agent-framework dependency. Optional runner, trace-adapter, and semantic-judge plugins let teams connect their own stack without coupling the core harness to a provider.

## What it checks

- **Intent accuracy** — did the agent choose the expected intent?
- **Required questions** — did it ask for each piece of information the scenario requires?
- **Tool accuracy** — did it invoke the expected set of tools?
- **Policy compliance** — did it avoid actions listed under `must_not`?
- **Unnecessary questions** — how often did it ask for information outside the required or allowed question set?
- **Scenario pass rate** — did each scenario satisfy all of its declared checks?
- **Run-to-run stability** — how consistent are those scores across repeated agent executions?
- **HTML reports** — can reviewers inspect a run or repeated-run analysis in a browser?
- **Multi-turn conversations** — can expectations be checked across turns while preserving order?
- **Coverage and mutation audits** — which behavior dimensions have scenarios, and do those scenarios catch representative regressions?
- **Framework adapters** — built-in normalizers for OpenAI Agents SDK spans and LangChain event/run records, alongside OpenTelemetry and custom adapters.
- **Safe tool fixtures** — deterministic mock tool responses are available to runner plugins; unknown calls fail closed without reaching external services.
- **Execution metadata** — optional model, prompt version, latency, token, and cost fields flow into reports and repeated-run trend summaries.
- **Judge calibration** — cache repeated semantic judgments, inspect score stability, and compare judge decisions across saved reports.
- **Coverage-guided suggestions** — generate review-only test candidates from coverage gaps without inventing expected labels.
- **Scenario version audits** — snapshot scenario contracts, detect breaking edits, and migrate scenario-set and per-scenario versions.
- **Failure triage and slices** — group regressions by cause and compare pass rates by intent, tag, model, and prompt version.
- **Execution budgets** — gate total cost/tokens and p95 scenario latency in local runs or CI.
- **Gold-set judge calibration** — measure semantic judge accuracy against human-labeled cases, including precision, recall, F1, and score calibration.
- **Reproducible run bundles** — package evaluated scenarios, results, reports, settings, and integrity hashes for later review or replay.
- **Pull request annotations** — surface scenario regressions as GitHub Actions annotations and add a metric comparison to the job summary.
- **Behavior drift monitoring** — compare sampled current traces with a baseline and fail when metric movement or scenario regressions exceed configured limits.
- **Model and prompt experiments** — score named result sets side by side, including quality, cost, token usage, and latency metadata.
- **Evaluation history** — append compact run or bundle summaries to JSONL and inspect pass-rate and metric trends over time.
- **Trace-to-scenario drafts** — convert observed traces into review-only YAML drafts while leaving expected behavior blank for human authors.

Built-in checks are deterministic and work over structured traces. Semantic judges are opt-in plugins; the core does not make provider calls or require model credentials.

## Quick start

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
agent-eval run scenarios/ --responses examples/responses.jsonl
```

Expected output for the included example run:

```text
3 scenarios

Passed: 3
Failed: 0

Intent accuracy:       100.0%
Tool accuracy:         100.0%
Policy compliance:     100.0%
Unnecessary questions: 0.0%
Answer quality:        50.0%
```

The CLI returns exit code `0` when every scenario passes, `1` when one or more scenarios fail, and `2` for invalid input. This makes it straightforward to use as a CI quality gate.

## How it works

```text
Your agent ── adapter ──> results.jsonl
                              │
scenarios/*.yaml ─────────────┤
                              ▼
                    agent-eval run
                              │
                    text or JSON report
```

1. Write scenario files that capture user inputs and the behavior you expect.
2. Run your agent and collect structured decisions from its trace.
3. Write one JSON object per scenario to a JSONL file.
4. Run `agent-eval` to calculate metrics and see which scenarios failed.

The adapter is the only framework-specific part. Capture actual tool calls and structured agent decisions; don't infer them from the final response text if execution traces are available. Agent Eval provides a `TraceAdapter` interface for code integrations and a path-mapping adapter for JSON traces, so frameworks do not need to become core dependencies.

### Adapt a framework trace

Map JSON trace paths to the canonical result fields:

```yaml
fields:
  scenario: metadata.scenario_id
  intent: agent.final.intent
  asked: agent.final.questions[*]
  tools: trace.tool_calls[*].name
  tool_calls: trace.tool_calls
  actions: trace.actions
  events: trace.events
```

Normalize a JSONL trace file before evaluation:

```bash
agent-eval adapt framework-traces.jsonl \
  --mapping adapter-mapping.yaml \
  --output results.jsonl
agent-eval run scenarios/ --responses results.jsonl --strict
```

Paths use dot notation for nested objects and `[*]` to collect a field from each list item. `scenario` and `intent` mappings are required; other result fields are optional. See [`examples/adapter_mapping.yaml`](examples/adapter_mapping.yaml) and [`examples/framework-traces.jsonl`](examples/framework-traces.jsonl) for a runnable example. For frameworks with non-JSON trace objects, implement the small `TraceAdapter` protocol and convert those objects to the same canonical result fields.

Framework packages can register optional adapters under the `agent_eval.adapters` Python entry-point group. Load one by its entry-point name:

```bash
agent-eval adapt-framework my-framework traces.jsonl --output results.jsonl
```

The plugin exposes `adapt(record) -> AgentResult`. This keeps framework SDKs and their version changes out of the core install.

For example, an integration package registers `[project.entry-points."agent_eval.adapters"]` with `my-framework = "my_adapter:Adapter"`. Adapter classes or factories are loaded only when selected by the command.

### Execute an agent directly

Install a runner plugin in the `agent_eval.runners` entry-point group, then execute scenarios and save both captured results and an evaluation report:

```bash
agent-eval run-agent scenarios/ --runner my-agent \
  --results run.jsonl --report report.html --format html
```

A runner implements `run_turn(scenario, user_input, history) -> AgentTurnResult`. Agent Eval calls it once for a single-turn scenario and sequentially for each `turns` entry, passing prior turn results as history. The runner owns provider setup, tool execution, timeouts, and credentials; it should return observed structured behavior instead of inferring tool use from answer text.

Register it from the runner package's `pyproject.toml`:

```toml
[project.entry-points."agent_eval.runners"]
my-agent = "my_agent.agent_eval:Runner"
```

Runners that accept a `tool_sandbox` keyword receive a scenario-scoped fixture sandbox. It only serves declared fixtures and raises `ToolFixtureError` for unmatched calls:

```yaml
tool_fixtures:
  - name: lookup_appointment
    arguments:
      patient_id: demo-patient
    response:
      appointment_id: appt-123
      date: "2026-10-14"
```

Call `tool_sandbox.call("lookup_appointment", {"patient_id": "demo-patient"})` from the runner. Fixtures may also specify `error` instead of `response` to exercise tool failures. Matching uses the declared argument mapping as a required subset, so extra runtime arguments are allowed.

Built-in trace normalization is available without installing framework SDKs:

```bash
agent-eval adapt-framework openai-agents openai-spans.jsonl --output results.jsonl
agent-eval adapt-framework langchain langchain-events.jsonl --output results.jsonl
```

Both adapters expect `agent_eval.scenario` and `agent_eval.intent` labels in trace metadata (or top-level `scenario` and `intent`). Framework trace schemas change over time; keep a small exported fixture from your installed SDK version and validate mappings before using the adapter in production.

## Define scenarios

Create a `.yaml` or `.yml` file in a scenario directory:

```yaml
scenario: appointment_reschedule
input:
  user: "Can I move my appointment to tomorrow?"
expected:
  intent: RESCHEDULE
  must_ask:
    - patient_identity
    - preferred_time
  tools:
    - lookup_appointment
must_not:
  - cancel_without_confirmation
allowed_questions:
  - appointment_date
tags: [scheduling, safety]
```

### Scenario fields

| Field | Required | Description |
| --- | --- | --- |
| `scenario` | Yes | Unique ID used to match the agent result. |
| `version` | No | Positive integer scenario contract version; defaults to `1` and should increase when its expected behavior changes. |
| `input` | Yes | Non-empty mapping passed to your agent adapter. |
| `expected.intent` | Yes | Expected intent label. Comparison ignores letter case. |
| `expected.must_ask` | No | Question identifiers the agent must ask. A missing required question fails the scenario. |
| `expected.tools` | No | Exact expected tool set. Omit this field when the scenario does not check tool selection. |
| `expected.tool_calls` | No | Exact ordered tool-call sequence, with optional argument matchers for each call. |
| `must_not` | No | Action identifiers that must not appear in the recorded actions. |
| `allowed_questions` | No | Additional question identifiers permitted for this scenario. |
| `expected.answer_graders` | No | Deterministic checks or installed grader plugins applied to the agent's final answer. |
| `tags` | No | Labels for selecting related scenarios from the CLI. |
| `turns` | No | Ordered conversation inputs with optional expectations for each turn. Use instead of `input`. |

Version the set with `scenarios/scenario-set.yaml`:

```yaml
scenario_set: support-agent
version: 1.2.0
schema_version: 1
```

`agent-eval scenario-version scenarios/ --format json --output snapshot.json` records stable scenario IDs, per-scenario versions, and contract hashes. Compare that snapshot with the last approved snapshot to classify added, removed, or changed contracts and enforce a semantic version bump:

```bash
agent-eval scenario-version scenarios/ --baseline approved-scenarios.json
```

Breaking contract changes recommend a major set version and require an increment to the changed scenario's `version`. Additions recommend a minor version; no contract changes recommend a patch. To migrate changed scenario versions and write the recommended set version in one step, use `--migrate` with a baseline. For a legacy/missing manifest, `--write-manifest` writes the current manifest schema. Snapshots contain hashes rather than scenario input contents.

For a multi-turn scenario, use `turns` instead of `input`. Top-level `expected` fields check the aggregate conversation and final intent; each turn can add specific assertions:

```yaml
scenario: appointment_booking_conversation
turns:
  - input:
      user: "I need to book an appointment."
    expected:
      intent: BOOK
      must_ask: [preferred_date]
  - input:
      user: "Tomorrow morning."
    expected:
      intent: BOOK
      must_ask: [patient_identity]
expected:
  intent: BOOK
```

Turn expectations can include `intent`, `must_ask`, `tools`, and `tool_calls`; turn-level `must_not` and `allowed_questions` are supported too. Recorded results keep an ordered `turns` array, and aggregate checks still use the combined trace.

Question, tool, and action names are identifiers agreed on by your adapter and scenario files. Tool order doesn't matter; an extra or missing tool makes the tool check fail. Identifiers are compared without letter case or surrounding whitespace.

For order- and argument-sensitive checks, declare `expected.tool_calls`:

```yaml
expected:
  intent: RESCHEDULE
  tool_calls:
    - name: lookup_appointment
      arguments:
        patient_id:
          exists: true
        requested_date:
          one_of: [tomorrow, "2026-10-14"]
    - name: reserve_appointment
      arguments:
        slot_id:
          contains: morning
```

The list is an exact sequence: missing, extra, or out-of-order calls fail. Argument values can be exact scalars or use one matcher: `equals`, `not_equals`, `one_of`, `contains`, or `exists`. Argument paths can use dots to inspect nested objects, such as `patient.id`.

Policy rules can require a confirmation or identity question before a sensitive action is permitted:

```yaml
must_not:
  - action: cancel_appointment
    unless:
      asked: cancellation_confirmation
```

This fails if the action occurred before the named question. For conditional rules, include ordered question/action events in the result so the evaluator can verify the sequence; a final `asked` list alone cannot prove ordering. Existing string entries in `must_not` remain unconditional prohibitions.

The repository includes scenarios for appointment rescheduling, prescription refill requests, and clinic hours. The healthcare examples are illustrative fixtures; replace them with scenarios and policy labels that match your application.

## Record agent results

Write one JSON object per line. Each `scenario` must match one scenario ID.

```json
{"scenario":"appointment_reschedule","intent":"RESCHEDULE","asked":["patient_identity","preferred_time"],"tools":["lookup_appointment"],"actions":[]}
```

### Result fields

| Field | Required | Description |
| --- | --- | --- |
| `scenario` | Yes | Scenario ID from the YAML file. |
| `intent` | Yes | Agent's selected or classified intent. |
| `asked` | No | Question identifiers the agent asked. |
| `tools` | No | Tool names invoked during this run. |
| `tool_calls` | No | Ordered tool calls as objects with `name` and an `arguments` mapping. This can supply the tool list when `tools` is omitted. |
| `actions` | No | Action identifiers checked against the scenario's `must_not` list. |
| `events` | No | Ordered `{type, name}` records, where type is `question` or `action`. Needed to verify that conditional policy prerequisites happened before an action. |
| `turns` | No | Ordered per-turn result records for a multi-turn scenario. Aggregate fields are derived from turns when omitted. |
| `answer` | No | Final answer text used only when the scenario declares answer graders. |
| `notes` | No | Optional diagnostic text displayed with a failing scenario. |
| `metadata` | No | Execution details such as `model`, `prompt_version`, `latency_ms`, `input_tokens`, `output_tokens`, and `cost_usd`. |

An adapter can capture results from an agent trace, tool middleware, or your application's own structured response. Keep the result file free of real patient information, credentials, and other sensitive data—scenario identifiers and behavior labels are enough for scoring.

## Run evaluations

Score a directory recursively or pass one scenario file:

```bash
agent-eval run scenarios/ --responses run.jsonl
agent-eval run scenarios/appointment_reschedule.yaml --responses one-run.jsonl
```

Write a machine-readable report for CI artifacts or another reporting tool:

```bash
agent-eval run scenarios/ --responses run.jsonl --format json > report.json
```

Use `--strict` to return an input error when any scenario result is missing. Without it, missing results appear as failed scenarios in the report. Unknown and duplicate scenario IDs in the result file are always input errors. Use `--quiet-details` to print only the text summary.

Set execution budgets to fail the run when a measured limit is exceeded or a requested metric is absent:

```bash
agent-eval run scenarios/ --responses run.jsonl \
  --max-cost-usd 2.00 --max-input-tokens 50000 --max-output-tokens 12000 \
  --max-p95-latency-ms 1500
```

Cost and token limits apply to totals across the selected scenarios. Latency uses the nearest-rank p95 across per-scenario `metadata.latency_ms` values. Gate results appear in text, JSON, and HTML reports; any failed budget returns exit code `1`. The same options are available for `run-agent`.

Cluster and slice failures to find common regressions:

```bash
agent-eval triage scenarios/ --responses run.jsonl --format json
```

The report groups failures by cause and summarizes pass rates by expected intent, scenario tag, and available model/prompt metadata. `triage` is informational and returns `0` after producing the report.

Create a portable run bundle with any recorded or plugin-backed evaluation:

```bash
agent-eval run scenarios/ --responses run.jsonl --strict \
  --bundle-dir artifacts/evaluation-2026-10-13
agent-eval verify-bundle artifacts/evaluation-2026-10-13
```

Bundles contain the selected scenario YAML files, scenario version snapshot, normalized `results.jsonl`, JSON report, run settings, and a SHA-256 inventory. Verification checks every listed file and the bundle run ID. Re-run an evaluation with `agent-eval run artifacts/evaluation-2026-10-13/scenarios --responses artifacts/evaluation-2026-10-13/results.jsonl --strict`. Bundle directories must be new; existing destinations are never overwritten. Bundles copy scenario inputs and captured results, so protect them like the source data.

Semantic judges can be repeated and cached during evaluation. Cache files should be treated as evaluation artifacts because they contain judge scores, rationales, and request fingerprints:

```bash
agent-eval run scenarios/ --responses run.jsonl --judge-repetitions 3 --judge-cache .agent-eval/judges.json
agent-eval judge-agreement report-a.json report-b.json
```

Repeated judge scores and their population standard deviation are recorded in grade metadata. `judge-agreement` compares pass/fail decisions and score ranges by scenario across JSON reports. It needs at least two reports containing semantic judge results.

Calibrate a semantic judge against a held-out YAML gold set with human-labeled expected decisions:

```bash
agent-eval calibrate-judge examples/judge-gold-set.yaml \
  --judge my-judge --model judge-model-v2 --threshold 0.7 --min-accuracy 0.9
```

Gold sets declare `cases` with a unique `id`, `input`, `answer`, and `expected_passed`; optional `expected_score` labels the target score. A top-level or case-level rubric is supported. Reports include confusion counts, accuracy, precision, recall, F1, Brier score, mean absolute score error, judge version, and individual disagreements. `--min-accuracy` makes calibration a CI gate. Keep the gold set separate from examples used to tune the judge or rubric. See [`examples/judge-gold-set.yaml`](examples/judge-gold-set.yaml).

Store compact evaluation summaries over time and inspect changes from the first retained run to the latest:

```bash
agent-eval history record --scenarios scenarios/ --responses run.jsonl \
  --store .agent-eval/history.jsonl --label release-2026-10
agent-eval history record --bundle artifacts/agent-eval-bundle \
  --store .agent-eval/history.jsonl --label release-2026-10
agent-eval history trend .agent-eval/history.jsonl --limit 30
```

`history record` accepts canonical results or a bundle. Bundles are integrity-verified before recording. The append-only JSONL store keeps aggregate metrics, counts, selected execution metadata, and labels; it does not copy scenario inputs or traces. Keep the store with your CI artifacts or durable storage if you need long-term trends.

Generate deterministic coverage-guided candidates for human review:

```bash
agent-eval suggest scenarios/ --limit 25 --output suggestions.json
```

Suggestions include a small input paraphrase and missing behavior dimensions. They are deliberately not executable scenarios: a reviewer must author and verify expected behavior before adopting a candidate.

Generate scenario drafts from exported JSONL traces using the same field mapping as `adapt`. Supply a path to the captured user input:

```bash
agent-eval draft-scenarios traces.jsonl \
  --mapping examples/draft-mapping.yaml \
  --input-path request.user \
  --output-dir scenario-drafts/
```

Each YAML file separates `observed` intent, questions, tools, and actions from the `expected` section. Expected values are `null` and the document is marked `draft: true`, so the trace is never silently promoted to an oracle. Replace those placeholders with human-approved behavior before moving a draft into the executable scenario set. Output directories must be new. Trace inputs can contain personal or confidential data; sanitize exports before creating drafts.

Compare a candidate run with a baseline to see scenario regressions and metric changes:

```bash
agent-eval compare scenarios/ \
  --baseline examples/responses-before.jsonl \
  --candidate examples/responses.jsonl \
  --strict
```

The comparison exits `1` when the candidate regresses a scenario, `0` otherwise, and `2` for invalid input. Add `--format json` for machine-readable output. Metric deltas are candidate minus baseline; a negative unnecessary-question delta is an improvement.

Emit inline GitHub Actions annotations for regressed scenarios and append a Markdown comparison to the job summary:

```bash
agent-eval compare scenarios/ \
  --baseline artifacts/baseline.jsonl \
  --candidate artifacts/candidate.jsonl \
  --strict --github-annotations
```

The command uses `GITHUB_STEP_SUMMARY` when available; pass `--summary-file path.md` to select a summary file explicitly. Each regressed scenario becomes an `::error` annotation, and the command keeps the normal regression exit status so it can gate a pull request.

Monitor a sampled production trace batch against a known-good canonical JSONL baseline:

```bash
agent-eval monitor scenarios/ \
  --baseline artifacts/release-baseline.jsonl \
  --current artifacts/production-sample.jsonl \
  --max-absolute-drift 0.05 --format json --output drift-report.json
```

The default threshold is five percentage points of absolute movement per available metric. Any scenario regression or metric beyond the threshold fails the command. Convert exported traces with `adapt-otel` or another adapter first; the OpenTelemetry adapter excludes sensitive attributes by default. Keep sampled inputs and result artifacts redacted and apply your normal retention controls.

Compare two or more model or prompt result sets with named variants:

```bash
agent-eval experiment scenarios/ \
  --variant model-a-prompt-v1=artifacts/model-a.jsonl \
  --variant model-b-prompt-v2=artifacts/model-b.jsonl \
  --format json --output experiment.json
```

Each variant reports its recorded model and prompt versions, scenario pass rate, behavior metrics, failing scenario IDs, and available aggregate cost/token plus mean and p95 latency metadata. The harness scores captured results; your runner or experiment pipeline is responsible for executing each model/prompt combination and recording its metadata.

Analyze variability across repeated executions by passing two or more captured result files:

```bash
agent-eval repeat scenarios/ \
  --runs runs/run-01.jsonl runs/run-02.jsonl runs/run-03.jsonl \
  --strict
```

Agent Eval analyzes supplied result files; it does not launch the agent or choose a model sampling configuration. The report includes each metric's mean, sample standard deviation, and observed range, plus per-scenario pass frequency and the run numbers that failed. `repeat` requires at least two runs and exits `1` if any run has failed scenarios.

Repeated-run reports also include Wilson score confidence intervals for the overall and per-scenario pass rates. Gate a release on the lower confidence bound with `--min-pass-rate`; the gate fails if the lower bound is below the requested threshold, even when the observed pass rate is higher. Set `--confidence` (default `0.95`) to a value between `0.5` and `1`, and `--min-runs` (default `2`) to require a minimum number of captures:

```bash
agent-eval repeat scenarios/ --runs runs/*.jsonl \
  --min-runs 10 --confidence 0.95 --min-pass-rate 0.98 --strict
```

The interval measures uncertainty in the aggregate scenario pass proportion, assuming each scenario result is a Bernoulli outcome. It does not correct for correlated runs, scenario selection bias, or incomplete scenario coverage. Choose a run count and threshold appropriate to the risk; a small sample can have a wide interval even with no observed failures.

### Filter by scenario tags

Add a `tags` list at the scenario's top level, then pass one or more `--tag` and `--exclude-tag` options to `run`, `compare`, or `repeat`. Inclusion uses OR semantics (a scenario matching any included tag is selected); exclusions always take precedence. Matching ignores case.

```bash
agent-eval run scenarios/ --responses run.jsonl --tag safety --tag medication --exclude-tag faq
```

Tags help run focused suites while preserving the same scenario definitions. A filtered run still rejects unknown result IDs and ignores result rows for scenarios outside the selected set.

### Grade answer quality

Answer graders are opt-in. Add grader specifications under `expected.answer_graders` and store the final answer as the optional `answer` string in each result. Every configured grader must pass for its scenario to pass; the report also includes the mean grader score as `answer_quality_score`.

```yaml
expected:
  intent: OFFICE_HOURS
  answer_graders:
    - type: contains_any
      values: ["5 PM", "17:00"]
      name: includes_closing_time
    - type: not_contains
      values: ["patient record"]
```

Built-in grader types are `contains_any`, `contains_all`, `not_contains`, `min_length`, and `max_length`. Text graders accept `case_sensitive` (default `false`); `contains_any` and `contains_all` can accept a `threshold` from 0 to 1. Length graders take a non-negative integer `value`. These are deterministic checks, not semantic truth or clinical safety assessments.

Custom graders can be installed as Python entry points in the `agent_eval.graders` group. Each entry-point name is the scenario's `type`; its loaded class/factory/object must expose `grade(answer: str, config: Mapping[str, Any]) -> GradeResult`. The `GradeResult` contains a normalized `score` from 0 to 1, `passed`, optional `feedback`, and optional `metadata`. Grader packages run in-process and should be treated as trusted code.

For semantic scoring, install a judge plugin in `agent_eval.judges` and declare `type: llm_judge`:

```yaml
expected:
  intent: OFFICE_HOURS
  answer_graders:
    - type: llm_judge
      judge: internal_judge
      model: evaluator-model-v1
      threshold: 0.8
      rubric: "Check that the answer is complete, directly answers the question, and does not invent hours."
```

The plugin implements `judge(request: JudgeRequest) -> GradeResult` and exposes a `version` string. Results record the configured model, judge name and version, rubric and SHA-256, score, and rationale (`feedback`). Provider selection, network access, privacy controls, and reproducibility of the underlying model remain the plugin's responsibility.

Register a judge with an entry point such as `[project.entry-points."agent_eval.judges"]` and `internal_judge = "my_package.judge:Judge"`. Judge plugins run only when a scenario explicitly configures `llm_judge`.

### Audit scenario coverage and sensitivity

Inspect which intent, tool, policy, tag, and check dimensions your scenarios declare:

```bash
agent-eval coverage scenarios/ --format json
```

Run a deterministic mutation audit against a passing result set to check whether the evaluator catches regressions such as wrong intent, missing required questions, changed tools, and prohibited actions:

```bash
agent-eval mutate scenarios/ --responses run.jsonl --format json
```

The command exits `1` when a generated mutation survives and `2` when there are no eligible mutations to audit. Missing or already-failing baseline scenarios are reported as not checked. Coverage summarizes what is declared; it does not prove that a scenario set represents production traffic.

### Ingest OpenTelemetry traces

Convert an OTLP/JSON trace export into the canonical JSONL result format:

```bash
agent-eval adapt-otel otel-traces.json --output results.jsonl
agent-eval run scenarios/ --responses results.jsonl --strict
```

The adapter groups spans by trace ID, identifies agent invocations and tool spans through the GenAI semantic convention attributes, and reads scenario and intent from `agent_eval.scenario` and `agent_eval.intent` by default. Override those attribute names with `--scenario-attribute` and `--intent-attribute`. Emit `agent_eval.event.type` (`question` or `action`) and `agent_eval.event.name` on spans or span events to capture ordered policy events.

Trace content can contain sensitive data. The adapter omits tool arguments and generated answer text by default. `--include-sensitive-attributes` opts in to extracting `gen_ai.tool.call.arguments`, `gen_ai.output.messages`, or the custom `agent_eval.answer` attribute; review and sanitize trace exports before storing or sharing result files.

Create a standalone, shareable HTML report for one run or repeated-run analysis:

```bash
agent-eval run scenarios/ --responses run.jsonl --format html --output report.html
agent-eval repeat scenarios/ --runs runs/*.jsonl --format html --output stability.html
```

HTML reports include summary metrics and scenario-level details, escape trace content, and have no external asset or network requirements.

```bash
agent-eval run scenarios/ --responses run.jsonl --strict
```

### CI example

Run the same command after your application creates `run.jsonl`. A failed scenario naturally returns a non-zero status and fails the job:

```yaml
- name: Evaluate agent behavior
  run: agent-eval run scenarios/ --responses run.jsonl --strict
```

The example assumes your preceding CI steps installed this package and generated the result file. Agent Eval intentionally does not start your agent or manage model credentials.

The included [GitHub Actions workflow](.github/workflows/ci.yml) runs the unit suite across Python 3.10–3.12, evaluates the checked-in sample results, checks a repeated-run gate, creates HTML reports, and uploads them as workflow artifacts.

## Understand the metrics

| Metric | Calculation |
| --- | --- |
| Intent accuracy | Correct intent results ÷ all scenarios. |
| Tool accuracy | Exact tool-set or ordered tool-call matches ÷ scenarios that declare `expected.tools` or `expected.tool_calls`. Displays `n/a` when no scenario checks tools. |
| Policy compliance | Satisfied prohibited-action checks ÷ all declared `must_not` checks. Displays 100% when there are no prohibited-action checks. |
| Unnecessary questions | Asked identifiers that are neither required nor allowed ÷ all recorded questions. Displays 0% when the agent asked no questions. |
| Scenario pass rate | Scenarios passing intent, required-question, unnecessary-question, declared tool, and prohibited-action checks ÷ all scenarios. |

A scenario fails if the intent is wrong, a required question is missing, an unnecessary question was asked, a declared tool set doesn't match, a prohibited action appears in the trace, a turn expectation fails, or any configured answer grader fails. Policy checks look for exact action identifiers; they don't interpret free-form text or decide whether an action was safe in context.

## Project structure

```text
agent_eval/
├── examples/
│   └── responses.jsonl   # Example agent trace
├── scenarios/            # YAML behavior specifications
├── src/agent_eval/        # CLI, loaders, scoring, and report formatting
├── LICENSE
├── README.md
└── pyproject.toml
```

## Current scope

Agent Eval is an early, local-first harness. It evaluates structured traces, uses exact identifier matching for deterministic checks, and produces aggregate metrics plus scenario-level failure details. Runner and judge plugins execute in-process and may make provider calls, so install only trusted integrations. Built-in answer graders are simple deterministic checks, and confidence intervals assume independent Bernoulli outcomes. Treat scores as evidence about the scenarios and captures you supplied, and expand the suite as you learn about real user behavior and failure modes.

The repository includes GitHub Actions CI for Python 3.10–3.12, unit validation, and the example evaluation command.

## Related projects

Agent Eval focuses on a small, deterministic scenario-and-trace workflow. Larger evaluation frameworks offer complementary approaches:

- [Promptfoo](https://github.com/promptfoo/promptfoo) — prompt and agent evaluations, comparisons, and red teaming.
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM metrics, agent trajectory evaluation, and integrations.
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — a broader framework for defining and running LLM evaluations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
