# Agent Eval

**A lightweight, provider-independent harness for evaluating AI agent behavior against repeatable scenarios.**

Describe expected behavior in YAML, record what your agent actually asked and did as JSON Lines, and score the run from the command line. Use it to catch changes in intent selection, tool use, required questions, and policy-sensitive actions before they reach users.

Agent Eval does not call a model or depend on an agent framework. A small adapter turns your agent's execution trace into the result format below, so the same scenarios can be run against different prompts, models, and orchestration stacks.

## What it checks

- **Intent accuracy** — did the agent choose the expected intent?
- **Required questions** — did it ask for each piece of information the scenario requires?
- **Tool accuracy** — did it invoke the expected set of tools?
- **Policy compliance** — did it avoid actions listed under `must_not`?
- **Unnecessary questions** — how often did it ask for information outside the required or allowed question set?
- **Scenario pass rate** — did each scenario satisfy all of its declared checks?

The checks are deterministic and work over structured traces. They don't require an LLM judge, network access, or a model API key.

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
```

### Scenario fields

| Field | Required | Description |
| --- | --- | --- |
| `scenario` | Yes | Unique ID used to match the agent result. |
| `input` | Yes | Non-empty mapping passed to your agent adapter. |
| `expected.intent` | Yes | Expected intent label. Comparison ignores letter case. |
| `expected.must_ask` | No | Question identifiers the agent must ask. A missing required question fails the scenario. |
| `expected.tools` | No | Exact expected tool set. Omit this field when the scenario does not check tool selection. |
| `expected.tool_calls` | No | Exact ordered tool-call sequence, with optional argument matchers for each call. |
| `must_not` | No | Action identifiers that must not appear in the recorded actions. |
| `allowed_questions` | No | Additional question identifiers permitted for this scenario. |

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
| `notes` | No | Optional diagnostic text displayed with a failing scenario. |

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

Compare a candidate run with a baseline to see scenario regressions and metric changes:

```bash
agent-eval compare scenarios/ \
  --baseline examples/responses-before.jsonl \
  --candidate examples/responses.jsonl \
  --strict
```

The comparison exits `1` when the candidate regresses a scenario, `0` otherwise, and `2` for invalid input. Add `--format json` for machine-readable output. Metric deltas are candidate minus baseline; a negative unnecessary-question delta is an improvement.

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

## Understand the metrics

| Metric | Calculation |
| --- | --- |
| Intent accuracy | Correct intent results ÷ all scenarios. |
| Tool accuracy | Exact tool-set or ordered tool-call matches ÷ scenarios that declare `expected.tools` or `expected.tool_calls`. Displays `n/a` when no scenario checks tools. |
| Policy compliance | Satisfied prohibited-action checks ÷ all declared `must_not` checks. Displays 100% when there are no prohibited-action checks. |
| Unnecessary questions | Asked identifiers that are neither required nor allowed ÷ all recorded questions. Displays 0% when the agent asked no questions. |
| Scenario pass rate | Scenarios passing intent, required-question, unnecessary-question, declared tool, and prohibited-action checks ÷ all scenarios. |

A scenario fails if the intent is wrong, a required question is missing, an unnecessary question was asked, a declared tool set doesn't match, or a prohibited action appears in the trace. Policy checks look for exact action identifiers; they don't interpret free-form text or decide whether an action was safe in context.

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

Agent Eval is an early, local-first harness. It evaluates recorded structured traces, uses exact identifier matching, and produces aggregate metrics plus scenario-level failure details. It does not execute agents, grade natural-language answer quality, estimate statistical confidence, or guarantee that a scenario collection covers every production risk. Treat the score as evidence about the scenarios you wrote, and expand the suite as you learn about real user behavior and failure modes.

## Next ideas

Potential additions, in a useful implementation order:

1. **Repeated-run analysis** — run scenarios multiple times and report variability, since sampling can make agent behavior nondeterministic.
2. **HTML reports** — browse scenario-level failures and compare history without parsing console output.
3. **Framework-specific packages** — optional integrations for common frameworks, built on the adapter protocol without adding framework dependencies to the core package.

The repository includes GitHub Actions CI for Python 3.10–3.12, unit validation, and the example evaluation command.

## Related projects

Agent Eval focuses on a small, deterministic scenario-and-trace workflow. Larger evaluation frameworks offer complementary approaches:

- [Promptfoo](https://github.com/promptfoo/promptfoo) — prompt and agent evaluations, comparisons, and red teaming.
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM metrics, agent trajectory evaluation, and integrations.
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — a broader framework for defining and running LLM evaluations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
