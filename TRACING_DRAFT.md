# Shared execution identity, separate consumer views

## Problem: consumers reconstruct the same execution differently

DSPy programs compose modules, predictors, LM calls, tools, and interpreters. A signature is not
an execution: the same predictor can run repeatedly, inside copied attempts, or concurrently.
Metrics, demonstration builders, optimizer research agents, observability SDKs, and feedback-driven
learning need to refer to that work without independently reconstructing its identity.

Unify execution identity and capture boundaries; preserve distinct views for optimization,
observability, accounting, and agent state. This is a prototype, not a production-ready storage
schema or a claim that every existing trace collector has been replaced.

| Consumer | Needs | Must continue to own |
| --- | --- | --- |
| Metric | Requested purpose and available evidence | Return semantics, acceptance thresholds, judge policy |
| Demonstration builder | Selected predictor occurrences | Which attempts become training examples |
| Optimizer research agent | Readable executions, tool results, failures | Candidate ancestry, experiments, replay environment |
| GEPA | Exact component occurrence | Which component to critique and how feedback affects search |
| ProAct | Delayed feedback with durable targets | Case/channel mapping, multiple reviews and revision policy |
| CMPND | Correlated lifecycle observations | Export, retention, database identity and assessments |
| Accounting / History | Relevant usage / task-facing presentation | Provider accounting / conversation and subtask state |

An execution record contains invocations, not semantic turns or signatures. Events, selected
attempts, sessions, and cases are not interchangeable with invocation nesting. Feedback annotates
observed work; it is not another causal execution step. Captured input/output is not necessarily
enough to replay an environment.

The optimizer research thread distinguishes an implemented MetaHarness reading file-backed logs
from proposed searchable-trace/component-replay tools. ProAct already accepts late case/channel
feedback; its CMPND example drops annotation targets when importing assessment bodies. Neither
system establishes a requirement for DSPy to own persistence or in-flight agent intervention.

## Three implementation rounds

1. **Ordinary callbacks as the sole recorder.** Implemented and rejected: a tool inside
   `dspy.context(callbacks=[])` disappears, and retained raw output changes after caller mutation.
   Capture now runs at the existing decorated invocation boundary, using the same call ID as
   callbacks. It is independent of optional observer configuration, not a second instrumentation
   decorator or an external tracing dependency.
2. **One execution record, linked consumer views.** Records include all observed attempts, but
   demonstration tuples still select the winner. Projecting all predictor spans into demonstrations
   would be incorrect. Predictor outputs carry their invocation reference. A capture-local weak
   name map records known copy provenance; GEPA no longer carries a parallel `trace_names` array.
   `bootstrap_trace_data` wraps calls instead of patching `program.forward` during evaluation.
3. **Detached records and separate targeted feedback.** Records are JSON snapshots, and references
   resolve to a trace, invocation, or JSON Pointer into a completed invocation's output. An output
   target requires an explicit invocation: a scope may contain multiple roots. Sealing prevents
   late workers rewriting a finished record. Parse failures now retain the exact failing sibling
   and its inputs when execution capture is enabled, instead of inferring identity from signature.

The final design deliberately keeps the two contracts separate:

```python
def metric(example_or_inputs, prediction, *, metric_context): ...

with dspy.capture_execution(program) as capture:
    prediction = program(question="...")
execution = capture.trace
root = execution.spans[0]  # This example has one root; consumers must not assume that generally.
feedback = execution.annotate(
    "Support this claim", span_id=root["span_id"], output_path="/answer",
)
assert execution.resolve(feedback.target) == execution.span(root["span_id"])["outputs"]["answer"]
```

`MetricContext` says what answer the caller needs (`score`, `accept`, or `feedback`) and supplies
evidence it already has. `ExecutionTrace` records observations; `ExecutionRef` addresses evidence;
`Feedback` pairs a reference with content. Feedback is not a universal metric return type.
Results remain caller-specific. No exporter, History API, universal verdict, or database is added.

**Migration policy:** preserving legacy metric signatures is no longer a design constraint.
The [migrating-dspy-metrics skill](.agents/skills/migrating-dspy-metrics/SKILL.md) guides application
agents to the context-only contract, including direct callers, wrappers, and behavioral checks.
The current implementation still contains the compatibility bridge; it is transitional code, not
a requirement of the target API. Removing it is separate from testing execution capture.

## Start from the decisions, not the signatures

The pre-prototype baseline is `ee02b214ab2aa27c458e5379ac7882da02dd4be6`. These are the direct invocation
boundaries; other optimizers delegate to them. Evidence and return interpretation are independent.

| Boundary | Legacy inputs/evidence | Question and result consumer | New purpose |
| --- | --- | --- | --- |
| `Evaluate.__call__` | gold, prediction; no trace argument | Sum raw results and average. Prediction arithmetic uses `.score`; result triples retain the raw object. Errors follow the evaluator budget. | score |
| `BootstrapFewShot._bootstrap_one_example` | gold, prediction, live predictor tuple list | With a truthy threshold, compare numerically; otherwise use truthiness to admit demonstrations. Threshold `0` means absent. | score with threshold; accept otherwise |
| `bootstrap_trace_data` → GRPO/GEPA | gold, prediction, **None**, despite collecting a trace | Retain scores for rewards/ranking; keep existing parse/error handling. | score |
| `BootstrapFinetune.compile` → same helper | same third **None** | Filter examples by stored score truthiness before preparing training data. | accept |
| `SIMBA.wrap_program` | gold, prediction; prediction may be None after failure | Numeric fitness or Prediction score plus metadata; metric error → zero. | score |
| `AvatarOptimizer.process_example` | gold, prediction | Fitness; error → zero. | score |
| `ReAnchor.metric_value` | gold, prediction | `.score` or float; caller requires finite values; errors propagate. | score |
| `BestOfN.forward`, `Refine.forward` | **inputs dict**, prediction | Rank attempts and stop at reward threshold. Reward/judge tuples do not become demonstrations. Refine's advice is separate. | score |
| GEPA evaluation | two arguments, or third None through trace helper | Extract program fitness and optional objective scores. | score |
| GEPA component feedback | gold, prediction, full trace, component name, singleton step list | Use feedback text; ignore differing component score in favor of program fitness. | feedback |
| Flex evaluation | declared GEPA args as None plus experimental `program_trace` | Program fitness and objective scores; candidate evidence needed even without trajectory retention. | score |

MIPRO, COPRO, random search, Optuna, InferRules, and BetterTogether use the evaluation/bootstrap
boundaries rather than introducing a fourth metric purpose. GRPO uses the trace helper's default
score request. Optimization strategy does not itself determine metric purpose.

**Essential differences:** a continuous fitness, a demonstration-admission decision, and targeted
diagnostic feedback answer different questions. Thresholded bootstrapping asks for fitness, not a
boolean. GEPA feedback needs a particular occurrence, not merely a predictor signature.

**Accidental differences:** argument count, using trace presence as a mode switch, hiding available
evidence behind `trace=None`, and inferring target identity from equal signatures. A complete scoring
trace is still scoring; an empty acceptance trace is still acceptance.

## Two credible designs

| Existing metric | Universal verdict: score/accepted/feedback/objectives | Explicit request, caller-owned result (chosen) |
| --- | --- | --- |
| Exact match returning bool | Must wrap an already sufficient answer; consumers need missing-field rules. | Same answer works for fitness or acceptance. |
| Partial match returning 0.25 | Must decide who supplies acceptance and whose threshold wins. | Fitness remains 0.25; acceptance branch supplies its own boolean when requested. |
| `SemanticF1` / `CompleteAndGrounded` | Automatic normalization changes Prediction truthiness. | Existing built-ins remain legacy; opt-in migration must intentionally choose boolean acceptance. |
| LM judge returning score, feedback, objectives | Either always compute expensive feedback or add a request discriminator anyway. | Compute feedback only when requested; retain existing GEPA/SIMBA result shapes. |
| GEPA component critique | Universal result still needs separate target context. | Name plus trace index identify the selected occurrence. |
| Flex efficiency metric | Universal result does not supply evidence or separate judge work. | Existing tuple capture and UsageTracker supply candidate-only evidence. |

The universal verdict is attractive for generic result consumers, but those consumers do not exist
here. It moves thresholds, fallback feedback, coercion, and missing-field policies into a new layer
while the optimizers still own their decisions. Standardize **the request**, not the answer. The
adapter calls once and returns the exact object. It never retries a metric's TypeError.

## Metric evidence is a view, not the entire execution

`MetricContext` retains its decision fields and adds optional recorded execution:

- `purpose`: `score`, `accept`, or `feedback`.
- `trace`: optional tuple of existing `(predictor, inputs, prediction)` steps. None means unavailable;
  an empty tuple means no steps recorded. Membership is frozen, payload objects are not deep-copied.
- `component_name` and `target`: optional GEPA name and zero-based index into this trace. Repeated
  calls have distinct indices. These are not global execution IDs or a persistence schema.
- `usage`: optional existing UsageTracker totals by model. Reported non-cached tokens, not an invoice.
- `execution`: optional detached `ExecutionTrace`. `target_ref` resolves the selected predictor
  tuple to its durable invocation identity, or is None when that occurrence was not recorded.

Evaluate, Avatar, and ReAnchor do not promise evidence. Existing bootstrap/ranking capture supplies
tuple evidence. Flex additionally supplies usage even with `capture_traces=False`; candidate capture
ends before the judge runs. A tuple count is **not an LM-request count**: cached predictions still
produce tuples; direct LM calls and failures need not produce them.

GEPA reads component names from the referenced invocation, then selects an occurrence. Legacy
metrics retain signature matching. BestOfN/Refine tell capture the provenance of their copies;
unnamed dynamic modules still have invocation IDs but no invented static component path.
Candidate usage continues using the existing tracker and joined-worker sharing policy. Recording
does not charge usage again. Totals do not establish accounting for detached or abandoned work.

Execution capture is automatic at the contextual bootstrap/GEPA/Flex candidate boundary and
available explicitly through `capture_execution()` or `bootstrap_trace_data(capture_execution=True)`.
Other metric callers need not have execution evidence. Existing demonstration tuple collection
remains: this prototype links its identity, not a claim to have removed every collection path.

## Context-only migration and current runtime behavior

The migration target requires `metric_context`, including on a Module's `forward`; do not add
dual signatures by default. The current bridge detects an explicitly declared parameter, not
`**kwargs`, and still preserves old positional calls and built-ins. These are implementation facts,
not a promise to retain that bridge. Agents must distinguish signature changes from policy changes:
`Prediction(score=False)` is truthy and a zero bootstrap threshold currently means absent. Resolve
intended acceptance decisions explicitly instead of mechanically changing return truthiness.

```python
# Target contract: migrate direct callers as well as the metric.
def metric(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    return score >= 0.8 if metric_context.purpose == "accept" else score

# Migrated GEPA metric: only feedback requests pay for a targeted critique.
def gepa_metric(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    if metric_context.purpose == "feedback":
        predictor, inputs, output = metric_context.trace[metric_context.target]
        return dspy.Prediction(score=score, feedback=f"Review {metric_context.component_name}: {output}")
    return score

# Experimental Flex exception: replace program_trace now, rather than migrate twice.
def efficient(gold, pred, *, metric_context):
    tokens = 0
    for counts in metric_context.usage.values():
        total = counts.get("total_tokens")
        if total is None:
            total = counts["prompt_tokens"] + counts["completion_tokens"]
        tokens += total
    return float(gold.answer == pred.answer) - tokens / 100000
```

The Flex example requires Flex's evidence guarantee; it is not a portable cost metric for Evaluate.
Old `program_trace` metrics fail with a migration message. Ordinary legacy Flex metrics still work.
Use the migration skill to update projects to the target contract rather than preserving every old
calling convention in DSPy. No release, dependency upgrade, or runtime removal is performed by
adding the skill. Removing the bridge and migrating DSPy's remaining internal callers/built-ins
remain implementation work; the earlier compatibility replay records the current bridge's behavior.

## What gets simpler, and what deliberately remains

- Removed the metric-only `_metric_predictor_names` registry and aligned `trace_names` arrays.
  Component provenance is recorded once per invocation; consumers resolve references.
- Removed temporary `program.forward` replacement/restoration from bootstrap collection.
  The wrapper owns each candidate boundary without mutating the candidate program.
- Callbacks and recorded invocations use one ID generator and the existing worker propagation.
  No second ContextVar hierarchy, decorator, event bus, or tracing SDK is introduced.
- Completed records contain JSON, not predictor objects or discarded cloned programs.
- Demonstration selection, metric results, existing UsageTracker, and History remain distinct.

The larger original prototype's OTel exporter, observer protocol, execution-selection model,
evaluation-attempt retention, History correlation, and client accounting attachments remain absent.
The new ExecutionTrace is a detached read record, not restoration of that whole framework.

## CMPND and ProAct exposed real adapter requirements

CMPND 0.40.0 generates independent wire span and trace UUIDs. A real local-only SDK trial disproved
the assumption that a DSPy callback call ID is already its wire span ID. For supported module/tool
start events, a companion callback after CMPND can record the mapping through public
`get_current_span()`. The trial verifies both mapped spans in exported packets, without backend
writes. This is not a general exporter: skipped/collapsed SDK operations and callback failures
still require explicit policy. LM usage observations separately export `sdk_call_id`.

A real ProAct ledger trial preserves target + content as JSON in a string feedback channel and
resolves the original output after round-trip. A second review on the same case/channel is rejected
by ProAct. This validates ingestion, not learning quality, revision semantics, or a production
adapter. CMPND root-output pointers also require checking that the selected invocation maps to the
SDK's root before exporting an output annotation; arbitrary span-output targets are richer than
CMPND's current assessment API.

## Frozen checks and remaining uncertainty

Before the metric-only revision, expectations were frozen in `metric-contract-expectations.json`
(SHA-256 `af4da78c20de71515f612f3b1db5f57f14c5920bf61532a096ea75347557f089`). Evidence and runnable
harnesses are attached to the PR's Amp thread; the focused contract checks are in
`tests/evaluate/test_metric_context.py` and `tests/flex/test_flex_gepa.py`. Before the three capture
rounds, additional expectations were frozen in `tracing-rounds/expectations.json`, SHA-256
`076d082ca60c67c10915c118909819f28742e728417e5e4ca9b134ecad8dbe0d`. The new public-boundary tests
are in `tests/utils/test_execution.py`.

The differential harness executes identical bytes against the baseline and revised packages in
separate processes: 611 cases, zero differences. Coercing returns to bool changes 182 cases,
including BestOfN incorrectly selecting 0.2 over 0.8. Cases cover ABIs, raw results, exceptions,
thresholds, bootstrap, ranking, trace collection, and GEPA score/feedback interpretation.
The new replay uses the metric-only head as baseline and again finds zero differences. Nine new
mutations are caught: lost parent, reused invocation ID, wrong occurrence, wrong purpose, erased
evidence, coerced result, accepted foreign feedback, mutable output, and late writes after sealing.
Two test-fixture mistakes were corrected: Parallel is an executor, not an instrumented module;
the two-attempt test now supplies two LM responses and asserts both attempts succeed.

Real Azure + Deno + cmpnd callback checks compare independent provider requests with candidate
evidence: cached and uncached two-step candidates have two tuples but one versus two requests;
deterministic candidates have none. Judge requests are excluded, with trajectory retention both
on and off. The fresh 12-trial run also checks exact invocation references and sealed record
stability across judge execution. cmpnd delivery is captured locally, not sent to a deployed backend.

A nine-repeat local deterministic-call probe measured medians of 9.50 ms before and 9.12 ms with
capture disabled per 1,000 calls (noise, not a speedup claim), versus 172.26 ms with capture and
JSON readout enabled. That record is about 282 KB. All 1,000 discarded module objects were collected
while retaining the record. This prototype trades capture cost for evidence; it is not free.

Limits: replay is not a proof over arbitrary user metrics. Refine replay covers one attempt, not
iterative advice. Full GRPO training and live fine-tuning are not run; tests exercise their metric
capture/filter boundaries. Existing partial-parse arithmetic remains outside this change.
Tuple evidence is incomplete and shallow. JSON payload conversion is lossy for custom objects;
payload redaction, capture size limits, alias paths, arbitrary user-thread propagation, schema
versioning, and complete provider accounting remain unresolved. Nested capture scopes are isolated,
not a graph of linked traces. Production backend compatibility and ProAct learning improvement are
not claimed. These are reasons to retain draft status, not hidden implementation guarantees.
