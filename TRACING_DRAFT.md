# Metric requests, not an execution tracing framework

This draft replaces the execution-tracing prototype. Its scope is one opt-in calling convention:

```python
def metric(example_or_inputs, prediction, *, metric_context): ...
```

`MetricContext` says what answer the caller needs (`score`, `accept`, or `feedback`) and supplies
evidence it already has. Results remain caller-specific. No recorder, exporter, new trace identity,
History API, or universal verdict is necessary for this contract.

**Migration policy:** preserving legacy metric signatures is no longer a design constraint.
The [migrating-dspy-metrics skill](.agents/skills/migrating-dspy-metrics/SKILL.md) guides application
agents to the context-only contract, including direct callers, wrappers, and behavioral checks.
The current implementation still contains the compatibility bridge; it is transitional code, not
a requirement of the target API. This skill addition does not itself remove runtime support.

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

## The small contract and its limits

`MetricContext` has five fields:

- `purpose`: `score`, `accept`, or `feedback`.
- `trace`: optional tuple of existing `(predictor, inputs, prediction)` steps. None means unavailable;
  an empty tuple means no steps recorded. Membership is frozen, payload objects are not deep-copied.
- `component_name` and `target`: optional GEPA name and zero-based index into this trace. Repeated
  calls have distinct indices. These are not global execution IDs or a persistence schema.
- `usage`: optional existing UsageTracker totals by model. Reported non-cached tokens, not an invoice.

Evaluate, Avatar, and ReAnchor do not promise evidence. Existing bootstrap/ranking capture supplies
tuple evidence. Flex additionally supplies usage even with `capture_traces=False`; candidate capture
ends before the judge runs. A tuple count is **not an LM-request count**: cached predictions still
produce tuples; direct LM calls and failures need not produce them.

GEPA records predictor names alongside existing tuples only for migrated metrics (or Flex usage
capture). Migrated feedback matches the name and selects an occurrence. Legacy metrics keep their
signature-based matching. BestOfN/Refine copy names from the original predictor paths into a scoped
lookup; they do not infer names from signatures or change the legacy tuples. Candidate usage opts
into sharing its existing tracker with DSPy workers; legacy per-worker isolation stays unchanged.
Totals cover completed/joined work, not detached or abandoned workers. These are private capture
details, not new public settings or an identity API. Dynamic unnamed predictors, aliases, and arbitrary user-created threads
are not given stronger identity guarantees. Evidence cannot be required at every boundary without
expanding this design into a different project.

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

## What this deletes from the previous prototype

- `dspy/utils/tracing.py`: Invocation/InvocationRef, ExecutionTrace, ExecutionObserver,
  ExecutionSelection, capture_trace, record_event, serialization, snapshotting, lifecycle state.
- `dspy/utils/otel.py`, the OTel optional dependency, and `docs/examples/tracing_cmpnd.py`.
- Callback instrumentation changes and streaming invocation refs.
- LMCallEvidence, client/decision/usage observer attachments, and history correlation fields.
- Evaluate's capture_execution flag, retained executions, attempts, accepted/root refs.
- BestOfN/Refine execution selection and loser-retention bookkeeping.
- MetricContext.execution/invocation_id resolution; implicit built-in metric migration.
- Corresponding tracing/exporter/streaming-identity tests. Ordinary existing tests remain.

cmpnd continues using existing callbacks. The metric context is a synchronous decision request, not
the permanent observability record. There is no new cmpnd ingestion, History, or OTel contract.

## Frozen checks and remaining uncertainty

Before implementing this revision, expectations were frozen in `metric-contract-expectations.json`
(SHA-256 `af4da78c20de71515f612f3b1db5f57f14c5920bf61532a096ea75347557f089`). Evidence and runnable
harnesses are attached to the PR's Amp thread; the focused contract checks are in
`tests/evaluate/test_metric_context.py` and `tests/flex/test_flex_gepa.py`.

The differential harness executes identical bytes against the baseline and revised packages in
separate processes: 611 cases, zero differences. Coercing returns to bool changes 182 cases,
including BestOfN incorrectly selecting 0.2 over 0.8. Cases cover ABIs, raw results, exceptions,
thresholds, bootstrap, ranking, trace collection, and GEPA score/feedback interpretation.
Additional mutations substitute purpose, empty evidence, a same-signature sibling, or a different
occurrence; each fails the migrated public-boundary checks. Review regressions exercise copied
BestOfN/Refine targets and sequential versus parallel token capture, including a distinct judge.

Real Azure + Deno + cmpnd callback checks compare independent provider requests with candidate
evidence: cached and uncached two-step candidates have two tuples but one versus two requests;
deterministic candidates have none. Judge requests are excluded, with trajectory retention both
on and off. cmpnd delivery is captured locally, not sent to a deployed backend.

Limits: replay is not a proof over arbitrary user metrics. Refine replay covers one attempt, not
iterative advice. Full GRPO training and live fine-tuning are not run; tests exercise their metric
capture/filter boundaries. Existing partial-parse arithmetic and forward-patching fragility remain.
Tuple evidence is incomplete and shallow; dynamic Flex identity and provider invoice accounting
remain out of scope. No performance or production-backend compatibility claim is made.
