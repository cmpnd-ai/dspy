# DSPy execution tracing — additive 3.x prototype

## Status and compatibility contract

This local design document describes the implemented working-tree prototype, not a released or stable API. Imports and examples below match the inspected code; all new tracing APIs are experimental and provisional. This is not permission to remove an existing interface.

**DSPy 3.x changes must be additive and opt-in.** Keep callback signatures and registration, legacy predictor tuples (`settings.trace`, `_trace`), history, usage tracking, existing metric calling conventions, and default Evaluate results compatible. A prospective 4.0 removal or breaking redesign requires a separate, explicitly authorized change. New records do not automatically replace every old accumulator.

The delivered core separates:

- **Identity:** `InvocationRef(trace_id, span_id)` identifies an execution occurrence. Legacy callback `call_id` is a separate opaque compatibility ID, not the span ID.
- **Facts:** invocation boundaries, entry/exit snapshots, errors, signature descriptions, metadata, domain attachments, and LM evidence.
- **Observation:** synchronous `ExecutionObserver.on_start/on_end` snapshots, independent of retaining a local capture.
- **Retention:** explicit `capture_trace()` views; Evaluate candidate payload retention requires `capture_execution=True`.
- **Eligibility:** `ExecutionSelection` holds invocation refs without rewriting observed facts. Legacy tuple projections remain optimizer adapters.

History consolidation, optimizer semantic callbacks, and artifact lineage are separate scope decisions, not implied deliveries of this prototype.

## Capture, identity, and selection

This example needs no provider or credentials:

```python
import dspy
from dspy.utils.tracing import ExecutionSelection

class Echo(dspy.Module):
    def forward(self, question):
        dspy.record_event("stage", {"name": "answer"})
        return dspy.Prediction(answer=question)

program = Echo()
with dspy.capture_trace(program) as execution:
    prediction = program(question="hello")

assert execution.closed and execution.complete
root = execution.get(execution.roots[0])
assert root.outputs.answer == "hello"
summary = execution.to_dict()  # include_payloads=False by default
assert "inputs" not in summary["invocations"][0]
selection = ExecutionSelection(frozenset(execution.roots))
steps = execution.predictor_trace(selection=selection)  # []: Echo has no Predict
```

`dspy` exports `Invocation`, `InvocationRef`, `ExecutionObserver`, `ExecutionSelection`, `ExecutionTrace`, `capture_trace`, `record_event`, and `MetricContext`. The OTel bridge has its own optional module.

`capture_trace(program=None, *, origin=None, optimization=True, enabled=True)` does not install a callback or change `settings.trace`. Disabled capture yields `None`. Nested captures share invocation objects and run identity, rather than making copies. `origin` maps known copied predictors back to their original program paths (used for BestOfN); it is not signature-based identity inference.

An invocation includes `ref`, `parent`, compatibility `call_id/run_id/parent_id`, operation, live component reference, inputs, outputs/error, wall-clock nanosecond timestamps, monotonic duration endpoints, metadata, signature, `data`, and optional `lm` evidence. Static component names are labels relative to a captured program, not call identity. Shared predictors called twice have distinct refs. Runtime-created predictors can be observed without becoming named optimizer targets.

`roots` returns refs whose parent is outside the view; `get(ref)` returns a retained invocation or `None`. `closed` means the scope exited; `complete` additionally requires all recorded calls to have ended. New calls after scope close are rejected by that view, but already recorded work can finish later. Join/await workers before relying on completeness or exporting a final snapshot. A root may have an external parent.

`predictor_trace()` returns completion-ordered `(predictor, inputs, prediction)` tuples. `predictor_invocations()` exposes their invocations. Entry snapshots avoid later input-history mutation. `selected=False` includes discarded attempts. An explicit `selection=ExecutionSelection(frozenset(refs))` selects those refs independently of default eligibility. `discard()` and `optimization=False` update the capture views' exclusion sets, not invocation facts. These mutable view defaults are not yet a full selector/provenance model.

`usage()` aggregates attached billable usage across all attempts, including losers, independently of optimization eligibility. Cache hits remain observable without being counted again as billed usage. It is not a provider invoice or a complete retry accounting system.

## Observer registration and context

```python
from dspy.utils.tracing import ExecutionObserver, InvocationRef

class LocalObserver(ExecutionObserver):
    def __init__(self):
        self.finished = []

    def on_start(self, invocation):
        pass

    def on_end(self, invocation):
        self.finished.append(invocation)

observer = LocalObserver()
with dspy.context(
    execution_observers=[observer],
    execution_metadata={"experiment": "local-check"},
):
    program(question="hello")  # no capture_trace or history required
assert observer.finished[-1].metadata["experiment"] == "local-check"

# Optional externally supplied parent; use valid canonical IDs for OTel consumers.
parent = InvocationRef("1" * 32, "2" * 16)
with dspy.context(execution_tracing=True, execution_parent=parent):
    program(question="child")
```

Register scoped observers using `dspy.context(execution_observers=[...])`; use `dspy.configure(execution_observers=[...])` for application-wide settings when appropriate. A nested list replaces the configured list unless the caller explicitly appends existing observers. `OTelBridge.context()` appends itself and avoids duplicate self-registration. `execution_tracing=True` enables identity without requiring a retained capture or observer. Metadata and external parent settings are opt-in; an active DSPy invocation takes precedence over an external parent.

Observers receive detached snapshots, with component and predictor tuple references removed. Mutating a notification does not mutate retained evidence. Notifications are synchronous; do not perform blocking network IO. Observer exceptions are logged rather than changing the program result. Consumers own queue bounds, delivery, retry, redaction, and finalization. Snapshotting still costs CPU/memory even when no capture is retained.

Existing decorated boundaries create records before callback-start notification and finish them before callback-end notification. Existing callbacks continue receiving their old IDs and arguments and need not opt into retention. DSPy worker settings propagation and async context inheritance carry tracing context; arbitrary user-created threads require explicit propagation. Scope exit is not a task-join mechanism.

## LM evidence is not history

`LMCallEvidence` in `dspy.clients.call_result` is populated from `CallResult` on the active LM invocation, independently of history retention. Fields currently include requested/reported model, cache hit, `usage_reported`, raw and normalized usage, response IDs, finish reasons, requested/reported service tier, and estimated `cost`. Multiple responses retain evidence rather than silently choosing the first response's usage. Reported service tier currently comes from the first response; normalization is not a universal provider contract.

Cached raw usage may exist while `usage_reported` is false for billable accounting. Missing usage is not zero usage. Estimated DSPy cost must not be labeled provider-origin cost. Provider transport retries and individual completion candidates are not yet separate child invocations.

Global, LM, and module history lists remain legacy interfaces with their own retention policy. Where history entries are emitted under tracing, compatibility call/run IDs and history attachments support correlation; exporters should use invocation refs and LM evidence, not history positions, model names, or signature equality. `disable_history` is not a privacy switch for execution capture. Usage and decision-evidence collectors also retain compatibility semantics: nested usage roll-up and exclusive nested evidence scopes cannot be replaced blindly by all descendants of a trace.

## Evaluate: optional provenance and explicit retention

```python
def exact(gold, pred):  # unchanged legacy two-argument metric
    return float(gold.answer == pred.answer)

devset = [dspy.Example(question="hello", answer="hello").with_inputs("question")]
result = dspy.Evaluate(
    devset=devset, metric=exact, num_threads=1,
    display_progress=False, capture_execution=True,
)(program)
gold, pred, score = result.results[0]  # still three elements
provenance = result.executions[0]
attempt = provenance.accepted
assert attempt is not None and attempt.status == "ok"
assert attempt.execution.complete
program_roots = attempt.program
metric_roots = attempt.metric
```

Capture is activated by `capture_execution=True`, an active tracing setting/observer/capture, or a metric explicitly accepting `metric_context`. Only the explicit Evaluate retention flag keeps each candidate `attempt.execution` in the returned result; otherwise it is cleared after scoring. New metrics can inspect a temporary candidate execution without forcing result payload retention. An enclosing capture or consumer can independently retain data.

When activated, `result.executions` aligns with examples. Each item has `attempts` and `accepted`; each attempt has `attempt_id`, program and metric root refs, optional retained candidate execution, `status` (`running`, `ok`, `error`), and error type name. Refs name roots, not all descendants. Metric work is captured separately from candidate work so judge calls do not enter candidate training tuples or usage. Metric trace payloads are not retained in the attempt; observers or an enclosing capture can consume them.

Straggler resubmission carries successful attempt identity with the accepted result; it does not choose whichever attempt started last. Failed examples have no accepted attempt and retain failure provenance when capture is enabled. Both successful and raising metrics retain root references. Outstanding workers and incomplete scopes remain a lifecycle concern.

With a legacy metric and tracing disabled, Evaluate preserves the old `score/results` shape: no `executions` field is added. Public result triples do not gain a fourth element. There is no delivered `result.traces` API.

## Easy metric migration, without a 3.x flag day

The adapter recognizes an explicit positional-or-keyword or keyword-only `metric_context` parameter (including a module's `forward`). Arbitrary `**kwargs` is not opt-in. It does not catch a metric's `TypeError` and invoke the metric a second time. Old callers' positional and keyword conventions remain behind the adapter.

Start with a dual-contract metric so old direct callers still work:

```python
def compatible_metric(gold, pred, trace=None, *, metric_context=None):
    score = float(gold.answer == pred.answer)
    bootstrap = (
        metric_context.purpose == "bootstrap"
        if metric_context is not None else trace is not None
    )
    return score >= 0.8 if bootstrap else score
```

This tests `trace is not None`, not truthiness: an empty legacy tuple list still signals bootstrap mode. Built-in `SemanticF1` and `CompleteAndGrounded` preserve that fallback, including direct `trace=True` calls. Explicit purpose overrides legacy trace availability. The prior draft's claim that this fallback was removed is obsolete.

When all callers use the adapter, the metric itself can be context-only **without removing legacy support elsewhere in DSPy**:

```python
def context_metric(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    if metric_context.purpose == "bootstrap":
        return score >= 0.8
    if metric_context.purpose == "feedback":
        return dspy.Prediction(score=score, feedback="Compare the answer with the reference.")
    return score
```

`MetricContext` carries `purpose` (`score`, `bootstrap`, `feedback`), optional candidate `execution`, `component_name`, and legacy lookup `invocation_id`. `target` resolves that call ID to an `InvocationRef` if retained evidence is available. Purpose is independent of evidence availability: a fully captured scoring call is still scoring, not bootstrap. Metrics must tolerate unavailable optional evidence. Return types remain caller-specific: score, demonstration acceptance, or score/feedback; the adapter does not invent one universal metric result schema.

### GEPA's five-argument bridge

GEPA accepts either an explicit context metric callable with two inputs plus context, or its legacy `(gold, pred, trace, pred_name, pred_trace)` contract. A legacy metric can retain optional defaults:

```python
def legacy_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float(gold.answer == pred.answer)
    if pred_name is not None:
        return dspy.Prediction(score=score, feedback=f"Review component {pred_name}.")
    return score
```

The ordinary scoring bridge preserves its previous call convention (including `trace=None` where applicable); component feedback supplies the captured full tuple trace, component name, and a one-step predictor tuple list. It does not replace the legacy third argument with an `ExecutionTrace`. Context metrics instead receive purpose `feedback`, candidate execution when available, component name, and a target identifying the selected captured call. Repeated calls are distinguished by occurrence, not only signature. Tuple-only trajectories retain the legacy signature-based matching fallback. GEPA Flex uses the scoring adapter but its legacy `program_trace` bridge remains; reliable target identity for rewritten/dynamic Flex programs is not a settled contract.

## Delivered integration scope

| Consumer | Implemented additive behavior | Compatibility boundary |
| --- | --- | --- |
| Predict | One producer projects structured snapshots and legacy tuples | Old tuple reference semantics and list limits stay; structured capture is separate |
| BootstrapFewShot | Explicit bootstrap metric context; capture if enabled or metric opts in | Old metric gets its third tuple-list argument |
| `bootstrap_trace_data` | Structured execution alongside tuple training view when enabled; failed-parse capture can force it | Fine-tuning/GRPO keep tuple consumers; format-failure support is a special capture path |
| SIMBA | Scoring context and optional structured capture | Demonstration/reflection tuples stay |
| GEPA | Context scoring/feedback and occurrence-aware component matching when captured | Five-argument and tuple-only paths stay |
| BestOfN / Refine | Capture attempts when tracing is enabled, separate winner eligibility from observed work | Winning legacy tuples still propagate; reward/advice work excluded from optimization selection |
| Decision evidence / usage | Attach facts at their producer as well as existing accumulators | Existing collectors retain their distinct nesting semantics |
| Streaming | Status and token chunks carry optional invocation refs; buffered final chunks retain producer identity | Existing fields remain; untraced calls have no canonical ref and cached predictions do not become token streams |

Optimizers continue using legacy tuple paths by default. Structured capture is enabled by tracing or an explicitly new metric, rather than imposed unconditionally (apart from specialized failed-parse capture needs). Observed execution boundaries are not a new optimizer semantic callback API: proposal/acceptance/selection decisions and their reasons need a separate design.

## Optional exporters: what is and is not demonstrated

### OpenTelemetry SDK completed-span bridge

Install `dspy[otel]` (or `opentelemetry-sdk` separately). This uses public SDK processor types, but not a second SDK-generated identity tree:

```python
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from dspy.utils.otel import OTelBridge

exporter = InMemorySpanExporter()
processor = SimpleSpanProcessor(exporter)
bridge = OTelBridge(processor)  # include_payloads=False
with bridge.context():
    program(question="hello")
spans = exporter.get_finished_spans()
assert spans
processor.force_flush()
processor.shutdown()
```

`OTelBridge` delivers completed `ReadableSpan`s to `processor.on_end`; it does **not** deliver live SDK spans to processor `on_start`. Processors requiring start-time mutation are unsupported. The caller owns flushing/shutdown. For production network export, use an appropriate batching processor rather than blocking hooks.

Enter `bridge.context()` before the DSPy root. It joins valid ambient OTel context using canonical IDs, preserves same-trace sampling flags and trace state, makes native OTel spans inside DSPy calls inherit the DSPy span, and restores context afterward. Standalone spans are sampled. Unsampled spans are not delivered to the processor. Payloads are off by default; explicit export includes inputs, outputs, error messages, signature, metadata, domain data, and LM evidence as JSON.

`tests/utils/test_tracing_otel.py` contains local SDK/in-memory checks, guarded by optional SDK availability. This is narrower than proving arbitrary processor compatibility, deployed collector behavior, or MLflow/OpenInference integration.

### cmpnd local consumer example

`docs/examples/tracing_cmpnd.py` implements an injected **in-memory**, ingestion-shaped sink, not a cmpnd SDK migration or backend test. `CmpndObserver` consumes start/end snapshots without capture/history, maps canonical trace IDs to destination UUIDv4 IDs explicitly, preserves span IDs and correlation refs, and shapes LM evidence into usage observations. Its `address(ref)` joins Evaluate attempt roots to destination addresses. Its cost field is an estimate, not provider-origin cost.

`finalize()` is an explicit caller-owned boundary after work has stopped and all workers have joined; root completion alone cannot finalize a multi-root trace. The example retains state for one bounded run but does not implement general queue/memory limits. It exports sensitive payloads and needs a destination-owned privacy policy before production use. Tests in `tests/utils/test_tracing_cmpnd.py` exercise this local consumer contract only. No SDK migration, HTTP delivery, retry behavior, live backend acceptance, or external integration compatibility is asserted.

## Remaining design risks and limits

1. **Privacy and retention:** capture stores input/output snapshots and attachments; neither snapshotting nor `to_dict(include_payloads=False)` redacts retained memory. Payload export requires opt-in but may expose secrets, user text, provider data, and exception messages. There is no comprehensive redaction, sampling, eviction, size-budget, or access-control policy. Observers also see payloads by default. Optimizers must not silently train from budget-truncated evidence.
2. **Portability versus persistence:** `to_dict()` emits a provisional `schema_version: 1` projection; `include_payloads=True` uses portable conversions and unavailable markers. This is not a replayable history, checkpoint contract, or durable schema guarantee. Captures still contain locks, live components, and identity maps. Persist refs/projections deliberately or omit execution fields; do not assume pickling a live execution works.
3. **Snapshot completeness:** opaque programs/clients/callables, cyclic references, and uncopyable values get `UnavailableValue` markers. Debugging evidence can be useful without being a valid training example. Explicit completeness/eligibility validation is still needed. Signature schema generation can also be unavailable.
4. **Lifecycle and attempt provenance:** `closed` differs from `complete`; abandoned workers, cancellation, late results, and exporter finalization require an explicit policy. Evaluate records attempts and accepted identity, but does not turn all detached work into a frozen aggregate automatically.
5. **Selection and component ownership:** explicit ref selections separate facts from eligibility, but default exclusions and retained records are mutable. Who selected which attempt and why, aliases, rewritten programs, dynamic modules, and cross-run component descriptors remain open.
6. **Coverage:** direct `.forward()` and custom replacement call boundaries can bypass instrumentation. Agent owners can attach events but built-in agents do not all emit intermediate state. `StreamResponse.invocation` carries the originating predictor ref across producer/consumer boundaries, including buffered final chunks. `StatusMessage` carries the callback ID and optional canonical ref supplied at the callback boundary. Cached predictions remain predictions, not synthetic token streams; history formats are unchanged.
7. **Separate future scopes:** history needs its own compatibility/retention migration; optimizer semantic callbacks need event ownership and payload contracts; artifact lineage (datasets, program versions, checkpoints, produced artifacts) is deferred. None should be inferred from a shared execution ID alone.

## Release and migration phases

1. **Experimental now, additive in 3.x:** ship only opt-in APIs with provisional labeling and precise limitations. Keep all legacy defaults, built-in fallback behavior, callbacks, tuples, history, usage, and result shapes working.
2. **Opt-in migration support:** document dual-contract metrics first, context-only metrics for migrated callers second; provide exporter registration, identity joins, privacy/retention guidance, and examples. Validate consumer contracts separately from real SDK/backend migrations. Settle lifecycle and selection semantics with compatibility tests.
3. **Warnings only after documentation and contracts settle:** decide each deprecation separately, provide actionable migration and a support window, and avoid speculative warnings now. There is no authorization here to warn on or remove all legacy trace use.
4. **Prospective 4.0 only, separately authorized:** consider breaking removals after evidence, explicit review, and a versioned migration plan. Consolidating old accumulators or making context-only metrics mandatory is not part of this 3.x prototype.

## Evidence for this document

The test matrix covers callback compatibility, explicit metric purpose, legacy direct metric calls, selected versus rejected attempts, shared predictor identity, async/thread propagation, cancellation, incomplete capture scopes, candidate/judge provenance, history-disabled LM evidence, cache accounting, streaming correlation, and real in-memory OTel export. The cmpnd example is a local contract exercise, not production SDK/backend validation. PR checks provide the revision-specific test results; no deployed backend results are claimed here.
