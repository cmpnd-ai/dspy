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

## Implementation rounds

1. **Ordinary callbacks as the sole recorder.** Implemented and rejected: a tool inside
   `dspy.context(callbacks=[])` disappears, and retained raw output changes after caller mutation.
   Capture now runs at the existing decorated invocation boundary, using the same call ID as
   callbacks. It is independent of optional observer configuration and external tracing dependencies.
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
4. **One typed observation, three projections (rejected as insufficient).** A private `_Invocation`
   unified completion, but consumers still rebuilt attribution from tuples, signatures, and JSON.
   This fixed inconsistent final outcomes without making consumers simpler. Centralizing capture
   alone did not justify the abstraction.
5. **Retain the named call, not another projection (current revision).** The same public
   `Invocation` reaches typed callbacks and selected predictor traces. It carries component and
   occurrence identity directly. GEPA no longer joins JSON or guesses by signature; bootstrap
   failures no longer reconstruct predictor identity; SIMBA and BootstrapFewShot no longer keep
   reverse maps. Finetuning and GRPO route by component name rather than call ordinal or signature
   hash. Predict owns demonstration retention; the generic observation object does not.

### The smaller consumer contract

```text
program(...) → Invocation(instance, inputs, outputs, component_name, ref, parent, times, error)
                  ├─ callback.on_end(call)     all observed calls; no start-event lookup
                  ├─ settings.trace           selected successful predictor calls
                  └─ capture.trace            optional detached JSON representation
```

There is one call object, not a new optimizer trace registry beside the observation. Three-element
unpacking remains a migration view of the object. Metric purpose and result interpretation are
unchanged. Name lookup works without retaining JSON; an internal name-only capture scope supplies
identity to ordinary predictor collection. At collection boundaries, `_trace_program` declares
the optimizer-relative root, including when collection is nested in an outer execution capture.
The existing weak name map owns provenance, including copied BestOfN/Refine attempts. Unknown
dynamic predictors remain unnamed rather than being matched to a similar signature.

`component_name` is specifically a `named_predictors()` address. Module/tool/LM nesting uses
parent/call IDs; those nodes do not need optimizer addresses. An initial version walked
`named_sub_modules()`, including accumulated history, on each root call. Removing that traversal
eliminated the largest measured regression without adding a graph cache or invalidation protocol.

The rejected alternative was to keep raw tuples and make every consumer resolve them through an
execution graph. It requires capture to answer an ordinary attribution question, loses typed
inputs on JSON round-trip, and leaves the joins in callers. The retained-call design costs one
public record type and two typed callback hooks; it deletes those consumer-side mechanisms.

| Consumer | Deleted reconstruction | Current operation |
| --- | --- | --- |
| GEPA | JSON span join and separate signature fallback | Filter calls by component; select an occurrence |
| Parse-failure collection | Signature scan, span lookup, prediction-attached identity | Retain the failed call with a caller-owned failure output |
| BootstrapFewShot | Student/teacher reverse identity maps | Add selected call to the named component's demonstrations |
| SIMBA | Cross-candidate reverse identity map | Read names directly for demonstrations and advice trajectories |
| Finetuning | Trace-position routing and formatting wrapper | Format every call belonging to the requested component |
| GRPO | Signature-hash routing and reconstructed identity | Group named calls and scores using the existing policy |
| Typed callbacks | Consumer's start-event lookup table | Read the completed call in `on_end(call)` |

Not every consumer shrinks: BestOfN's selection policy stays intact. Refine still needs local
names for advice and a signature lookup before an adapter runs; removing those requires changing
the hint-injection contract, not merely improving completed evidence. CMPND still owns wire IDs
and export. This is a narrower claim than eliminating all tracing-related code.

### What the shared-observation experiment preserves and changes

- **Preserved:** legacy callback signatures and tuple access, `_trace=False`, trace disabling/limits,
  nested capture isolation, parentage, retry winner selection, and metric return interpretation.
  Capturing JSON is still optional; tuple collection does not retain a complete execution graph.
- **Changed intentionally:** a custom `Predict.forward` that replaces its parsed prediction now
  contributes the returned value, not an intermediate prediction. A call that raises after parsing
  contributes no successful demonstration tuple. Callbacks and snapshots already saw these final
  outcomes; all three views now agree.
- **Direct base Predict calls:** `forward`/`aforward` also use the observation boundary, retaining
  their existing tuple collection and now producing callbacks/spans. Normal `__call__`/`acall`
  reuse the outer invocation rather than recording twice. Calling `forward` directly remains
  discouraged; an undecorated custom override still bypasses the module's outer lifecycle.
- **New API:** `Invocation`, `BaseCallback.on_start(call)`, and `on_end(call)`. A callback can consume
  completed calls without maintaining its own start-event table. Default hooks dispatch to the old
  callbacks, including duck-typed SDK callbacks. No event bus, observer registry, exporter, or storage
  is added. CMPND still owns its callback-to-backend translation.

Round 5 verification uses the preceding draft as baseline. The 611-case differential replay has
zero differences; coercing returns to bool changes 182 cases. Eight additional mutations fail the
intended assertions: wrong purpose, missing execution evidence, stripped result metadata, wrong
failing sibling, wrong collection root, wrong selected occurrence, admitted failed prediction,
and signature-based GRPO routing. The last mutation initially survived a weak fixture: constructing
two signatures from identical strings creates distinct classes. Both predictors now share the
same Signature object, and the test detects the real collision.

Public compile tests inspect finetuning rows and GRPO groups at the provider boundary: repeated,
reordered, and skipped calls route correctly; same-signature components stay separate. The two
attribution regressions fail against the preceding draft; the multitask control passes. No provider
training is run. The full suite passes with **2,576 passed, 544 skipped, 2 xfailed**:

```sh
PATH="$PWD/.venv/bin:/usr/lib/git-core:$PATH" .venv/bin/pytest -q -n 4
.venv/bin/pytest -q tests/utils/test_execution.py tests/callback/test_callback.py \
  tests/teleprompt/test_bootstrap_finetune.py tests/teleprompt/test_grpo.py \
  tests/teleprompt/test_gepa_builtin_instruction_proposer.py
```

The focused command passes 110 tests. A real CMPND 0.40.0 SDK probe confirms three selected calls
are the same three objects seen by typed callbacks, linked to three SDK-exported spans and the
local JSON record. Export transport is intercepted: zero backend writes. Duck-typed SDK callbacks
require dispatch to declared hooks; their dynamic fallback must not silently swallow legacy events.

**Cost remains:** two paired runs, each with five 500-call DummyLM samples, measured these median
microseconds per call (before → after): disabled 313 → 355 and 348 → 341; selected evidence
294 → 456 and 354 → 568; JSON capture 932 → 1,007 and 995 → 1,021. Selected collection therefore
costs about 162–214 extra microseconds here (55–60%). These are noisy local measurements, not
provider-latency claims or a performance improvement. The rejected history-walking version cost
about 2,965 microseconds per selected call. Consumer simplification is real; it is not free.

**Migration policy:** preserving legacy metric signatures is no longer a design constraint.
The [migrating-dspy-metrics skill](.agents/skills/migrating-dspy-metrics/SKILL.md) guides application
agents to the context-only contract, including direct callers, wrappers, and behavioral checks.
The current implementation still contains the compatibility bridge; it is transitional code, not
a requirement of the target API. Removing it is separate from testing execution capture.

## What the implemented objects actually are

There are two separate contracts: **observe an execution** and **ask a metric a question**.
The metric can receive an execution record as evidence; it is not the owner of that record.
The implementation lives in [`execution.py`](dspy/utils/execution.py) and
[`metric_context.py`](dspy/evaluate/metric_context.py).

| Object | Actual representation and lifetime | What consumers do with it |
| --- | --- | --- |
| `Invocation` | Live dataclass: instance, kind, bound arguments, prepared predictor inputs, output/error, parent/call IDs, times, component name and reference. An active ContextVar lets Predict supply prepared inputs. | Typed callbacks receive it; `settings.trace` retains selected predictor calls. Use `.inputs`, `.outputs`, `.component_name`, and `.ref` directly. Treat it as read-only; it retains live Python objects. |
| `capture_execution(program=None)` | Context manager yielding a private `_Capture`: mutable span dictionary, lock, weak component-name map, and trace ID. Stored in DSPy's existing scoped settings while active. | Run work inside the scope; read `capture.trace`. Do not use the private recorder methods as an API. |
| `ExecutionTrace` | Frozen dataclass holding a JSON string. Every `capture.trace` read makes a new detached snapshot. `to_dict()`, `spans`, and `span(id)` return fresh values. | Inspect, serialize, or hand evidence to another consumer without retaining live program objects. It is not a live subscription. |
| A span | A plain dictionary inside that JSON record, not the live `Invocation`. One observed call to an instrumented boundary. | Follow `parent_span_id`, inspect inputs/outputs/error, or select by kind/component. |
| `ExecutionRef` | Frozen dataclass: `trace_id`, optional `span_id`, optional `output_path`. Contains no record or program object. | Address the whole record, one invocation, or a JSON Pointer into that invocation's output. Resolve against an explicitly supplied record. |
| `MetricContext` | Frozen request dataclass: purpose plus optional tuple of `Invocation`s, target index/component name, usage, and execution snapshot. Membership is frozen; calls and usage are not deeply immutable. | Decide whether to score, accept demonstrations, or produce feedback; use whatever evidence this caller supplies. `target_ref` links the selected call to the supplied record. |
| `Feedback` | Frozen dataclass containing only `target: ExecutionRef` and `content: str`. Created and validated by `execution.annotate(...)`. | Carry an annotation separately from execution and metric results. No implicit storage, dispatch, channel, author, or revision behavior. |

There is no public `ExecutionContext` or `TraceExecution` type in this revision. The active owner is
the capture handle; the readable value is `ExecutionTrace`; the metric request is `MetricContext`.
`Feedback` does not replace a metric's `Prediction(score=..., feedback=...)` return value.

An **execution scope** is whatever code ran inside one capture context. It can contain several
roots. A **span** is one invocation, not one module definition, signature, agent turn, or provider
request. Calling one predictor twice creates two spans. A predictor using a cached LM can still
create predictor/LM-boundary observations without a new provider request. Arbitrary Python helper
functions do not become spans; they need an existing instrumented DSPy boundary.

The record shape is currently this, with IDs and times shortened for illustration:

```json
{
  "trace_id": "execution-A",
  "closed": true,
  "spans": [{
    "span_id": "call-B",
    "parent_span_id": null,
    "kind": "module",
    "name": "Answer",
    "component_name": null,
    "start_time": 100.0,
    "end_time": 100.1,
    "inputs": {"args": [], "kwargs": {"question": "evidence"}},
    "outputs": {"answer": "EVIDENCE"},
    "error": null
  }]
}
```

Kinds come from callback dispatch: module, predict, lm, tool, adapter format/parse, interpreter
operations, evaluate, and compile. Names are class names except tools, which retain their declared
name. Component paths identify known predictor ownership, not occurrence identity. Inputs follow
the decorated method's bound arguments, so their shape varies by boundary. Times are wall-clock
seconds. Failure is an error type/message; an unfinished span has `end_time=None`.

`closed=True` means the **scope is sealed**, not that every worker succeeded or finished. A snapshot
taken while running never updates itself. Nested capture scopes are isolated; they are not merged
into the outer record. These are implemented semantics, not a proposed persistence/OTel schema.

## One invocation feeds callbacks, selected predictor evidence, and JSON records

For a normal `program(...)` call, [`with_callbacks`](dspy/utils/callback.py) already surrounds
`Module.__call__`. Its shared lifecycle creates one typed observation when a consumer needs it:

```text
with capture_execution(program) as capture
  program(question=...)
    with_callbacks wrapper                         create Invocation with ID and parent
      capture.start(invocation)                    optional input snapshot
      callback.on_start(invocation)                default dispatches legacy start hook
      set active invocation and call ID
      Module.__call__
        program.forward
          predictor(...) / tool(...) / child(...)  same boundary for each child
          Predict preprocessing                    attach prepared typed inputs
      restore parent's active invocation and ID    also when the call raises
      Predict._record_trace(invocation)            retain eligible predictor call
      capture.end(invocation)                      optional JSON output snapshot
      callback.on_end(invocation)                  complete call; default dispatches legacy end hook
  scope exit                                       seal recorder, clear weak name map
execution = capture.trace                          detached JSON-backed value
```

The async wrapper follows the same order around `await`. Existing DSPy worker propagation carries
scoped settings and active call identity into joined workers; the recorder locks its shared state.
`Parallel` is an executor, not an extra recorded module, so its children parent to the calling module.
Arbitrary user-created threads do not get a new propagation guarantee.

Callbacks are still synchronous lifecycle hooks, not a stream read back from `ExecutionTrace`.
They work without capture, and capture works without callbacks. Disabling optional observers does
not erase requested execution evidence. There is no generic event bus or replay of callbacks from
a saved record. Code inside the capture scope is observed regardless of whether it is called a
"candidate", "judge", or "tool"; boundaries, not metric purpose, determine inclusion.

## Candidate collection closes before the outer metric runs

[`bootstrap_trace_data`](dspy/teleprompt/bootstrap_trace.py) owns a candidate boundary per example.
It wraps the program instead of changing its `forward` method:

```text
bootstrap_trace_data(program, dataset, metric=metric)
  Evaluate(CapturedProgram(), wrapped_metric)
    for each example (possibly in DSPy workers)
      CapturedProgram.forward
        collect
          enter trace=[]; optional usage tracker; execution capture
          program(**example.inputs())              candidate root and descendants
          collect prediction, tuple trace, usage
          exit contexts                            execution now sealed
        return prediction, tuples, usage, execution snapshot
      wrapped_metric
        call_metric(metric, gold, pred, MetricContext("score", evidence...))
          metric(gold, pred, metric_context=...)    OUTSIDE candidate capture
    return rows containing prediction, trace, execution, optional usage/score
```

The evaluation wrapper itself starts outside the candidate capture. A separate outer callback SDK
may observe evaluation and judging as well; that does not put them in this candidate's record.
Returned row `trace` contains live predictor invocations; row `execution` is detached JSON.
Callers that retain the entire row still retain program/input/output objects. On a predictor failure,
the boundary attaches the failing call to the exception. Collection makes a view of that call with
its caller-owned `FailedPrediction` output, retaining its identity, inputs, and error. Neither JSON
joins nor signature matching are involved. A `History` remains a `History` for adapter reuse.

| Entry point | What this draft actually supplies |
| --- | --- |
| Explicit `with capture_execution(...)` | Records instrumented work inside the scope; does not construct a `MetricContext` or enable token accounting. |
| Plain `Evaluate` | `MetricContext("score")`; does not automatically capture/attach execution evidence. An ambient capture can observe the evaluation but is not injected into this context. |
| `BootstrapFewShot` | Existing tuple evidence and explicit purpose; no automatic execution record. |
| `bootstrap_trace_data` | Records when the metric declares `metric_context`, `capture_usage=True`, or `capture_execution=True`. |
| Ordinary GEPA | Its trace-retaining evaluation uses the helper above; ordinary non-retaining evaluation uses `Evaluate`. Contextual component feedback receives the retained record and exact target. |
| Contextual Flex evaluation | Collects execution and usage even when `capture_traces=False`; that flag controls returned trajectories, not evidence available to the metric. |
| `BestOfN` / `Refine` reward | Tuple evidence plus a snapshot of an already-active capture, if any; does not itself enable capture. |

## Consumer examples use the same observations for different decisions

The Python blocks below run in order. They use deterministic tools and `DummyLM`, so the DSPy
examples need no provider credentials. The ProAct block needs that optional external package.

### An application captures and inspects a completed execution

```python
import json
from dataclasses import asdict
import dspy

class Answer(dspy.Module):
    def forward(self, question):
        answer = dspy.Tool(lambda text: text.upper(), name="uppercase")(text=question)
        return dspy.Prediction(answer=answer)

program = Answer()
with dspy.capture_execution(program) as capture:
    prediction = program(question="evidence")
execution = capture.trace
root, = [span for span in execution.spans if span["parent_span_id"] is None]
tool, = [span for span in execution.spans if span["kind"] == "tool"]
assert tool["name"] == "uppercase" and tool["outputs"] == "EVIDENCE"
assert tool["parent_span_id"] == root["span_id"]
restored = dspy.ExecutionTrace.from_dict(json.loads(json.dumps(execution.to_dict())))
assert restored.span(root["span_id"])["outputs"] == {"answer": "EVIDENCE"}
```

The single-root assertion belongs to this example, not the general API. No `Signature` was needed
to observe this deterministic module. JSON inspection is useful even when there are zero predictor
tuples; it does not imply the record contains enough environment state to rerun the program.

### A metric receives a request, not a different kind of execution

```python
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

def metric(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    if metric_context.purpose == "accept":
        return score == 1.0
    return score

gold = dspy.Example(question="evidence", answer="EVIDENCE").with_inputs("question")
rows = bootstrap_trace_data(program, [gold], metric=metric, num_threads=1)
assert rows[0]["score"] == 1.0
assert rows[0]["trace"] == []                  # No predictors, but real execution.
assert len(rows[0]["execution"].spans) == 2
assert rows[0]["execution"].to_dict()["closed"]
```

This metric ignores evidence it does not need. A cost-sensitive Flex metric can read
`metric_context.usage`; it should not count spans to infer provider charges. A metric with its own
LM judge runs after this candidate scope closes. Return values are still owned by each caller:
numeric fitness, acceptance boolean, or GEPA's existing score/feedback object.

### GEPA feedback addresses an occurrence, not an equal signature

The actual feedback path is:

```text
DspyAdapter.make_reflective_dataset
  filter typed calls by component_name; keep their trace indices
  choose an occurrence (failure first; otherwise the existing sampling policy)
  MetricContext("feedback", component_name=..., target=tuple_index, execution=...)
  GEPA feedback_fn -> call_metric -> user's metric
  use returned feedback text; retain program fitness as the score
```

Here is a runnable demonstration of that request using a real retained execution. The explicit
`target=1` below stands in for GEPA's selection, not a new selection policy:

```python
from dspy.utils.dummies import DummyLM

class Twice(dspy.Module):
    def __init__(self):
        self.solve = dspy.Predict("question -> answer")

    def forward(self, question):
        self.solve(question="first")
        return self.solve(question=question)

def critique(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    if metric_context.purpose != "feedback":
        return score
    call = metric_context.trace[metric_context.target]
    return dspy.Prediction(score=score, feedback=f"Review {call.component_name}: {call.outputs}")

with dspy.context(lm=DummyLM([{"answer": "draft"}, {"answer": "EVIDENCE"}])):
    row, = bootstrap_trace_data(Twice(), [gold], metric=critique, num_threads=1)
context = dspy.MetricContext(
    "feedback", trace=row["trace"], target=1, component_name="solve", execution=row["execution"],
)
selected = context.execution.resolve(context.target_ref)
assert selected["inputs"]["kwargs"]["question"] == "evidence"
assert selected["outputs"] == {"answer": "EVIDENCE"}
assert "EVIDENCE" in critique(gold, row["prediction"], metric_context=context).feedback
```

That example records this topology (names at the left, span kinds at the right):

```text
Twice                                      module
  solve, first invocation                  predict
    ChatAdapter.format                     adapter_format
    DummyLM call                           lm
    ChatAdapter.parse                      adapter_parse
  solve, second invocation                 predict
    ChatAdapter.format                     adapter_format
    DummyLM call                           lm
    ChatAdapter.parse                      adapter_parse
```

The two `solve` calls share a predictor and signature, but not a span ID. Format, LM, and parse are
children of each predictor call, not separate turns. This is one concrete adapter path; other
adapters and fallback parsing can produce different children without changing the record model.
Predictor component paths use the optimizer's `named_predictors()` spelling, including quoted
dictionary keys such as `steps['answer']`; copied attempts inherit that same path.

`target` is local to the selected call list; `target_ref` reads that call's `.ref` when a matching
execution record is supplied. No identity is attached to the Prediction. The critique above needs
only typed evidence; the separate assertions demonstrate resolving the same occurrence in JSON.
If no corresponding record was supplied, `target_ref` is `None`.

### A future optimizer can collect evidence without defining a metric

```python
observations = bootstrap_trace_data(program, [gold], capture_execution=True, num_threads=1)
record = observations[0]["execution"]
operations = [
    (span["name"], span["inputs"], span["outputs"], span["error"])
    for span in record.spans if span["kind"] == "tool"
]
assert operations[0][0] == "uppercase"
assert operations[0][2] == "EVIDENCE"
```

This is an implemented collection API, **not an implemented research optimizer**. That optimizer
would own candidate IDs, experiment comparisons, storage/search, and deciding which operation to
retry. It can retain only `record.to_dict()` to avoid holding the predictor objects in the row.

### A delayed reviewer annotates output; ProAct owns learning state

```python
feedback = execution.annotate("Cite evidence", span_id=root["span_id"], output_path="/answer")
wire = json.loads(json.dumps(asdict(feedback)))
later = dspy.Feedback(dspy.ExecutionRef(**wire["target"]), wire["content"])
assert restored.resolve(later.target) == "EVIDENCE"
```

`annotate` returns a value; it does not append feedback to the record or send it anywhere. The
application must retain the matching record to resolve the reference later. A foreign execution,
missing span, or invalid output pointer is rejected. `execution.annotate("Overall review")` targets
the whole record; supplying only `span_id` targets the invocation rather than one output field.

This small, explicit adapter preserves the target in ProAct's string feedback channel:

```python
from proact import FeedbackChannel, Ledger

ledger = Ledger({"review": FeedbackChannel(description="Groundedness")})
case = ledger.add_case({"question": "evidence"}, dict(prediction), producer="candidate-v1")
event = ledger.add_feedback(case.id, "review", json.dumps(asdict(feedback)))
payload = json.loads(event.content)
assert restored.resolve(dspy.ExecutionRef(**payload["target"])) == "EVIDENCE"
```

This adapter is exercised against ProAct, not shipped inside DSPy. Case IDs, channels, record lookup,
reviewer identity, and training policy remain external. ProAct currently rejects a second feedback
entry for the same case/channel; this abstraction does not solve revision semantics for it.

### An observer consumes lifecycle hooks, not the completed record

```python
from dspy.utils.callback import BaseCallback

events = []
class Observer(BaseCallback):
    def on_end(self, call):
        if call.kind == "tool":
            events.append(call)

with dspy.context(callbacks=[Observer()]), dspy.capture_execution(program) as observed:
    program(question="callbacks")
call = events[0]
assert call.parent_id is not None
assert call.end_time >= call.start_time
assert observed.trace.span(call.call_id)["outputs"] == call.outputs == "CALLBACKS"
```

The observer sees the live output object; the record stores its converted snapshot. An observability
provider can export start/end events without retaining this DSPy record. Conversely an offline
consumer can use the record without running that provider. CMPND does not ingest `ExecutionTrace`
directly: its independently allocated IDs need mapping (see the adapter findings below).

## Retry stacks show why recorded work and selected demonstrations differ

With an ambient capture, the existing BestOfN control flow looks like this:

```text
BestOfN.forward
  copy candidate; transfer known component paths into capture's weak name map
  attempt 1: candidate(...) -> predictor occurrence A -> prediction A + tuples A
  reward(inputs, prediction A, context)              snapshot taken BEFORE this reward
  attempt 2: candidate(...) -> predictor occurrence B -> prediction B + tuples B
  reward(inputs, prediction B, context)              snapshot taken BEFORE this reward
  select winner; append only winner's tuples to outer demonstration trace
scope exit -> execution retains both attempts and any observed internal reward work
```

Both copied predictors may have the same `component_name`; their span IDs differ. The winner's
tuple keeps its exact occurrence reference. Refine likewise transfers copy provenance; its advice
generation can also be observed inside the scope.

**Important boundary distinction:** the outer evaluation metric is excluded by candidate scope
closure. A BestOfN reward or Refine advice call is *inside* the candidate and can be recorded. A
reward's pre-call snapshot excludes that reward's upcoming work, but may contain earlier rewards
and attempts; it is not an isolated per-attempt record. Usage collected around the whole candidate
likewise includes internal work. No generic "judge filtering" policy is implemented.

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
- `trace`: optional tuple of `Invocation` objects. None means unavailable;
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

GEPA reads component names directly from each call, then selects an occurrence. Both legacy and
contextual metrics use this attribution. BestOfN/Refine tell capture the provenance of their copies;
unnamed dynamic modules still have invocation IDs but no invented static component path.
Candidate usage continues using the existing tracker and joined-worker sharing policy. Recording
does not charge usage again. Totals do not establish accounting for detached or abandoned work.

Execution capture is automatic at the contextual bootstrap/GEPA/Flex candidate boundary and
available explicitly through `capture_execution()` or `bootstrap_trace_data(capture_execution=True)`.
Other metric callers need not have execution evidence. Existing demonstration selection remains;
its retained entries are now the same typed observations delivered to callbacks.

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

- Removed the metric-only `_metric_predictor_names` registry and aligned `trace_names` arrays,
  the prediction-attached `_execution_ref` bridge, GEPA's JSON-to-tuple join and signature fallback,
  bootstrap failure's predictor scan, SIMBA/BootstrapFewShot's reverse identity maps, and finetuning's
  trace-ordinal formatting helper. Component and occurrence identity travel with typed evidence.
- GRPO no longer hashes signatures for routing or reconstructs identity after selecting a group.
  Finetuning selects every call to the requested component, including repeated and skipped calls.
  These correct previously ambiguous attribution; reward/grouping policies are unchanged.
- Removed temporary `program.forward` replacement/restoration from bootstrap collection.
  The wrapper owns each candidate boundary without mutating the candidate program.
- Callbacks and recorded invocations use one ID generator and the existing worker propagation.
  One active-call ContextVar permits Predict's input handoff; it is not a separate parent hierarchy.
  Sync/async callbacks share their lifecycle and one kind classifier instead of two dispatch tables.
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
The earlier capture-round replay used the metric-only head as baseline and found zero differences.
Its nine mutations were caught: lost parent, reused invocation ID, wrong occurrence, wrong purpose, erased
evidence, coerced result, accepted foreign feedback, mutable output, and late writes after sealing.
Two test-fixture mistakes were corrected: Parallel is an executor, not an instrumented module;
the two-attempt test now supplies two LM responses and asserts both attempts succeed.

Real Azure + Deno + cmpnd callback checks compare independent provider requests with candidate
evidence: cached and uncached two-step candidates have two tuples but one versus two requests;
deterministic candidates have none. Judge requests are excluded, with trajectory retention both
on and off. The fresh 12-trial run also checks exact invocation references and sealed record
stability across judge execution. This was rerun after the retained-call changes: all 12 trials pass,
with 179 candidate tokens for cached calls, 358 for uncached calls, and zero for deterministic calls.
cmpnd delivery is captured locally, not sent to a deployed backend.

An earlier nine-repeat local deterministic-call probe measured medians of 9.50 ms before and 9.12 ms with
capture disabled per 1,000 calls (noise, not a speedup claim), versus 172.26 ms with capture and
JSON readout enabled. That record is about 282 KB. All 1,000 discarded module objects were collected
while retaining the record. This is historical capture-round evidence, not the current overhead:
the round 5 measurements above include automatic typed collection and its additional naming cost.

Limits: replay is not a proof over arbitrary user metrics. Refine replay covers one attempt, not
iterative advice. Full GRPO training and live fine-tuning are not run; tests exercise compilation
through the submitted training rows/groups. Existing partial-parse arithmetic remains outside this change.
Tuple evidence is incomplete and shallow. JSON payload conversion is lossy for custom objects;
payload redaction, capture size limits, alias paths, arbitrary user-thread propagation, schema
versioning, and complete provider accounting remain unresolved. Nested capture scopes are isolated,
not a graph of linked traces. Production backend compatibility and ProAct learning improvement are
not claimed. These are reasons to retain draft status, not hidden implementation guarantees.
