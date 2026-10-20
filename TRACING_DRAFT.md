# Invocation-backed metrics: shared identity, separate consumer decisions

This draft describes the current checkout, not a released-version guarantee. Metrics, callbacks,
demonstration builders, accounting, and optional JSON share execution identity, not consumer policy.
This phase adds graph/accounting plumbing; it is **not a claim of net LOC reduction**.

## Implemented contract

[`MetricContext`](dspy/evaluate/metric_context.py) is a frozen request with exactly three fields:

| Field | Meaning |
| --- | --- |
| `purpose` | `"score"` (default) or `"feedback"`. There is no acceptance purpose. |
| `invocation` | Candidate root `Invocation`, or `None` if unavailable. |
| `target` | Exact selected `Invocation` for feedback, or `None`; never a trace index. |

Evidence lives on [`Invocation`](dspy/utils/execution.py), not on duplicate context fields:

| Member | Meaning |
| --- | --- |
| `children` | Immediate observed calls; `None` means the subtree was not retained. |
| `descendants()` | Traverses descendants; raises if a subtree was not retained. |
| `trace` | Selected predictor calls, not every observed operation; `None` means unavailable. |
| `usage` | Aggregate noncached, reported model totals for this call and its descendants; `None` means unavailable. |
| `own_usage` | Directly observed usage, without descendant aggregation. |
| `cache_hit` | Observed LM cache status, or `None` when not established. |
| `cost` | This LM invocation's estimated USD cost; zero for cache hits, `None` if unknown. Not a subtree total. |
| `get_total_cost()` | Estimated USD cost across the retained subtree, including this call; `None` if any LM cost is unknown. |
| `get_total_lm_calls()` | Number of observed LM invocations, including this call, cache hits and failures. |

For example, a metric can read `metric_context.invocation.get_total_cost()` and
`metric_context.invocation.get_total_lm_calls()` without accessing LM history.
Both helpers require a retained subtree and raise `ValueError` otherwise. They include
all observed attempts, not just selected predictor evidence. A subtree with no LM calls
returns zero for both. Call count is at the DSPy LM boundary: provider retries and
candidate fan-out inside one invocation do not increase it. Cost reuses existing LM
estimates, not a new pricing system or a provider invoice; unreported work is not priced.

Existing fields remain: `instance`, `kind`, bound `arguments`, prepared predictor `inputs`,
`outputs`, `exception`, `parent_id`, `call_id`, start/end times, `component_name`, and `ref`.
Prepared inputs exclude LM/config overrides; arguments need not. Component paths name owned
predictors, not occurrences. Repeated calls and copied attempts have distinct call identities.
Treat calls as read-only: the graph retains live Python objects, not immutable deep snapshots.
Tuple access `(instance, inputs, outputs)` remains a compatibility view, not the target API.

`capture_execution(program=None)` publicly exposes `.roots` and `.trace`: roots are live
invocations; `.trace` is a detached JSON-backed `ExecutionTrace`. Each read is a fresh snapshot;
its accessors return copies. A scope can have several roots. JSON conversion can be lossy and
is not enough to replay an environment. `closed=True` seals the scope, not proof workers succeeded.
Nested public capture scopes have isolated JSON records. Internal `capture_invocation` retains
a candidate graph and forwards observations to an existing recorder instead of hiding them.
An invocation reference is not a guarantee that a corresponding JSON record was requested.

## One lifecycle, including accounting

Typed callbacks receive the **same Invocation** retained by graph/trace consumers. They do not
read events back from JSON. Capture works without callbacks; callbacks work without JSON capture.
All actual LM usage producers (`base_lm`, structured execution, and typesafe clients) now call
`record_usage` on the active LM invocation. Cache hits do not add reported usage to totals.
Callback completion propagates totals to parents and feeds existing `UsageTracker` convenience
scopes. Recording does not charge a second copy. Totals are reported tokens, not an invoice or
a guarantee that every provider reported usage; predictor counts are not provider-request counts.
Automatic `track_usage=True` no longer creates module-local UsageTrackers: Predictions receive
their module invocation's aggregate directly, including joined workers. Nested module outputs
can now expose their own totals. Explicit `track_usage()` scopes retain their existing semantics.

```text
bootstrap_trace_data(program, dataset, metric)
  Evaluate(CapturedProgram(), wrapped_metric)
    per example: CapturedProgram.forward
      trace=[]; capture_invocation(program, recording=explicit_JSON_flag)
        program(**inputs)                         candidate root
          with_callbacks                          one Invocation per boundary
            predictor / tool / LM / child module  immediate children
              LM producer -> record_usage         own_usage; cache_hit
            completion -> parent totals           existing UsageTracker scopes too
          retain selected predictor trace
      close candidate capture                     graph + accounting ready
      return prediction, root, optional JSON
    wrapped_metric -> call_metric
      metric(gold, pred, metric_context=MetricContext(invocation=root))
                                                    outside candidate boundary
  rows: invocation=root; trace/usage compatibility projections; optional execution JSON
```

`bootstrap_trace_data` no longer opens `UsageTracker` or uses `_share_usage_tracker`.
It always captures candidates; `capture_usage=True` exposes the compatibility usage key, while
only explicit `capture_execution=True` requests a row's JSON execution record.
Failed-parse collection can supply a caller-owned `FailedPrediction` view of a selected call,
preserving identity, inputs, and exception. Existing parse/error policies remain caller-owned.

| Consumer | Capture and decision |
| --- | --- |
| Evaluate, BootstrapFewShot, Avatar, ReAnchor | Capture candidates for explicitly contextual metrics; request ordinary scores. |
| SIMBA, bootstrap_trace_data (including GRPO/finetune) | Capture candidates; request scores; retain existing failure/result handling. |
| BestOfN / Refine | Capture each attempt before its reward; first metric argument is the inputs dict. |
| GEPA / Flex | Candidate evidence, including when returned trajectories are disabled; feedback targets an exact occurrence. |

An arbitrary callable passed to Evaluate still gets `invocation=None`: an incidental predictor
inside an uninstrumented function is not evidence of the whole function's execution.

BestOfN/Refine attempt metrics exclude their own reward work. Whole-candidate usage includes
internal rewards, Refine advice, and discarded attempts. The outer candidate excludes its outer
judge because capture closes first. There is no generic judge-filtering policy. Selected
demonstration trace and full observed graph deliberately differ. Joined DSPy workers propagate
context; arbitrary detached/user threads do not gain a new accounting guarantee.

## Runnable examples (in order, no provider credentials)

```python
import dspy
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

class Answer(dspy.Module):
    def forward(self, question):
        answer = dspy.Tool(lambda text: text.upper(), name="uppercase")(text=question)
        return dspy.Prediction(answer=answer)

def exact(gold, pred, *, metric_context):
    assert metric_context.purpose == "score"
    return float(gold.answer == pred.answer)

program = Answer()
gold = dspy.Example(question="evidence", answer="EVIDENCE").with_inputs("question")
row, = bootstrap_trace_data(program, [gold], metric=exact, num_threads=1)
root = row["invocation"]
assert row["score"] == 1.0 and root.trace == ()
assert any(call.kind == "tool" for call in root.descendants())
assert "execution" not in row  # Typed evidence does not require JSON.

with dspy.capture_execution(program) as capture:
    program(question="evidence")
observed, = capture.roots
record = capture.trace
assert record.span(observed.call_id)["outputs"] == {"answer": "EVIDENCE"}
feedback = record.annotate("Cite evidence", span_id=observed.call_id, output_path="/answer")
assert record.resolve(feedback.target) == "EVIDENCE"
```

GEPA selects the occurrence; the metric must not search by signature or choose a sibling:

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
    if metric_context.purpose == "feedback":
        call = metric_context.target
        if call is None:
            raise ValueError("Feedback needs an exact occurrence")
        return dspy.Prediction(score=score, feedback=f"Review {call.component_name}: {call.outputs}")
    return score

with dspy.context(lm=DummyLM([{"answer": "draft"}, {"answer": "EVIDENCE"}])):
    row, = bootstrap_trace_data(Twice(), [gold], metric=critique, num_threads=1)
target = row["invocation"].trace[1]  # Demonstrates GEPA's occurrence selection, not a new policy.
context = dspy.MetricContext("feedback", invocation=row["invocation"], target=target)
assert "EVIDENCE" in critique(gold, row["prediction"], metric_context=context).feedback
```

GEPA normalization is unchanged: consume component feedback, not its differing score as fitness.
`Feedback(target, content)` annotations are separate from metric results; annotation does not send/store them.

## Migration and caller-owned acceptance

Require keyword-only `metric_context`, including `Module.forward`; migrate wrappers/direct calls.
The remaining `call_metric` bridge recognizes explicit declarations, not `**kwargs` alone, and
returns the exact result in one call. No TypeError retry, universal verdict, or acceptance mode.

BootstrapFewShot compares numerically only with a truthy `metric_threshold`; otherwise it uses
result truthiness. Threshold zero still means absent. BootstrapFinetune filters stored score
truthiness. `Prediction(score=False)` is truthy: replacing it with `False` changes admission.
If finetuning needs a boolean cutoff, supply a caller-specific wrapper, not another purpose:

```python
def finetune_metric(gold, pred, *, metric_context):
    return exact(gold, pred, metric_context=metric_context) >= 0.8

assert finetune_metric(gold, row["prediction"], metric_context=dspy.MetricContext())
```

Keep numeric scoring for ranking/evaluation and thresholded bootstrap; demonstrate intentional
admission changes below/at/above cutoff. See the [migration skill](.agents/skills/migrating-dspy-metrics/SKILL.md).

## Simplification, historical evidence, and limits

Consumers read root graph/usage and exact targets instead of maintaining aligned name arrays,
JSON-to-tuple joins, signature matching, prediction-attached references, or separate metric
evidence bundles. Bootstrap collection uses a wrapper rather than replacing `program.forward`.
The cost is retained graph state, lifecycle propagation, locking, accounting, and optional JSON
conversion. Existing History, explicit UsageTracker scopes, caller policies, and compatibility bridge remain.

The rejected alternative keeps trace, usage, JSON, target index, and component name on the metric
context. It avoids retaining a tree but makes each caller assemble and align separate evidence.
The chosen context points to the candidate and selected occurrence; consumers needing only totals
can use automatic accounting without retaining a tree. Contextual metrics retain the candidate
tree. Return values remain caller-specific rather than adding a universal verdict.

This revision's executable review bundle is `.amp/in/artifacts/invocation-contract/` in the Amp
thread: 611 legacy replay cases with zero drift, six detected purpose/evidence/target/result/usage/cache
mutations, and 12 passing Azure/Deno/CMPND trials. Candidate tokens were 179 cached, 358 uncached,
and zero deterministic; every outer judge was excluded. Exports were intercepted locally.
`pytest -q -n 4`: 2,600 passed, 544 skipped, 2 xfailed. Both documentation files' Python blocks ran.

An identical public-API probe exposed an intentional accounting correction: with two workers,
the previous Prediction reported 16 of 26 tokens; it now reports all 26 (5 + 3 + 7 + 11), matching
the invocation. Single-worker totals stayed 26. This is separate from the unchanged metric replay.
Five-sample contextual collection medians were 1.312 → 0.974 ms/example without mandatory JSON.
Final raw-call samples measured 481–488 µs selected, 1.076–1.273 ms JSON, and 350–360 µs disabled.
Paired deltas were noisy; these are not general speedup claims. Retaining 1,000 live capture roots
raised the allocation probe's peak from 3.47 to 5.79 MB; keeping only detached JSON released every
program in both revisions. Typed evidence has a real lifetime/memory cost.

CMPND 0.40.0 trials found independent wire trace/span UUIDs: DSPy IDs cannot be exported as SDK
IDs without mapping. A companion callback after supported module/tool starts used public
`get_current_span()`; intercepted exports made zero backend writes. Skipped/collapsed operations,
callback failures, and duck-typed hook dispatch need explicit handling. LM observations separately
export `sdk_call_id`. Root-output assessments require verifying that the invocation maps to the
SDK root; arbitrary span-output pointers exceed CMPND's current assessment API.
ProAct's historical adapter serialized target/content into a string feedback channel; record
lookup, cases/channels, reviewer identity, learning, and revision policy remain external. Its trial
rejected a second review on the same case/channel. Neither trial establishes a production adapter.

Unresolved: lossy payloads, redaction/size limits, aliases, schema versioning, detached work, complete
accounting, backend/replay/learning quality. No full GRPO/live finetuning checks; retain draft status.
