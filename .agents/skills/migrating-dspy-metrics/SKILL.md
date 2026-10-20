---
name: migrating-dspy-metrics
description: Migrates DSPy metrics, wrappers, and direct callers to invocation-backed MetricContext requests while preserving caller-owned scoring and acceptance policies. Use when replacing trace, pred_name, pred_trace, or program_trace metric arguments.
---

# Migrating DSPy metrics

Migrate to one explicit calling convention, not an application compatibility adapter:

```python
def metric(example_or_inputs, prediction, *, metric_context): ...
```

## 1. Verify the target and inventory consumers

Inspect the pinned checkout or installed package before editing. Confirm MetricContext has
`purpose` (`"score"` or `"feedback"`), `invocation` (candidate root or None), and `target`
(exact Invocation or None). There is no acceptance purpose or context trace/usage/execution field.
Do not assume a released version supplies this API, upgrade dependencies without authorization,
or patch site-packages. Report a missing API as a blocker. Reconcile target-version differences.

Find metrics passed to Evaluate, optimizers, BestOfN/Refine rewards, and application loops.
Follow aliases, partials, decorators, factories, callable objects, Module.forward, wrappers,
and direct calls. Record each caller's result interpretation and exception policy. Check native
library metrics for context support; do not claim context-only migration while relying on their
legacy bridge. Leave unrelated functions named trace alone.

| Caller | Request and interpretation |
| --- | --- |
| Evaluate, Avatar, ReAnchor | Score; explicitly contextual metrics receive captured candidates. Aggregation/coercion/errors remain caller-specific. |
| BootstrapFewShot | Score; truthy metric_threshold triggers numerical comparison, otherwise result truthiness. Zero threshold means absent. |
| BootstrapFinetune | Score via bootstrap_trace_data; filters stored score truthiness. |
| GRPO / bootstrap_trace_data; SIMBA | Score with captured candidates; preserve ranking, metadata, and failure policies. SIMBA can supply prediction=None. |
| BestOfN / Refine | Score per captured attempt; first argument is an inputs dict, not gold labels. |
| GEPA | Score for program fitness/objectives; feedback for one exact occurrence. Component score does not replace program fitness. |
| Flex | Score with candidate evidence even without returned trajectories; preserve scoring-result feedback used by code reflection. |

## 2. Freeze behavior and resolve acceptance policy

Record old scores, selected attempts, accepted demonstrations/training rows, feedback targets,
metadata, and exceptions through actual callers. Use asymmetric scores, cutoff boundaries,
empty/unavailable evidence, and repeated predictor calls. If the old environment is unavailable,
use recorded expectations and limit the claim; never fabricate a baseline.

Both bootstrap and finetune request ordinary scores. Do not infer acceptance from purpose,
trace presence, usage, a target, or the optimizer's identity. Replace old trace-dependent mode
switches with numeric scoring and an explicit caller policy. A signature migration can otherwise
silently turn an old cutoff into admission of every nonzero partial score.

- Preserve ranges, objective_scores, feedback, and Prediction metadata unless intentionally changed.
- `Prediction(score=False)` is truthy. Returning False instead changes admission, not just syntax.
- Preserve the caller's truthy-threshold/zero-threshold semantics unless a change is authorized.
- If finetune requires a boolean cutoff, use a caller-specific metric wrapper. Do not invent an
  acceptance purpose, caller IDs, evidence heuristics, or a universal verdict object.
- Demonstrate intentional policy changes with old/new admission results below, at, and above
  the actual cutoff; report them separately from compatibility-preserving changes.

## 3. Migrate definitions, wrappers, and direct calls

Require keyword-only metric_context, including Module.forward. Transitional DSPy dispatch opts in
by explicit declaration, not **kwargs alone; the runtime compatibility bridge remains. Forward
the keyword through every owned wrapper and supply requests at direct calls. Never catch TypeError
and retry a different signature: a judge could run twice. Do not add dual signatures by default.

This runnable example keeps scoring separate from a deliberate finetune cutoff:

```python
import dspy

def quality(gold, pred, *, metric_context):
    return pred.quality  # Substitute the application's existing numeric grader.

def finetune_metric(gold, pred, *, metric_context):
    return quality(gold, pred, metric_context=metric_context) >= 0.8

context = dspy.MetricContext("score")  # Direct calls may have no captured candidate.
gold = dspy.Example(answer="expected")
for value, expected in [(0.79, False), (0.8, True), (0.81, True)]:
    pred = dspy.Prediction(quality=value)
    assert quality(gold, pred, metric_context=context) == value
    assert finetune_metric(gold, pred, metric_context=context) is expected
    assert bool(value)  # Legacy finetune truthiness would admit all three.
```

Use the application's actual threshold, not 0.8 blindly. Keep numeric quality for ranking,
evaluation, and thresholded bootstrap; a boolean wrapper is not a universal replacement. If the
grader returns Prediction, preserve it for scoring and explicitly select `.score` in a policy
wrapper only after settling the intended truthiness behavior.

### GEPA feedback targets

Replace pred_name with metric_context.target.component_name and pred_trace with the selected
Invocation itself, not an index into a second trace. GEPA selects the occurrence; never search by
signature, component name alone, or first/last call. Preserve result normalization and existing
whole-program feedback/objective metadata. Do not migrate the internal PredictorFeedbackFn ABI
as though it were the user metric API.

```python
def critique(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    if metric_context.purpose == "feedback":
        call = metric_context.target
        if call is None:
            raise ValueError("Component feedback needs an exact occurrence")
        return dspy.Prediction(score=score, feedback=f"Review {call.component_name}: {call.outputs}")
    return score
```

### Evidence and usage

Read selected predictor calls from `metric_context.invocation.trace`, aggregate usage from
`.usage`, immediate calls from `.children`, and traversal from `.descendants()`. Check the root
for None first. Traversal raises when a subtree was not retained; None evidence is not empty
evidence. Do not silently use `or {}`/`or []` and treat unavailable work as free execution.

Calls expose instance, prepared inputs, outputs, exception, component_name, call_id/parent_id,
times, and optional ref. `.arguments` contains bound arguments, potentially LM/config overrides.
`.own_usage` is directly observed usage; `.usage` aggregates noncached reported model totals;
`.cache_hit` records observed LM cache status. Treat calls as read-only live objects. Tuple access
is transitional. Failed parses may have caller-owned FailedPrediction output views. Repeated
calls/copies and equal signatures are not interchangeable; unowned predictors can be unnamed.

```python
def token_budget(gold, pred, *, metric_context):
    root = metric_context.invocation
    if root is None or root.usage is None:
        raise ValueError("Candidate usage is required")
    tokens = 0
    for counts in root.usage.values():
        total = counts.get("total_tokens")
        if total is None:
            total = counts["prompt_tokens"] + counts["completion_tokens"]
        tokens += total
    return float(gold.answer == pred.answer) - tokens / 100000
```

Keep reported zero totals; missing component counts should fail rather than default to zero.
Reported tokens are not an invoice or proof all providers reported. Predictor counts are not
LM request counts (cache hits still produce predictor calls). Changing a step penalty to a token
penalty changes units and requires an explicit policy decision.

Replace experimental program_trace or obsolete context evidence fields with the invocation
members actually needed. Typed evidence needs no JSON. Public capture_execution exposes roots
and detached `.trace` JSON; request bootstrap_trace_data JSON explicitly with capture_execution=True.
Its row invocation is the root; trace/usage keys are compatibility projections. Do not reconstruct
evidence from global LM history or add another tracker/recorder. Usage producers call record_usage
on active LM invocations; callback completion propagates totals and feeds UsageTracker scopes.
Automatic `track_usage=True` attaches each module's invocation total to its Prediction without
opening a separate tracker. This includes joined workers and may expose nested-module totals
previously unavailable. Explicit `track_usage()` scopes keep their existing semantics.

Evaluate/bootstrap/Avatar/ReAnchor capture for explicitly contextual metrics; SIMBA and
bootstrap_trace_data capture candidates, BestOfN/Refine capture each attempt before reward.
Evaluate supplies no candidate invocation for uninstrumented callables; their child predictor
calls are not a complete execution of the callable. Handle unavailable evidence explicitly.
Attempt evidence excludes its own reward. Whole-candidate usage includes internal rewards,
advice, and discarded attempts but excludes the outer judge. Do not promise generic judge
exclusion. Internal capture_invocation forwards to existing recorders; nested public JSON capture
scopes remain isolated. Invocation refs do not imply a retained JSON record. Context is not History,
persistence, or an observability SDK contract.

## 4. Demonstrate behavior through real consumers

Compare old and migrated decisions; list intentional changes separately. Exercise applicable cases:

- Score with populated, empty, and unavailable evidence; numeric ranges and raw result metadata.
- Below/at/above cutoff; zero/negative scores, threshold zero, Prediction truthiness, finetune rows.
- GEPA repeated/same-signature siblings, exact selected object, program versus component score.
- Retained/unretained trajectories, deterministic work, cache hits, outer judge exclusion,
  per-attempt reward exclusion versus whole-candidate internal reward/advice accounting.
- Module metrics, wrappers, direct calls, inputs-dict rewards, and existing exception policies.

Use temporary negative controls (erase evidence, select a sibling, coerce a score, drop a wrapper
keyword) and ensure relevant checks fail, then remove mutations. Keep only caller-facing regression
tests, not a committed scratch harness. Prefer deterministic fixtures; use real judges only where
needed. Historical DSPy verification is not proof of a new application's migration.

Search for stale arguments, evidence fields, mode switches, wrappers, direct calls, and examples.
Leave unrelated callbacks, History, export/storage, and optimizer algorithms alone. Report target
revision, entry points, intentional acceptance changes with evidence, commands/results, and
unresolved requirements. Do not claim application migration removed DSPy's compatibility bridge
or changed a published release.
