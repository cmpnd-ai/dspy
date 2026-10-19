---
name: migrating-dspy-metrics
description: Migrates DSPy metrics and their callers to the explicit MetricContext contract, separating scoring, demonstration acceptance, and feedback from trace availability. Use when upgrading metric APIs or replacing trace, pred_name, pred_trace, or program_trace metric arguments.
---

# Migrating DSPy metrics

Migrate to one calling convention, not a compatibility wrapper:

```python
def metric(example_or_inputs, prediction, *, metric_context): ...
```

This skill targets DSPy's MetricContext contract. Verify it exists in the user's target checkout
or installed package; do not assume a particular released version has it. Do not upgrade or
replace their DSPy dependency without authorization. If the target lacks the API, identify that
blocker before editing working metrics to require it.

## 1. Inventory consumers before editing definitions

Find metrics passed to Evaluate, optimizers, BestOfN/Refine rewards, and application-owned loops.
Follow aliases, partials, decorators, callable objects, Module.forward, factories, and direct calls.
Read what each consumer does with the result: sum, compare, test truthiness, or read feedback.
Do not rewrite unrelated callbacks or functions just because they accept an argument named trace.

For each metric, record its callers, purpose, available evidence, return shape, and exception policy.
When working outside the DSPy repository, inspect the pinned target's implementation or documentation.
The contract below is the migration target; reconcile changes in that target before proceeding.
For library-supplied metrics, check the target class rather than patching site-packages. Report any
missing native context support as remaining library migration work; do not claim the application
is context-only while it still depends on DSPy's legacy bridge.

| Caller | Purpose | Evidence / decision |
| --- | --- | --- |
| Evaluate; Avatar; ReAnchor | score | Evidence may be unavailable; fitness and aggregation remain caller-owned. |
| BootstrapFewShot with a truthy metric_threshold | score | Caller compares fitness against its threshold. |
| BootstrapFewShot without a truthy metric_threshold | accept | Caller admits demonstrations by truthiness. A zero threshold currently means absent, not a zero cutoff. |
| BootstrapFinetune | accept | Caller filters training data by truthiness. |
| GRPO / bootstrap_trace_data default; SIMBA | score | Captured predictor evidence does not imply acceptance. SIMBA can call the metric with prediction=None after failure. |
| BestOfN / Refine reward | score | First argument is an inputs dict, not gold labels; reward ranks attempts. Refine advice is separate. |
| GEPA candidate evaluation | score | Program fitness and optional objective scores. |
| GEPA component critique | feedback | Named component and selected occurrence; feedback score does not replace program fitness. |
| Flex candidate evaluation | score | Candidate trace and reported usage, including when returned trajectories are not retained. |

## 2. Freeze behavior and identify intentional changes

Before editing, run representative old metrics through their real callers. Record scores, selected
attempts, accepted demonstrations/training rows, feedback text/target, metadata, and exceptions.
Use asymmetric scores, threshold boundaries, empty evidence, and repeated predictor occurrences.
Record what cannot be reproduced; do not fabricate a baseline.

Distinguish a signature migration from a policy change:

- A metric using trace presence as a mode switch should use purpose instead. But old fine-tuning
  could pass trace=None while asking for acceptance. Do not assume its old behavior matched
  few-shot acceptance just because both now request accept.
- Prediction(score=False) is truthy. Replacing it with False changes actual admission behavior.
  Report that discrepancy and settle the intended policy, rather than silently fixing or preserving it.
- Preserve numeric ranges, thresholds, objective_scores, feedback, and other result metadata unless
  the requested migration explicitly changes them. Do not turn every result into float or bool.
- If different callers require different acceptance policies that one metric cannot express, make
  those policies explicit in caller-specific metrics or obtain a policy decision. Do not add caller
  IDs, infer the optimizer from evidence, or invent a universal result object to conceal the conflict.

Compatibility with old signatures is not required by default. Ask about supporting both versions
only when the project explicitly needs it; do not add fallback arguments or runtime signature sniffing.

## 3. Rewrite the metric and all owned call sites

Declare required keyword-only metric_context, including on a Module's forward. An explicit declaration
also works with transitional DSPy callers; **kwargs alone is not opt-in on those versions. Update
wrappers to forward the keyword, and update direct calls to supply an explicit request. Do not catch
TypeError and retry on another signature: an LM judge could run twice.

### Plain and acceptance-aware metrics

Pure scoring functions can ignore the context when their answer is valid for their consumers:

```python
def exact(gold, pred, *, metric_context):
    return gold.answer == pred.answer

# Formerly: return score >= 0.8 if trace is not None else score
def quality(gold, pred, *, metric_context):
    score = compare_answers(gold.answer, pred.answer)
    return score >= 0.8 if metric_context.purpose == "accept" else score

score = quality(gold, pred, metric_context=dspy.MetricContext("score"))
accepted = quality(gold, pred, metric_context=dspy.MetricContext("accept", trace=()))
```

Use the metric's existing acceptance threshold, not 0.8 blindly. Thresholded bootstrap requests
score; let the caller apply its threshold once. For metrics returning Prediction metadata, preserve
that object for scoring. Return an actual acceptance decision only after resolving the truthiness
policy above. Never treat presence of trace, usage, component_name, or target as the purpose.

### GEPA's five-argument metrics

Replace pred_name with metric_context.component_name. The selected old pred_trace step becomes
metric_context.trace[metric_context.target]; target is an index, including valid index zero.
Do not index by component name, search by signature, or choose the first/last call yourself.

```python
def metric(gold, pred, *, metric_context):
    score = grade(gold, pred)
    if metric_context.purpose == "feedback":
        trace, target = metric_context.trace, metric_context.target
        if trace is None or target is None:
            raise ValueError("Component feedback requires its selected trace occurrence")
        predictor, inputs, output = trace[target]
        text = critique(metric_context.component_name, inputs, output)
        return dspy.Prediction(score=score, feedback=text)
    return score
```

Preserve existing whole-program feedback during score requests if a caller consumes it. In
particular, Flex code reflection uses the scoring result's feedback; do not blindly move all
feedback generation into the component-feedback branch. Keep program fitness and objective scores
separate from a component-specific critique. Do not migrate DSPy's internal PredictorFeedbackFn
callback ABI as though it were the user metric's five-argument API.

### Flex and evidence-dependent metrics

Replace experimental program_trace with metric_context.trace, not with a new execution recorder.
For the former tracing prototype, replace execution.predictor_trace() with trace and execution.usage()
with usage only after checking the fields/units actually consumed. Invocation refs, span IDs, event
streams, and provider-request counts have no replacement in this metric context; report such needs.

```python
def token_budget(gold, pred, *, metric_context):
    usage = metric_context.usage
    if usage is None:
        raise ValueError("This metric needs candidate usage, as supplied by Flex evaluation")
    tokens = sum(model_usage.get("total_tokens", 0) for model_usage in usage.values())
    return float(gold.answer == pred.answer) - tokens / 100000
```

None means unavailable; empty trace/usage means nothing was recorded. Do not use `or {}` or `or []`
to turn unavailable evidence into free execution. Reported usage is not an invoice or proof every
provider reported tokens. Tuple length counts predictor steps, not LM requests: cache hits still
produce steps. Preserve a deliberate predictor-step penalty; changing it to a token penalty changes
units and requires an explicit policy. Never relabel tuple length as a request count.

Candidate evidence excludes judge work and is independent of trajectory retention. Do not reconstruct
it from global LM history or callbacks. Trace membership is snapshotted but payloads are shared;
treat them as read-only. Context is not History, a durable trace, or an observability SDK contract.

## 4. Prove the migration, not just that it imports

Run the same examples through the target callers. Compare old and migrated outputs/decisions,
listing intentional differences separately. When the old environment is unavailable, use recorded
expectations and explicitly limit the compatibility claim.

Exercise applicable cases, not an indiscriminate Cartesian product:

- Score with a nonempty trace; accept with an empty trace; score with no evidence.
- Scores below, at, and above the relevant cutoff; zero/negative scores; threshold-zero policy.
- Raw bool/float and Prediction with metadata; errors remain errors with the same caller policy.
- GEPA same-signature siblings, repeated calls, target zero, and whole-program versus component score.
- Flex retained/unretained trajectories, deterministic work, cache hits, and judge usage exclusion.
- Module metrics, wrappers, and direct calls; inputs dictionaries for inference-time rewards.

Introduce temporary negative controls: swap purpose, erase supplied evidence, select a sibling or
another occurrence, and coerce the result. Confirm the relevant checks fail for the intended reason,
then remove the mutations. A passing test that ignores purpose or never reaches the metric proves little.
Keep only regression tests of behavior the application relies on; do not commit the mutation harness
merely as a record. Prefer deterministic fixtures first; use real judges when the contract depends on them.

Finally search for remaining legacy metric arguments, mode tests, wrappers, direct calls, and stale
examples. Leave unrelated trace collection, History, callbacks, OTel, persistence, and optimizer
algorithms alone. Report the target revision, migrated entry points, intentional policy changes,
commands/results, and unresolved evidence or caller-policy requirements. Do not claim that migrating
an application removed DSPy's own compatibility bridge or changed a published release.
