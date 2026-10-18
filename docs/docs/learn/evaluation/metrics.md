---
sidebar_position: 5
---

# Metrics

DSPy is a machine learning framework, so you must think about your **automatic metrics** for evaluation (to track your progress) and optimization (so DSPy can make your programs more effective).


## What is a metric and how do I define a metric for my task?

A metric is just a function that will take examples from your data and the output of your system and return a score that quantifies how good the output is. What makes outputs from your system good or bad? 

For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks.

However, for most applications, your system will output long-form outputs. There, your metric should probably be a smaller DSPy program that checks multiple properties of the output (quite possibly using AI feedback from LMs).

Getting this right on the first try is unlikely, but you should start with something simple and iterate. 


## Simple metrics

A DSPy metric is just a function in Python that takes `example` (e.g., from your training or dev set) and the output `pred` from your DSPy program, and outputs a `float` (or `int` or `bool`) score.

Your metric should accept an optional third argument called `trace`. Existing metrics and direct calls using this argument remain supported; no migration is required before a major release.

The local tracing prototype additionally supports a keyword-only `metric_context`. Its `purpose` is `"score"`, `"bootstrap"`, or `"feedback"`; its optional `execution` contains captured program calls. When context is supplied, purpose—not trace availability—determines behavior. A compatibility adapter preserves the previous calling convention for metrics that do not opt into context.

Here's a simple example of a metric that's comparing `example.answer` and `pred.answer`. This particular metric will return a `bool`.

```python
def validate_answer(example, pred, trace=None, *, metric_context=None):
    return example.answer.lower() == pred.answer.lower()
```

Some people find these utilities (built-in) convenient:

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

Your metrics could be more complex, e.g. check for multiple properties. The dual-version pattern below returns a numeric score for evaluation and a boolean acceptance decision when bootstrapping demonstrations. Without context, it preserves the legacy rule: `trace is None` returns a score; any non-`None` trace (even an empty list) requests a boolean. With explicit context, both purposes can have a captured execution.

```python
def validate_context_and_answer(example, pred, trace=None, *, metric_context=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    bootstrap = metric_context.purpose == "bootstrap" if metric_context is not None else trace is not None
    if bootstrap:
        return answer_match and context_match
    return (answer_match + context_match) / 2.0
```

Keep `trace=None` and this fallback while supporting existing DSPy versions and direct callers. `SemanticF1` and `CompleteAndGrounded` follow the same rule, returning a `Prediction` whose `score` is numeric or boolean. Explicit `"score"` or `"feedback"` returns a numeric score even with a trace; explicit `"bootstrap"` returns a threshold decision even without one.

For prospective DSPy 4.0, removal of the legacy trace calling convention is a **planned major-version change**, not current behavior. Once your minimum supported version and callers use context, migration is straightforward: remove the `trace` parameter and fallback, and use `metric_context.purpose` directly. Do not make that breaking change while supporting older callers.

Defining a good metric is an iterative process, so doing some initial evaluations and looking at your data and outputs is key.


## Evaluation

Once you have a metric, you can run evaluations in a simple Python loop.

```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
```

If you need some utilities, you can also use the built-in `Evaluate` utility. It can help with things like parallel evaluation (multiple threads) or showing you a sample of inputs/outputs and the metric scores.

```python
from dspy.evaluate import Evaluate

# Set up the evaluator, which can be re-used in your code.
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5)

# Launch evaluation.
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
```


## Intermediate: Using AI feedback for your metric

For most applications, your system will output long-form outputs, so your metric should check multiple dimensions of the output using AI feedback from LMs.

This simple signature could come in handy.

```python
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

For example, below is a simple metric that checks a generated tweet (1) answers a given question correctly and (2) whether it's also engaging. We also check that (3) `len(tweet) <= 280` characters.

```python
def metric(gold, pred, trace=None, *, metric_context=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    
    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    bootstrap = metric_context.purpose == "bootstrap" if metric_context is not None else trace is not None
    if bootstrap:
        return score >= 2
    return score / 2.0
```

When selecting demonstrations, `metric_context.purpose == "bootstrap"`, so we accept only when `score >= 2`. Candidate scoring during optimization still uses `"score"` and returns `score / 2.0`. Without context, the original `trace is not None` strict-judging behavior is retained.


## Advanced: Using a DSPy program as your metric

If your metric is itself a DSPy program, one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.



### Advanced: Accessing the `trace` and execution

In the legacy calling convention, evaluation does not pass a program trace to the metric; bootstrapping passes predictor input/output tuples so you can validate intermediate steps. In the prototype, evaluation and optimization both capture program execution, available through explicit context. The metric runs outside the program's captured scope, so an LM judge does not add its own steps to the execution it is judging.

`metric_context.execution.predictor_trace()` projects the selected predictor inputs and outputs into the familiar tuple shape. `execution.invocations` also includes module, tool, and LM boundaries, errors, and discarded attempts. It is not correct to count predictor tuples as provider requests: retries, cache hits, and multiple completions have different accounting semantics.


```python
def validate_hops(example, pred, trace=None, *, metric_context=None):
    if metric_context is not None:
        execution = metric_context.execution
        trace = execution.predictor_trace() if execution is not None else []
    trace = trace if trace is not None else []
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```
