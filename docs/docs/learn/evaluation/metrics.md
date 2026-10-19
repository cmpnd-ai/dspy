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

Your metric should also accept an optional third argument called `trace`. You can ignore this for a moment, but it will enable some powerful tricks if you want to use your metric for optimization.

Here's a simple example of a metric that's comparing `example.answer` and `pred.answer`. This particular metric will return a `bool`.

```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

Some people find these utilities (built-in) convenient:

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

Your metrics could be more complex, e.g. check for multiple properties. The metric below will return a `float` if `trace is None` (i.e., if it's used for evaluation or optimization), and will return a `bool` otherwise (i.e., if it's used to bootstrap demonstrations).

```python
def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # if we're doing evaluation or optimization
        return (answer_match + context_match) / 2.0
    else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
        return answer_match and context_match
```

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
def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    
    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0
```

When compiling, `trace is not None`, and we want to be strict about judging things, so we will only return `True` if `score >= 2`. Otherwise, we return a score out of 1.0 (i.e., `score / 2.0`).


## Advanced: Using a DSPy program as your metric

If your metric is itself a DSPy program, one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.



### Advanced: Accessing the `trace`

When your metric is used during evaluation runs, DSPy will not try to track the steps of your program.

But during compiling (optimization), DSPy will trace your LM calls. The trace will contain inputs/outputs to each DSPy predictor and you can leverage that to validate intermediate steps for optimization.


```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```

## Experimental: explicit metric requests

The target API explicitly declares `metric_context` to separate the question being asked
from available trace evidence. Migrate metrics and their direct callers together:

```python
def metric(gold, pred, *, metric_context):
    score = float(gold.answer == pred.answer)
    return score >= 0.8 if metric_context.purpose == "accept" else score

score = metric(gold, pred, metric_context=dspy.MetricContext("score"))
```

`dspy.MetricContext` contains:

- `purpose`: `"score"` for fitness/ranking, `"accept"` for demonstration admission,
  or `"feedback"` for GEPA's component critique. Thresholded bootstrapping asks for
  a score; unthresholded bootstrapping and fine-tuning filtering ask for acceptance.
- `trace`: optional existing `(predictor, inputs, prediction)` tuples. `None` means
  unavailable; an empty tuple means no steps recorded. Membership is snapshotted,
  but predictor/input/output objects remain shared. Treat evidence as read-only.
- `component_name` and `target`: optional GEPA component name and occurrence index
  into `trace`. Use `metric_context.trace[metric_context.target]` for that step.
- `usage`: optional reported non-cached token totals by model, not invoice accounting.

Evidence availability never determines purpose. Evaluate does not capture a trace for
this API; existing bootstrap/ranking collectors supply tuples. Flex additionally supplies
usage even when trajectories are not retained. Predictor tuples are not provider requests.

Only a declared parameter opts in; `**kwargs` alone does not. Module metrics declare it
on `forward`. For a fully migrated metric, remove the legacy `trace` argument and require
`metric_context`. BestOfN and Refine pass an inputs dict rather than a gold Example.

**Results remain caller-specific:** return a numeric fitness for scoring, a boolean for
acceptance, or the existing `Prediction(score=..., feedback=...)` for GEPA feedback.
The bridge does not coerce results, apply thresholds, or retry exceptions. Legacy
truthiness is unchanged: notably, `Prediction(score=False)` is truthy. Built-in metrics
are not silently migrated by the current bridge. That bridge still accepts legacy metrics;
it is not a requirement of the target API.

### Agent-assisted migration

The repository includes `.agents/skills/migrating-dspy-metrics/SKILL.md`. Copy the
`migrating-dspy-metrics` directory into your project's `.agents/skills/` (or your agent's
supported skill directory), then ask: **"Use migrating-dspy-metrics to migrate our DSPy
metrics and direct callers to the context-only API."**

The skill checks the target DSPy API before editing, follows metrics through wrappers and
optimizers, and tests scoring, acceptance, and feedback separately. It does not add legacy
signature fallbacks by default. It also identifies behavioral decisions a signature rewrite
cannot settle, such as a false-valued Prediction that previously admitted demonstrations.
