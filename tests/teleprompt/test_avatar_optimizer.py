from unittest.mock import MagicMock

import pytest

import dspy
from dspy.teleprompt.avatar_optimizer import AvatarOptimizer


def _zero_metric(example, prediction):
    return 0


@pytest.fixture
def make_optimizer(monkeypatch):
    # `dspy.TypedPredictor` is not exported by this version of dspy, so stub it so
    # the real `AvatarOptimizer.__init__` (including argument validation) runs.
    monkeypatch.setattr(dspy, "TypedPredictor", lambda signature: MagicMock(), raising=False)

    def _make(**kwargs):
        kwargs.setdefault("metric", _zero_metric)
        return AvatarOptimizer(**kwargs)

    return _make


class _FakeSignature:
    def __init__(self, instructions="ORIGINAL-INSTRUCTION"):
        self.instructions = instructions

    def with_instructions(self, new_instruction):
        return _FakeSignature(new_instruction)


class _FakeActor:
    def __init__(self):
        self.signature = _FakeSignature()


class _FakeStudent:
    def __init__(self):
        self.actor = _FakeActor()
        self.tools = []


def _scores_to_results(scores):
    examples = []
    results = []
    for i, score in enumerate(scores):
        example = dspy.Example(input=f"q{i}").with_inputs("input")
        examples.append(example)
        results.append((example, None, float(score)))
    return examples, results


def _eval_stub(results, avg_score=None):
    if avg_score is None:
        avg_score = sum(r[2] for r in results) / len(results) if results else 0.0

    def _fn(devset, actor, return_outputs=False, num_threads=None):
        return (avg_score, results)

    return _fn


def test_init_rejects_invalid_optimize_for():
    # Guards against the silent no-op footgun: a typo or wrong casing previously
    # fell through the `else 999` sentinel and a never-matching apply guard,
    # returning the student unchanged.
    for bad in ["minn", "Min", "maximum", "MAX", "", None]:
        with pytest.raises(ValueError, match="optimize_for"):
            AvatarOptimizer(metric=_zero_metric, optimize_for=bad)


def test_get_pos_neg_results_max_routes_high_to_pos_low_to_neg(make_optimizer, monkeypatch):
    optimizer = make_optimizer(optimize_for="max", lower_bound=0, upper_bound=1)
    examples, results = _scores_to_results([1, 1, 0, 0])
    monkeypatch.setattr(optimizer, "thread_safe_evaluator", _eval_stub(results, avg_score=0.5))

    avg, pos, neg = optimizer._get_pos_neg_results(_FakeStudent(), examples)

    assert avg == 0.5
    assert [r.score for r in pos] == [1.0, 1.0]
    assert [r.score for r in neg] == [0.0, 0.0]


def test_get_pos_neg_results_min_routes_low_to_pos_high_to_neg(make_optimizer, monkeypatch):
    # For optimize_for="min" lower scores are better, so the good (low-scoring)
    # examples must be `pos_inputs` and the bad (high-scoring) ones `neg_inputs`
    # -- the inverse of "max". Before the fix both directions routed high->pos /
    # low->neg, inverting the feedback signal for "min".
    optimizer = make_optimizer(optimize_for="min", lower_bound=0, upper_bound=1)
    examples, results = _scores_to_results([1, 1, 0, 0])
    monkeypatch.setattr(optimizer, "thread_safe_evaluator", _eval_stub(results, avg_score=0.5))

    _, pos, neg = optimizer._get_pos_neg_results(_FakeStudent(), examples)

    assert [r.score for r in pos] == [0.0, 0.0]
    assert [r.score for r in neg] == [1.0, 1.0]


def test_get_pos_neg_results_min_all_high_raises_no_positive(make_optimizer, monkeypatch):
    # Before the fix, optimize_for="min" with all-worst (all high) scores produced
    # no *negatives* (inverted); after the fix it correctly produces no *positives*
    # because every example is bad for a "min" metric.
    optimizer = make_optimizer(optimize_for="min", lower_bound=0, upper_bound=1)
    examples, results = _scores_to_results([1, 1, 1])
    monkeypatch.setattr(optimizer, "thread_safe_evaluator", _eval_stub(results))

    with pytest.raises(ValueError, match="No positive examples found"):
        optimizer._get_pos_neg_results(_FakeStudent(), examples)


def test_compile_max_routes_correct_pos_neg_to_comparator(make_optimizer, monkeypatch):
    optimizer = make_optimizer(optimize_for="max", max_iters=3, max_positive_inputs=10, max_negative_inputs=10)
    examples, results = _scores_to_results([1, 1, 0, 0])
    monkeypatch.setattr(optimizer, "thread_safe_evaluator", _eval_stub(results, avg_score=0.5))

    comparator_calls = []

    def fake_comparator(**kwargs):
        comparator_calls.append(kwargs)
        return MagicMock(feedback="feedback")

    monkeypatch.setattr(optimizer, "comparator", fake_comparator)
    monkeypatch.setattr(
        optimizer, "feedback_instruction", lambda **kwargs: MagicMock(new_instruction="NEW-INSTRUCTION")
    )

    result = optimizer.compile(_FakeStudent(), trainset=examples)

    assert len(comparator_calls) == 3
    for call in comparator_calls:
        assert [r.score for r in call["pos_input_with_metrics"]] == [1.0, 1.0]
        assert [r.score for r in call["neg_input_with_metrics"]] == [0.0, 0.0]
    assert result.actor.signature.instructions == "NEW-INSTRUCTION"


def test_compile_min_routes_correct_pos_neg_to_comparator(make_optimizer, monkeypatch):
    # End-to-end: for optimize_for="min" the comparator (whose signature treats
    # pos=good / neg=bad) must receive low scores as positives and high scores as
    # negatives, the generated instruction must be applied, and it must propagate
    # to subsequent iterations' comparator input.
    optimizer = make_optimizer(optimize_for="min", max_iters=3, max_positive_inputs=10, max_negative_inputs=10)
    examples, results = _scores_to_results([1, 1, 0, 0])
    monkeypatch.setattr(optimizer, "thread_safe_evaluator", _eval_stub(results, avg_score=0.5))

    comparator_calls = []

    def fake_comparator(**kwargs):
        comparator_calls.append(kwargs)
        return MagicMock(feedback="feedback")

    monkeypatch.setattr(optimizer, "comparator", fake_comparator)
    monkeypatch.setattr(
        optimizer, "feedback_instruction", lambda **kwargs: MagicMock(new_instruction="NEW-INSTRUCTION")
    )

    result = optimizer.compile(_FakeStudent(), trainset=examples)

    assert len(comparator_calls) == 3
    for call in comparator_calls:
        assert [r.score for r in call["pos_input_with_metrics"]] == [0.0, 0.0]
        assert [r.score for r in call["neg_input_with_metrics"]] == [1.0, 1.0]
    assert result.actor.signature.instructions == "NEW-INSTRUCTION"
    assert comparator_calls[0]["instruction"] == "ORIGINAL-INSTRUCTION"
    assert comparator_calls[1]["instruction"] == "NEW-INSTRUCTION"
    assert comparator_calls[2]["instruction"] == "NEW-INSTRUCTION"
