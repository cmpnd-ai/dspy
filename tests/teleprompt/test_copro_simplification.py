import importlib
from typing import ClassVar

import dspy
from dspy.teleprompt.copro_optimizer import COPRO
from dspy.utils.dummies import DummyLM


class Student(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


class FakeEvaluate:
    scores: ClassVar[list[float]] = []

    def __init__(self, **kwargs):
        pass

    def __call__(self, program, **kwargs):
        score = float(len(self.scores) + 1)
        self.scores.append(score)
        return dspy.Prediction(score=score)


def test_compile_tracks_stats_and_uses_optional_prompt_model(monkeypatch):
    copro_module = importlib.import_module("dspy.teleprompt.copro_optimizer")
    monkeypatch.setattr(copro_module, "Evaluate", FakeEvaluate)
    FakeEvaluate.scores = []

    task_model = DummyLM([])
    prompt_model = DummyLM(
        [
            {
                "proposed_instruction": "first optimized instruction",
                "proposed_prefix_for_output_field": "first optimized prefix",
            },
            {
                "proposed_instruction": "second optimized instruction",
                "proposed_prefix_for_output_field": "second optimized prefix",
            },
            {
                "proposed_instruction": "third optimized instruction",
                "proposed_prefix_for_output_field": "third optimized prefix",
            },
        ]
    )
    optimizer = COPRO(
        prompt_model=prompt_model,
        metric=lambda example, prediction: True,
        breadth=2,
        depth=2,
        track_stats=True,
    )

    with dspy.context(lm=task_model):
        compiled = optimizer.compile(Student(), trainset=[])

    assert len(prompt_model.history) == 2
    assert task_model.history == []
    assert compiled.total_calls == len(FakeEvaluate.scores) == 4
    assert [candidate["score"] for candidate in compiled.candidate_programs] == [4.0, 3.0, 2.0, 1.0]

    predictor_id = next(iter(compiled.results_best))
    assert compiled.results_best[predictor_id] == {
        "depth": [0, 1],
        "max": [2.0, 4.0],
        "average": [1.5, 2.5],
        "min": [1.0, 1.0],
        "std": [0.5, 1.118033988749895],
    }
    assert compiled.results_latest[predictor_id] == {
        "depth": [0, 1],
        "max": [2.0, 4.0],
        "average": [1.5, 3.5],
        "min": [1.0, 3.0],
        "std": [0.5, 0.5],
    }
