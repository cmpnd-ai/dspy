from unittest.mock import Mock, call, patch

import dspy
from dspy.teleprompt.utils import create_n_fewshot_demo_sets, eval_candidate_program
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        pass


def test_eval_candidate_program_full_trainset():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(return_value=0)
    batch_size = 10

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    evaluate.assert_called_once()
    _, called_kwargs = evaluate.call_args
    assert len(called_kwargs["devset"]) == len(trainset)
    assert called_kwargs["callback_metadata"] == {"metric_key": "eval_full"}
    assert result == 0


def test_eval_candidate_program_minibatch():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(return_value=0)
    batch_size = 3

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    evaluate.assert_called_once()
    _, called_kwargs = evaluate.call_args
    assert len(called_kwargs["devset"]) == batch_size
    assert called_kwargs["callback_metadata"] == {"metric_key": "eval_minibatch"}
    assert result == 0

def test_eval_candidate_program_failure():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(side_effect=ValueError("Error"))
    batch_size = 3

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    assert result.score == 0.0


def test_create_n_fewshot_demo_sets_passes_metric_threshold_for_unshuffled():
    """Verify that metric_threshold is passed to BootstrapFewShot for the unshuffled (seed=-1) case.

    Regression test for https://github.com/stanfordnlp/dspy/issues/9308
    """
    student = DummyModule()
    student.predictor = dspy.Predict("input -> output")
    trainset = [dspy.Example(input="test", output="test").with_inputs("input")]

    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)

    with patch("dspy.teleprompt.utils.BootstrapFewShot") as mock_bootstrap:
        mock_instance = Mock()
        mock_instance.compile.return_value = student
        mock_bootstrap.return_value = mock_instance

        create_n_fewshot_demo_sets(
            student=student,
            num_candidate_sets=4,  # -3, -2, -1, 0 → hits seed=-1
            trainset=trainset,
            max_labeled_demos=1,
            max_bootstrapped_demos=1,
            metric=lambda ex, pred, trace=None: 1.0,
            teacher_settings={},
            metric_threshold=0.9,
        )

        # Find the call where seed == -1 (unshuffled few-shot)
        # BootstrapFewShot should be called at least twice: once for seed=-1, once for seed>=0
        calls = mock_bootstrap.call_args_list
        assert len(calls) >= 1, "BootstrapFewShot was never called"

        # Every BootstrapFewShot call should include metric_threshold
        for call in calls:
            _, kwargs = call
            assert "metric_threshold" in kwargs, (
                f"metric_threshold missing from BootstrapFewShot call: {kwargs}"
            )
            assert kwargs["metric_threshold"] == 0.9, (
                f"metric_threshold={kwargs['metric_threshold']}, expected 0.9"
            )


def test_create_demo_sets_uses_shared_rng_for_negative_seed_fallbacks():
    student = DummyModule()
    student.predictor = dspy.Predict("input -> output")
    trainset = list(range(5))
    teacher = Mock()
    rng = Mock()
    sizes = iter([2, 3, 1])

    def shuffle(items):
        items.reverse()

    rng.shuffle.side_effect = shuffle
    rng.randint.side_effect = lambda *_: next(sizes)

    with patch("dspy.teleprompt.utils.BootstrapFewShot") as bootstrap:
        bootstrap.return_value.compile.return_value = student
        result = create_n_fewshot_demo_sets(
            student=student,
            num_candidate_sets=4,
            trainset=trainset,
            max_labeled_demos=2,
            max_bootstrapped_demos=4,
            metric=Mock(),
            teacher_settings={"x": 1},
            teacher=teacher,
            include_non_bootstrapped=False,
            rng=rng,
        )

    assert rng.method_calls == [
        call.shuffle(list(reversed(trainset))),
        call.randint(1, 4),
        call.shuffle(list(reversed(trainset))),
        call.randint(1, 4),
        call.shuffle(list(reversed(trainset))),
        call.randint(1, 4),
    ]
    assert [call.kwargs["max_bootstrapped_demos"] for call in bootstrap.call_args_list] == [2, 3, 4, 1]
    assert [call.kwargs["trainset"] for call in bootstrap.return_value.compile.call_args_list] == [
        list(reversed(trainset)),
        list(reversed(trainset)),
        trainset,
        list(reversed(trainset)),
    ]
    assert all(call.kwargs["teacher"] is teacher for call in bootstrap.return_value.compile.call_args_list)
    assert len(result[0]) == 4
