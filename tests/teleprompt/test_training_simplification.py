from unittest.mock import Mock

import pytest

from dspy.teleprompt.grpo import GRPO


@pytest.mark.parametrize("step", [-1, 0, 1, 2])
@pytest.mark.parametrize("has_validation,report_train", [(True, True), (True, False), (False, True), (False, False)])
def test_validation_scheduling_datasets_and_score_units(monkeypatch, step, has_validation, report_train):
    optimizer = GRPO(
        exclude_demos=True,
        num_train_steps=3,
        num_steps_for_val=2,
        report_train_scores=report_train,
        use_train_as_val=report_train and not has_validation,
    )
    trainset = ["train1", "train2", "train3"]
    valset = ["validation"] if has_validation else None
    scores = [0.7, 0.2, 0.4, 0.9] if has_validation and report_train else [0.6]
    evaluation = Mock(score=60.0, results=[(None, None, score) for score in scores])
    evaluate = Mock(return_value=Mock(return_value=evaluation))
    monkeypatch.setattr("dspy.teleprompt.grpo.Evaluate", evaluate)
    logger = Mock()
    student = Mock()

    optimizer.report_validation_metrics(student, trainset, valset, logger, step_idx=step)

    if step == 0 or not (has_validation or report_train):
        evaluate.assert_not_called()
        return
    expected_dataset = (valset or []) + (trainset if report_train else [])
    assert evaluate.call_args.kwargs["devset"] == expected_dataset
    assert evaluate.call_args.kwargs["max_errors"] == (10 if has_validation else 30)
    evaluate.return_value.assert_called_once_with(student, metric=optimizer.metric)
    when = "before training loop" if step == -1 else f"after training step {step + 1}/3"
    if has_validation and report_train:
        logger.info.assert_any_call(f"Student program training set score {when}: 0.5")
        logger.info.assert_any_call(f"Student program validation set score {when}: 0.7")
    else:
        dataset = "validation" if has_validation else "training"
        logger.info.assert_any_call(f"Student program {dataset} set score {when}: 60.0")
