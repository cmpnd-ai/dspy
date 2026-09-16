from types import SimpleNamespace

import pytest

import dspy
from dspy.teleprompt import mipro_optimizer_v2
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2


@pytest.mark.parametrize("minibatch", [False, True])
def test_optimize_prompt_parameters_preserves_trial_log_state(monkeypatch, minibatch):
    """Exercise real Optuna trial numbering while isolating program evaluation."""
    pytest.importorskip("optuna")
    scores = iter([0.1, 0.2, 0.3])
    batch_sizes = []

    def evaluate_candidate(batch_size, valset, program, evaluate, rng):
        batch_sizes.append(batch_size)
        return SimpleNamespace(score=next(scores))

    def save_program(program, log_dir, trial_num, note=None):
        suffix = f"-{note}" if note else ""
        return f"{log_dir}/{trial_num}{suffix}.json"

    monkeypatch.setattr(mipro_optimizer_v2, "eval_candidate_program", evaluate_candidate)
    monkeypatch.setattr(mipro_optimizer_v2, "save_candidate_program", save_program)

    program = dspy.Predict("x -> y")
    optimizer = MIPROv2(metric=lambda *_: True, prompt_model=object(), task_model=object(), log_dir="logs")
    result = optimizer._optimize_prompt_parameters(
        program=program,
        instruction_candidates={0: [program.signature.instructions]},
        demo_candidates=None,
        evaluate=None,
        valset=[1, 2, 3, 4],
        num_trials=1,
        minibatch=minibatch,
        minibatch_size=2,
        minibatch_full_eval_steps=1,
        seed=7,
    )

    assert result.score == (0.3 if minibatch else 0.2)
    assert result.trial_logs[1] == {
        "full_eval_program_path": "logs/-1.json",
        "full_eval_score": 0.1,
        "total_eval_calls_so_far": 4,
        "full_eval_program": result.trial_logs[1]["full_eval_program"],
    }
    assert result.trial_logs[1]["full_eval_program"] is not program

    if minibatch:
        assert batch_sizes == [4, 2, 4]
        assert set(result.trial_logs) == {1, 2, 3}
        assert result.trial_logs[2].keys() == {
            "0_predictor_instruction",
            "mb_program_path",
            "mb_score",
            "total_eval_calls_so_far",
            "mb_program",
        }
        assert result.trial_logs[2]["mb_program_path"] == "logs/2.json"
        assert result.trial_logs[2]["mb_score"] == 0.2
        assert result.trial_logs[2]["total_eval_calls_so_far"] == 6
        assert result.trial_logs[2]["mb_program"] is not result.mb_candidate_programs[0]["program"]
        assert result.trial_logs[3]["full_eval_program_path"] == "logs/3-full_eval.json"
        assert result.trial_logs[3]["full_eval_score"] == 0.3
        assert result.trial_logs[3]["total_eval_calls_so_far"] == 10
    else:
        assert batch_sizes == [4, 4]
        assert set(result.trial_logs) == {1, 2}
        assert result.trial_logs[2].keys() == {
            "0_predictor_instruction",
            "full_eval_program_path",
            "full_eval_score",
            "total_eval_calls_so_far",
            "full_eval_program",
        }
        assert result.trial_logs[2]["full_eval_program_path"] == "logs/2.json"
        assert result.trial_logs[2]["full_eval_score"] == 0.2
        assert result.trial_logs[2]["total_eval_calls_so_far"] == 8
        assert result.trial_logs[2]["full_eval_program"] is not result.candidate_programs[0]["program"]
