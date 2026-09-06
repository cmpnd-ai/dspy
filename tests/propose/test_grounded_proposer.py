import pytest

import dspy
from dspy.predict import Predict
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.utils.dummies import DummyLM


class RecordingDummyLM(DummyLM):
    """DummyLM that records the last user-message content sent to it (or any copy of it).

    ``GroundedProposer.propose_instruction_for_predictor`` sends the generation prompt to a
    *copy* of the prompt model (``self.prompt_model.copy(...)``). ``BaseLM.copy`` performs a
    shallow copy, so ``captured_prompts`` is shared by reference between the original and the
    rollout copy, allowing the test to inspect the exact prompt sent to the LM.
    """

    def __init__(self, answers, **kwargs):
        super().__init__(answers, **kwargs)
        self.captured_prompts = []

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        self.captured_prompts.append(messages[-1]["content"])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def _make_proposer(*, use_tip=True, set_tip_randomly=False):
    """Build a minimal ``GroundedProposer`` whose only variable input field is ``tip``.

    Disables dataset summary, program awareness, task demos, and instruction history so that
    the only conditionally-included input field is ``tip``, making prompt assertions precise.
    """
    program = Predict("question -> answer")
    lm = RecordingDummyLM([{"proposed_instruction": "instruction"}] * 20)
    proposer = GroundedProposer(
        prompt_model=lm,
        program=program,
        trainset=[],
        verbose=False,
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=False,
        use_instruct_history=False,
        use_tip=use_tip,
        set_tip_randomly=set_tip_randomly,
        set_history_randomly=False,
    )
    return proposer, lm, program


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
def test_propose_instructions_for_program(demo_candidates):
    # Set large number here so that lm always returns the same response
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")
    trainset = []

    proposer = GroundedProposer(prompt_model=prompt_model, program=program, trainset=trainset, verbose=False)
    result = proposer.propose_instructions_for_program(
        trainset=trainset, program=program, demo_candidates=demo_candidates, trial_logs={}, N=1
    )
    assert isinstance(result, dict)
    assert len(result) == len(program.predictors())
    for pred_instructions in result.values():
        assert pred_instructions == ["instruction"]


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
def test_propose_instruction_for_predictor(demo_candidates):
    class TrackingDummyLM(DummyLM):
        def copy(self, **kwargs):
            self.last_copy_kwargs = kwargs
            return super().copy(**kwargs)

    prompt_model = TrackingDummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")

    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        verbose=False,
        init_temperature=0.7,
    )
    result = proposer.propose_instruction_for_predictor(
        program=program,
        predictor=None,
        pred_i=0,
        demo_candidates=demo_candidates,
        demo_set_i=0,
        trial_logs={},
        tip=None,
    )
    assert result == "instruction"
    assert prompt_model.last_copy_kwargs["temperature"] == 0.7


def test_propose_instruction_for_predictor_omits_tip_field_when_tip_is_none():
    """``use_tip=True`` with ``tip=None`` must not render ``[[ ## tip ## ]]\nNone`` in the prompt.

    Regression test for the ``set_tip_randomly=False`` path (supported per PR #3919), where
    ``selected_tip`` stays ``None`` while ``use_tip`` keeps its constructor default ``True``.
    Before the fix the ``tip`` input field was still declared on the signature and ``None`` was
    forwarded into the prompt, where the adapter rendered it as the literal string ``"None"``.
    """
    proposer, lm, program = _make_proposer(use_tip=True, set_tip_randomly=False)
    result = proposer.propose_instruction_for_predictor(
        program=program,
        predictor=None,
        pred_i=0,
        demo_candidates=None,
        demo_set_i=0,
        trial_logs={},
        tip=None,
    )
    assert result == "instruction"
    assert "[[ ## tip ## ]]" not in lm.captured_prompts[-1]


def test_propose_instruction_for_predictor_includes_tip_field_when_tip_supplied():
    """A real tip value must still declare the ``tip`` field and render the tip text.

    Guards against an over-correction that would always omit the ``tip`` field.
    """
    proposer, lm, program = _make_proposer(use_tip=True, set_tip_randomly=False)
    tip = "Don't be afraid to be creative when creating the new instruction!"
    result = proposer.propose_instruction_for_predictor(
        program=program,
        predictor=None,
        pred_i=0,
        demo_candidates=None,
        demo_set_i=0,
        trial_logs={},
        tip=tip,
    )
    assert result == "instruction"
    prompt = lm.captured_prompts[-1]
    assert "[[ ## tip ## ]]" in prompt
    assert tip in prompt


def test_propose_instructions_for_program_omits_tip_when_set_tip_randomly_false():
    """End-to-end: ``set_tip_randomly=False`` keeps ``selected_tip=None``; the prompt must not
    contain the ``tip`` block (the user-facing configuration from PR #3919)."""
    proposer, lm, program = _make_proposer(use_tip=True, set_tip_randomly=False)
    result = proposer.propose_instructions_for_program(
        trainset=[],
        program=program,
        demo_candidates=None,
        trial_logs={},
        N=1,
    )
    assert result == {0: ["instruction"]}
    assert "[[ ## tip ## ]]" not in lm.captured_prompts[-1]
