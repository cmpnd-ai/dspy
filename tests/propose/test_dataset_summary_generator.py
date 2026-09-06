import pytest

import dspy
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.utils.dummies import DummyLM


def _build_trainset(n):
    return [dspy.Example(question=f"What is {i}+{i}?", answer=f"{2 * i}") for i in range(n)]


def _summarizer_prompt(trainset, answers):
    """Run ``create_dataset_summary`` and return the prompt sent to ObservationSummarizer (last LM call)."""
    lm = DummyLM(answers)
    create_dataset_summary(trainset=trainset, view_data_batch_size=10, prompt_model=lm)
    messages, _ = lm.get_convo(len(lm.history) - 1)
    return "\n".join(m.get("content", "") for m in messages)


# Sentinel forms the detector must recognize and skip (not append to observations).
# The DatasetDescriptorWithPriorObservations prompt asks the model to ``say 'COMPLETE'``
# (single-quoted), so quoted forms occur in practice; the detector must strip the quotes.
SENTINELS = ["COMPLETE", "'COMPLETE'", '"COMPLETE"', "COMPLETE."]

# Real observations whose first 8 chars match "COMPLETE" but must NOT be treated as the sentinel.
REAL_OBSERVATIONS = [
    "Completely analyzed the data",
    "Completed the review of all samples",
]


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_sentinel_is_skipped_not_appended(sentinel):
    """A COMPLETE sentinel (bare, quoted, or with trailing punctuation) must be skipped
    rather than appended to the observations fed to ObservationSummarizer."""
    # trainset=20 -> 1 loop iteration (range(10, 20, 10) == [10]).
    prompt = _summarizer_prompt(
        _build_trainset(20),
        [
            {"observations": "Initial observation."},  # initial DatasetDescriptor
            {"observations": sentinel},  # loop iteration: sentinel, must be skipped
            {"summary": "the summary"},  # ObservationSummarizer
        ],
    )
    # Case-sensitive "COMPLETE" avoids matching the chat adapter's "[[ ## completed ## ]]" marker.
    assert "COMPLETE" not in prompt, f"Sentinel {sentinel!r} was appended instead of skipped:\n{prompt}"


@pytest.mark.parametrize("observation", REAL_OBSERVATIONS)
def test_real_observation_starting_with_complete_is_preserved(observation):
    """A legitimate observation starting with "Complete" must NOT be mistaken for the
    sentinel; it must be appended and reach ObservationSummarizer."""
    prompt = _summarizer_prompt(
        _build_trainset(20),
        [
            {"observations": "Initial observation."},
            {"observations": observation},  # real observation, must be kept
            {"summary": "the summary"},
        ],
    )
    assert observation in prompt, f"Real observation {observation!r} was dropped as a sentinel:\n{prompt}"


def test_repeated_quoted_sentinel_triggers_early_stop():
    """Quoted ``'COMPLETE'`` on consecutive batches must increment ``skips`` and
    short-circuit via ``skips >= 5`` instead of running every batch and leaking the quoted
    tokens into the observations fed to ObservationSummarizer."""
    # trainset=90 -> 8 loop iterations; 5 quoted sentinels trigger the ``skips >= 5`` early-stop.
    lm = DummyLM(
        [{"observations": "Initial observation."}]
        + [{"observations": "'COMPLETE'"}] * 5
        + [{"summary": "final summary"}]
    )
    summary = create_dataset_summary(
        trainset=_build_trainset(90),
        view_data_batch_size=10,
        prompt_model=lm,
    )
    # 1 (initial) + 5 (loop sentinels) + 1 (summarizer) == 7 total LM calls.
    assert len(lm.history) == 7, f"Expected early-stop at 7 LM calls, got {len(lm.history)}"
    messages, _ = lm.get_convo(len(lm.history) - 1)
    summarizer_prompt = "\n".join(m.get("content", "") for m in messages)
    assert "COMPLETE" not in summarizer_prompt
    assert summary == "final summary"
