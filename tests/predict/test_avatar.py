import re

import dspy
from dspy.predict.avatar.avatar import Avatar
from dspy.predict.avatar.models import ActionOutput, Tool
from dspy.teleprompt.avatar_optimizer import AvatarOptimizer
from dspy.utils.dummies import DummyLM


class _FakeRetriever:
    """Minimal retriever exposing the `.run(input)` interface `Avatar._call_tool` relies on."""

    def __init__(self, response="retrieved context"):
        self.response = response
        self.calls = []

    def run(self, query, *args, **kwargs):
        self.calls.append(query)
        return f"{self.response}: {query}"


class _ScriptedFieldLM(DummyLM):
    """A stateless `DummyLM` that answers based on the output field the prompt requests.

    The ChatAdapter ends each user turn with a line like:
      "Respond with the corresponding output fields, starting with the field `[[ ## <name> ## ]]` ..."
    This class parses that field name and returns a matching scripted answer, so the avatar's
    actor loop and the optimizer's comparator / feedback predictors all receive correctly-shaped
    responses regardless of call ordering (the optimizer evaluates examples in a thread pool).
    """

    _field_re = re.compile(r"starting with the field `?\[\[ ## (\w+) ## \]\]")

    def __init__(self, responses):
        super().__init__(answers=[{}])
        self._responses = responses

    def forward(self, prompt=None, messages=None, **kwargs):
        content = messages[-1]["content"] if messages else prompt
        match = self._field_re.search(content)
        field = match.group(1) if match else None
        self.answers = iter([self._responses.get(field, {"answer": "Paris"})])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def _finish_after_one_search_responses(*, answer="Paris", tool="Search", query="what is the capital of France?"):
    return {
        "action_1": {"action_1": {"tool_name": tool, "tool_input_query": query}},
        "action_2": {"action_2": {"tool_name": "Finish", "tool_input_query": ""}},
        "answer": {"answer": answer},
    }


def test_avatar_construction_does_not_crash():
    # Regression: Avatar.__init__ called dspy.TypedPredictor, which was removed in PR #1949, raising
    # AttributeError on every instantiation. It must now build a dspy.Predict actor instead.
    retriever = _FakeRetriever()
    tools = [Tool(tool=retriever, name="Search", desc="Search the web")]

    avatar = Avatar(dspy.Signature("question -> answer"), tools, max_iters=3)

    assert isinstance(avatar.actor, dspy.Predict)
    assert avatar.max_iters == 3
    # The "Finish" tool is appended automatically to terminate the action/observation loop.
    assert avatar.tools[-1].name == "Finish"
    # The user's input field is folded into the actor signature.
    assert "question" in avatar.actor.signature.input_fields
    assert "action_1" in avatar.actor.signature.output_fields


def test_avatar_optimizer_construction_does_not_crash():
    # Regression: AvatarOptimizer.__init__ called dspy.TypedPredictor (removed in PR #1949), so the
    # exported teleprompter crashed on first instantiation. Both predictors must now be dspy.Predict.
    optimizer = AvatarOptimizer(metric=lambda *args, **kwargs: 1.0)

    assert isinstance(optimizer.comparator, dspy.Predict)
    assert isinstance(optimizer.feedback_instruction, dspy.Predict)
    assert optimizer.metric is not None


def test_avatar_forward_returns_prediction_with_output_fields():
    retriever = _FakeRetriever(response="Paris is the capital of France")
    tools = [Tool(tool=retriever, name="Search", desc="Search the web")]
    lm = _ScriptedFieldLM(_finish_after_one_search_responses(answer="Paris"))
    dspy.configure(lm=lm)

    avatar = Avatar(dspy.Signature("question -> answer"), tools, max_iters=3)
    result = avatar(question="What is the capital of France?")

    assert isinstance(result, dspy.Prediction)
    # The result exposes exactly the signature's output fields plus the synthesized `actions` log.
    assert set(result.keys()) == {"answer", "actions"}
    assert result.answer == "Paris"

    # The tool was actually dispatched with the query the actor produced.
    assert retriever.calls == ["what is the capital of France?"]

    # The action/observation trajectory is recorded as ActionOutput objects.
    assert len(result.actions) == 1
    assert isinstance(result.actions[0], ActionOutput)
    assert result.actions[0].tool_name == "Search"
    assert result.actions[0].tool_input_query == "what is the capital of France?"
    assert result.actions[0].tool_output == "Paris is the capital of France: what is the capital of France?"

    # forward() resets the mutated actor signature back to its pristine clone, so the avatar is
    # reusable and the next call starts fresh from action_1.
    assert "action_1" in avatar.actor.signature.output_fields
    assert "action_2" not in avatar.actor.signature.output_fields


def test_avatar_optimizer_compile_runs_end_to_end():
    # Exercises the formerly-broken `self.comparator = dspy.TypedPredictor(Comparator)` line and the
    # full compile() loop: the optimizer must evaluate the student Avatar, call the comparator for a
    # feedback signal, call the feedback-instruction predictor, and return a student whose actor
    # signature carries the generated instruction.
    retriever = _FakeRetriever()
    student = Avatar(
        dspy.Signature("question -> answer"),
        [Tool(tool=retriever, name="Search", desc="Search the web")],
        max_iters=3,
    )

    examples = [dspy.Example(question=f"pos_q{i}").with_inputs("question") for i in range(2)] + [
        dspy.Example(question=f"neg_q{i}").with_inputs("question") for i in range(2)
    ]

    def metric(example, prediction, *args, **kwargs):
        # Deterministic split that yields both positive (>= upper_bound) and negative (<= lower_bound)
        # examples, which AvatarOptimizer._get_pos_neg_results requires.
        return 1.0 if example.question.startswith("pos_") else 0.0

    lm = _ScriptedFieldLM(
        {
            **_finish_after_one_search_responses(answer="Paris"),
            "feedback": {"feedback": "Prefer the Search tool for factual questions."},
            "new_instruction": {"new_instruction": "Use the Search tool, then answer concisely."},
        }
    )
    dspy.configure(lm=lm, num_threads=1)

    optimizer = AvatarOptimizer(metric=metric, max_iters=1)
    compiled = optimizer.compile(student=student, trainset=examples)

    assert isinstance(compiled, Avatar)
    assert isinstance(compiled.actor, dspy.Predict)
    # The generated instruction was written into the compiled actor's signature.
    assert "Use the Search tool, then answer concisely." in compiled.actor.signature.instructions
    # The optimizer-finalized clone is kept in sync with the actor.
    assert compiled.actor_clone.signature.instructions == compiled.actor.signature.instructions
