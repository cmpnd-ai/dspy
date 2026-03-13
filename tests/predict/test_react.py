import asyncio
import time

import pytest
from pydantic import BaseModel

import dspy
from dspy.adapters.types.tool import ToolCalls
from dspy.utils.dummies import DummyLM


def _tool_calls(*calls: tuple[str, dict]) -> ToolCalls:
    return ToolCalls.from_dict_list([{"name": name, "args": args} for name, args in calls])


def test_react_submit_with_pydantic_args_and_history_shape():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    class InvitationSignature(dspy.Signature):
        participant_name: str = dspy.InputField()
        event_info: CalendarEvent = dspy.InputField()
        invitation_letter: str = dspy.OutputField()

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])
    steps = iter(
        [
            dspy.Prediction(
                next_message="I'll write the invitation.",
                tool_calls=_tool_calls(
                    (
                        "write_invitation_letter",
                        {
                            "participant_name": "Alice",
                            "event_info": {
                                "name": "Science Fair",
                                "date": "Friday",
                                "participants": {"Alice": "female", "Bob": "male"},
                            },
                        },
                    )
                ),
            ),
            dspy.Prediction(
                next_message=None,
                tool_calls=_tool_calls(
                    (
                        "submit",
                        {"invitation_letter": "It's my honor to invite Alice to event Science Fair on Friday"},
                    )
                ),
            ),
        ]
    )
    react.react = lambda **kwargs: next(steps)

    result = react(
        participant_name="Alice",
        event_info=CalendarEvent(
            name="Science Fair",
            date="Friday",
            participants={"Alice": "female", "Bob": "male"},
        ),
    )

    assert result.invitation_letter == "It's my honor to invite Alice to event Science Fair on Friday"
    assert result.history.messages[0]["role"] == "user"
    assert "Alice" in result.history.messages[0]["content"]
    assert "Respond with the corresponding output fields" not in result.history.messages[0]["content"]
    assert result.history.messages[1]["tool_calls"][0]["function"]["name"] == "write_invitation_letter"
    assert result.history.messages[2]["role"] == "tool"
    assert result.history.messages[2]["name"] == "write_invitation_letter"
    assert result.history.messages[3]["tool_calls"][0]["function"]["name"] == "submit"
    assert result.history.messages[4]["name"] == "submit"
    assert "Submitted final outputs successfully" in result.history.messages[4]["content"]
    assert "[[ ## invitation_letter ## ]]" in result.history.messages[5]["content"]

    assert result.trajectory["tool_name_0"] == "write_invitation_letter"
    assert result.trajectory["observation_0"] == "It's my honor to invite Alice to event Science Fair on Friday"
    assert result.trajectory["tool_name_1"] == "submit"


def test_react_fallback_extract_appends_synthetic_submit():
    react = dspy.ReAct("question -> answer", tools=[])
    react.react = lambda **kwargs: dspy.Prediction(next_message="I think I can answer.", tool_calls=None)
    react.extract = lambda **kwargs: dspy.Prediction(reasoning="fallback", answer="Paris")

    result = react(question="What is the capital of France?")

    assert result.answer == "Paris"
    assert result.history.messages[1] == {"role": "assistant", "content": "I think I can answer."}
    assert result.history.messages[2]["tool_calls"][0]["function"]["name"] == "submit"
    assert result.history.messages[3]["name"] == "submit"
    assert "Submitted final outputs successfully" in result.history.messages[3]["content"]
    assert "[[ ## answer ## ]]" in result.history.messages[4]["content"]
    assert result.trajectory["tool_name_0"] == "submit"
    assert result.trajectory["tool_args_0"] == {"answer": "Paris"}


def test_react_invalid_submit_yields_tool_error_then_can_continue():
    react = dspy.ReAct("question -> answer", tools=[])
    steps = iter(
        [
            dspy.Prediction(next_message=None, tool_calls=_tool_calls(("submit", {}))),
            dspy.Prediction(next_message=None, tool_calls=_tool_calls(("submit", {"answer": "Paris"}))),
        ]
    )
    react.react = lambda **kwargs: next(steps)

    result = react(question="What is the capital of France?")

    submit_errors = [message for message in result.history.messages if message.get("role") == "tool"]
    assert len(submit_errors) == 2
    assert "missing output fields" in submit_errors[0]["content"]
    assert "Submitted final outputs successfully" in submit_errors[1]["content"]
    assert result.answer == "Paris"


def test_react_history_preserves_prior_user_turn_for_pronouns():
    captured_histories = []
    react = dspy.ReAct("question -> answer", tools=[])
    react.react = lambda **kwargs: captured_histories.append(kwargs["history"]) or dspy.Prediction(
        next_message=None,
        tool_calls=_tool_calls(("submit", {"answer": "20C"})),
    )

    prior_history = dspy.History.raw(
        messages=[
            {"role": "user", "content": "What is the weather in Paris in F?"},
            {
                "role": "assistant",
                "content": "Checking the weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "68"},
            {"role": "assistant", "content": "[[ ## answer ## ]]\nIt is 68F in Paris.\n\n[[ ## completed ## ]]\n"},
        ]
    )

    result = react(question="Convert that number to C.", history=prior_history)

    seen_history = captured_histories[0]
    assert seen_history.messages[0]["content"] == "What is the weather in Paris in F?"
    assert seen_history.messages[3]["content"].startswith("[[ ## answer ## ]]")
    assert seen_history.messages[-1]["role"] == "user"
    assert "Convert that number to C." in seen_history.messages[-1]["content"]
    assert result.answer == "20C"


def test_react_parallel_tool_calls_preserve_original_order():
    def slow_tool():
        time.sleep(0.05)
        return "slow"

    def fast_tool():
        time.sleep(0.01)
        return "fast"

    react = dspy.ReAct("question -> answer", tools=[slow_tool, fast_tool], parallel_tool_calls=True)
    steps = iter(
        [
            dspy.Prediction(
                next_message="Checking both tools.",
                tool_calls=_tool_calls(("slow_tool", {}), ("fast_tool", {})),
            ),
            dspy.Prediction(next_message=None, tool_calls=_tool_calls(("submit", {"answer": "done"}))),
        ]
    )
    react.react = lambda **kwargs: next(steps)

    result = react(question="Check both tools.")

    first_tool_messages = [message for message in result.history.messages if message.get("role") == "tool"][:2]
    assert [message["name"] for message in first_tool_messages] == ["slow_tool", "fast_tool"]
    assert [message["content"] for message in first_tool_messages] == ["slow", "fast"]


@pytest.mark.extra
def test_tool_observation_preserves_custom_type_in_raw_history():
    pytest.importorskip("PIL.Image")
    from PIL import Image

    class SpyChatAdapter(dspy.ChatAdapter):
        pass

    def make_images():
        return dspy.Image("https://example.com/test.png"), dspy.Image(Image.new("RGB", (100, 100), "red"))

    adapter = SpyChatAdapter()
    lm = DummyLM([{"reasoning": "image ready", "answer": "done"}], adapter=adapter)
    dspy.configure(lm=lm, adapter=adapter)

    react = dspy.ReAct("question -> answer", tools=[make_images])
    steps = iter(
        [
            dspy.Prediction(next_message="I should inspect the images.", tool_calls=_tool_calls(("make_images", {}))),
            dspy.Prediction(next_message="I can answer now.", tool_calls=None),
        ]
    )
    react.react = lambda **kwargs: next(steps)

    react(question="Draw me something red")

    tool_messages = [message for message in lm.history[-1]["messages"] if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert sum(1 for part in tool_messages[0]["content"] if part.get("type") == "image_url") == 2


def test_react_accepts_legacy_trajectory_alias():
    captured_histories = []
    react = dspy.ReAct("question -> answer", tools=[])
    react.react = lambda **kwargs: captured_histories.append(kwargs["history"]) or dspy.Prediction(
        next_message=None,
        tool_calls=_tool_calls(("submit", {"answer": "Paris"})),
    )

    trajectory = {
        "thought_0": "I should search.",
        "tool_name_0": "search",
        "tool_args_0": {"query": "capital of France"},
        "observation_0": "Paris",
    }

    result = react(question="What is the capital of France?", trajectory=trajectory)

    seen_history = captured_histories[0]
    assert seen_history.messages[0]["tool_calls"][0]["function"]["name"] == "search"
    assert seen_history.messages[1]["role"] == "tool"
    assert seen_history.messages[-1]["role"] == "user"
    assert result.answer == "Paris"


@pytest.mark.asyncio
async def test_react_async_tool_calling_and_submit():
    async def add(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[add])
    steps = iter(
        [
            dspy.Prediction(next_message="Let me add that.", tool_calls=_tool_calls(("add", {"a": 1, "b": 2}))),
            dspy.Prediction(next_message=None, tool_calls=_tool_calls(("submit", {"c": 3}))),
        ]
    )

    async def mock_react(**kwargs):
        return next(steps)

    react.react.acall = mock_react

    result = await react.acall(a=1, b=2)

    assert result.c == 3
    assert result.history.messages[1]["tool_calls"][0]["function"]["name"] == "add"
    assert result.history.messages[2]["content"] == "3"
    assert result.history.messages[3]["tool_calls"][0]["function"]["name"] == "submit"
