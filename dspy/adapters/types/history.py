from typing import Annotated, Any, Callable, Literal

import pydantic

from dspy.adapters.types.tool import ToolCalls


class InputEvent(pydantic.BaseModel):
    event: Literal["input"] = "input"
    inputs: dict[str, Any]


class Observation(pydantic.BaseModel):
    """A single tool observation with an optional error flag."""
    value: Any
    is_error: bool = False


class ActionEvent(pydantic.BaseModel):
    event: Literal["action"] = "action"
    thought: str | None = None
    tool_calls: ToolCalls | None = None
    observations: list[Observation] = []


class OutputEvent(pydantic.BaseModel):
    event: Literal["output"] = "output"
    outputs: dict[str, Any]


class LegacyEvent(pydantic.BaseModel):
    """Backward-compat wrapper for plain dict messages from old History format."""
    event: Literal["legacy"] = "legacy"
    data: dict[str, Any]


HistoryEvent = Annotated[InputEvent | ActionEvent | OutputEvent | LegacyEvent, pydantic.Field(discriminator="event")]


class History(pydantic.BaseModel):
    """Conversation history with typed semantic events and pluggable compaction."""

    messages: list[HistoryEvent]

    model_config = pydantic.ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_legacy_messages(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "messages" not in data:
            return data
        raw = data["messages"]
        if not isinstance(raw, list):
            return data
        coerced = []
        for msg in raw:
            if not isinstance(msg, dict) or "event" in msg:
                coerced.append(msg)
                continue
            # Legacy plain dict — wrap as LegacyEvent so it passes validation
            coerced.append({"event": "legacy", "data": msg})
        return {**data, "messages": coerced}

    @pydantic.model_serializer(mode="wrap")
    def _serialize_legacy_messages(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if "messages" in data:
            serialized = []
            for msg in data["messages"]:
                if isinstance(msg, dict) and msg.get("event") == "legacy":
                    serialized.append(msg["data"])
                else:
                    serialized.append(msg)
            data["messages"] = serialized
        return data

    def __init__(self, *args: Any, compact_fn: Callable[["History"], None] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_compact_fn", compact_fn)

    def compact_if_needed(self) -> None:
        fn = getattr(self, "_compact_fn", None)
        if fn is not None:
            fn(self)

    def append_input(self, inputs: dict[str, Any]) -> None:
        self.messages.append(InputEvent(inputs=inputs))

    def append_action(self, *, thought: str, tool_calls: ToolCalls | None, observations: list[Observation]) -> None:
        self.messages.append(ActionEvent(thought=thought, tool_calls=tool_calls, observations=observations))

    def append_output(self, outputs: dict[str, Any]) -> None:
        self.messages.append(OutputEvent(outputs=outputs))

    def has_open_episode(self) -> bool:
        last_boundary = None
        for m in self.messages:
            if isinstance(m, InputEvent):
                last_boundary = "input"
            elif isinstance(m, OutputEvent):
                last_boundary = "output"
        return last_boundary == "input"


def truncate_oldest_actions(history: History, *, max_tokens: int = 200_000, keep_n: int = 3) -> None:
    est = len(str(history.messages)) // 4
    if est <= max_tokens:
        return
    actions = [(i, m) for i, m in enumerate(history.messages) if isinstance(m, ActionEvent)]
    to_drop = len(actions) - keep_n
    if to_drop <= 0:
        return
    drop_indices = {i for i, _ in actions[:to_drop]}
    history.messages[:] = [m for i, m in enumerate(history.messages) if i not in drop_indices]


def make_truncate_oldest_actions(max_tokens: int = 200_000, keep_n: int = 3) -> Callable[[History], None]:
    def _compact(history: History) -> None:
        truncate_oldest_actions(history, max_tokens=max_tokens, keep_n=keep_n)
    return _compact
