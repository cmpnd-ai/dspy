from typing import Annotated, Any, Callable, Literal

import pydantic


class InputEvent(pydantic.BaseModel):
    event: Literal["input"] = "input"
    inputs: dict[str, Any]


class ActionEvent(pydantic.BaseModel):
    event: Literal["action"] = "action"
    thought: str | None = None
    tool_calls: Any = None  # ToolCalls type, use Any to avoid circular import
    observations: list[tuple[Any, bool]] = []


class OutputEvent(pydantic.BaseModel):
    event: Literal["output"] = "output"
    outputs: dict[str, Any]


HistoryEvent = Annotated[InputEvent | ActionEvent | OutputEvent, pydantic.Field(discriminator="event")]


class History(pydantic.BaseModel):
    """Conversation history with typed semantic events and pluggable compaction."""

    messages: list[HistoryEvent]

    model_config = pydantic.ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    def __init__(self, *args: Any, compact_fn: Callable[["History"], None] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_compact_fn", compact_fn)

    def compact_if_needed(self) -> None:
        fn = getattr(self, "_compact_fn", None)
        if fn is not None:
            fn(self)

    def append_input(self, inputs: dict[str, Any]) -> None:
        self.messages.append(InputEvent(inputs=inputs))

    def append_action(self, *, thought: str, tool_calls: Any, observations: list[tuple[Any, bool]]) -> None:
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
