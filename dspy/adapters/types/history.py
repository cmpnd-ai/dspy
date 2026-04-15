from typing import Any, Callable

import pydantic


class History(pydantic.BaseModel):
    """Conversation history with semantic events (REQUEST/ACTION/FINAL) and pluggable compaction."""

    messages: list[dict[str, Any]]

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

    def append_request(self, inputs: dict[str, Any]) -> None:
        self.messages.append({"__dspy_history_event__": "REQUEST", **inputs})

    def append_action(self, *, thought: str, tool_calls: Any, observations: list[tuple[Any, bool]]) -> None:
        self.messages.append({
            "__dspy_history_event__": "ACTION",
            "thought": thought,
            "tool_calls": tool_calls,
            "observations": observations,
        })

    def append_final(self, outputs: dict[str, Any]) -> None:
        self.messages.append({"__dspy_history_event__": "FINAL", **outputs})

    def has_open_episode(self) -> bool:
        last_boundary = None
        for m in self.messages:
            evt = m.get("__dspy_history_event__")
            if evt in ("REQUEST", "FINAL"):
                last_boundary = evt
        return last_boundary == "REQUEST"


def truncate_oldest_actions(history: History, *, max_tokens: int = 200_000, keep_n: int = 3) -> None:
    est = len(str(history.messages)) // 4
    if est <= max_tokens:
        return
    actions = [(i, m) for i, m in enumerate(history.messages) if m.get("__dspy_history_event__") == "ACTION"]
    to_drop = len(actions) - keep_n
    if to_drop <= 0:
        return
    drop_indices = {i for i, _ in actions[:to_drop]}
    history.messages[:] = [m for i, m in enumerate(history.messages) if i not in drop_indices]


def make_truncate_oldest_actions(max_tokens: int = 200_000, keep_n: int = 3) -> Callable[[History], None]:
    def _compact(history: History) -> None:
        truncate_oldest_actions(history, max_tokens=max_tokens, keep_n=keep_n)
    return _compact
