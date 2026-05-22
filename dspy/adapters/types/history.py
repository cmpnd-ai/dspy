from typing import Any, Callable

import pydantic
from pydantic import Field, model_validator


class Observation(pydantic.BaseModel):
    """External result produced by executing or evaluating a history frame."""

    value: Any
    source: str | None = None
    call_id: str | None = None
    name: str | None = None
    is_error: bool = False


class HistoryFrame(pydantic.BaseModel):
    """A partial DSPy example/prediction frame with optional observations."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    complete: bool = False
    source: str | None = None

    model_config = pydantic.ConfigDict(extra="forbid")


HistoryEntry = HistoryFrame | dict[str, Any]


class History(pydantic.BaseModel):
    """Reusable DSPy field-frame history."""

    frames: list[HistoryEntry] = Field(default_factory=list)

    model_config = pydantic.ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_messages_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "messages" in data and "frames" not in data:
            data = dict(data)
            data["frames"] = data.pop("messages")
        return data

    def __init__(self, *args: Any, compact_fn: Callable[["History"], None] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_compact_fn", compact_fn)

    @property
    def messages(self) -> list[HistoryEntry]:
        return self.frames

    def compact_if_needed(self) -> None:
        fn = getattr(self, "_compact_fn", None)
        if fn is not None:
            fn(self)

    def append_inputs(self, inputs: dict[str, Any], *, source: str | None = None) -> HistoryFrame:
        frame = HistoryFrame(inputs=dict(inputs), source=source)
        self.frames.append(frame)
        return frame

    def append_outputs(
        self,
        outputs: dict[str, Any],
        *,
        observations: list[Observation] | None = None,
        complete: bool = False,
        source: str | None = None,
    ) -> HistoryFrame:
        frame = HistoryFrame(
            outputs=dict(outputs),
            observations=list(observations or []),
            complete=complete,
            source=source,
        )
        self.frames.append(frame)
        return frame

    def append_observation(
        self,
        value: Any,
        *,
        source: str | None = None,
        call_id: str | None = None,
        name: str | None = None,
        is_error: bool = False,
    ) -> Observation:
        observation = Observation(value=value, source=source, call_id=call_id, name=name, is_error=is_error)
        if self.frames and isinstance(self.frames[-1], HistoryFrame):
            self.frames[-1].observations.append(observation)
        else:
            self.frames.append(HistoryFrame(observations=[observation], source=source))
        return observation

    def append_input(self, inputs: dict[str, Any]) -> HistoryFrame:
        return self.append_inputs(inputs)

    def append_output(self, outputs: dict[str, Any]) -> HistoryFrame:
        return self.append_outputs(outputs, complete=True)

    def has_open_episode(self) -> bool:
        last_boundary = None
        for frame in self.frames:
            if isinstance(frame, dict):
                continue
            if frame.inputs:
                last_boundary = "input"
            if frame.complete:
                last_boundary = "output"
        return last_boundary == "input"


def truncate_oldest_actions(history: History, *, max_tokens: int = 200_000, keep_n: int = 3) -> None:
    if len(str(history.frames)) // 4 <= max_tokens:
        return

    action_starts = [
        idx
        for idx, frame in enumerate(history.frames)
        if isinstance(frame, HistoryFrame) and frame.observations
    ]
    drop_count = len(action_starts) - keep_n
    if drop_count <= 0:
        return

    drop_indices = set(action_starts[:drop_count])
    history.frames[:] = [frame for idx, frame in enumerate(history.frames) if idx not in drop_indices]


def make_truncate_oldest_actions(max_tokens: int = 200_000, keep_n: int = 3) -> Callable[[History], None]:
    def _compact(history: History) -> None:
        truncate_oldest_actions(history, max_tokens=max_tokens, keep_n=keep_n)

    return _compact
