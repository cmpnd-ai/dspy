from typing import Any, Callable

import pydantic
from pydantic import Field, model_validator

from dspy.adapters.types.tool import ToolCalls
from dspy.core.types import LMMessage, LMTextPart, LMToolResultPart


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

    @model_validator(mode="after")
    def _normalize_tool_calls_outputs(self) -> "HistoryFrame":
        normalized_outputs = {}
        changed = False
        for key, value in self.outputs.items():
            if isinstance(value, dict) and set(value.keys()) == {"tool_calls"} and isinstance(value["tool_calls"], list):
                normalized_outputs[key] = ToolCalls.model_validate(value)
                changed = True
            else:
                normalized_outputs[key] = value

        if changed:
            self.outputs = normalized_outputs
        return self


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

    def to_lm_messages(self, adapter: Any, signature: type[Any], *, use_native_tool_calls: bool = False) -> list[LMMessage]:
        messages: list[LMMessage] = []
        for frame_idx, entry in enumerate(self.frames):
            frame = self._entry_to_frame(signature, entry)
            if frame.inputs:
                content = adapter.format_user_message_content(signature, frame.inputs)
                if self._has_content(content):
                    messages.append(self._content_message("user", content))

            if frame.outputs:
                messages.extend(self._format_frame_outputs(adapter, signature, frame, frame_idx, use_native_tool_calls))
            elif frame.observations:
                messages.append(self._content_message("user", self._format_observations(frame.observations)))
        return messages

    def _format_frame_outputs(
        self,
        adapter: Any,
        signature: type[Any],
        frame: HistoryFrame,
        frame_idx: int,
        use_native_tool_calls: bool,
    ) -> list[LMMessage]:
        tool_calls = self._find_tool_calls(frame.outputs)
        if use_native_tool_calls and tool_calls is not None:
            parts = []
            content = self._format_frame_native_content(adapter, signature, frame.outputs)
            if content:
                parts.append(LMTextPart(text=content))
            parts.extend(tool_calls.to_lm_parts(id_prefix=f"call_{frame_idx}"))

            messages = [LMMessage(role="assistant", parts=parts)]
            non_native_observations = []
            for observation in frame.observations:
                if observation.call_id is not None:
                    messages.append(self._native_tool_observation(observation))
                else:
                    non_native_observations.append(observation)
            if non_native_observations:
                messages.append(self._content_message("user", self._format_observations(non_native_observations)))
            return messages

        messages = [self._content_message("assistant", self._format_outputs(adapter, signature, frame.outputs))]
        if frame.observations:
            messages.append(self._content_message("user", self._format_observations(frame.observations)))
        return messages

    @staticmethod
    def _entry_to_frame(signature: type[Any], entry: HistoryEntry) -> HistoryFrame:
        if isinstance(entry, HistoryFrame):
            return entry

        inputs = {key: value for key, value in entry.items() if key in signature.input_fields}
        outputs = {key: value for key, value in entry.items() if key in signature.output_fields}
        unknown = {key: value for key, value in entry.items() if key not in inputs and key not in outputs}
        if unknown and not outputs:
            outputs = unknown
        elif unknown:
            outputs = {**outputs, **unknown}
        return HistoryFrame(inputs=inputs, outputs=outputs, complete=True)

    def _format_outputs(self, adapter: Any, signature: type[Any], outputs: dict[str, Any]) -> str:
        signature_outputs = {key: value for key, value in outputs.items() if key in signature.output_fields}
        unknown_outputs = {key: value for key, value in outputs.items() if key not in signature.output_fields}
        if signature_outputs and not unknown_outputs:
            return adapter.format_assistant_message_content(
                signature,
                signature_outputs,
                missing_field_message="Not supplied for this conversation history message. ",
            )

        sections = []
        if signature_outputs:
            sections.append(
                adapter.format_assistant_message_content(
                    signature,
                    signature_outputs,
                    missing_field_message="Not supplied for this conversation history message. ",
                ).strip()
            )
        for key, value in unknown_outputs.items():
            sections.append(f"[[ ## {key} ## ]]\n{self._format_observation_content(value)}")
        sections.append("[[ ## completed ## ]]")
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _find_tool_calls(outputs: dict[str, Any]) -> ToolCalls | None:
        for value in outputs.values():
            if isinstance(value, ToolCalls):
                return value
        return None

    def _format_frame_native_content(self, adapter: Any, signature: type[Any], outputs: dict[str, Any]) -> str | None:
        non_tool_outputs = {key: value for key, value in outputs.items() if not isinstance(value, ToolCalls)}
        if not non_tool_outputs:
            return None
        if len(non_tool_outputs) == 1:
            return str(next(iter(non_tool_outputs.values())))
        return self._format_outputs(adapter, signature, non_tool_outputs)

    def _format_observations(self, observations: list[Observation]) -> str:
        rendered = []
        for idx, observation in enumerate(observations):
            label = "Error" if observation.is_error else "Observation"
            content = self._format_observation_content(observation.value)
            subject = self._observation_subject(observation, idx)
            if "\n" in content:
                rendered.append(f"{subject}:\n{label}:\n{content}")
            else:
                rendered.append(f"{subject}:\n{label}: {content}")
        observations_text = "\n\n".join(rendered)
        return f"[[ ## observations ## ]]\n{observations_text}"

    @staticmethod
    def _observation_subject(observation: Observation, idx: int) -> str:
        if observation.call_id is not None or observation.name is not None or observation.source == "tool":
            tool_name = observation.name or f"unknown_{idx + 1}"
            return f"Tool call {idx + 1} (`{tool_name}`)"
        if observation.source:
            return f"{observation.source} observation {idx + 1}"
        return f"Observation {idx + 1}"

    @staticmethod
    def _format_observation_content(content: Any) -> str:
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    def _native_tool_observation(self, observation: Observation) -> LMMessage:
        return LMMessage(
            role="tool",
            parts=[
                LMToolResultPart(
                    call_id=observation.call_id,
                    name=observation.name,
                    content=[LMTextPart(text=self._format_observation_content(observation.value))],
                    is_error=observation.is_error,
                )
            ],
        )

    @staticmethod
    def _has_content(content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        return bool(content)

    @staticmethod
    def _content_message(role: str, content: Any) -> LMMessage:
        return LMMessage.model_validate({"role": role, "content": content})


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
