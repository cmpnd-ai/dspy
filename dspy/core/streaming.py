"""Normalized streaming types and response assembly for language models."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types import (
    LMAudioPart,
    LMCitationPart,
    LMImagePart,
    LMOutput,
    LMPart,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMUsage,
    _parse_json_object,
)


class LMDelta(BaseModel):
    """Base class for streamed content deltas."""

    type: str


class LMTextDelta(LMDelta):
    type: Literal["text_delta"] = "text_delta"
    text: str


class LMThinkingDelta(LMDelta):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class LMToolCallDelta(LMDelta):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str | None = None
    name: str | None = None
    args_delta: str | None = None


class LMCitationDelta(LMDelta):
    type: Literal["citation_delta"] = "citation_delta"
    citation: LMCitationPart


class LMImageDelta(LMDelta):
    type: Literal["image_delta"] = "image_delta"
    image: LMImagePart


class LMAudioDelta(LMDelta):
    type: Literal["audio_delta"] = "audio_delta"
    audio: LMAudioPart


LMAnyDelta = Annotated[
    LMTextDelta | LMThinkingDelta | LMToolCallDelta | LMCitationDelta | LMImageDelta | LMAudioDelta,
    Field(discriminator="type"),
]


class LMStreamEvent(BaseModel):
    """Base class for normalized LM stream events."""

    type: str


class LMStreamStartEvent(LMStreamEvent):
    type: Literal["start"] = "start"
    model: str | None = None


class LMStreamDeltaEvent(LMStreamEvent):
    type: Literal["delta"] = "delta"
    output_index: int = Field(default=0, ge=0)
    part_index: int = Field(ge=0)
    delta: LMAnyDelta


class LMStreamOutputEndEvent(LMStreamEvent):
    type: Literal["output_end"] = "output_end"
    output_index: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    truncated: bool = False


class LMStreamEndEvent(LMStreamEvent):
    type: Literal["end"] = "end"
    usage: LMUsage | dict[str, Any] | None = None
    cost: float | None = None
    response: LMResponse | None = None


class LMStreamErrorEvent(LMStreamEvent):
    type: Literal["error"] = "error"
    error: Exception

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LMOutputBuilder:
    """Assemble streamed LM events into a final `LMResponse`."""

    def __init__(self):
        self.model: str | None = None
        self._parts: dict[int, list[LMPart | None]] = {}
        self._finish_reasons: dict[int, str | None] = {}
        self._truncated: dict[int, bool] = {}

    def apply(self, event: LMStreamEvent) -> LMResponse | None:
        if isinstance(event, LMStreamStartEvent):
            self.model = event.model
            return None
        if isinstance(event, LMStreamDeltaEvent):
            self._apply_delta(event)
            return None
        if isinstance(event, LMStreamOutputEndEvent):
            self._finish_reasons[event.output_index] = event.finish_reason
            self._truncated[event.output_index] = event.truncated
            return None
        if isinstance(event, LMStreamEndEvent):
            if event.response is not None:
                return event.response
            return self.to_response(usage=event.usage, cost=event.cost)
        if isinstance(event, LMStreamErrorEvent):
            raise event.error
        return None

    def to_response(self, *, usage: LMUsage | dict[str, Any] | None = None, cost: float | None = None) -> LMResponse:
        output_indices = set(self._parts) | set(self._finish_reasons) | set(self._truncated)
        if not output_indices:
            output_indices = {0}
        max_index = max(output_indices)
        expected_indices = set(range(max_index + 1))
        if output_indices != expected_indices:
            missing = sorted(expected_indices - output_indices)
            raise ValueError(f"Stream output indices must be contiguous from 0; missing indices: {missing}.")

        outputs = []
        for output_index in range(max_index + 1):
            part_buffer = self._parts.get(output_index, [])
            missing_part_indices = [index for index, part in enumerate(part_buffer) if part is None]
            if missing_part_indices:
                raise ValueError(
                    f"Stream part indices for output {output_index} must be contiguous; "
                    f"missing indices: {missing_part_indices}."
                )
            parts = [_finalize_stream_part(part) for part in part_buffer]
            outputs.append(
                LMOutput(
                    parts=parts,
                    finish_reason=self._finish_reasons.get(output_index),
                    truncated=self._truncated.get(output_index, False),
                )
            )
        return LMResponse(model=self.model, outputs=outputs, usage=usage, cost=cost)

    def _apply_delta(self, event: LMStreamDeltaEvent) -> None:
        parts = self._parts.setdefault(event.output_index, [])
        while len(parts) <= event.part_index:
            parts.append(None)

        current = parts[event.part_index]
        delta = event.delta
        if isinstance(delta, LMThinkingDelta):
            if current is not None and not isinstance(current, LMThinkingPart):
                raise ValueError("Cannot apply thinking delta to a non-thinking stream part.")
            text = (current.text if isinstance(current, LMThinkingPart) else "") + delta.text
            parts[event.part_index] = LMThinkingPart(text=text)
        elif isinstance(delta, LMTextDelta):
            if current is not None and not isinstance(current, LMTextPart):
                raise ValueError("Cannot apply text delta to a non-text stream part.")
            text = (current.text if isinstance(current, LMTextPart) else "") + delta.text
            parts[event.part_index] = LMTextPart(text=text)
        elif isinstance(delta, LMToolCallDelta):
            if current is not None and not isinstance(current, LMToolCallPart):
                raise ValueError("Cannot apply tool-call delta to a non-tool-call stream part.")
            buffer = current.provider_data.get("args_buffer", "") if isinstance(current, LMToolCallPart) else ""
            buffer += delta.args_delta or ""
            parts[event.part_index] = LMToolCallPart(
                id=delta.id if delta.id is not None else getattr(current, "id", None),
                name=delta.name if delta.name is not None else getattr(current, "name", ""),
                args=_parse_json_object(buffer),
                provider_data={"args_buffer": buffer},
            )
        elif isinstance(delta, LMCitationDelta):
            if current is not None and not isinstance(current, LMCitationPart):
                raise ValueError("Cannot apply citation delta to a different stream part type.")
            parts[event.part_index] = delta.citation
        elif isinstance(delta, LMImageDelta):
            if current is not None and not isinstance(current, LMImagePart):
                raise ValueError("Cannot apply image delta to a different stream part type.")
            parts[event.part_index] = delta.image
        elif isinstance(delta, LMAudioDelta):
            if current is not None and not isinstance(current, LMAudioPart):
                raise ValueError("Cannot apply audio delta to a different stream part type.")
            parts[event.part_index] = delta.audio


class LMStream:
    """Synchronous LM stream with a final `LMResponse` result."""

    def __init__(
        self,
        *,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        finalize: Callable[[LMRequest, LMResponse], LMResponse],
    ):
        self.request = request
        self._events = events
        self._finalize = finalize
        self._builder = LMOutputBuilder()
        self._result: LMResponse | None = None

    def __iter__(self) -> Iterator[LMStreamEvent]:
        for event in self._events:
            response = self._builder.apply(event)
            if response is not None:
                self._result = self._finalize(self.request, response)
            yield event

    def result(self) -> LMResponse:
        if self._result is None:
            raise RuntimeError("Stream has not completed yet.")
        return self._result


class AsyncLMStream:
    """Asynchronous LM stream with a final `LMResponse` result."""

    def __init__(
        self,
        *,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        finalize: Callable[[LMRequest, LMResponse], LMResponse],
    ):
        self.request = request
        self._events = events
        self._finalize = finalize
        self._builder = LMOutputBuilder()
        self._result: LMResponse | None = None

    async def __aiter__(self) -> AsyncIterator[LMStreamEvent]:
        async for event in self._events:
            response = self._builder.apply(event)
            if response is not None:
                self._result = self._finalize(self.request, response)
            yield event

    def result(self) -> LMResponse:
        if self._result is None:
            raise RuntimeError("Stream has not completed yet.")
        return self._result


def _finalize_stream_part(part: LMPart) -> LMPart:
    if isinstance(part, LMToolCallPart) and "args_buffer" in part.provider_data:
        return part.model_copy(update={"args": _parse_json_object_strict(part.provider_data["args_buffer"])})
    return part


def _parse_json_object_strict(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Streamed tool-call arguments must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Streamed tool-call arguments must be a JSON object.")
    return parsed
