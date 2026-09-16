"""Typed LM history records and history-only legacy rendering.

History rendering deliberately preserves media references as recorded. Unlike
wire rendering, it must never open local media paths.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pprint import pformat
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMPart,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMToolCallPart,
    LMToolResultPart,
    LMVideoPart,
    _split_data_uri,
)


class LMHistoryEntry(BaseModel, Mapping[str, Any]):
    """A typed history record that can be read like a dictionary.

    Store the canonical request and response, then derive legacy convenience
    fields such as `outputs`, `usage`, `messages`, and `kwargs` on demand.
    Because this class implements `Mapping`, existing history code can keep
    using `entry["messages"]`, `entry.get("prompt")`, `entry.items()`, and
    `dict(entry)`.
    """

    request: LMRequest
    response: LMResponse
    timestamp: str
    uuid: str
    model_type: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def drop_derived_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in _HISTORY_DERIVED_KEYS:
                data.pop(key, None)
        return data

    @property
    def outputs(self) -> list[Any]:
        return self.response.to_outputs()

    @property
    def usage(self) -> dict[str, Any]:
        return self.response.usage_as_dict()

    @property
    def cost(self) -> float | None:
        return self.response.cost

    @property
    def model(self) -> str:
        return self.request.model

    @property
    def prompt(self) -> str | None:
        if len(self.request.messages) != 1:
            return None
        message = self.request.messages[0]
        if message.role == "user" and len(message.parts) == 1 and isinstance(message.parts[0], LMTextPart):
            return message.parts[0].text
        return None

    @property
    def messages(self) -> list[dict[str, Any]] | None:
        return _request_messages_as_openai(self.request)

    @property
    def kwargs(self) -> dict[str, Any]:
        data = self.request.config.model_dump(exclude_none=True)
        extensions = data.pop("extensions", {}) or {}
        return {**extensions, **data}

    @property
    def response_model(self) -> str | None:
        return self.response.model

    def __getitem__(self, key: str) -> Any:
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())

    def __repr__(self) -> str:
        return f"LMHistoryEntry(\n{pformat(self._essential_mapping(), width=100, sort_dicts=False)}\n)"

    def __str__(self) -> str:
        return repr(self)

    def to_dict(self, *, mode: str = "python", exclude_none: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Return this history entry as a plain dictionary."""
        if kwargs:
            return self.model_dump(mode=mode, exclude_none=exclude_none, **kwargs)
        data = self._mapping()
        if mode != "python":
            data = json.loads(json.dumps(data, default=_json_default))
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data

    def _essential_mapping(self) -> dict[str, Any]:
        data = self.model_dump(mode="python", exclude_none=True)
        data.update(self.model_extra or {})
        return data

    def _mapping(self) -> dict[str, Any]:
        data = self._essential_mapping()
        data.update({key: getattr(self, key) for key in _HISTORY_DERIVED_KEYS})
        return {key: value for key, value in data.items() if value is not None}


_HISTORY_DERIVED_KEYS = ("outputs", "usage", "cost", "model", "prompt", "messages", "kwargs", "response_model")


def _request_messages_as_openai(request: LMRequest) -> list[dict[str, Any]]:
    messages = []
    for message in request.messages:
        if message.role == "assistant":
            tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
            content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
            item: dict[str, Any] = {
                "role": "assistant",
                "content": _parts_as_content(content_parts) if content_parts else None,
            }
            if tool_calls:
                item["tool_calls"] = [_tool_call_as_openai(call) for call in tool_calls]
        elif message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
            result = message.parts[0]
            item = {"role": "tool", "content": _tool_result_content(result)}
            if result.call_id is not None:
                item["tool_call_id"] = result.call_id
            if result.name is not None:
                item["name"] = result.name
        else:
            item = {"role": message.role, "content": _parts_as_content(message.parts)}
        if message.name is not None and "name" not in item:
            item["name"] = message.name
        messages.append(item)
    return messages


def _tool_call_as_openai(call: LMToolCallPart) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.args)},
    }
    if call.id is not None:
        data["id"] = call.id
    return data


def _tool_result_content(result: LMToolResultPart) -> str:
    return "".join(
        part.text
        if isinstance(part, LMTextPart)
        else json.dumps(part.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
        for part in result.content
    )


def _parts_as_content(parts: list[LMPart]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], LMTextPart):
        return parts[0].text
    return [_part_as_content(part) for part in parts]


def _part_as_content(part: LMPart) -> dict[str, Any]:
    if isinstance(part, LMTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, LMImagePart):
        return {"type": "image_url", "image_url": {"url": _part_source(part)}}
    if isinstance(part, LMAudioPart):
        audio: dict[str, Any] = {"format": _media_format(part.media_type)}
        if part.data is not None:
            if part.data.startswith("data:"):
                media_type, audio["data"] = _split_data_uri(part.data)
                audio["format"] = _media_format(media_type)
            else:
                audio["data"] = part.data
        else:
            for key in ("url", "file_id", "path"):
                if getattr(part, key) is not None:
                    audio[key] = getattr(part, key)
                    break
        return {"type": "input_audio", "input_audio": audio}
    if isinstance(part, LMVideoPart):
        video: dict[str, Any] = {"media_type": part.media_type}
        for key in ("data", "url", "file_id", "path"):
            if getattr(part, key) is not None:
                video[key] = _part_source(part) if key == "data" else getattr(part, key)
                break
        return {"type": "video", "video": video}
    if isinstance(part, LMDocumentPart):
        data: dict[str, Any] = {"type": "document"}
        data["source"] = part.source if part.source is not None else _part_source(part)
        if part.source is None:
            data["media_type"] = part.media_type
        if part.citations:
            data["citations"] = part.citations
        for key in ("title", "context"):
            value = getattr(part, key)
            if value is not None:
                data[key] = value
        return data
    if isinstance(part, LMBinaryPart):
        values = {
            "data": _part_source(part),
            "file_id": part.file_id,
            "filename": part.filename,
            "media_type": part.media_type,
        }
        return {"type": "binary", "binary": {key: value for key, value in values.items() if value is not None}}
    return part.model_dump(exclude_none=True)


def _part_source(part: LMImagePart | LMAudioPart | LMVideoPart | LMDocumentPart | LMBinaryPart) -> str | None:
    if part.data is not None:
        return part.data if part.data.startswith("data:") else f"data:{part.media_type};base64,{part.data}"
    return part.url or part.file_id or part.path


def _media_format(media_type: str) -> str:
    return media_type.split("/", 1)[1] if "/" in media_type else media_type


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return str(value)
