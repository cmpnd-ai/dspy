from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic.fields import FieldInfo

from dspy.core.types import LMConfig, LMMessage, LMOutput, LMPart, LMToolSpec

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.signatures.signature import Signature


@dataclass(frozen=True)
class _CallContext:
    adapter: Adapter
    use_native_function_calling: bool
    allow_parallel_tool_calls: bool | None
    native_response_types: tuple[type[object], ...]
    lm: BaseLM
    lm_kwargs: dict[str, object] = field(default_factory=dict)
    lm_default_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _TypeRenderContext:
    field_name: str
    field_info: FieldInfo
    signature: type[Signature]
    values: dict[str, object]
    adapter: Adapter
    call_context: _CallContext
    role: Literal["input", "output"]


@dataclass(frozen=True)
class _TypeParseContext:
    field_name: str
    field_info: FieldInfo
    signature: type[Signature]
    adapter: Adapter
    call_context: _CallContext
    parts: list[LMPart]
    lm_output: LMOutput


_OutputParser = Callable[[_TypeParseContext], object]


@dataclass
class _AdapterCallPlan:
    source_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, object]
    lm_kwargs: dict[str, object]
    messages: list[LMMessage] = field(default_factory=list)
    tools: list[LMToolSpec] = field(default_factory=list)
    config: LMConfig | None = None
    output_parsers: dict[str, _OutputParser] = field(default_factory=dict)

    @classmethod
    def from_signature(
        cls,
        signature: type[Signature],
        inputs: dict[str, object],
        lm_kwargs: dict[str, object] | None = None,
    ) -> _AdapterCallPlan:
        return cls(
            source_signature=signature,
            render_signature=signature,
            inputs=dict(inputs),
            lm_kwargs=dict(lm_kwargs or {}),
        )

    def delete_field(self, field_name: str) -> None:
        if field_name in self.render_signature.fields:
            self.render_signature = self.render_signature.delete(field_name)

    def merge_config(self, config: LMConfig | None) -> None:
        if config is None:
            return
        self.config = _merge_lm_config(self.config, config)


class _TypeFeatureHandler:
    def prepare(self, call: _AdapterCallPlan, ctx: _CallContext) -> None:
        pass

    def parse(
        self,
        values: dict[str, object],
        output: LMOutput,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> None:
        pass


def _merge_lm_config(left: LMConfig | None, right: LMConfig | None) -> LMConfig | None:
    if left is None:
        return right
    if right is None:
        return left

    data = left.model_dump()
    right_data = right.model_dump(exclude_none=True, exclude_unset=True)
    for key in ("reasoning", "tool_choice", "cache", "prompt_cache"):
        if key in right_data and isinstance(data.get(key), dict) and isinstance(right_data[key], dict):
            right_data[key] = {**data[key], **right_data[key]}
    data.update(right_data)
    data["extensions"] = {**left.extensions, **right.extensions}
    return LMConfig(**data)
