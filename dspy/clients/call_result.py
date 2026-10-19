"""DSPy execution metadata around unmodified lm15 responses."""

from dataclasses import dataclass, field
from typing import Any

from dspy._vendor.lm15.serde import response_from_dict, response_to_dict
from dspy.clients.legacy_outputs import plain, value
from dspy.clients.lm15_boundary import response_value
from dspy.lm15 import DataPart, Request, Response, TextPart, ThinkingPart

CACHE_FORMAT = "dspy-lm15-result-v1"


@dataclass(frozen=True)
class LMCallEvidence:
    """Per-invocation observations, independent of history retention and pricing policy."""

    model_requested: str
    model_reported: str | None
    cache_hit: bool
    usage_reported: bool
    usage_raw: Any
    usage: dict
    response_ids: tuple[str, ...]
    finish_reasons: tuple[str, ...]
    service_tier_requested: str | None
    service_tier_reported: str | None
    cost: float | None


def _has_reported_usage(usage):
    # Normalized native and legacy usage omit absent dimensions or carry None.
    # Nested details can also be empty; a reported zero is still evidence.
    if isinstance(usage, dict):
        return any(_has_reported_usage(item) for item in usage.values())
    return usage is not None


def observe_result(lm, result, kwargs):
    from dspy.utils.tracing import current_invocation, snapshot

    invocation = current_invocation()
    if invocation is None:
        return
    responses = result.responses or ((result.raw,) if result.raw is not None else ())
    raw_usage = [plain(value(response, "usage")) for response in responses]
    ids = []
    reasons = []
    for response in responses:
        identity = value(response, "id") or value(response, "response_id")
        if not identity and isinstance(response, Response):
            # Native chat streaming can retain the provider's ID only in raw metadata.
            identity = value(response.provider_data, "id")
        if identity:
            ids.append(str(identity))
        reason = value(response, "finish_reason") or value(response, "stop_reason")
        if reason:
            reasons.append(str(reason))
        for choice in value(response, "choices", ()) or ():
            if reason := value(choice, "finish_reason"):
                reasons.append(str(reason))
    invocation.lm = LMCallEvidence(
        model_requested=lm.model, model_reported=result.response_model, cache_hit=result.cache_hit,
        usage_reported=not result.cache_hit and _has_reported_usage(result.usage),
        usage_raw=snapshot(raw_usage[0] if len(raw_usage) == 1 else raw_usage), usage=snapshot(result.usage),
        response_ids=tuple(ids), finish_reasons=tuple(reasons),
        service_tier_requested=kwargs.get("service_tier", lm.kwargs.get("service_tier")),
        service_tier_reported=value(responses[0], "service_tier") if responses else None,
        cost=result.cost,
    )


class AttributeDict(dict):
    """Legacy output fields remain accessible by key and by attribute."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self, **kwargs):
        return plain(self)


def attributes(obj):
    if isinstance(obj, dict):
        return AttributeDict({key: attributes(item) for key, item in obj.items()})
    if isinstance(obj, list):
        return [attributes(item) for item in obj]
    return obj


def legacy_output(response: Response, *, logprobs=False):
    import json

    # Adapters still consume JSON text; typed callers keep the original DataPart.
    texts = [part.text if isinstance(part, TextPart) else json.dumps(part.value, ensure_ascii=False)
             for part in response.message.parts if isinstance(part, (TextPart, DataPart))]
    thinking = [part.text for part in response.message.parts if isinstance(part, ThinkingPart)]
    output = {"text": "".join(texts) if texts else None}
    if any(thinking):
        output["reasoning_content"] = "".join(thinking)
    if response.tool_calls:
        output["tool_calls"] = [attributes({
            "id": call.id, "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.input)},
        }) for call in response.tool_calls]
    if response.citations:
        output["citations"] = [
            {key: val for key, val in {"text": part.text, "title": part.title, "url": part.url}.items() if val is not None}
            for part in response.citations
        ]
    if logprobs:
        output["logprobs"] = attributes({"content": [
            {"token": token.token, "logprob": token.logprob,
             "bytes": list(token.bytes) if token.bytes is not None else None,
             "top_logprobs": [{"token": top.token, "logprob": top.logprob,
                               "bytes": list(top.bytes) if top.bytes is not None else None} for top in token.top]}
            for token in response.logprobs or ()
        ]})
    return output


def usage_dict(response: Response, provider=None):
    u = response.usage
    inputs, outputs = u.input_tokens, u.output_tokens
    # LiteLLM's public prompt/completion counters include cached Anthropic
    # input and Gemini thinking output. lm15's counters are provider-verbatim.
    if provider and "anthropic" in provider and inputs is not None:
        inputs += (u.cache_read_tokens or 0) + (u.cache_write_tokens or 0)
    if provider in {"gemini", "vertex", "vertex-express", "xai"} and outputs is not None:
        outputs += u.reasoning_tokens or 0
    data = {key: val for key, val in {
        "prompt_tokens": inputs, "completion_tokens": outputs,
        "total_tokens": (inputs + outputs if provider and "anthropic" in provider and inputs is not None and outputs is not None
                         else u.total_tokens),
    }.items() if val is not None}
    prompt_details = {key: val for key, val in {
        "cached_tokens": u.cache_read_tokens, "cache_creation_tokens": u.cache_write_tokens,
        "audio_tokens": u.input_audio_tokens,
    }.items() if val is not None}
    completion_details = {key: val for key, val in {
        "reasoning_tokens": u.reasoning_tokens, "audio_tokens": u.output_audio_tokens,
    }.items() if val is not None}
    if data or prompt_details or completion_details:
        data["prompt_tokens_details"] = prompt_details or None
        data["completion_tokens_details"] = completion_details or None
    return data


@dataclass
class CallResult:
    responses: tuple[Response, ...] = ()
    outputs: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost: float | None = None
    raw: Any = None
    response_model: str | None = None
    cache_hit: bool = False
    cost_details: dict = field(default_factory=dict)

    @classmethod
    def native(cls, response, *, model_type="chat", logprobs=False, provider=None):
        output = legacy_output(response, logprobs=logprobs)
        outputs = [output] if model_type == "responses" or len(output) > 1 else [output["text"]]
        return cls((response,), outputs, usage_dict(response, provider), raw=response, response_model=response.model)

    @classmethod
    def legacy(cls, lm, raw, request=None, *, kwargs=None):
        kwargs = kwargs or {}
        outputs = (lm._process_response(raw) if lm.model_type == "responses"
                   else lm._process_completion(raw, {**lm.kwargs, **kwargs}))
        return cls(outputs=outputs, usage=plain(dict(value(raw, "usage", {}) or {})),
                   cost=getattr(raw, "_hidden_params", {}).get("response_cost"), raw=raw,
                   response_model=value(raw, "model", lm.model))

    def typed(self, request: Request, model_type):
        if self.responses:
            if len(self.responses) != 1:
                raise ValueError("An explicit Request returns exactly one Response")
            return self.responses[0]
        return response_value(self.raw, model_type, request)

    def dump(self):
        return {"_dspy_format": CACHE_FORMAT,
                "responses": [response_to_dict(r, include_provider_data=True) for r in self.responses],
                "outputs": plain(self.outputs), "usage": plain(self.usage), "cost": self.cost,
                "cost_details": plain(self.cost_details),
                "raw": None if self.responses else plain(self.raw), "response_model": self.response_model}

    @classmethod
    def load(cls, record):
        responses = tuple(response_from_dict(r) for r in record["responses"])
        return cls(responses, attributes(record["outputs"]), {}, record["cost"],
                   responses[0] if len(responses) == 1 else attributes(record.get("raw")),
                   record["response_model"], True, record.get("cost_details", {}))

    def provider_response(self):
        if self.raw is not None:
            if not isinstance(self.raw, Response):
                return self.raw
        choices = []
        for index, output in enumerate(self.outputs):
            output = output if isinstance(output, dict) else {"text": output}
            choices.append({"index": index, "finish_reason": "stop", "message": {
                "content": output.get("text"), "tool_calls": output.get("tool_calls"),
                "reasoning_content": output.get("reasoning_content"),
            }, "logprobs": output.get("logprobs")})
        return attributes({"model": self.response_model, "choices": choices, "usage": self.usage,
                           "cache_hit": self.cache_hit})


def combine(results, *, model_type):
    from dspy.utils.usage_tracker import UsageTracker

    tracker = UsageTracker()
    for result in results:
        tracker.add_usage("combined", result.usage)
    outputs = [out for result in results for out in result.outputs]
    if any(isinstance(out, dict) for out in outputs):
        outputs = [out if isinstance(out, dict) else {"text": out} for out in outputs]
    costs = [r.cost for r in results]
    return CallResult(tuple(r for item in results for r in item.responses), outputs,
                      tracker.get_total_tokens().get("combined", {}),
                      sum(costs) if costs and all(c is not None for c in costs) else None,
                      raw=results[0].raw if len(results) == 1 else None,
                      response_model=results[0].response_model if results else None,
                      cost_details={"candidates": [r.cost_details for r in results]} if any(r.cost_details for r in results) else {})
