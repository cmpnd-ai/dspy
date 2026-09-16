import warnings

import pytest
from litellm.utils import Choices, Message, ModelResponse

import dspy
from dspy.utils.usage_tracker import track_usage


def _provider_response(text="hello"):
    return ModelResponse(
        choices=[Choices(message=Message(role="assistant", content=text), finish_reason="stop")],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        model="provider-model",
    )


class _LegacyLM(dspy.BaseLM):
    forward_contract = "legacy"

    def __init__(self):
        super().__init__("request-model", temperature=0.3)
        self.calls = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls.append(("sync", prompt, messages, kwargs))
        return _provider_response()

    async def aforward(self, prompt=None, messages=None, **kwargs):
        self.calls.append(("async", prompt, messages, kwargs))
        return _provider_response()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sync", "async"])
async def test_legacy_conversion_transport_uses_shared_arguments_and_accounts_once(transport):
    lm = _LegacyLM()
    request = dspy.LMRequest.from_call(model="request-model", prompt="question", temperature=0.7, cache=False)

    with track_usage() as usage_tracker:
        response = lm(request) if transport == "sync" else await lm.acall(request)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "hello"
    assert response.model == "provider-model"
    assert lm.calls == [(transport, "question", None, {"temperature": 0.7, "cache": False})]
    assert len(lm.history) == 1
    assert lm.history[0].request == request
    assert lm.history[0].response == response
    assert len(usage_tracker.usage_data) == 1
    usage = usage_tracker.get_total_tokens()["request-model"]
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sync", "async"])
async def test_inherited_legacy_contract_typed_result_preserves_shape_and_warning_location(transport):
    class UndeclaredLegacyLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return dspy.LMResponse.from_text("typed", model=self.model)

        async def aforward(self, prompt=None, messages=None, **kwargs):
            return dspy.LMResponse.from_text("typed", model=self.model)

    lm = UndeclaredLegacyLM("model")
    request = dspy.LMRequest.from_call(model="model", prompt="question")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = lm(request) if transport == "sync" else await lm.acall(request)

    assert response.text == "typed"
    assert len(lm.history) == 1
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    # The added finalization frame must not move the warning from the same callback boundary used before this refactor.
    assert caught[0].filename.endswith("dspy/utils/callback.py")
