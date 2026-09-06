from unittest import mock

import pytest

import dspy
from tests.adapters.conftest import format_messages_and_lm_kwargs


def test_two_step_adapter_format_exact_messages_for_simple_signature_with_demo():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TwoStepAdapter(dspy.utils.DummyLM([{"answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter, QA, [{"question": "Q1", "answer": "A1"}], {"question": "Q2"}
    )

    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve tasks based on user input.\n"
            "As input, you will be provided with:\n"
            "1. `question` (str):\n"
            "Your outputs must contain:\n"
            "1. `answer` (str):\n"
            "You should lay out your outputs in detail so that your answer can be understood by "
            "another agent\n"
            "Specific instructions: Given the fields `question`, produce the fields `answer`.",
        },
        {"role": "user", "content": "question: Q1"},
        {"role": "assistant", "content": "answer: A1"},
        {"role": "user", "content": "question: Q2"},
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_format_exact_messages_with_typed_outputs():
    class TypedSignature(dspy.Signature):
        question: str = dspy.InputField()
        count: int = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TwoStepAdapter(dspy.utils.DummyLM([{"count": 1, "answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(adapter, TypedSignature, [], {"question": "Q"})

    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve tasks based on user input.\n"
            "As input, you will be provided with:\n"
            "1. `question` (str):\n"
            "Your outputs must contain:\n"
            "1. `count` (int): \n"
            "2. `answer` (str):\n"
            "You should lay out your outputs in detail so that your answer can be understood by "
            "another agent\n"
            "Specific instructions: Given the fields `question`, produce the fields `count`, "
            "`answer`.",
        },
        {"role": "user", "content": "question: Q"},
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_call():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    program = dspy.Predict(TestSignature)

    mock_main_lm = mock.MagicMock(spec=dspy.LM)
    mock_main_lm.return_value = ["text from main LM"]
    mock_main_lm.kwargs = {"temperature": 1.0}
    mock_main_lm.model = "openai/gpt-4o-mini"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.return_value = [
        """
[[ ## solution ## ]] result
[[ ## answer ## ]] 12
[[ ## completed ## ]]
"""
    ]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"

    dspy.configure(lm=mock_main_lm, adapter=dspy.TwoStepAdapter(extraction_model=mock_extraction_lm))

    result = program(question="What is 5 + 7?")

    assert result.answer == 12

    # main LM call
    mock_main_lm.assert_called_once()
    _, call_kwargs = mock_main_lm.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "1. `question` (str)" in content
    assert "1. `solution` (str)" in content
    assert "2. `answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content

    # extraction LM call
    mock_extraction_lm.assert_called_once()
    _, call_kwargs = mock_extraction_lm.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "text from main LM" in content


@pytest.mark.asyncio
async def test_two_step_adapter_async_call():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    program = dspy.Predict(TestSignature)

    mock_main_lm = mock.MagicMock(spec=dspy.LM)
    mock_main_lm.acall.return_value = ["text from main LM"]
    mock_main_lm.kwargs = {"temperature": 1.0}
    mock_main_lm.model = "openai/gpt-4o-mini"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.acall.return_value = [
        """
[[ ## solution ## ]] result
[[ ## answer ## ]] 12
[[ ## completed ## ]]
"""
    ]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"

    with dspy.context(lm=mock_main_lm, adapter=dspy.TwoStepAdapter(extraction_model=mock_extraction_lm)):
        result = await program.acall(question="What is 5 + 7?")

    assert result.answer == 12

    # main LM call
    mock_main_lm.acall.assert_called_once()
    _, call_kwargs = mock_main_lm.acall.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "1. `question` (str)" in content
    assert "1. `solution` (str)" in content
    assert "2. `answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content

    # extraction LM call
    mock_extraction_lm.acall.assert_called_once()
    _, call_kwargs = mock_extraction_lm.acall.call_args
    assert len(call_kwargs["messages"]) == 2

    # assert first message
    assert call_kwargs["messages"][0]["role"] == "system"
    content = call_kwargs["messages"][0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content

    # assert second message
    assert call_kwargs["messages"][1]["role"] == "user"
    content = call_kwargs["messages"][1]["content"]
    assert "text from main LM" in content


def test_two_step_adapter_parse():
    class ComplexSignature(dspy.Signature):
        input_text: str = dspy.InputField()
        tags: list[str] = dspy.OutputField(desc="List of relevant tags")
        confidence: float = dspy.OutputField(desc="Confidence score")

    first_response = "main LM response"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.return_value = [
        """
        {
            "tags": ["AI", "deep learning", "neural networks"],
            "confidence": 0.87
        }
    """
    ]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"
    adapter = dspy.TwoStepAdapter(mock_extraction_lm)
    dspy.configure(adapter=adapter, lm=mock_extraction_lm)

    result = adapter.parse(ComplexSignature, first_response)

    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


def test_two_step_adapter_parse_errors():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    first_response = "main LM response"

    mock_extraction_lm = mock.MagicMock(spec=dspy.LM)
    mock_extraction_lm.return_value = ["invalid response"]
    mock_extraction_lm.kwargs = {"temperature": 1.0}
    mock_extraction_lm.model = "openai/gpt-4o"

    adapter = dspy.TwoStepAdapter(mock_extraction_lm)

    with pytest.raises(dspy.AdapterParseError, match="Failed to parse response"):
        adapter.parse(TestSignature, first_response)


def _make_mock_lm(
    *,
    model,
    model_type="chat",
    supports_reasoning=False,
    supports_function_calling=False,
    supports_response_schema=False,
    supported_params=None,
    kwargs=None,
    sync_return=None,
    async_return=None,
):
    lm = mock.MagicMock(spec=dspy.LM)
    lm.model = model
    lm.model_type = model_type
    lm.kwargs = kwargs or {"temperature": 1.0}
    lm.supports_reasoning = supports_reasoning
    lm.supports_function_calling = supports_function_calling
    lm.supports_response_schema = supports_response_schema
    lm.supported_params = supported_params if supported_params is not None else set()
    if sync_return is not None:
        lm.return_value = sync_return
    if async_return is not None:
        lm.acall.return_value = async_return
    return lm


def _reasoning_signature():
    class ReasoningSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        reasoning: dspy.Reasoning = dspy.OutputField()

    return ReasoningSignature


@pytest.mark.asyncio
async def test_two_step_adapter_async_call_captures_native_reasoning():
    # Regression: the async path used to bypass `_call_preprocess`/`_call_postprocess`, so
    # `reasoning` stayed in the extractor signature and the extraction LM could not produce
    # it from the main LM's `text` -> `dspy.AdapterParseError`. It must now be populated from
    # the main LM's native `reasoning_content`, matching the sync path.
    signature = _reasoning_signature()

    main_lm = _make_mock_lm(
        model="openai/o3-mini",
        supports_reasoning=True,
        async_return=[{"text": "Paris", "reasoning_content": "I thought about geography."}],
        sync_return=[{"text": "Paris", "reasoning_content": "I thought about geography."}],
    )
    extraction_lm = _make_mock_lm(
        model="openai/gpt-4o-mini",
        async_return=["[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n"],
    )

    program = dspy.Predict(signature)
    with dspy.context(lm=main_lm, adapter=dspy.TwoStepAdapter(extraction_model=extraction_lm)):
        result = await program.acall(question="What is the capital of France?")

    assert result.answer == "Paris"
    assert result.reasoning == "I thought about geography."

    # `reasoning_effort` should be forwarded to the main LM (preprocess side-effect).
    _, main_call_kwargs = main_lm.acall.call_args
    assert main_call_kwargs["reasoning_effort"] == "low"

    # The extraction LM must not be asked to produce `reasoning`; it is stripped via preprocess.
    _, extraction_call_kwargs = extraction_lm.acall.call_args
    extraction_system = extraction_call_kwargs["messages"][0]["content"]
    assert "`answer` (str)" in extraction_system
    assert "`reasoning`" not in extraction_system


@pytest.mark.asyncio
async def test_two_step_adapter_async_call_non_reasoning_lm_keeps_reasoning_in_extractor():
    # When the main LM is not reasoning-capable, `reasoning` must NOT be stripped (no native
    # reasoning feature); the extraction LM produces it from the main LM's text.
    signature = _reasoning_signature()

    main_lm = _make_mock_lm(
        model="openai/gpt-4o-mini",
        supports_reasoning=False,
        async_return=[{"text": "Paris, because of geography."}],
    )
    extraction_lm = _make_mock_lm(
        model="openai/gpt-4o-mini",
        async_return=[
            "[[ ## answer ## ]]\nParis\n\n[[ ## reasoning ## ]]\nBecause of geography.\n\n[[ ## completed ## ]]\n"
        ],
    )

    program = dspy.Predict(signature)
    with dspy.context(lm=main_lm, adapter=dspy.TwoStepAdapter(extraction_model=extraction_lm)):
        result = await program.acall(question="What is the capital of France?")

    assert result.answer == "Paris"
    assert result.reasoning == "Because of geography."

    # No native reasoning -> reasoning_effort should not be forwarded to the main LM.
    _, main_call_kwargs = main_lm.acall.call_args
    assert "reasoning_effort" not in main_call_kwargs

    # The extraction LM must still be asked to produce `reasoning`.
    _, extraction_call_kwargs = extraction_lm.acall.call_args
    extraction_system = extraction_call_kwargs["messages"][0]["content"]
    assert "`answer` (str)" in extraction_system
    assert "`reasoning` (str)" in extraction_system


@pytest.mark.asyncio
async def test_two_step_adapter_async_call_gpt5_chat_keeps_reasoning_in_extractor():
    # GPT-5* on the chat completion API keeps `reasoning` in the extractor signature (Litellm
    # caveat in `Reasoning.adapt_to_native_lm_feature`). The async preprocess path must honor
    # that caveat identically to the sync path rather than over-stripping `reasoning`.
    signature = _reasoning_signature()

    main_lm = _make_mock_lm(
        model="openai/gpt-5",
        model_type="chat",
        supports_reasoning=True,
        async_return=[{"text": "Paris, because of geography."}],
    )
    extraction_lm = _make_mock_lm(
        model="openai/gpt-4o-mini",
        async_return=[
            "[[ ## answer ## ]]\nParis\n\n[[ ## reasoning ## ]]\nBecause of geography.\n\n[[ ## completed ## ]]\n"
        ],
    )

    program = dspy.Predict(signature)
    with dspy.context(lm=main_lm, adapter=dspy.TwoStepAdapter(extraction_model=extraction_lm)):
        result = await program.acall(question="What is the capital of France?")

    assert result.answer == "Paris"
    assert result.reasoning == "Because of geography."

    # No reasoning_effort forwarded (gpt-5 chat caveat skips native reasoning).
    _, main_call_kwargs = main_lm.acall.call_args
    assert "reasoning_effort" not in main_call_kwargs

    # `reasoning` stays in the extractor signature for gpt-5 chat.
    _, extraction_call_kwargs = extraction_lm.acall.call_args
    extraction_system = extraction_call_kwargs["messages"][0]["content"]
    assert "`reasoning` (str)" in extraction_system
