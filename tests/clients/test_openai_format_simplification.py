import pytest

from dspy.clients.openai_format import (
    common_config_kwargs,
    output_audio_to_part,
    output_file_to_part,
    output_image_to_part,
    responses_config_kwargs,
    text_config_kwargs,
    tool_to_openai,
    tool_to_openai_responses,
)
from dspy.core.types import LMConfig, LMToolSpec


def test_config_mappers_share_values_but_keep_endpoint_specific_policy():
    config = LMConfig(
        temperature=1,
        max_tokens=17,
        top_p=0.75,
        stop=[],
        n=2,
        logprobs=False,
        response_format={"type": "json_object"},
        reasoning={"effort": "low", "summary": "auto"},
        extensions={"temperature": 99, "stop": ["extension-stop"], "text": {"verbosity": "low"}},
    )

    chat = common_config_kwargs(config, model="gpt-5-mini")
    responses = responses_config_kwargs(config, model="gpt-4.1")
    text = text_config_kwargs(config)

    for result in (chat, responses, text):
        assert result["temperature"] == 1
        assert result["top_p"] == 0.75
        assert result["n"] == 2
        assert result["logprobs"] is False
        # A falsey configured stop does not replace an extension value.
        assert result["stop"] == ["extension-stop"]
    assert chat["max_completion_tokens"] == 17
    assert chat["response_format"] == {"type": "json_object"}
    assert chat["reasoning_effort"] == "low"
    assert responses["max_output_tokens"] == 17
    assert responses["reasoning"] == {"effort": "low", "summary": "auto"}
    assert responses["text"] == {"verbosity": "low", "format": {"type": "json_object"}}
    assert text["max_tokens"] == 17
    assert "reasoning" not in text and "response_format" not in text


def test_tool_dialects_wrap_one_protected_function_representation():
    tool = LMToolSpec(
        name="canonical",
        description="description",
        parameters={"type": "object"},
        strict=False,
        provider_data={
            "type": "custom",
            "name": "override",
            "description": "override",
            "parameters": {"type": "string"},
            "strict": True,
            "vendor": "kept",
        },
    )
    function = {
        "name": "canonical",
        "description": "description",
        "parameters": {"type": "object"},
        "strict": False,
        "vendor": "kept",
    }
    assert tool_to_openai(tool) == {"type": "function", "function": function}
    assert tool_to_openai_responses(tool) == {"type": "function", **function}


@pytest.mark.parametrize(
    ("converter", "value", "expected_type", "expected_data", "extra"),
    [
        (
            output_image_to_part,
            {
                "b64_json": "data:image/webp;base64,aW1hZ2U=",
                "image_url": {"url": "https://nested/image"},
                "url": "https://flat/image",
                "file_id": "image-id",
                "detail": "high",
            },
            "image/webp",
            "aW1hZ2U=",
            {"detail": "high"},
        ),
        (
            output_audio_to_part,
            {"audio": {"data": "data:audio/mpeg;base64,YXVkaW8=", "url": "https://audio", "file_id": "audio-id"}},
            "audio/mpeg",
            "YXVkaW8=",
            {},
        ),
        (
            output_file_to_part,
            {
                "file": {
                    "file_data": "data:text/plain;base64,ZmlsZQ==",
                    "url": "https://file",
                    "file_id": "file-id",
                    "filename": "result.txt",
                }
            },
            "text/plain",
            "ZmlsZQ==",
            {"filename": "result.txt"},
        ),
    ],
)
def test_output_media_data_uri_has_priority_and_preserves_wrapper_fields(
    converter, value, expected_type, expected_data, extra
):
    part = converter(value)
    assert part.data == expected_data
    assert part.url is None and part.file_id is None
    assert part.media_type == expected_type
    for key, expected in extra.items():
        assert getattr(part, key) == expected


@pytest.mark.parametrize(
    ("converter", "url_value", "id_value", "default_type"),
    [
        (
            output_image_to_part,
            {"image_url": {"url": "https://image"}, "file_id": "ignored-id"},
            {"file_id": "image-id"},
            "image/png",
        ),
        (
            output_audio_to_part,
            {"audio": {"url": "https://audio", "file_id": "ignored-id"}},
            {"audio": {"file_id": "audio-id"}},
            "audio/wav",
        ),
        (
            output_file_to_part,
            {"file": {"url": "https://file", "file_id": "ignored-id"}},
            {"file": {"id": "file-id"}},
            "application/octet-stream",
        ),
    ],
)
def test_output_media_url_then_file_id_priority_and_defaults(converter, url_value, id_value, default_type):
    url_part = converter(url_value)
    assert url_part.url.startswith("https://")
    assert url_part.file_id is None
    assert url_part.media_type == default_type

    id_part = converter(id_value)
    assert id_part.file_id.endswith("-id")
    assert id_part.media_type == default_type


@pytest.mark.parametrize(
    ("converter", "kind"),
    [(output_image_to_part, "image"), (output_audio_to_part, "audio"), (output_file_to_part, "file")],
)
def test_output_media_missing_source_errors_are_specific(converter, kind):
    with pytest.raises(ValueError, match=rf"Provider {kind} output did not include data, url, or file_id\."):
        converter({})


@pytest.mark.parametrize(
    ("converter", "empty_data"),
    [
        (output_image_to_part, {"data": ""}),
        (output_audio_to_part, {"b64_json": ""}),
        (output_file_to_part, {"data": ""}),
    ],
)
def test_explicit_empty_data_is_rejected_instead_of_falling_back_to_url(converter, empty_data):
    with pytest.raises(ValueError, match="data must be non-empty"):
        converter({**empty_data, "url": "https://fallback.example/file"})


def test_file_output_empty_file_id_uses_item_id():
    assert output_file_to_part({"file_id": "", "id": "valid-item"}).file_id == "valid-item"
