import pickle

from dspy.core.history import LMHistoryEntry as ExtractedLMHistoryEntry
from dspy.core.types import (
    Assistant,
    LMAudioPart,
    LMDocumentPart,
    LMHistoryEntry,
    LMImagePart,
    LMRequest,
    LMResponse,
    User,
)


def test_types_keeps_history_entry_compatibility_export_and_legacy_pickle_global():
    assert LMHistoryEntry is ExtractedLMHistoryEntry
    assert pickle.loads(b"cdspy.core.types\nLMHistoryEntry\n.") is ExtractedLMHistoryEntry


def test_history_rendering_preserves_media_path_without_reading_it(monkeypatch, tmp_path):
    media_path = tmp_path / "must-not-be-read.png"

    def fail_read(*args, **kwargs):
        raise AssertionError("history rendering must not read media files")

    monkeypatch.setattr(type(media_path), "read_bytes", fail_read)
    request = LMRequest(model="model", messages=[User(LMImagePart(path=str(media_path)))])
    entry = LMHistoryEntry(
        request=request,
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

    assert entry.messages == [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": str(media_path)}}]}
    ]


def test_history_keeps_empty_document_labels_and_legacy_audio_data():
    entry = LMHistoryEntry(
        request=LMRequest(
            model="model",
            messages=[
                User(
                    LMDocumentPart(url="https://example.com/report.pdf", title="", context=""),
                    LMAudioPart(data="data:legacy-without-comma"),
                )
            ],
        ),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )
    assert entry.messages[0]["content"] == [
        {
            "type": "document",
            "source": "https://example.com/report.pdf",
            "media_type": "application/pdf",
            "title": "",
            "context": "",
        },
        {"type": "input_audio", "input_audio": {"format": "octet-stream", "data": "data:legacy-without-comma"}},
    ]


def test_history_preserves_role_specific_empty_content():
    entry = LMHistoryEntry(
        request=LMRequest(model="model", messages=[User(), Assistant()]),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )
    assert entry.messages == [{"role": "user", "content": []}, {"role": "assistant", "content": None}]
