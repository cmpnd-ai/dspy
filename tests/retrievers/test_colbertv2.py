from unittest.mock import MagicMock, patch

import pytest

from dspy.dsp.colbertv2 import ColBERTv2, colbertv2_get_request_v2, colbertv2_post_request_v2


def test_get_request_raises_on_server_error():
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": True, "message": "connection failed"}

    with patch("dspy.dsp.colbertv2.requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="connection failed"):
            colbertv2_get_request_v2("http://test", "query", k=3)


def test_post_request_raises_on_server_error():
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": True, "message": "server error"}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        with pytest.raises(ValueError, match="server error"):
            colbertv2_post_request_v2("http://test2", "query", k=3)


def test_get_request_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}]}

    with patch("dspy.dsp.colbertv2.requests.get", return_value=mock_response):
        result = colbertv2_get_request_v2("http://test3", "query", k=3)
        assert result[0]["long_text"] == "doc1"


def test_post_request_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}]}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        result = colbertv2_post_request_v2("http://test4", "query", k=3)
        # The canonical ColBERTv2 server returns `text`-keyed passages; the wrapper mirrors the
        # GET path and additionally aliases `text` -> `long_text` for downstream consumers.
        assert result[0]["text"] == "doc1"
        assert result[0]["long_text"] == "doc1"


def test_post_request_adds_long_text_alias_like_get():
    """POST must add the same `long_text` alias the GET path adds (regression guard for the
    asymmetry where only GET aliased `text` -> `long_text`)."""
    server_response = {"topk": [{"text": "doc1", "score": 0.9}, {"text": "doc2", "score": 0.8}]}
    post_mock = MagicMock()
    post_mock.json.return_value = server_response
    get_mock = MagicMock()
    get_mock.json.return_value = server_response

    with patch("dspy.dsp.colbertv2.requests.post", return_value=post_mock):
        post_result = colbertv2_post_request_v2("http://test5", "query", k=3)

    with patch("dspy.dsp.colbertv2.requests.get", return_value=get_mock):
        get_result = colbertv2_get_request_v2("http://test6", "query", k=3)

    assert [d["long_text"] for d in post_result] == ["doc1", "doc2"]
    assert post_result == get_result


def test_colbertv2_call_post_simplify_returns_long_text_strings():
    """ColBERTv2(post_requests=True)(..., simplify=True) reads psg['long_text'] and must not
    raise KeyError (the failure reported when the POST path omitted the alias)."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"topk": [{"text": "doc1", "score": 0.9}, {"text": "doc2", "score": 0.8}]}

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        rm = ColBERTv2(url="http://test8", post_requests=True)
        passages = rm("query", k=3, simplify=True)

    assert passages == ["doc1", "doc2"]


def test_retrieve_with_post_requests_returns_passages_via_long_text():
    """End-to-end: Retrieve with rm=ColBERTv2(post_requests=True) must not raise
    AttributeError on `psg.long_text` (propagates through @with_callbacks)."""
    import dspy
    from dspy.retrievers.retrieve import Retrieve

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "topk": [
            {"text": "doc1", "score": 0.9},
            {"text": "doc2", "score": 0.8},
            {"text": "doc3", "score": 0.7},
        ]
    }

    with patch("dspy.dsp.colbertv2.requests.post", return_value=mock_response):
        dspy.settings.configure(rm=ColBERTv2(url="http://test10", post_requests=True))
        prediction = Retrieve(k=3)("query")

    assert prediction.passages == ["doc1", "doc2", "doc3"]
