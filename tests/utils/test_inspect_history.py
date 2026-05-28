"""Tests for dspy.utils.inspect_history.pretty_print_history."""

import io

from dspy.utils.inspect_history import pretty_print_history


def _make_history(messages, outputs=None):
    """Helper to build a minimal history entry."""
    if outputs is None:
        outputs = [{"text": "ok", "tool_calls": None}]
    return [{"messages": messages, "outputs": outputs, "timestamp": "2025-01-01"}]


def test_none_content_does_not_crash():
    """Assistant messages with content=None (native FC) must not crash."""
    history = _make_history(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q":"test"}'}}
                ],
            },
        ]
    )
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "Assistant message:" in output
    assert "search" in output


def test_tool_calls_displayed_on_assistant_message():
    """tool_calls attached to an assistant message should be printed."""
    history = _make_history(
        [
            {"role": "user", "content": "find something"},
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": '{"query":"dspy"}'}}
                ],
            },
        ]
    )
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "Tool calls:" in output
    assert "web_search" in output
    assert '{"query":"dspy"}' in output


def test_tool_call_id_shown_on_tool_message():
    """Tool role messages with tool_call_id should display it."""
    history = _make_history(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "tool",
                "content": "result data",
                "tool_call_id": "call_abc123",
            },
        ]
    )
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "tool_call_id=call_abc123" in output
    assert "result data" in output


def test_assistant_message_without_content_key():
    """Assistant messages may have tool_calls but no 'content' key at all."""
    history = _make_history(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": '{"id":42}'}}
                ],
            },
        ]
    )
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "lookup" in output


def test_regular_messages_still_work():
    """Normal user/assistant string messages render correctly."""
    history = _make_history(
        [
            {"role": "user", "content": "What is DSPy?"},
            {"role": "assistant", "content": "DSPy is a framework."},
        ]
    )
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "User message:" in output
    assert "What is DSPy?" in output
    assert "Assistant message:" in output
    assert "DSPy is a framework." in output
