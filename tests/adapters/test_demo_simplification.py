import dspy
from dspy.adapters.chat_adapter import ChatAdapter


def test_complete_demos_preserve_custom_adapter_default_prefix():
    class CustomAdapter(ChatAdapter):
        def format_user_message_content(self, signature, inputs, prefix="custom default", **kwargs):
            return super().format_user_message_content(signature, inputs, prefix=prefix, **kwargs)

    signature = dspy.Signature("question, context -> answer")
    messages = CustomAdapter().format_demos(
        signature,
        [
            {"question": "complete", "context": "context", "answer": "answer"},
            {"question": "incomplete", "answer": "partial"},
            {"question": "no output"},
        ],
    )
    assert len(messages) == 4
    assert messages[0]["content"].startswith("This is an example of the task")
    assert "incomplete" in messages[0]["content"]
    assert messages[2]["content"].startswith("custom default")
    assert "complete" in messages[2]["content"]
