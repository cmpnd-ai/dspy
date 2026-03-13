import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import pydantic

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.lm import LM


class HistoryCompaction(pydantic.BaseModel):
    max_visible_tokens: int = pydantic.Field(gt=0)
    keep_last_messages: int = pydantic.Field(gt=0)

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


_DEFAULT_RAW_HISTORY_COMPACTION = HistoryCompaction(max_visible_tokens=8_000, keep_last_messages=8)
_SUMMARY_PREFIX = "Summary of earlier conversation:\n"


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Examples:
        ```
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]
    mode: Literal["demo", "raw"] = "demo"
    compaction: HistoryCompaction | None = None
    summary: str | None = None
    compacted_count: int = 0

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    @classmethod
    def demo(cls, messages: list[dict[str, Any]] | None = None) -> "History":
        return cls(messages=messages or [], mode="demo")

    @classmethod
    def raw(
        cls,
        messages: list[dict[str, Any]] | None = None,
        compaction: HistoryCompaction | None = None,
    ) -> "History":
        compaction = compaction or _DEFAULT_RAW_HISTORY_COMPACTION
        return cls(messages=messages or [], mode="raw", compaction=compaction)

    def visible_messages(self) -> list[dict[str, Any]]:
        if self.compacted_count == 0:
            return copy.deepcopy(self.messages)

        visible_messages = []
        if self.summary is not None:
            visible_messages.append({"role": "user", "content": f"{_SUMMARY_PREFIX}{self.summary}"})
        visible_messages.extend(self.messages[self.compacted_count :])
        return copy.deepcopy(visible_messages)

    def with_messages(
        self,
        new_messages: list[dict[str, Any]],
        *,
        lm: "LM | None",
        adapter: "Adapter | None",
    ) -> "History":
        updated = type(self)(
            messages=[*self.messages, *new_messages],
            mode=self.mode,
            compaction=self.compaction,
            summary=self.summary,
            compacted_count=self.compacted_count,
        )
        return updated._maybe_compact(lm=lm, adapter=adapter)

    def _maybe_compact(self, *, lm: "LM | None", adapter: "Adapter | None") -> "History":
        if self.mode != "raw" or self.compaction is None:
            return self

        visible_tokens = self._count_visible_tokens(self.visible_messages(), lm)
        if visible_tokens is None or visible_tokens <= self.compaction.max_visible_tokens:
            return self

        new_compacted_count = len(self.messages) - self.compaction.keep_last_messages
        if new_compacted_count <= self.compacted_count:
            return self

        hidden_messages: list[dict[str, Any]] = []
        if self.summary is not None:
            hidden_messages.append({"role": "user", "content": f"{_SUMMARY_PREFIX}{self.summary}"})
        hidden_messages.extend(self.messages[self.compacted_count : new_compacted_count])

        if not hidden_messages or lm is None or adapter is None:
            return self

        summary = self._summarize_prefix(hidden_messages, lm=lm, adapter=adapter)
        return type(self)(
            messages=self.messages,
            mode=self.mode,
            compaction=self.compaction,
            summary=summary,
            compacted_count=new_compacted_count,
        )

    def _count_visible_tokens(self, messages: list[dict[str, Any]], lm: "LM | None") -> int | None:
        if lm is None or getattr(lm, "model", None) is None:
            return None

        try:
            from litellm.utils import token_counter

            return token_counter(model=lm.model, messages=messages)
        except Exception:
            return None

    def _summarize_prefix(
        self,
        hidden_messages: list[dict[str, Any]],
        *,
        lm: "LM",
        adapter: "Adapter",
    ) -> str:
        compacted_history = type(self)(messages=hidden_messages, mode="raw", compaction=None)

        import dspy

        with dspy.context(lm=lm, adapter=adapter, trace=[]):
            return _get_history_summarizer()(history=compacted_history).summary

    @pydantic.model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        data: dict[str, Any] = {"messages": self.messages}
        if self.mode != "demo":
            data["mode"] = self.mode
        if self.mode == "raw" and self.compaction is not None:
            data["compaction"] = self.compaction.model_dump()
        if self.summary is not None:
            data["summary"] = self.summary
        if self.compacted_count:
            data["compacted_count"] = self.compacted_count
        return data


@lru_cache(maxsize=1)
def _get_history_summarizer():
    import dspy

    class SummarizeHistory(dspy.Signature):
        """Summarize the earlier conversation faithfully for future continuation.

        Preserve concrete facts, tool results, unresolved questions, and any relevant state.
        Keep the summary concise.
        """

        history: History = dspy.InputField()
        summary: str = dspy.OutputField()

    return dspy.Predict(SummarizeHistory)
