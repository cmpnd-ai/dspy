from typing import Any, Callable

import pydantic

from dspy.dsp.utils import settings
from dspy.predict.predict import Predict, Prediction
from dspy.signatures.signature import InputField, OutputField, Signature


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

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def __init__(self, *args: Any, compact_if_needed: Callable[["History"], "History"] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.compact_if_needed = compact_if_needed or self._default_compact_if_needed

    #NOTE: We assume that whatever is being called here is the lm that will be used to summarize.
    def _default_compact_if_needed(self: "History") -> "History":
        return self.__deepcopy__()

    def add_message(self, signature: type[Signature], inputs: dict[str, Any], prediction: Prediction, tool_observations: list[tuple[str, bool]]):
        # WHAT IS MESSAGES AHH
        pass






def estimate_tokens(text: str, model: str) -> int:
    # Do we want to use a worse method and avoid the dependency?
    # TODO: Add a dependency on tiktoken
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def summarize_if_needed(history: History, max_tokens: int = 200000, summarizer: Predict = Predict(SummarizationSignature)) -> History:
    # If someone wants to optimize the compaction w an optimizeable signature, how would they specify this in ReActV2?
    class SummarizationSignature(Signature):
        """Given the below conversation history, generate a summary that would be helpful to continue the conversation""" # TODO: look at the CC compaction prompt
        history: History = InputField()
        summary: str = OutputField()

    token_count = estimate_tokens(str(history.messages), settings.lm.model_name)
    if token_count > max_tokens:
        return History(messages=[{"summary": summarizer(history)}])
    return history

def truncate_if_needed(history: History, max_tokens: int = 200000) -> History:
    token_count = estimate_tokens(str(history.messages), settings.lm.model_name)
    messages = history.model_copy(update={"messages": []}).messages
    while token_count > max_tokens:
        if len(messages) == 0:
            raise ValueError(f"History is too long to truncate: {token_count} > {max_tokens}. Consider using a larger max_tokens or a different compaction strategy.")
        messages.pop()
        token_count = estimate_tokens(str(messages), settings.lm.model_name)
    return history.model_copy(update={"messages": messages})
