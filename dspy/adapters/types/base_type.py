import json
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin

import pydantic

from dspy.clients.base_lm import BaseLM

if TYPE_CHECKING:
    from litellm import ModelResponseStream

    from dspy.signatures.signature import Signature

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Examples:

        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        """Description of the custom type"""
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation):
        """Extract all custom types from the annotation.

        This is used to extract all custom types from the annotation of a field, while the annotation can
        have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
        """
        # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
        # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass

        origin = get_origin(annotation)
        if origin is None:
            return []

        result = []
        # Recurse into all type args
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))

        return result

    @pydantic.model_serializer()
    def serialize_model(self):
        formatted = self.format()
        if isinstance(formatted, list):
            return (
                f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(formatted, ensure_ascii=False)}{CUSTOM_TYPE_END_IDENTIFIER}"
            )
        return formatted

    @classmethod
    def adapt_to_native_lm_feature(
        cls,
        signature: type["Signature"],
        field_name: str,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
    ) -> type["Signature"]:
        """Adapt the custom type to the native LM feature if possible.

        When the LM and configuration supports the related native LM feature, e.g., native tool calling, native
        reasoning, etc., we adapt the signature and `lm_kwargs` to enable the native LM feature.

        Args:
            signature: The DSPy signature for the LM call.
            field_name: The name of the field in the signature to adapt to the native LM feature.
            lm: The LM instance.
            lm_kwargs: The keyword arguments for the LM call, subject to in-place updates if adaptation if required.

        Returns:
            The adapted signature. If the custom type is not natively supported by the LM, return the original
            signature.
        """
        return signature

    @classmethod
    def is_streamable(cls) -> bool:
        """Whether the custom type is streamable."""
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk: "ModelResponseStream") -> Optional["Type"]:
        """
        Parse a stream chunk into the custom type.

        Args:
            chunk: A stream chunk.

        Returns:
            A custom type object or None if the chunk is not for this custom type.
        """
        return None

    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Type"]:
        """Parse a LM response into the custom type.

        Args:
            response: A LM response.

        Returns:
            A custom type object.
        """
        return None
