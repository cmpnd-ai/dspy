from typing import TYPE_CHECKING, Any

from dspy.adapters.types.citation import Citations
from dspy.adapters.types.document import Document

if TYPE_CHECKING:
    from dspy.experimental.monty import MontyProgram

__all__ = [
    "Citations",
    "Document",
    "MontyProgram",
]


def __getattr__(name: str) -> Any:
    if name == "MontyProgram":
        from dspy.experimental.monty import MontyProgram

        return MontyProgram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
