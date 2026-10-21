import functools
from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

if TYPE_CHECKING:
    from dspy.utils.execution import Invocation

P = ParamSpec("P")
T = TypeVar("T")

ACTIVE_CALL_ID: ContextVar[str | None] = ContextVar("active_call_id", default=None)
ACTIVE_INVOCATION: ContextVar["Invocation | None"] = ContextVar("active_invocation", default=None)
_ACTIVE_COMPILES = ContextVar("active_compiles", default=())


def _bind_active_call_id(fn: Callable[P, T]) -> Callable[P, T]:
    """Bind execution identity and parentage to each synchronous worker call."""
    parent_call_id = ACTIVE_CALL_ID.get()
    parent_invocation = ACTIVE_INVOCATION.get()

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        token = ACTIVE_CALL_ID.set(parent_call_id)
        invocation_token = ACTIVE_INVOCATION.set(parent_invocation)
        try:
            return fn(*args, **kwargs)
        finally:
            ACTIVE_INVOCATION.reset(invocation_token)
            ACTIVE_CALL_ID.reset(token)

    return wrapper
