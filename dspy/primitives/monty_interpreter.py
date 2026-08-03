"""Monty-backed code interpreter for RLM execution."""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
from typing import Any, Callable

from pydantic_monty import (
    AbstractOS,
    Monty,
    MontyCrashedError,
    MontyRuntimeError,
    MontySession,
    MontySyntaxError,
    MountDir,
    ResourceLimits,
)

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:\s*(?:python|py)\s*)?\n(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class MontyInterpreter:
    """Execute persistent Python snippets in Monty's isolated worker pool.

    Monty provides a constrained Python runtime with no filesystem, network,
    or environment access unless those capabilities are explicitly supplied.
    A shared interpreter is safe to use from multiple threads: each thread gets
    an isolated persistent session backed by the same worker pool.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        resource_limits: ResourceLimits | None = None,
        mounts: MountDir | list[MountDir] | None = None,
        os_access: AbstractOS | None = None,
        request_timeout: float | None = 120.0,
        max_processes: int | None = None,
    ) -> None:
        self._tools = dict(tools or {})
        self.output_fields = output_fields
        self._tools_registered = False
        self._resource_limits = resource_limits
        self._mounts = mounts
        self._os_access = os_access
        self._request_timeout = request_timeout
        self._max_processes = max_processes
        self._pool: Monty | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._thread_local = threading.local()
        self._live_sessions: dict[int, MontySession] = {}
        self._closed = False
        self._terminal_error: str | None = None

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return self._tools

    def _ensure_session(self) -> MontySession:
        if self._terminal_error is not None:
            raise CodeInterpreterError(self._terminal_error)
        if self._closed:
            raise CodeInterpreterError("interpreter has been shut down")

        local = self._thread_local
        if getattr(local, "generation", None) != self._generation:
            local.session = None
            local.generation = self._generation
        if getattr(local, "session", None) is not None:
            return local.session

        with self._lock:
            if self._pool is None:
                self._pool = Monty(request_timeout=self._request_timeout, max_processes=self._max_processes)
                self._pool.__enter__()
            pool = self._pool
        try:
            session = pool.checkout(limits=self._resource_limits)
            session.__enter__()
        except Exception as error:
            raise CodeInterpreterError(f"failed to start Monty interpreter: {error}") from error
        local.session = session
        with self._lock:
            self._live_sessions[id(session)] = session
        return session

    def start(self) -> None:
        self._ensure_session()

    @staticmethod
    def _invoke_tool(tool: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        advertised = inspect.signature(tool)
        bound = advertised.bind(*args, **kwargs)
        keyword_wrapper = (
            inspect.isfunction(tool)
            and tool.__code__.co_argcount == 0
            and tool.__code__.co_kwonlyargcount == 0
            and bool(tool.__code__.co_flags & inspect.CO_VARKEYWORDS)
        )
        result = tool(**bound.arguments) if keyword_wrapper else tool(*bound.args, **bound.kwargs)
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
            return loop.run_until_complete(result)
        return result

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute code, retaining guest state until shutdown."""
        match = _CODE_FENCE_RE.match(code)
        if match:
            code = match.group(1)

        printed: list[str] = []
        submissions: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def submit(*args: Any, **kwargs: Any) -> None:
            submissions.append((args, kwargs))

        external_lookup: dict[str, Callable[..., Any]] = {"SUBMIT": submit}
        for name, tool in self.tools.items():
            def invoke(*args: Any, _tool: Callable[..., Any] = tool, **kwargs: Any) -> Any:
                return self._invoke_tool(_tool, *args, **kwargs)

            external_lookup[name] = invoke

        try:
            result = self._ensure_session().feed_run(
                code,
                inputs=variables or None,
                external_lookup=external_lookup,
                print_callback=lambda _stream, value: printed.append(value),
                mount=self._mounts,
                os=self._os_access,
            )
        except MontySyntaxError as error:
            raise SyntaxError(str(error)) from error
        except MontyCrashedError as error:
            self._terminal_error = (
                f"{error}; interpreter state was lost. Create a new interpreter for a fresh session."
            )
            self.shutdown()
            raise CodeInterpreterError(self._terminal_error) from error
        except MontyRuntimeError as error:
            raise CodeExecutionError(error.display("type-msg")) from error

        if submissions:
            args, kwargs = submissions[0]
            return _handle_submit(args, kwargs, self.output_fields)
        if printed:
            return "".join(printed).removesuffix("\n")
        return result

    def shutdown(self) -> None:
        if self._closed:
            return
        with self._lock:
            sessions = list(self._live_sessions.values())
            self._live_sessions.clear()
            pool, self._pool = self._pool, None
            self._generation += 1
            self._closed = True
        for session in sessions:
            try:
                session.__exit__(None, None, None)
            except Exception:
                pass
        if pool is not None:
            pool.__exit__(None, None, None)

    def __enter__(self) -> MontyInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()


def _handle_submit(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_fields: list[dict[str, Any]] | None,
) -> FinalOutput:
    names = [field["name"] for field in output_fields or []]
    if names:
        if args:
            if kwargs or len(args) != len(names):
                raise CodeExecutionError("SUBMIT arguments do not match output fields")
            kwargs = dict(zip(names, args, strict=True))
        elif set(kwargs) != set(names):
            raise CodeExecutionError("SUBMIT arguments do not match output fields")
        return FinalOutput(kwargs)
    if kwargs or len(args) != 1:
        raise CodeExecutionError("SUBMIT requires exactly one positional output")
    return FinalOutput(args[0])
