"""Regression tests for async tool execution from ``PythonInterpreter``.

These tests exercise ``_await_in_sync`` and ``PythonInterpreter.invoke_tool``
directly. They never start the Deno/Pyodide subprocess (``invoke_tool`` only
needs ``self.tools``), so they run in the default CI job without the
``--deno`` flag.

The key scenario they lock down is the one that regressed in the
``RLM.aforward()`` entrypoint: a tool call reaches ``invoke_tool`` ->
``_await_in_sync`` while the host event loop is already running. The previous
implementation called ``loop.run_until_complete(coroutine)`` on that loop,
which CPython rejects with ``RuntimeError: This event loop is already
running`` and leaks the coroutine.
"""

import asyncio
import warnings

import pytest

from dspy.primitives.python_interpreter import PythonInterpreter, _await_in_sync


def _make_interpreter(tools: dict[str, object]) -> PythonInterpreter:
    """Build a PythonInterpreter without starting the Deno subprocess.

    ``invoke_tool`` only needs ``self.tools``; it never touches the subprocess,
    so the exact ``RLM.aforward -> invoke_tool -> _await_in_sync`` path can be
    exercised without Deno installed.
    """
    return PythonInterpreter(deno_command=["deno"], tools=tools)


# =============================================================================
# Running event loop (RLM.aforward path) — the bug scenario
# =============================================================================


@pytest.mark.asyncio
async def test_await_in_sync_inside_running_loop():
    """A coroutine awaited via _await_in_sync from inside a running loop resolves.

    Before the fix this raised ``RuntimeError: This event loop is already running``.
    """

    async def coro():
        await asyncio.sleep(0)
        return "value"

    assert _await_in_sync(coro()) == "value"


@pytest.mark.asyncio
async def test_await_in_sync_inside_running_loop_preserves_exception_type():
    """The original exception type propagates (not wrapped in RuntimeError)."""

    async def coro():
        await asyncio.sleep(0)
        raise ValueError("boom:7")

    with pytest.raises(ValueError, match="boom:7"):
        _await_in_sync(coro())


@pytest.mark.asyncio
async def test_await_in_sync_inside_running_loop_no_coroutine_leak_on_success():
    """A successfully resolved coroutine leaves no 'never awaited' warning."""

    async def coro():
        await asyncio.sleep(0)
        return "ok"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _await_in_sync(coro())

    assert not any("was never awaited" in str(w.message) for w in caught)


@pytest.mark.asyncio
async def test_await_in_sync_inside_running_loop_no_coroutine_leak_on_error():
    """The coroutine is closed on failure, so no 'never awaited' RuntimeWarning leaks."""

    async def coro():
        await asyncio.sleep(0)
        raise ValueError("boom:7")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            _await_in_sync(coro())

    assert not any("was never awaited" in str(w.message) for w in caught), (
        f"coroutine leaked: {[str(w.message) for w in caught]}"
    )


# =============================================================================
# PythonInterpreter.invoke_tool: the real call chain
# =============================================================================


@pytest.mark.asyncio
async def test_invoke_tool_async_in_running_loop():
    """invoke_tool awaits an async tool while the host loop is running (the bug)."""

    async def score(payload, factor: int = 2):
        await asyncio.sleep(0)
        return payload["value"] * factor

    interp = _make_interpreter({"score": score})
    assert interp.invoke_tool("score", {"payload": {"value": 21}, "factor": 3}) == 63


def test_invoke_tool_async_no_running_loop():
    """invoke_tool awaits an async tool when no loop is running (RLM.forward path)."""

    async def slow_search(query: str):
        await asyncio.sleep(0)
        return f"answer:{query}"

    interp = _make_interpreter({"slow_search": slow_search})
    assert interp.invoke_tool("slow_search", {"query": "hello"}) == "answer:hello"


def test_invoke_tool_sync_function_unchanged():
    """Sync tools return directly; _await_in_sync is never consulted."""

    def add(a: int, b: int) -> str:
        return f"{a + b}"

    interp = _make_interpreter({"add": add})
    assert interp.invoke_tool("add", {"a": 1, "b": 2}) == "3"
