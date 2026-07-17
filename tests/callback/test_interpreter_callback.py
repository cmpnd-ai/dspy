"""Tests for interpreter-level callbacks (`on_interpreter_*`).

Test organization mirrors tests/primitives/test_python_interpreter.py:
- Unit tests (no Deno required): MockInterpreter, PythonInterpreter tool-dispatch seam,
  dispatch-routing integrity.
- Integration tests (@pytest.mark.deno): real PythonInterpreter execute/tool/lifecycle events
  and a full dspy.RLM run.
"""

import json

import pytest

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives import python_interpreter
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter


@pytest.fixture(autouse=True)
def reset_settings():
    original_settings = dspy.settings.copy()
    yield
    dspy.configure(**original_settings)


class RecordingCallback(BaseCallback):
    """Records every handler invocation, capturing the active (parent) call id at call time."""

    def __init__(self):
        self.calls = []

    def _record(self, handler, call_id, **kwargs):
        self.calls.append({
            "handler": handler,
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            **kwargs,
        })

    # Module / LM / Tool handlers (used to assert dispatch integrity).
    def on_module_start(self, call_id, instance, inputs):
        self._record("on_module_start", call_id, instance=instance, inputs=inputs)

    def on_module_end(self, call_id, outputs, exception):
        self._record("on_module_end", call_id, outputs=outputs, exception=exception)

    def on_lm_start(self, call_id, instance, inputs):
        self._record("on_lm_start", call_id, instance=instance, inputs=inputs)

    def on_lm_end(self, call_id, outputs, exception):
        self._record("on_lm_end", call_id, outputs=outputs, exception=exception)

    def on_tool_start(self, call_id, instance, inputs):
        self._record("on_tool_start", call_id, instance=instance, inputs=inputs)

    def on_tool_end(self, call_id, outputs, exception):
        self._record("on_tool_end", call_id, outputs=outputs, exception=exception)

    # Interpreter handlers.
    def on_interpreter_execute_start(self, call_id, instance, inputs):
        self._record("on_interpreter_execute_start", call_id, instance=instance, inputs=inputs)

    def on_interpreter_execute_end(self, call_id, outputs, exception):
        self._record("on_interpreter_execute_end", call_id, outputs=outputs, exception=exception)

    def on_interpreter_tool_call_start(self, call_id, instance, inputs):
        self._record("on_interpreter_tool_call_start", call_id, instance=instance, inputs=inputs)

    def on_interpreter_tool_call_end(self, call_id, outputs, exception):
        self._record("on_interpreter_tool_call_end", call_id, outputs=outputs, exception=exception)

    def on_interpreter_startup_start(self, call_id, instance, inputs):
        self._record("on_interpreter_startup_start", call_id, instance=instance, inputs=inputs)

    def on_interpreter_startup_end(self, call_id, outputs, exception):
        self._record("on_interpreter_startup_end", call_id, outputs=outputs, exception=exception)

    def on_interpreter_shutdown_start(self, call_id, instance, inputs):
        self._record("on_interpreter_shutdown_start", call_id, instance=instance, inputs=inputs)

    def on_interpreter_shutdown_end(self, call_id, outputs, exception):
        self._record("on_interpreter_shutdown_end", call_id, outputs=outputs, exception=exception)

    def handlers(self):
        return [c["handler"] for c in self.calls]

    def by_handler(self, handler):
        return [c for c in self.calls if c["handler"] == handler]


class _FakeStdin:
    """Captures lines written by _handle_tool_call so tests can assert JSON-RPC responses."""

    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()


# ============================================================================
# Unit Tests: execute() events via MockInterpreter (no Deno required)
# ============================================================================


def test_execute_fires_start_and_end_with_code_and_output():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    interp = MockInterpreter(responses=["hello world\n"])
    result = interp.execute("print('hello world')", variables={"x": 1})

    assert result == "hello world\n"
    assert callback.handlers() == ["on_interpreter_execute_start", "on_interpreter_execute_end"]

    start = callback.by_handler("on_interpreter_execute_start")[0]
    assert start["inputs"]["code"] == "print('hello world')"
    assert start["inputs"]["variables"] == {"x": 1}

    end = callback.by_handler("on_interpreter_execute_end")[0]
    assert end["outputs"] == "hello world\n"
    assert end["exception"] is None
    # start/end share the same call_id.
    assert start["call_id"] == end["call_id"]


def test_execute_submit_path_reports_final_output():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    interp = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
    result = interp.execute("SUBMIT('42')")

    assert isinstance(result, FinalOutput)
    end = callback.by_handler("on_interpreter_execute_end")[0]
    assert isinstance(end["outputs"], FinalOutput)
    assert end["outputs"].output == {"answer": "42"}
    assert end["exception"] is None


def test_execute_error_path_surfaces_exception_and_propagates():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    interp = MockInterpreter(responses=[CodeInterpreterError("NameError: name 'x' is not defined")])

    with pytest.raises(CodeInterpreterError, match="NameError"):
        interp.execute("print(x)")

    end = callback.by_handler("on_interpreter_execute_end")[0]
    assert isinstance(end["exception"], CodeInterpreterError)
    assert end["outputs"] is None


# ============================================================================
# Unit Tests: tool-dispatch seam via PythonInterpreter (no Deno process needed)
# ============================================================================


def test_invoke_tool_fires_events_with_tool_name_and_kwargs():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    def my_tool(a: str = "", b: str = "") -> str:
        return f"{a}:{b}"

    interp = PythonInterpreter(tools={"my_tool": my_tool})
    # invoke_tool is the decorated seam; it does not require a running Deno process.
    result = interp.invoke_tool("my_tool", {"a": "x", "b": "y"})

    assert result == "x:y"
    assert callback.handlers() == ["on_interpreter_tool_call_start", "on_interpreter_tool_call_end"]

    start = callback.by_handler("on_interpreter_tool_call_start")[0]
    assert start["inputs"] == {"tool_name": "my_tool", "kwargs": {"a": "x", "b": "y"}}

    end = callback.by_handler("on_interpreter_tool_call_end")[0]
    assert end["outputs"] == "x:y"
    assert end["exception"] is None


def test_plain_closure_tool_fires_events():
    """A plain (non-dspy.Tool) callable, like RLM's llm_query, still fires tool-call events."""
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    def llm_query(prompt: str = "") -> str:
        return f"answer to {prompt}"

    interp = PythonInterpreter(tools={"llm_query": llm_query})
    result = interp.invoke_tool("llm_query", {"prompt": "hi"})

    assert result == "answer to hi"
    start = callback.by_handler("on_interpreter_tool_call_start")[0]
    assert start["inputs"]["tool_name"] == "llm_query"
    assert callback.by_handler("on_interpreter_tool_call_end")[0]["outputs"] == "answer to hi"


def test_raising_tool_delivers_exception_and_sandbox_still_gets_jsonrpc_error():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    def failing_tool() -> str:
        raise RuntimeError("Tool failed!")

    interp = PythonInterpreter(tools={"failing_tool": failing_tool})
    interp.deno_process = _FakeProcess()

    # _handle_tool_call preserves its JSON-RPC error conversion: the exception reaches the end
    # handler, and is then caught and written to the sandbox exactly as before.
    interp._handle_tool_call({"id": 7, "params": {"name": "failing_tool", "kwargs": {}}})

    # The end handler saw the real exception...
    end = callback.by_handler("on_interpreter_tool_call_end")[0]
    assert isinstance(end["exception"], RuntimeError)
    assert str(end["exception"]) == "Tool failed!"
    assert end["outputs"] is None

    # ...and the sandbox still received a JSON-RPC error response for request id 7.
    assert len(interp.deno_process.stdin.written) == 1
    written = json.loads(interp.deno_process.stdin.written[0])
    assert written["id"] == 7
    assert "error" in written
    assert written["error"]["data"]["type"] == "RuntimeError"
    assert written["error"]["message"] == "Tool failed!"


def test_unknown_tool_still_converted_to_jsonrpc_error():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    interp = PythonInterpreter(tools={})
    interp.deno_process = _FakeProcess()

    interp._handle_tool_call({"id": 3, "params": {"name": "nope", "kwargs": {}}})

    end = callback.by_handler("on_interpreter_tool_call_end")[0]
    assert isinstance(end["exception"], CodeInterpreterError)

    written = json.loads(interp.deno_process.stdin.written[0])
    assert written["id"] == 3
    assert written["error"]["data"]["type"] == "CodeInterpreterError"


# ============================================================================
# Unit Tests: nesting (tool call nests under the enclosing execute call)
# ============================================================================


def test_tool_call_nests_under_execute_call_id():
    """A decorated tool call invoked synchronously inside a decorated execute() nests under it.

    This exercises the same ACTIVE_CALL_ID mechanism PythonInterpreter relies on: invoke_tool
    runs on the same thread inside execute(), so its parent is the execute call_id.
    """
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    def my_tool(x: str = "") -> str:
        return x

    interp = PythonInterpreter(tools={"my_tool": my_tool})

    def execute_fn(code, variables):
        # Simulate the sandbox calling back into a host tool mid-execution.
        return interp.invoke_tool("my_tool", {"x": "nested"})

    mock = MockInterpreter(execute_fn=execute_fn)
    mock.execute("my_tool(x='nested')")

    execute_start = callback.by_handler("on_interpreter_execute_start")[0]
    tool_start = callback.by_handler("on_interpreter_tool_call_start")[0]

    # The tool call's parent (active call id at handler time) is the execute call id.
    assert tool_start["parent_call_id"] == execute_start["call_id"]
    # The execute call itself has no interpreter parent here.
    assert execute_start["parent_call_id"] is None


# ============================================================================
# Unit Tests: lifecycle + idempotent shutdown
# ============================================================================


def test_startup_and_shutdown_events_fire_and_shutdown_is_idempotent():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    # PythonInterpreter.shutdown() with no live process is a no-op; decoration must not change that.
    interp = PythonInterpreter()
    interp.shutdown()
    interp.shutdown()  # idempotent: safe to call twice

    handlers = callback.handlers()
    assert handlers == [
        "on_interpreter_shutdown_start",
        "on_interpreter_shutdown_end",
        "on_interpreter_shutdown_start",
        "on_interpreter_shutdown_end",
    ]
    for end in callback.by_handler("on_interpreter_shutdown_end"):
        assert end["exception"] is None


def _stub_spawn(monkeypatch):
    """Stub the Deno subprocess spawn + health check so startup can be tested without Deno."""

    class _FakeAliveProcess:
        def poll(self):
            return None  # report the process as alive

    spawns = {"count": 0}

    def fake_popen(*args, **kwargs):
        spawns["count"] += 1
        return _FakeAliveProcess()

    monkeypatch.setattr(python_interpreter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(PythonInterpreter, "_health_check", lambda self: None)
    return spawns


def test_lazy_spawn_emits_startup_events_once(monkeypatch):
    """The lazy Deno spawn inside execute()/_ensure_deno_process (the default RLM path) emits
    startup events, exactly once per actual spawn."""
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])
    spawns = _stub_spawn(monkeypatch)

    interp = PythonInterpreter()
    # _ensure_deno_process is what execute() calls before talking to the sandbox (lazy start).
    interp._ensure_deno_process()
    assert spawns["count"] == 1
    assert callback.handlers() == ["on_interpreter_startup_start", "on_interpreter_startup_end"]
    assert callback.by_handler("on_interpreter_startup_end")[0]["exception"] is None

    # Process already running: no re-spawn and no new startup events.
    interp._ensure_deno_process()
    assert spawns["count"] == 1
    assert callback.handlers() == ["on_interpreter_startup_start", "on_interpreter_startup_end"]


def test_explicit_start_emits_startup_events_once(monkeypatch):
    """Explicit start() also emits startup events (via the same spawn seam), once per spawn."""
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])
    spawns = _stub_spawn(monkeypatch)

    interp = PythonInterpreter()
    interp.start()
    assert spawns["count"] == 1
    assert callback.handlers() == ["on_interpreter_startup_start", "on_interpreter_startup_end"]

    # start() is idempotent: no duplicate startup events when already running.
    interp.start()
    assert spawns["count"] == 1
    assert callback.handlers() == ["on_interpreter_startup_start", "on_interpreter_startup_end"]


def test_mock_interpreter_lifecycle_events():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    interp = MockInterpreter(responses=["ok"])
    interp.start()
    interp.execute("print(1)")
    interp.shutdown()

    assert callback.handlers() == [
        "on_interpreter_startup_start",
        "on_interpreter_startup_end",
        "on_interpreter_execute_start",
        "on_interpreter_execute_end",
        "on_interpreter_shutdown_start",
        "on_interpreter_shutdown_end",
    ]


# ============================================================================
# Unit Tests: dispatch integrity + zero-callback path
# ============================================================================


def test_dispatch_integrity_lm_tool_module_not_misrouted():
    """Interpreter events must not fire module handlers, and LM/Tool/Module keep their own."""
    callback = RecordingCallback()
    dspy.configure(
        lm=DummyLM({"How are you?": {"answer": "test output", "reasoning": "No more responses"}}),
        callbacks=[callback],
    )

    # Module + LM + Adapter events.
    cot = dspy.ChainOfThought("question -> answer")
    cot(question="How are you?")

    # Tool events.
    def tool_1(query: str) -> str:
        return "result 1"

    dspy.Tool(tool_1)(query="x")

    # Interpreter events.
    interp = MockInterpreter(responses=["ok"])
    interp.execute("print(1)")

    handlers = callback.handlers()
    # Module/LM/Tool events are all present.
    assert "on_module_start" in handlers
    assert "on_lm_start" in handlers
    assert "on_tool_start" in handlers
    # Interpreter events are present...
    assert "on_interpreter_execute_start" in handlers

    # ...and no module-start event was emitted by the interpreter (they all come from the
    # ChainOfThought module, never masquerading through the new dispatch branch).
    for call in callback.by_handler("on_module_start"):
        assert not isinstance(call["instance"], MockInterpreter)


def test_interpreter_only_run_emits_no_module_events():
    """Running only an interpreter must not trip any module/lm/tool handlers."""
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    interp = MockInterpreter(responses=["ok"])
    interp.start()
    interp.execute("print(1)")
    interp.shutdown()

    handlers = set(callback.handlers())
    assert not (handlers & {
        "on_module_start", "on_module_end",
        "on_lm_start", "on_lm_end",
        "on_tool_start", "on_tool_end",
    })


def test_zero_callback_path_unchanged():
    # No callbacks registered: behavior and results are unchanged, nothing recorded.
    callback = RecordingCallback()  # not registered anywhere

    interp = MockInterpreter(responses=["out", FinalOutput({"answer": "42"})])
    interp.start()
    assert interp.execute("print(1)", variables={"a": 1}) == "out"
    result = interp.execute("SUBMIT('42')")
    assert isinstance(result, FinalOutput)
    interp.shutdown()

    assert callback.calls == []
    # Recording of call history (the mock's own bookkeeping) is unaffected.
    assert interp.call_history == [("print(1)", {"a": 1}), ("SUBMIT('42')", {})]


# ============================================================================
# Integration: dspy.RLM run emits interpreter execute events (no Deno required)
# ============================================================================


def _make_mock_predictor(responses):
    class MockPredictor:
        def __init__(self):
            self.idx = 0

        def __call__(self, **kwargs):
            result = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**result)

    return MockPredictor()


def test_rlm_run_emits_one_execute_event_per_iteration():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    mock = MockInterpreter(responses=[
        "explored\n",
        "still exploring\n",
        FinalOutput({"answer": "42"}),
    ])
    rlm = RLM("query -> answer", max_iters=5, interpreter=mock)
    rlm.generate_action = _make_mock_predictor([
        {"reasoning": "Explore", "code": "print('explore 1')"},
        {"reasoning": "Explore", "code": "print('explore 2')"},
        {"reasoning": "Submit", "code": "SUBMIT('42')"},
    ])

    result = rlm.forward(query="test")
    assert result.answer == "42"

    execute_starts = callback.by_handler("on_interpreter_execute_start")
    execute_ends = callback.by_handler("on_interpreter_execute_end")
    # One execute event per RLM iteration (3 iterations here).
    assert len(execute_starts) == 3
    assert len(execute_ends) == 3

    # Interpreter events are interleaved with the module events emitted by RLM.
    handlers = callback.handlers()
    assert "on_interpreter_execute_start" in handlers
    # The final iteration's output is the FinalOutput.
    assert isinstance(execute_ends[-1]["outputs"], FinalOutput)

    # RLM injects llm_query as a plain closure into the interpreter; those are non-Tool callables,
    # so no on_tool_* events fire for them (the mock does not dispatch tools).
    assert "on_tool_start" not in handlers


# ============================================================================
# Deno integration tests (real PythonInterpreter)
# ============================================================================


@pytest.mark.deno
def test_deno_execute_events_and_nesting():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    calls = {"count": 0}

    def host_tool(value: str = "") -> str:
        calls["count"] += 1
        return f"host:{value}"

    with PythonInterpreter(tools={"host_tool": host_tool}) as interp:
        # A successful execute with code present in inputs and output in outputs.
        result = interp.execute("print('hi')")
        assert result == "hi\n"

        exec_start = callback.by_handler("on_interpreter_execute_start")[0]
        assert exec_start["inputs"]["code"] == "print('hi')"
        exec_end = callback.by_handler("on_interpreter_execute_end")[0]
        assert exec_end["outputs"] == "hi\n"

        # The Deno process was spawned lazily inside this first execute(), so the startup
        # events fired and nest under the execute call_id.
        startup_start = callback.by_handler("on_interpreter_startup_start")
        assert len(startup_start) == 1
        assert startup_start[0]["parent_call_id"] == exec_start["call_id"]

        # A tool call dispatched from inside execute nests under that execute call_id.
        callback.calls.clear()
        out = interp.execute("print(host_tool(value='x'))")
        assert "host:x" in out

        tool_starts = callback.by_handler("on_interpreter_tool_call_start")
        assert len(tool_starts) == 1
        assert tool_starts[0]["inputs"]["tool_name"] == "host_tool"
        assert tool_starts[0]["inputs"]["kwargs"] == {"value": "x"}

        this_execute = callback.by_handler("on_interpreter_execute_start")[0]
        assert tool_starts[0]["parent_call_id"] == this_execute["call_id"]

        # SUBMIT path yields a FinalOutput in the end handler.
        callback.calls.clear()
        submit_result = interp.execute("SUBMIT('done')")
        assert isinstance(submit_result, FinalOutput)
        assert isinstance(callback.by_handler("on_interpreter_execute_end")[0]["outputs"], FinalOutput)

    # Startup and shutdown events fired for the real interpreter across the context.
    assert callback.by_handler("on_interpreter_shutdown_start")


@pytest.mark.deno
def test_deno_execute_error_surfaces_exception():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    with PythonInterpreter() as interp:
        with pytest.raises(CodeInterpreterError):
            interp.execute("1/0")

    end = callback.by_handler("on_interpreter_execute_end")[-1]
    assert isinstance(end["exception"], CodeInterpreterError)
    assert end["outputs"] is None
