from unittest.mock import Mock, patch

import pytest

from dspy.experimental import MontyProgram
from dspy.experimental.monty._bridge import DEFAULT_LIMITS, Invocation, Policy, normalize_tools
from dspy.experimental.monty._interpreter import MontyInterpreter
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.primitives.sandbox_serializable import SandboxSerializable
from dspy.utils import DummyLM

pytest.importorskip("pydantic_monty")


def source(body, init="pass"):
    return f"""class TestModule(dspy.Module):
    def __init__(self):
        super().__init__()
        {init}

    def forward(self, **kwargs):
        {body}
"""


def invocation(tools=(), *, max_tools=100, depth=2):
    policy = Policy(100, max_tools, depth, DEFAULT_LIMITS, 120.0)
    return Invocation(policy=policy, tools=normalize_tools(list(tools)), lm=None)


def test_baseline_predict_and_typed_output():
    program = MontyProgram("question -> answer: int")
    program.set_lm(DummyLM([{"answer": "4"}]))
    assert program(question="2+2?").answer == 4


def test_baseline_without_inputs_is_valid_source():
    program = MontyProgram("-> answer")
    program.set_lm(DummyLM([{"answer": "ready"}]))
    assert program().answer == "ready"


def test_flex_source_predict():
    program = MontyProgram(
        "question -> answer",
        source(
            "result = self.p(question=kwargs['question'])\n        return dspy.Prediction(answer=result.answer)",
            "self.p = dspy.ChainOfThought('question -> answer')",
        ),
    )
    program.set_lm(DummyLM([{"reasoning": "add", "answer": "4"}]))
    assert program(question="2+2?").answer == "4"


def test_dspy_signature_syntax_preserves_instructions():
    program = MontyProgram(
        "question -> answer",
        source(
            "return self.p(question=kwargs['question'])",
            "self.p = dspy.Predict(dspy.Signature('question -> answer', 'Answer very briefly.'))",
        ),
    )
    program.set_lm(DummyLM([{"answer": "brief"}]))
    assert program(question="question").answer == "brief"


def test_depth_two_sub_rlm_uses_real_dspy_syntax():
    program = MontyProgram(
        "context, question -> answer",
        source("return self.root(context=kwargs['context'], question=kwargs['question'])", "self.root = dspy.RLM('context, question -> answer', max_iters=1)"),
    )
    program.set_lm(
        DummyLM(
            [
                {
                    "reasoning": "Delegate to a child RLM",
                    "code": (
                        "child = dspy.RLM('context, question -> answer', max_iters=1)\n"
                        "result = child(context=context, question=question)\n"
                        "SUBMIT(answer=result.answer)"
                    ),
                },
                {"reasoning": "Answer in the child", "code": "SUBMIT(answer='nested answer')"},
            ]
        )
    )
    assert program(context="facts", question="question").answer == "nested answer"


def test_direct_tool_and_budget():
    def add(a, b):
        return a + b

    program = MontyProgram(
        "a, b -> answer: int",
        source("return dspy.Prediction(answer=add(a=kwargs['a'], b=kwargs['b']))"),
        tools=[add],
        max_tool_calls=1,
    )
    assert program(a=20, b=22).answer == 42


def test_interpreter_tool_is_callable_and_delegatable():
    def add(a, b):
        return a + b

    current = invocation([add])
    interpreter = MontyInterpreter(current)
    interpreter.tools["add"] = lambda **kwargs: -1
    try:
        assert interpreter.execute("add(20, b=22)") == 42
        interpreter.execute("child = dspy.RLM('question -> answer', tools=[dspy.Tool(add)], max_iters=1)")
    finally:
        interpreter.shutdown()

    _, child = current.registry["predictor_0"]
    assert set(child.tools) == {"add"}


def test_interpreter_honors_dynamically_injected_tools_and_budget():
    interpreter = MontyInterpreter(invocation(max_tools=1))
    interpreter.tools["add"] = lambda a, b: a + b
    try:
        assert interpreter.execute("add(20, 22)") == 42
        with pytest.raises(CodeExecutionError, match="limit of 1 tool calls"):
            interpreter.execute("add(a=1, b=1)")
    finally:
        interpreter.shutdown()


def test_rlm_binary_sandbox_serializable_setup_uses_monty_compatible_transport():
    class BinaryValue(SandboxSerializable):
        def sandbox_setup(self):
            return ""

        def to_sandbox(self):
            return b"\xff\x00"

        def sandbox_assignment(self, var_name, data_expr):
            return f"{var_name} = {data_expr}"

        def rlm_preview(self, max_chars=500):
            return "binary"

    interpreter = MontyInterpreter(invocation())
    try:
        RLM("data -> answer")._prepare_serializable_vars({"data": BinaryValue()}, interpreter)
        assert interpreter.execute("data.hex()") == "ff00"
    finally:
        interpreter.shutdown()


def test_compound_predictors_share_tool_budget():
    calls = []

    def record(value: str) -> str:
        calls.append(value)
        return value

    current = invocation([record], max_tools=1)
    first = current.construct("ReAct", "question -> answer", {"tools": ["record"], "max_iters": 1})
    current.registry[first][1].tools["record"](value="first")
    second = current.construct("ReActV2", "question -> answer", {"tools": ["record"], "max_iters": 1})
    with pytest.raises(RuntimeError, match="limit of 1 tool calls"):
        current.registry[second][1].tools["record"](value="second")
    assert calls == ["first"]


def test_llm_query_accepts_advertised_positional_call():
    interpreter = MontyInterpreter(invocation())
    interpreter.tools["llm_query"] = lambda prompt: f"answer: {prompt}"
    try:
        assert interpreter.execute("llm_query('question')") == "answer: question"
    finally:
        interpreter.shutdown()


def test_nested_interpreter_exposes_only_selected_tools():
    def first():
        return "first"

    def second():
        return "second"

    current = invocation([first, second])
    handle = current.construct("RLM", "question -> answer", {"tools": ["first"], "max_iters": 1})
    child = current.registry[handle][1]._interpreter_factory().invocation
    assert set(child.tools) == {"first"}


def test_third_code_execution_level_is_rejected():
    root = invocation()
    outer_handle = root.construct("RLM", "question -> answer", {"max_iters": 1})
    depth_one = root.registry[outer_handle][1]._interpreter_factory().invocation
    child_handle = depth_one.construct("RLM", "question -> answer", {"max_iters": 1})
    depth_two = depth_one.registry[child_handle][1]._interpreter_factory().invocation
    with pytest.raises(RuntimeError, match="maximum depth of 2"):
        depth_two.construct("RLM", "question -> answer", {"max_iters": 1})


def test_call_time_max_iters_cannot_bypass_constructor_limit():
    current = invocation()
    handle = current.construct("ReAct", "question -> answer", {"max_iters": 1})
    with pytest.raises(ValueError, match="construction"):
        current.call(handle, {"question": "q", "max_iters": 1_000_000})


def test_untyped_submit_matches_code_interpreter_contract():
    interpreter = MontyInterpreter(invocation())
    try:
        assert interpreter.execute("SUBMIT({'answer': 42})") == FinalOutput({"output": {"answer": 42}})
        with pytest.raises(CodeExecutionError, match="exactly one positional"):
            interpreter.execute("SUBMIT(answer=42)")
    finally:
        interpreter.shutdown()


def test_submit_cannot_be_swallowed_by_guest_exception_handler():
    interpreter = MontyInterpreter(invocation())
    interpreter.output_fields = [{"name": "answer"}]
    try:
        result = interpreter.execute(
            "try:\n    SUBMIT(answer='ok')\nexcept Exception:\n    pass\nraise RuntimeError('must not execute')"
        )
        assert result == FinalOutput({"answer": "ok"})
    finally:
        interpreter.shutdown()


@pytest.mark.parametrize(
    "signature",
    ["value: unknown_package.Type -> answer", "value: UnknownType -> answer"],
)
def test_guest_predictor_annotations_cannot_import_host_modules(signature):
    current = invocation()
    with patch("dspy.signatures.signature.importlib.import_module") as import_module:
        with pytest.raises(ValueError, match=r"annotation|host modules"):
            current.construct("Predict", signature, {})
    import_module.assert_not_called()


def test_interpreter_is_persistent_and_shutdown_is_terminal():
    interpreter = MontyInterpreter(invocation())
    interpreter.execute("value = 41")
    assert interpreter.execute("value + 1") == 42
    with pytest.raises(CodeExecutionError):
        interpreter.execute("missing_name")
    assert interpreter.execute("value") == 41
    interpreter.shutdown()
    interpreter.shutdown()
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("value")


def test_worker_timeout_is_terminal():
    policy = Policy(100, 100, 2, DEFAULT_LIMITS, 0.1)
    interpreter = MontyInterpreter(Invocation(policy=policy, tools={}, lm=None))
    with pytest.raises(CodeInterpreterError, match="state was lost"):
        interpreter.execute("while True:\n    pass")
    with pytest.raises(CodeInterpreterError, match="session has ended"):
        interpreter.execute("1 + 1")


def test_startup_failure_is_wrapped_and_pool_is_cleaned_up():
    pool = Mock()
    pool.__enter__ = Mock(side_effect=RuntimeError("worker unavailable"))
    pool.__exit__ = Mock()
    interpreter = MontyInterpreter(invocation())

    with patch("pydantic_monty.Monty", return_value=pool):
        with pytest.raises(CodeInterpreterError, match=r"failed to start.*worker unavailable"):
            interpreter.start()

    pool.__exit__.assert_called_once()
    assert interpreter._pool is interpreter._checkout is interpreter._session is None


def test_tool_and_input_names_cannot_collide():
    def question():
        return "question"

    with pytest.raises(ValueError, match="conflict"):
        MontyProgram("question -> answer", tools=[question])


def test_reserved_rlm_tool_names_are_rejected():
    def query(prompt):
        return prompt

    query.__name__ = "llm_query"
    with pytest.raises(ValueError, match="reserved"):
        MontyProgram("question -> answer", tools=[query])


@pytest.mark.parametrize(
    "bad",
    [
        "import os",
        "class A(dspy.Module): pass\nclass B(dspy.Module): pass",
        "class A(object):\n def __init__(self): pass\n def forward(self): pass",
        "class A(dspy.Module):\n def __init__(self): pass\n async def forward(self): pass",
        "@decorator\nclass A(dspy.Module):\n def __init__(self): pass\n def forward(self): pass",
        "class A(dspy.Module):\n value = side_effect()\n def __init__(self): pass\n def forward(self): pass",
    ],
)
def test_source_validation(bad):
    with pytest.raises(ValueError):
        MontyProgram("-> answer", bad)


@pytest.mark.asyncio
async def test_async_is_explicitly_unsupported():
    program = MontyProgram("-> answer", source("return {'answer': 'ok'}"))
    with pytest.raises(NotImplementedError, match="synchronous"):
        await program.acall()


def test_state_reset_and_opaque_parameter():
    program = MontyProgram("-> answer", source("return {'answer': 'ok'}"))
    assert program.named_predictors() == []
    state = program.dump_state()
    program._bind_code(source("return {'answer': 'changed'}"))
    program.load_state(state)
    assert program.module_src == state["module_src"]
    program.reset()
    assert program.get_lm() is None


def test_read_only_source_and_signature():
    program = MontyProgram("-> answer")
    with pytest.raises(AttributeError):
        program.module_src = "changed"
    with pytest.raises(AttributeError):
        program.signature = "changed"


def test_depth_configuration_is_hard_capped_at_two():
    with pytest.raises(ValueError, match="between 0 and 2"):
        MontyProgram("-> answer", max_nested_depth=3)
