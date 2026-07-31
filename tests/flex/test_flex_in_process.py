"""dspy.Flex against an interpreter that declares ``runs_in_process``.

The bridge exists to carry dspy across a value boundary. An interpreter that shares this
process's object space has no such boundary, declares so, and gets a path with no shim,
no handles and no JSON round trip. These tests pin what that path guarantees:

* the submission sees the real ``dspy``, so it builds ordinary predictors;
* the ``Prediction`` it returns comes back as the object it is, not as a reconstruction;
* an object handed in through ``variables`` arrives as itself;
* the generated code's side of the contract is still checked.

The interpreter here is deliberately the simplest thing that satisfies the protocol --
``exec`` into a persistent namespace with last-expression semantics -- because that is
what "no boundary" means, and anything more would be testing the double.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

import dspy
from dspy.flex import Flex
from dspy.primitives.code_interpreter import CodeInterpreterError, runs_in_process
from dspy.utils.dummies import DummyLM


class InProcessInterpreter:
    """A ``CodeInterpreter`` that execs in this process, with no marshalling.

    Not a sandbox and does not pretend to be one: real backends of this shape get their
    isolation from the environment around the interpreter (a container, a wasm
    component), which a test double has no way to reproduce and no need to.
    """

    runs_in_process = True

    def __init__(self) -> None:
        self.tools: dict = {}
        self._ns: dict | None = None
        self.submissions: list[str] = []

    def start(self) -> None:
        if self._ns is None:
            self._ns = {"__name__": "flex_in_process_sandbox"}
            self._ns.update(self.tools)

    def execute(self, code: str, variables: dict | None = None):
        self.start()
        assert self._ns is not None
        self.submissions.append(code)
        self._ns.update(variables or {})
        tree = ast.parse(code)
        tail = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<submission>", "exec"), self._ns)  # noqa: S102
        if tail is None:
            return ""
        return eval(compile(tail, "<submission>", "eval"), self._ns)  # noqa: S307

    def shutdown(self) -> None:
        self._ns = None


class Doubler(dspy.Signature):
    """Return double the input value."""

    value: int = dspy.InputField()
    result: int = dspy.OutputField()


MODULE_SRC = textwrap.dedent(
    """
    class Doubled(dspy.Module):
        def __init__(self):
            self.step = dspy.Predict("value: int -> result: int")

        def forward(self, value):
            out = self.step(value=value)
            return dspy.Prediction(result=out.result)
    """
)


def _flex(src: str = MODULE_SRC, **kwargs):
    flex = Flex(Doubler, interpreter_factory=InProcessInterpreter, **kwargs)
    flex._bind_code(src)
    return flex


def test_capability_defaults_to_false_and_is_read_not_required():
    """An interpreter that says nothing is a value boundary, as every existing one is."""

    class Silent:
        tools: dict = {}

        def start(self): ...
        def execute(self, code, variables=None): ...
        def shutdown(self): ...

    assert runs_in_process(Silent()) is False
    assert runs_in_process(InProcessInterpreter()) is True


def test_the_submission_sees_the_real_dspy():
    """No shim is installed, so ``dspy`` in generated code is this dspy."""
    flex = _flex(
        textwrap.dedent(
            """
            class Probe(dspy.Module):
                def __init__(self):
                    self.step = dspy.Predict("value: int -> result: int")

                def forward(self, value):
                    assert type(self.step).__module__.startswith("dspy."), type(self.step)
                    assert isinstance(self.step, dspy.Predict)
                    return dspy.Prediction(result=type(self.step).__name__)
            """
        )
    )
    with dspy.context(lm=DummyLM([{"result": "4"}])):
        assert flex(value=2).result == "Predict"


def test_the_prediction_comes_back_as_the_object_it_is():
    """Identity, not equality: the bridged path can only ever rebuild an equal Prediction."""
    made: list = []

    def record(pred):
        made.append(pred)
        return ""

    flex = _flex(
        textwrap.dedent(
            """
            class Passthrough(dspy.Module):
                def forward(self, value):
                    out = dspy.Prediction(result=value)
                    record(pred=out)
                    return out
            """
        ),
        tools=[record],
    )
    with dspy.context(lm=DummyLM([{"result": "x"}])):
        returned = flex(value=7)
    assert returned is made[0]


def test_an_input_object_arrives_as_itself():
    """``variables`` binds by reference, which is what the capability claims."""

    class Marker:
        pass

    marker = Marker()
    flex = _flex(
        textwrap.dedent(
            """
            class Echo(dspy.Module):
                def forward(self, value):
                    return dspy.Prediction(result=value)
            """
        )
    )
    with dspy.context(lm=DummyLM([{"result": "x"}])):
        assert flex(value=marker).result is marker


def test_a_real_predictor_call_reaches_the_lm():
    """The point of the whole thing: generated code calling a predictor it built.

    The value arrives typed, and it was never untyped: the inner predictor's own adapter
    parsed it against ``result: int``. The bridged path reaches the same answer by a
    different route -- serialize to JSON, then rebuild each field with ``parse_value``
    against the outer signature -- which is why it needs the annotation and this does not.
    """
    flex = _flex()
    with dspy.context(lm=DummyLM([{"result": "84"}])):
        result = flex(value=42).result
    assert result == 84
    assert isinstance(result, int)


def test_user_tools_are_still_registered_by_name():
    """The tools contract belongs to the interpreter, not to the shim, so it survives."""

    def shout(text: str) -> str:
        return text.upper()

    flex = _flex(
        textwrap.dedent(
            """
            class UsesTool(dspy.Module):
                def forward(self, value):
                    return dspy.Prediction(result=shout(text="hi"))
            """
        ),
        tools=[shout],
    )
    with dspy.context(lm=DummyLM([{"result": "x"}])):
        assert flex(value=1).result == "HI"


def test_a_non_prediction_return_is_refused():
    flex = _flex(
        textwrap.dedent(
            """
            class Wrong(dspy.Module):
                def forward(self, value):
                    return {"result": 1}
            """
        )
    )
    with dspy.context(lm=DummyLM([{"result": "x"}])):
        with pytest.raises(CodeInterpreterError, match="must return a dspy.Prediction"):
            flex(value=1)


def test_a_missing_output_field_is_refused():
    flex = _flex(
        textwrap.dedent(
            """
            class Short(dspy.Module):
                def forward(self, value):
                    return dspy.Prediction(other=1)
            """
        )
    )
    with dspy.context(lm=DummyLM([{"result": "x"}])):
        with pytest.raises(CodeInterpreterError, match=r"missing declared output field\(s\) \['result'\]"):
            flex(value=1)


def test_the_interpreter_is_fresh_per_forward_and_shut_down():
    """Same lifecycle as the bridged path; only the marshalling differs."""
    made: list[InProcessInterpreter] = []

    def factory():
        interp = InProcessInterpreter()
        made.append(interp)
        return interp

    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(MODULE_SRC)
    with dspy.context(lm=DummyLM([{"result": "2"}, {"result": "4"}])):
        flex(value=1)
        flex(value=2)
    assert len(made) == 2
    assert all(interp._ns is None for interp in made)


def test_no_shim_is_submitted():
    """The absence is the feature: nothing defines a stand-in ``dspy`` on this path."""
    made: list[InProcessInterpreter] = []

    def factory():
        interp = InProcessInterpreter()
        made.append(interp)
        return interp

    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(MODULE_SRC)
    with dspy.context(lm=DummyLM([{"result": "2"}])):
        flex(value=1)
    submitted = "\n".join(made[0].submissions)
    assert "_DspyProxy" not in submitted
    assert "__dspy_construct__" not in submitted
    assert "__dspy_call__" not in submitted
