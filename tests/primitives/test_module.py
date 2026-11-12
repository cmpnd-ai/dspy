import json
import tempfile
import threading
from pathlib import Path

import pytest

import dspy
from dspy.primitives.example import Example
from dspy.primitives.module import Module, set_attribute_by_name
from dspy.utils import DummyLM


class HopModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict1 = dspy.Predict("question -> query")
        self.predict2 = dspy.Predict("query -> answer")

    def forward(self, question):
        query = self.predict1(question=question).query
        return self.predict2(query=query)


def test_module_initialization():
    module = Module()
    assert module._compiled is False, "Module _compiled attribute should be False upon initialization"


def test_named_predictors():
    module = HopModule()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2, "Should identify correct number of Predict instances"
    names, _preds = zip(*named_preds, strict=False)
    assert "predict1" in names and "predict2" in names, "Named predictors should include 'predict1' and 'predict2'"


def test_predictors():
    module = HopModule()
    preds = module.predictors()
    assert len(preds) == 2, "Should return correct number of Predict instances"
    assert all(isinstance(p, dspy.Predict) for p in preds), "All returned items should be instances of PredictMock"


def test_forward():
    program = HopModule()
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"query": "let me check"},
                "let me check": {"answer": "2"},
            }
        )
    )
    result = program(question="What is 1+1?").answer
    assert result == "2"


def test_nested_named_predictors():
    class Hop2Module(dspy.Module):
        def __init__(self):
            super().__init__()
            self.hop = HopModule()

    module = Hop2Module()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2
    names, _preds = zip(*named_preds, strict=False)
    assert "hop.predict1" in names
    assert "hop.predict2" in names


def test_empty_module():
    module = Module()
    assert list(module.named_sub_modules()) == [("self", module)]


def test_single_level():
    module = Module()
    module.sub = Module()
    expected = [("self", module), ("self.sub", module.sub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_levels():
    module = Module()
    module.sub = Module()
    module.sub.subsub = Module()
    expected = [("self", module), ("self.sub", module.sub), ("self.sub.subsub", module.sub.subsub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_sub_modules():
    module = Module()
    module.sub1 = Module()
    module.sub2 = Module()
    expected = [("self", module), ("self.sub1", module.sub1), ("self.sub2", module.sub2)]
    assert sorted(module.named_sub_modules()) == sorted(expected)


def test_non_base_module_attributes():
    module = Module()
    module.sub = Module()
    module.not_a_sub = "Not a self"
    expected = [("self", module), ("self.sub", module.sub)]
    assert list(module.named_sub_modules()) == expected


def test_complex_module_traversal():
    root = Module()
    root.sub_module = Module()
    root.sub_module.nested_list = [Module(), {"key": Module()}]
    root.sub_module.nested_tuple = (Module(), [Module(), Module()])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
        "self.sub_module.nested_tuple[1][1]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}

    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_complex_module_traversal_with_same_module():
    root = Module()
    root.sub_module = Module()
    root.sub_module.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    root.sub_module.nested_tuple = (Module(), [same_module, same_module])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",  # NOTE: named_sub_modules allows recursive structures
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",  # NEW: named_sub_modules allows recursive structures, but named_parameters does not
    }
    found_names = {name for name, _ in root.named_sub_modules()}

    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_complex_module_set_attribute_by_name():
    root = Module()
    root.sub_module = Module()
    root.sub_module.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    root.sub_module.nested_tuple = (Module(), [same_module, same_module])

    set_attribute_by_name(root, "test_attrib", True)
    assert root.test_attrib is True
    set_attribute_by_name(root, "sub_module.test_attrib", True)
    assert root.sub_module.test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[0].test_attrib", True)
    assert root.sub_module.nested_list[0].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[1]['key'].test_attrib", True)
    assert root.sub_module.nested_list[1]["key"].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[0].test_attrib", True)
    assert root.sub_module.nested_tuple[0].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[1][0].test_attrib", True)
    assert root.sub_module.nested_tuple[1][0].test_attrib is True
    assert root.sub_module.nested_tuple[1][1].test_attrib is True


class DuplicateModule(Module):
    def __init__(self):
        super().__init__()
        self.p0 = dspy.Predict("question -> answer")
        self.p1 = self.p0


def test_named_parameters_duplicate_references():
    module = DuplicateModule()
    # Only testing for whether exceptions are thrown or not
    # As Module.named_parameters() is recursive, this is mainly for catching infinite recursion
    module.named_parameters()


def test_save_and_load_react_state(tmp_path):
    source = dspy.ReAct("question -> answer: int", tools=[], max_iters=1)
    instructions = "Solve the arithmetic question and submit the integer answer."
    source.react.signature = source.react.signature.with_instructions(instructions)
    source.react.demos = [Example(
        question="What is 2 + 3?",
        next_thought="Add the two integers.",
        tool_calls=dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": 5}}]),
    )]
    path = tmp_path / "react.json"
    source.save(path)

    loaded = dspy.ReAct("question -> answer: int", tools=[], max_iters=1)
    loaded.load(path)
    assert loaded.react.signature.instructions == instructions
    assert len(loaded.react.demos) == 1
    assert loaded.react.demos[0]["tool_calls"]["tool_calls"][0]["args"] == {"answer": 5}

    lm = DummyLM([{
        "next_thought": "Multiply the two integers.",
        "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": 42}}]),
    }])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = loaded(question="What is 6 times 7?")

    assert result.answer == 42
    assert result.termination_reason == "submit"
    assert isinstance(result.history, dspy.History)
    assert len(result.history.messages) == 1
    assert result.history.messages[0]["question"] == "What is 6 times 7?"
    assert result.history.messages[0]["tool_calls"].tool_calls[0].name == "submit"
    prompt = str(lm.history[0]["messages"])
    assert instructions in prompt
    assert "What is 2 + 3?" in prompt



def test_load_state_is_transactional():
    """
    Regression test for https://github.com/stanfordnlp/dspy/issues/9589

    load_state must be all-or-nothing. If it fails mid-load (missing key
    or malformed value), the module must be completely unchanged.
    """

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.a = dspy.ChainOfThought(Sig)
            self.b = dspy.ChainOfThought(Sig)

    source = Prog()
    sentinel = Example(question="q1", answer="a1").with_inputs("question")
    source.a.predict.demos = [sentinel]
    source.b.predict.demos = [sentinel]

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.json"
        source.save(str(path), save_program=False)

        raw = json.loads(path.read_text())
        corrupted = {k: v for k, v in raw.items() if "b." not in k}
        path.write_text(json.dumps(corrupted))

        template = Prog()
        assert template.a.predict.demos == []

        with pytest.raises(KeyError):
            template.load(str(path))

        assert template.a.predict.demos == [], "load_state partially mutated module before failing"


# ---------------------------------------------------------------------------
# acall() fallback for sync-only modules
# ---------------------------------------------------------------------------


class SyncOnlyModule(dspy.Module):
    """A module that defines only ``forward`` — no ``aforward``."""

    def forward(self, x):
        return dspy.Prediction(result=x * 2)


class ParentWithPredict(dspy.Module):
    """A sync-only module whose ``forward`` calls a ``Predict`` sub-module."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("x -> y")

    def forward(self, x):
        return self.predict(x=x)


@pytest.mark.asyncio
async def test_acall_fallback_on_sync_only_module():
    """acall on a sync-only module returns the same result as __call__."""
    module = SyncOnlyModule()
    sync_result = module(x=21)
    async_result = await module.acall(x=21)
    assert async_result.result == sync_result.result == 42


@pytest.mark.asyncio
async def test_acall_fallback_has_aforward():
    """Every Module instance has aforward, even sync-only ones."""
    module = SyncOnlyModule()
    assert hasattr(module, "aforward")


@pytest.mark.asyncio
async def test_acall_fallback_runs_in_worker_thread():
    """The default aforward runs forward in a worker thread, not the event loop thread."""
    loop_thread = threading.get_ident()

    class ThreadCapture(dspy.Module):
        def forward(self, x):
            return dspy.Prediction(thread_id=threading.get_ident())

    module = ThreadCapture()
    result = await module.acall(x=1)
    assert result.thread_id != loop_thread


@pytest.mark.asyncio
async def test_acall_fallback_no_double_history():
    """The fallback calls self.forward directly (not self.__call__), so history is recorded once."""
    with dspy.context(lm=DummyLM([{"y": "84"}])):
        parent = ParentWithPredict()
        await parent.acall(x=42)
        assert len(parent.history) == 1
        assert len(parent.predict.history) == 1


@pytest.mark.asyncio
async def test_acall_fallback_no_forward_warning(capsys):
    """Accessing self.forward inside the default aforward does not emit the discouragement warning."""
    module = SyncOnlyModule()
    await module.acall(x=1)
    captured = capsys.readouterr()
    assert "directly is discouraged" not in captured.err


@pytest.mark.asyncio
async def test_acall_fallback_with_track_usage():
    """track_usage works correctly with the asyncify fallback path."""
    with dspy.context(lm=DummyLM([{"y": "84"}]), track_usage=True):
        parent = ParentWithPredict()
        result = await parent.acall(x=42)
        assert result.y == "84"
        assert result.get_lm_usage() is not None


@pytest.mark.asyncio
async def test_acall_fallback_on_built_in_sync_modules():
    """Built-in sync-only modules all have aforward via the Module default."""
    with dspy.context(lm=DummyLM([{"answer": "2"}])):
        pot = dspy.ProgramOfThought("question -> answer")
        assert hasattr(pot, "aforward")

    # BestOfN
    predict = dspy.Predict("question -> answer")
    best_of_n = dspy.BestOfN(module=predict, N=2, reward_fn=lambda *a: 1.0, threshold=1.0)
    assert hasattr(best_of_n, "aforward")

    # Refine
    refine = dspy.Refine(module=predict, N=2, reward_fn=lambda *a: 1.0, threshold=1.0)
    assert hasattr(refine, "aforward")

    # MultiChainComparison
    mcc = dspy.MultiChainComparison("question -> answer")
    assert hasattr(mcc, "aforward")
