"""Source inspection and instruction-history contracts used by proposal generation."""

import inspect
import linecache
import subprocess
import sys
from types import ModuleType

import dspy
from dspy.propose.utils import create_predictor_level_history_string, get_dspy_source_code


class FileProgram(dspy.Module):
    def forward(self, value):
        return value + 3


def test_source_from_regular_file():
    assert get_dspy_source_code(FileProgram()) == "\n\n" + inspect.getsource(FileProgram)


def test_import_does_not_replace_standard_inspection():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import inspect; original = inspect.getfile; import dspy; assert inspect.getfile is original",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_notebook_source_uses_own_method_and_preserves_decorators(monkeypatch):
    module = ModuleType("dspy_test_notebook")
    module.dspy = dspy
    module.decorate = lambda cls: cls
    monkeypatch.setitem(sys.modules, module.__name__, module)
    expected = (
        "@decorate\nclass NotebookProgram(dspy.Module):\n    def forward(self, value):\n        return value + 7\n"
    )
    source = (
        "class NotebookProgram(dspy.Module):\n    def forward(self, value):\n        return value - 1\n\n" + expected
    )
    filename = "<dspy-test-notebook-cell>"
    monkeypatch.setitem(linecache.cache, filename, (len(source), None, source.splitlines(keepends=True), filename))
    exec(compile(source, filename, "exec"), module.__dict__)

    assert get_dspy_source_code(module.NotebookProgram()) == "\n\n" + expected


def test_instruction_history_averages_duplicates_and_preserves_tie_order(tmp_path):
    base = dspy.Predict("x -> y")
    logs = {0: {"score": 100}}  # Trials without a saved program do not participate.
    for index, (instruction, score) in enumerate(
        [("shared", 2), ("tie", 5), ("shared", 8), ("best", 7), ("low", 1)], start=1
    ):
        program = dspy.Predict(base.signature.with_instructions(instruction))
        path = tmp_path / f"{index}.json"
        program.save(path)
        logs[index] = {"program_path": path, "score": score}

    assert create_predictor_level_history_string(base, 0, logs, 3) == (
        "tie | Score: 5.0\n\nshared | Score: 5.0\n\nbest | Score: 7.0\n\n"
    )
    assert create_predictor_level_history_string(base, 0, logs, 0) == ""
