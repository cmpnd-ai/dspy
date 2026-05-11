import subprocess
import sys

import pytest


def _check_module_not_loaded(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import dspy, sys; print('{module_name}' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "False", (
        f"{module_name!r} was imported during `import dspy`. "
        "Heavy / optional dependencies must be imported lazily inside functions, "
        "not at module top level."
    )


def test_import_dspy_does_not_load_litellm():
    """Ensure `import dspy` does not eagerly import litellm.

    litellm adds ~400-550ms to import time. It should only be loaded when
    actually needed (e.g. on the first LM() call), not at `import dspy` time.
    If this test fails, someone likely added a module-level `import litellm`
    in a module that is transitively imported by dspy/__init__.py.
    """
    _check_module_not_loaded("litellm")


@pytest.mark.parametrize("module_name", ["openai", "regex", "jiter"])
def test_import_dspy_does_not_load_optional_extras(module_name):
    """Ensure `import dspy` does not eagerly import dspy-runtime optional extras.

    These deps are absent in dspy-runtime by default; importing them during
    `import dspy` would break dspy-runtime users. Use lazy imports inside
    functions/methods instead.
    """
    _check_module_not_loaded(module_name)
