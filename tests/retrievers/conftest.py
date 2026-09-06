"""Shared fixtures and setup for retriever tests.

The ``databricks-sdk`` package is an optional dependency that is not installed
in the test environment. The ``databricks_rm`` module evaluates
``find_spec("databricks.sdk")`` at import time, which raises ``ModuleNotFoundError``
(rather than returning ``None``) when the ``databricks`` parent package itself is
absent. To keep retriever tests hermetic and importable without the optional
dependency, we register lightweight stub modules in ``sys.modules`` before any
test module that imports ``dspy.retrievers.databricks_rm`` is collected. This
makes ``find_spec`` resolve to a spec so the import-time guard sets
``_databricks_sdk_installed = True``; tests that exercise the non-SDK requests
path explicitly patch ``_databricks_sdk_installed`` back to ``False``.
"""

import sys
import types
from importlib.util import spec_from_loader


def _stub_databricks_sdk() -> None:
    for name in ("databricks", "databricks.sdk"):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__spec__ = spec_from_loader(name, loader=None)
        module.__path__ = []  # type: ignore[assignment]
        sys.modules[name] = module


_stub_databricks_sdk()
