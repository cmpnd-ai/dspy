"""`import dspy` must not execute dependencies that a reduced-stdlib build cannot provide.

Guards the deferred imports in dspy.clients, dspy.adapters, and dspy.streaming. Runs in a
subprocess because the pytest process has already imported most of these.
"""

import os
import subprocess
import sys
import textwrap

# Executed at import: pull in sqlite3 (diskcache), ssl (anyio), or zlib (requests/urllib3), or
# ship as a native extension with no wheel for every platform (jsonschema -> rpds, jiter).
_MUST_NOT_EXECUTE = [
    "anyio",
    "cachetools",
    "diskcache",
    "jiter",
    "jsonschema",
    "openai",
    "requests",
    "sqlite3",
    "urllib3",
]

_PROBE = textwrap.dedent(
    """
    import sys

    import dspy  # noqa: F401
    from dspy.utils.lazy_import import _LazyModule

    executed = [
        name
        for name in {names!r}
        if name in sys.modules and not isinstance(sys.modules[name], _LazyModule)
    ]
    print(",".join(executed))
    """
)


def test_import_dspy_does_not_execute_deferred_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(names=_MUST_NOT_EXECUTE)],
        capture_output=True,
        text=True,
        env={**os.environ, "DSPY_DISABLE_CACHE": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr

    executed = [name for name in result.stdout.strip().split(",") if name]
    assert executed == [], f"`import dspy` executed deferred dependencies: {', '.join(executed)}"
