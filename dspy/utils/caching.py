import os
from pathlib import Path


def default_cache_dir() -> str:
    """Resolve the cache directory without touching the filesystem.

    Evaluated at import time, so it must not raise. `Path.home()` raises RuntimeError
    where no home directory can be determined (WebAssembly guests, some containers),
    and `tempfile.gettempdir()` is no safer -- it probes for a writable directory and
    raises FileNotFoundError when none exists. So the last resort is a plain relative
    path, resolved only if something actually opens the cache.
    """
    explicit = os.environ.get("DSPY_CACHEDIR")
    if explicit:
        return explicit
    try:
        return os.path.join(Path.home(), ".dspy_cache")
    except RuntimeError:
        return ".dspy_cache"


DSPY_CACHEDIR = default_cache_dir()


def create_subdir_in_cachedir(subdir: str) -> str:
    """Create a subdirectory in the DSPy cache directory."""
    subdir = os.path.join(DSPY_CACHEDIR, subdir)
    subdir = os.path.abspath(subdir)
    os.makedirs(subdir, exist_ok=True)
    return subdir
