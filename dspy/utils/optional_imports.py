"""Helpers for optional dependency imports."""

import types


def import_numpy(feature: str = "") -> types.ModuleType:
    """Import and return the numpy module, raising a clear error if not installed.

    Args:
        feature: Optional description of the feature requiring numpy, used in the error message.

    Returns:
        The numpy module.

    Raises:
        ImportError: If numpy is not installed, with instructions to install it.
    """
    try:
        import numpy as np
    except ImportError:
        msg = "numpy is required"
        if feature:
            msg += f" for {feature}"
        msg += ". Install it with: pip install dspy[numpy]"
        raise ImportError(msg)
    return np
