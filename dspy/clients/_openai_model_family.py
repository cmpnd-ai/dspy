"""Shared OpenAI reasoning-model classifier.

Both :mod:`dspy.clients.lm` and :mod:`dspy.clients.openai_format` need to decide
whether a model string names an OpenAI reasoning-family model (``o1``/``o3``/
``o4``/``o5``, ``gpt-5`` excluding ``gpt-5-chat``). They previously carried two
separate copies that diverged in provider-prefix stripping and family grammar:

* ``lm.py`` stripped any provider prefix (``model.split("/")[-1]``) and matched an
  anchored regex.
* ``openai_format.py`` stripped only the literal ``openai/`` prefix and matched
  a loose ``startswith``, so non-``openai/`` prefixes like ``azure/o3`` were
  misclassified as non-reasoning.

This module is the single source of truth so future drift is a single
definition, not two. :mod:`dspy.clients.openai_format` cannot import
:mod:`dspy.clients.lm` (which already imports the other way), hence this neutral
third module.
"""

from __future__ import annotations

import re

_OPENAI_REASONING_MODEL_RE = re.compile(
    r"^(?:o1-preview|o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$"
)


def is_openai_reasoning_model(model: str | None) -> bool:
    """Return ``True`` iff ``model`` names an OpenAI reasoning-family model.

    Any LiteLLM provider prefix (``openai/``, ``azure/``, ``vertex_ai/`` ...) is
    stripped; the family is decided by the bare model suffix. ``None`` and
    non-strings return ``False`` (the adapter path may pass a missing model).
    """
    if not isinstance(model, str):
        return False
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return _OPENAI_REASONING_MODEL_RE.match(model_family) is not None
