"""dspy._vendor.lm15.login.declared: the kimi-code base URL regression guard.

``kimi-code`` is a *declared provider* -- an Anthropic Messages wire route
reached only through a managed login (device-flow OAuth against
``auth.kimi.com``), so no test in the suite had exercised its built request
URL.  The Anthropic Messages adapter (``AnthropicLM``) builds the wire URL as
``f"{base_url.rstrip('/')}/messages"`` and never inserts a version segment,
so an Anthropic-dialect ``base_url`` must already end in ``/v1`` (every row
of ``ANTHROPIC_PRESET_BASE_URLS`` does; the adapter default is
``https://api.anthropic.com/v1``).  ``KIMI_CODE_BASE_URL`` was declared
without ``/v1`` (dspy#10540), so every request went to
``https://api.kimi.com/coding/messages`` (HTTP 404) instead of the real
``https://api.kimi.com/coding/v1/messages`` endpoint.  These tests pin the
fixed URL through the adapter and guard the ``.../v1`` convention.
"""

import pytest

from dspy._vendor.lm15.compat import ANTHROPIC_PRESET_BASE_URLS
from dspy._vendor.lm15.login.declared import KIMI_CODE, KIMI_CODE_BASE_URL
from dspy._vendor.lm15.providers.anthropic import AnthropicLM
from dspy._vendor.lm15.types import Message, Request


@pytest.mark.parametrize("stream", [False, True])
def test_kimi_code_build_request_targets_the_v1_messages_path(stream):
    # The adapter builds the URL as f"{base_url.rstrip('/')}/messages" and
    # never inserts a version segment, so base_url must already end in /v1.
    # With the fix this reaches /coding/v1/messages (the real endpoint, which
    # answers 401 without credentials) instead of /coding/messages (404
    # resource_not_found_error).  The sync and async adapters share one
    # build_request (AsyncAnthropicLM delegates to its inner AnthropicLM),
    # so this exercises both wire paths.
    lm = AnthropicLM(api_key="k", access=KIMI_CODE.access)
    wire = lm.build_request(Request(model="kimi-k2", messages=(Message.user("hi"),)), stream=stream)
    assert wire.method == "POST"
    assert wire.url == "https://api.kimi.com/coding/v1/messages"
    assert wire.url == f"{KIMI_CODE_BASE_URL.rstrip('/')}/messages"
    assert "/coding/messages" not in wire.url  # the 404 path is gone


def test_every_anthropic_dialect_base_url_ends_in_v1():
    # Every Anthropic-dialect base URL -- the adapter default and the preset
    # table -- ends in /v1; kimi-code (a declared Anthropic-dialect provider)
    # now follows the same convention.  Guards a future regression that drops
    # the segment from any row or from the kimi-code declaration.
    for name, url in ANTHROPIC_PRESET_BASE_URLS.items():
        assert url.endswith("/v1"), f"anthropic preset {name!r}: {url!r} must end in /v1"
    assert KIMI_CODE_BASE_URL.endswith("/v1")
    # The adapter's own default root has the same shape: it builds the wire
    # URL as f"{base_url.rstrip('/')}/messages" and never inserts a version
    # segment, so its default base URL already ends in /v1.
    assert AnthropicLM(api_key="k").base_url == "https://api.anthropic.com/v1"
