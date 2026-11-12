"""Async xAI credential routing parity (R3).

A logged-out or expired xAI subscription login must BLOCK the ambient
``XAI_API_KEY`` on ``AsyncLMRouter`` exactly as it does on the sync
``LMRouter`` (rule R3, ratified 2026-09-22): a failed subscription is never
silently replaced by a metered key.

The router's ``oauth-unless-explicit`` branch probes
``getattr(cls, "stored_credential_state", None)`` -- a four-state method
(``usable`` | ``unusable`` | ``logged_out`` | ``absent``) that used to
live only on the sync ``BaseProviderLM``.  The async mirrors exposed only
the older boolean ``has_stored_credential``, so ``AsyncLMRouter`` always
took the boolean fallback, collapsing ``logged_out``/``unusable`` to
``absent`` and silently spending the metered env key.
"""

import pytest

from dspy._vendor.lm15 import AsyncLMRouter, LMRouter, MissingCredentialError, RouterConfig
from dspy._vendor.lm15.access import LoadedCredential
from dspy._vendor.lm15.providers.async_base import AsyncBaseProviderLM, AsyncXaiLM
from dspy._vendor.lm15.providers.xai import XaiLM

ENV = {"XAI_API_KEY": "sk-metered-key"}

ROUTERS = [
    pytest.param("sync", LMRouter, id="sync"),
    pytest.param("async", AsyncLMRouter, id="async"),
]

STATES = ["usable", "unusable", "logged_out", "absent"]


def _set_state(monkeypatch, state):
    """Patch the offline xAI stored-credential probe to return ``state``.

    ``_STORED_STATES["xai"]`` is the single lookup the router's four-state
    probe consults (via ``access.stored_credential_state``), so patching it
    controls both the sync and async classmethods without touching files.
    """
    import dspy._vendor.lm15.access as access

    monkeypatch.setitem(access._STORED_STATES, "xai", lambda: state)


def _stored_loader(monkeypatch):
    """Patch the xAI credential loader so a 'usable' stored login
    constructs hermetically (no file read, no network refresh).

    Only needed for the ``usable`` row: the router builds the LM straight
    from the local OAuth constructor, which otherwise calls
    ``get_xai_access_token`` (real files / network).
    """
    import dspy._vendor.lm15.access as access

    def fake_loader(_path):
        return LoadedCredential(lambda: "sk-stored-token", source="stored")

    monkeypatch.setitem(access._CREDENTIAL_LOADERS, "xai", fake_loader)


# ── The fix: the async side now exposes the four-state probe ──────────


def test_async_base_defines_the_four_state_probe():
    """The probe is defined on the async base so every async mirror
    inherits it (defense for any future oauth-unless-explicit provider)."""
    assert "stored_credential_state" in AsyncBaseProviderLM.__dict__
    assert callable(AsyncBaseProviderLM.stored_credential_state)


def test_async_xai_probe_is_callable_and_rebinds_to_sync_sibling():
    """The router does ``getattr(cls, "stored_credential_state", None)``
    and only blocks when it is callable.  Before the fix this was ``None``
    on ``AsyncXaiLM``; it must now be the sync sibling's four-state probe
    (same rebind convention as the existing ``has_stored_credential`` --
    accessing a classmethod builds a fresh bound method each time, so
    compare the underlying function)."""
    probe = getattr(AsyncXaiLM, "stored_credential_state", None)
    assert probe is not None
    assert callable(probe)
    assert AsyncXaiLM.stored_credential_state.__func__ is XaiLM.stored_credential_state.__func__
    assert AsyncXaiLM.stored_credential_state.__self__ is XaiLM


# ── R3 blocking parity through the router (sync vs async) ────────────


@pytest.mark.parametrize("direction,router_cls", ROUTERS)
@pytest.mark.parametrize("state", STATES)
def test_xai_subscription_routing_parity(direction, router_cls, state, monkeypatch):
    """The full routing matrix must match between ``LMRouter`` and
    ``AsyncLMRouter`` (R3, AUTH-1):

    * ``usable``    -> build with the stored login (env key NOT used);
    * ``unusable``  -> BLOCK the env key, raise MissingCredentialError;
    * ``logged_out``-> BLOCK the env key, raise MissingCredentialError;
    * ``absent``    -> build with the ambient env key (allowed).

    Before the fix, the async ``unusable``/``logged_out`` rows silently
    built with the metered ``XAI_API_KEY`` instead of raising.
    """
    _set_state(monkeypatch, state)
    if state == "usable":
        _stored_loader(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    router = router_cls(RouterConfig(env=dict(ENV)))

    if state in ("unusable", "logged_out"):
        expected = "signed out" if state == "logged_out" else "expired and cannot be renewed"
        with pytest.raises(MissingCredentialError, match=expected):
            router.lm("xai:grok-4")
        return

    lm = router.lm("xai:grok-4")
    assert lm.provider == "xai"
    if state == "usable":
        # The stored subscription outranks the ambient env key: the env
        # key is NOT used; the stored-login per-request provider is.
        assert lm.api_key != "sk-metered-key"
        assert callable(lm.api_key)
        assert getattr(lm, "_inner", lm)._credential_source == "stored"
    else:  # absent
        # Nothing stored: the env key is the allowed fallback.
        assert lm.api_key == "sk-metered-key"


@pytest.mark.parametrize("direction,router_cls", ROUTERS)
def test_explicit_api_key_still_outranks_blocking(direction, router_cls, monkeypatch):
    """An explicit ``RouterConfig(api_keys=...)`` entry is deliberate use
    and must still win even when a subscription login is signed out: R3
    blocks *ambient* keys, not a deliberate pass.
    """
    _set_state(monkeypatch, "logged_out")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    router = router_cls(RouterConfig(env=dict(ENV), api_keys={"xai": "sk-explicit"}))
    lm = router.lm("xai:grok-4")
    assert lm.api_key == "sk-explicit"
