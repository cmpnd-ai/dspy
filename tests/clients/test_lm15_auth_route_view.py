"""Tests for ``Auth._route_view`` route resolution in the vendored lm15 manager.

A slot record in the store's ``_lm15.slots`` map may be keyed by a provider id
that is no longer registered with this lm15 build (a stale slot left behind by a
sibling build or a provider-set change between releases).  ``_route_view``'s
owner-search must skip such stale owners rather than aborting route resolution
with a ``method_unavailable`` error, matching the tolerance ``connections()``
already applies to the same state.

These tests use an in-memory store only; no network, provider keys, or live
services are required.
"""

from __future__ import annotations

import pytest

from dspy._vendor.lm15.login import Auth, AuthOperationError, ConnectionStatus, ForgetResult, ProviderDescriptor
from dspy._vendor.lm15.login.store import META_KEY, STORE_VERSION, MemoryStore


def _record(connection_id: str | None, routes, **kwargs) -> dict:
    record = {
        "generation": "1",
        "connection_id": connection_id,
        "revision": "1",
        "kind": "api_key",
        "method_id": "api_key",
        "instance_id": "public",
        "label": connection_id or "slot",
        "created_at": "2026-01-01T00:00:00Z",
        "routes": list(routes),
        "settings": {},
        "state": "ready",
        "renewal": "none",
    }
    record.update(kwargs)
    return record


def _auth_with_slots(slots: dict) -> Auth:
    store = MemoryStore()
    store.mutate(lambda doc: {META_KEY: {"version": STORE_VERSION, "slots": slots}})
    return Auth(store)


def test_status_skips_stale_slot_for_unregistered_provider():
    auth = _auth_with_slots(
        {
            "gone-provider": _record("cn_goneStaleConn01", ["openai"]),
        }
    )
    # A stale owner advertising a route must not abort resolution; the route
    # has no live connection, so status reports absent.
    status = auth.status("openai")
    assert isinstance(status, ConnectionStatus)
    assert status.provider == "openai"
    assert status.presence == "absent"
    assert status.connection is None


def test_request_auth_skips_stale_slot_and_reports_login_required():
    auth = _auth_with_slots(
        {
            "gone-provider": _record("cn_goneStaleConn01", ["openai"]),
        }
    )
    # The wire-time resolver shares _route_view; a route with no live connection
    # must surface login_required, not a misleading method_unavailable.
    with pytest.raises(AuthOperationError) as exc:
        auth.request_auth("openai")
    assert exc.value.reason == "login_required"
    assert exc.value.provider == "openai"


def test_logout_skips_stale_slot_for_unregistered_provider():
    auth = _auth_with_slots(
        {
            "gone-provider": _record("cn_goneStaleConn01", ["openai"]),
        }
    )
    # logout resolves via _route_view too; it must not abort on the stale owner.
    result = auth.logout("openai")
    assert isinstance(result, ForgetResult)
    assert result.provider == "openai"
    assert result.forgot is False


def test_stale_owner_is_skipped_but_legitimate_owner_still_resolves():
    # A stale `gone-provider` slot advertising `meta-chat` sits alongside a real
    # `meta` slot that also advertises it. The stale owner is skipped and `meta`
    # (the live owner) wins resolution for the route.
    auth = _auth_with_slots(
        {
            "meta": _record("cn_meta01", ["meta", "meta-chat", "meta-anthropic"]),
            "gone-provider": _record("cn_gone", ["meta-chat"]),
        }
    )
    status = auth.status("meta-chat")
    assert status.presence == "saved"
    assert status.connection is not None
    assert status.connection.provider == "meta"
    assert status.connection.id == "cn_meta01"


def test_non_method_unavailable_descriptor_error_propagates(monkeypatch):
    # Only the "method_unavailable" reason (an unregistered owner) is tolerated.
    # Any other AuthOperationError from the descriptor lookup must still escape,
    # so the fix cannot mask unrelated failures.
    auth = _auth_with_slots(
        {
            "owner-x": _record("cn_x", ["shared"]),
        }
    )

    def raising_descriptor(provider: str) -> ProviderDescriptor:
        raise AuthOperationError(
            f"{provider}: a non-tolerated failure",
            reason="connection_changed",
            stage="resolution",
            recovery="select_connection",
            provider=provider,
        )

    monkeypatch.setattr(auth, "descriptor", raising_descriptor)
    with pytest.raises(AuthOperationError) as exc:
        auth._route_view(auth.store.read(), "shared")
    assert exc.value.reason == "connection_changed"
