"""Regression tests for the AWS SSO token refresh path in
``dspy._vendor.lm15.cloud.chains``.

These pin the behaviour the PR ``fix(dspy): persist refreshed AWS SSO token to
cache file`` introduced:

* a rotated ``refreshToken`` is persisted to the cache file;
* a second resolution does not refresh again when the cached token is fresh;
* ``ChainContext.write`` replaces a ``~``-prefixed entry by normalized path
  instead of shadowing it with a second absolute-path key (for dict and
  non-dict Mappings alike);
* while waiting for the cross-process lock, a token refreshed by another
  process is reused instead of refreshing again.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from dspy._vendor.lm15.cloud.chains import ChainContext, _sso_acquire, _sso_refresh_token
from dspy._vendor.lm15.credentials import format_rfc3339


def _token(*, access: str = "old-access", refresh: str = "old-refresh",
           expires_at: str | None = None) -> dict:
    return {
        "accessToken": access,
        "refreshToken": refresh,
        "clientId": "cid",
        "clientSecret": "csec",
        "expiresAt": expires_at or format_rfc3339(datetime.now(timezone.utc) - timedelta(hours=1)),
    }


def _refresh_body(*, access: str = "new-access", refresh: str | None = "new-refresh",
                  expires_in: int = 28800) -> dict:
    body: dict = {"accessToken": access, "expiresIn": expires_in}
    if refresh is not None:
        body["refreshToken"] = refresh
    return body


def _federation_body() -> dict:
    return {"roleCredentials": {
        "AccessKeyId": "AKIATEST", "SecretAccessKey": "secret",
        "SessionToken": "tok",
        "Expiration": format_rfc3339(datetime.now(timezone.utc) + timedelta(hours=1)),
    }}


class _FakeHttp:
    """Records refresh vs federation round trips and serves scripted bodies."""

    def __init__(self, refresh: list[dict] | None = None, federation: list[dict] | None = None) -> None:
        self.refresh_calls = 0
        self.federation_calls = 0
        self._refresh = list(refresh or [])
        self._federation = list(federation or [])

    def __call__(self, method, url, headers, body, timeout):
        if "oidc" in url and url.endswith("/token"):
            self.refresh_calls += 1
            return 200, {"content-type": "application/json"}, json.dumps(self._refresh.pop(0)).encode()
        if "federation/credentials" in url:
            self.federation_calls += 1
            return 200, {"content-type": "application/json"}, json.dumps(self._federation.pop(0)).encode()
        raise AssertionError(f"unexpected HTTP {method} {url}")


def _sso_config_text(start_url: str = "https://example.awsapps.com/start") -> str:
    return "\n".join([
        "[default]",
        f"sso_start_url = {start_url}",
        "sso_region = us-east-1",
        "sso_account_id = 111122223333",
        "sso_role_name = MyRole",
        "",
    ])


def _sso_files(cache_token: dict, *, start_url: str = "https://example.awsapps.com/start") -> dict:
    cache_key = hashlib.sha1(start_url.encode("utf-8")).hexdigest()
    return {
        "~/.aws/config": _sso_config_text(start_url),
        f"~/.aws/sso/cache/{cache_key}.json": json.dumps(cache_token),
    }


def test_sso_refresh_persists_rotated_refresh_token():
    now = datetime.now(timezone.utc)
    http = _FakeHttp(refresh=[_refresh_body(access="new-access", refresh="rotated-refresh")],
                    federation=[_federation_body()])
    ctx = ChainContext(env={}, files=_sso_files(_token()), http=http,
                       now=lambda: now)
    cache_path = "~/.aws/sso/cache/" + hashlib.sha1(b"https://example.awsapps.com/start").hexdigest() + ".json"

    access = _sso_acquire(ctx)

    assert access.access_key_id == "AKIATEST"
    raw = ctx.read(cache_path)
    assert raw is not None
    saved = json.loads(raw)
    assert saved["accessToken"] == "new-access"
    assert saved["refreshToken"] == "rotated-refresh"
    # The rotated refresh token replaces (not joins) the old one.
    assert "old-refresh" not in saved["refreshToken"]


def test_sso_acquire_does_not_refresh_twice_when_cache_is_fresh():
    now = datetime.now(timezone.utc)
    http = _FakeHttp(refresh=[_refresh_body(access="new-access", refresh="rotated-refresh")],
                    federation=[_federation_body(), _federation_body()])
    ctx = ChainContext(env={}, files=_sso_files(_token()), http=http,
                       now=lambda: now)

    first = _sso_acquire(ctx)
    second = _sso_acquire(ctx)

    assert first.access_key_id == second.access_key_id == "AKIATEST"
    assert http.refresh_calls == 1, "the second resolution must reuse the saved token, not refresh again"
    assert http.federation_calls == 2


@pytest.mark.parametrize("mapping_factory", [dict, MappingProxyType],
                         ids=["dict", "mapping-proxy"])
def test_chain_context_write_replaces_normalized_path_alias(mapping_factory):
    now = datetime.now(timezone.utc)
    cache_key = hashlib.sha1(b"https://example.awsapps.com/start").hexdigest()
    cache_path = f"~/.aws/sso/cache/{cache_key}.json"
    files = mapping_factory({
        "~/.aws/config": _sso_config_text(),
        cache_path: json.dumps(_token()),
    })
    http = _FakeHttp(refresh=[_refresh_body(access="written-access", refresh="written-refresh")],
                    federation=[_federation_body()])
    ctx = ChainContext(env={}, files=files, http=http, now=lambda: now)

    _sso_acquire(ctx)

    # The ~ -prefixed key is the only entry for this cache path; read returns
    # the refreshed content, not a stale alias.
    assert ctx.read(cache_path) is not None
    if isinstance(ctx.files, dict):
        paths = {str(ctx.path(k)) for k in ctx.files}
    else:
        paths = {str(ctx.path(k)) for k in dict(ctx.files)}
    assert sum(1 for p in paths if p == str(ctx.path(cache_path))) == 1
    saved = json.loads(ctx.read(cache_path))
    assert saved["accessToken"] == "written-access"
    assert saved["refreshToken"] == "written-refresh"


def test_sso_refresh_reuses_token_refreshed_by_another_process_under_lock(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    start_url = "https://example.awsapps.com/start"
    cache_key = hashlib.sha1(start_url.encode("utf-8")).hexdigest()
    cache_path = f"~/.aws/sso/cache/{cache_key}.json"
    home = tmp_path
    cache_file = home / ".aws" / "sso" / "cache" / f"{cache_key}.json"
    cache_file.parent.mkdir(parents=True)
    # Another process already wrote a fresh token while we waited for the lock.
    fresh = {
        "accessToken": "winner-access",
        "refreshToken": "winner-refresh",
        "clientId": "cid", "clientSecret": "csec",
        "expiresAt": format_rfc3339(now + timedelta(hours=2)),
    }
    cache_file.write_text(json.dumps(fresh), encoding="utf-8")
    monkeypatch.setenv("LM15_LOCK_DIR", str(tmp_path / "locks"))

    class _BoomHttp:
        def __call__(self, *args):
            raise AssertionError("refresh must not be called when a fresh token is already cached")

    # files=None forces the real-filesystem, locked path in _sso_refresh_token.
    ctx = ChainContext(env={}, home=home, files=None, http=_BoomHttp(), now=lambda: now)

    access = _sso_refresh_token(ctx, cache_path, _token(), "us-east-1")

    assert access == "winner-access"
