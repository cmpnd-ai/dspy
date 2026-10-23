"""Regression tests for the GitHub Copilot ``enterprise_domain`` replacement-login bug.

The vendored login manager (``dspy/_vendor/lm15/login/``) stores a provider
*instance* setting, ``enterprise_domain``, on each saved Copilot connection and
reads it back at renewal time to decide which host serves the Copilot token
exchange (``https://api.<domain>/copilot_internal/v2/token``).

On a *replacement* login (``Auth.login(..., replace=<id>)``), the manager's
``_commit`` merges the previous connection's settings back in::

    merged = {**slot.settings, **settings, **result.settings} if slot.connection_id else {**settings, **result.settings}

``CopilotFlow.login`` used to return ``settings={"enterprise_domain": domain}``
only when the resolved domain differed from the default (``github.com``), and
``{}`` otherwise. So a blank (or explicit ``github.com``) answer on a replacement
login that was meant to switch a GHE connection *back* to ``github.com`` could
not clear the previously saved GHE domain: neither the caller's ``settings``
(empty at every in-repo call site) nor ``result.settings`` (``{}`` for the
default) carried ``enterprise_domain``, so the old GHE value survived the merge.
The login itself ran against ``github.com`` (the flow resolves the working
domain from caller settings + answers only), but the saved settings kept
routing renewals at the GHE Copilot endpoint, which rejected the
``github.com``-issued GitHub token with a 401 and flipped the connection to
``needs_login``.

The fix: the flow now always emits the resolved domain in
``LoginResult.settings`` so the replacement merge is a true override/clear.

These tests drive the real ``Auth`` and the real ``CopilotFlow`` through the
public ``Auth.login`` / ``Auth.request_auth`` APIs; only HTTP and the clock are
faked, via the ``Auth(opener=..., clock=..., monotonic=..., sleep=...)`` seams
(the same contract the vendored vet harness
``dspy/_vendor/lm15/_vet_managed.py`` uses). No network and no real GitHub.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from typing import Any

import pytest

from dspy._vendor.lm15.login import Auth, AuthOperationError
from dspy._vendor.lm15.login.engine import LoginContext
from dspy._vendor.lm15.login.flows.copilot import (
    DEFAULT_API_BASE,
    DEFAULT_DOMAIN,
    METHOD_DEVICE,
    CopilotFlow,
)
from dspy._vendor.lm15.login.store import META_KEY, STORE_VERSION, MemoryStore
from dspy._vendor.lm15.login.types import ConnectionStatus

PROVIDER = "github-copilot"
GHE = "company.ghe.com"


# ─── Test seams ────────────────────────────────────────────────────────


class _Clock:
    """Two hands (wall + monotonic) plus a fake sleep, all sharing state."""

    def __init__(self, start_ms: int) -> None:
        self.start_s = start_ms / 1000.0
        self.elapsed_s = 0.0

    def wall(self) -> float:
        return self.start_s + self.elapsed_s

    def mono(self) -> float:
        return self.elapsed_s

    def sleep(self, seconds: float) -> None:
        self.elapsed_s += seconds

    def advance(self, seconds: float) -> None:
        self.elapsed_s += seconds


class _Reply:
    """A scripted urllib-like reply (mirrors ``_vet_managed._Reply``)."""

    def __init__(self, status: int, body: bytes, content_type: str) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type} if content_type else {}
        self._body = body

    def read(self, _n: int = -1) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


class _CopilotServer:
    """A URL-aware scripted opener for the three Copilot HTTP calls.

    Recognises the device-code, token-poll and Copilot-exchange endpoints by
    path suffix and serves a canned JSON reply, recording every request. The
    Copilot-token reply's ``expires_at`` is computed from the shared fake clock
    so a short-lived token can be made renewable on demand. A GHE Copilot
    endpoint can be made to fail (e.g. 401) via ``ghe_copilot_status`` to model
    a foreign-host token being rejected there.
    """

    def __init__(
        self,
        clock: _Clock,
        *,
        copilot_status: int = 200,
        ghe_copilot_status: int | None = None,
        copilot_token: str = "ghu_COP_TOKEN",
    ) -> None:
        self.clock = clock
        self.copilot_status = copilot_status
        self.ghe_copilot_status = ghe_copilot_status
        self.copilot_token = copilot_token
        self.requests: list[dict[str, str]] = []

    def __call__(self, request: Any, _timeout: float) -> _Reply:
        method = request.get_method()
        url = request.full_url
        host = urllib.parse.urlsplit(url).hostname or ""
        self.requests.append({"method": method, "url": url, "host": host})
        if method == "POST" and url.endswith("/login/device/code"):
            return _Reply(
                200,
                json.dumps(
                    {
                        "device_code": "devcode-XYZ",
                        "user_code": "USER-CODE",
                        "verification_uri": f"https://{host}/login/device",
                        "interval": 1,
                        "expires_in": 900,
                    }
                ).encode(),
                "application/json",
            )
        if method == "POST" and url.endswith("/login/oauth/access_token"):
            return _Reply(200, json.dumps({"access_token": "ghu_GHTOKEN"}).encode(), "application/json")
        if method == "GET" and url.endswith("/copilot_internal/v2/token"):
            is_ghe = host not in ("api.github.com", "github.com")
            if is_ghe and self.ghe_copilot_status is not None:
                return _Reply(self.ghe_copilot_status, b"", "application/json")
            return _Reply(
                self.copilot_status,
                json.dumps(
                    {
                        "token": self.copilot_token,
                        "expires_at": self.clock.wall() + 60,
                    }
                ).encode(),
                "application/json",
            )
        raise AssertionError(f"unexpected request: {method} {url}")

    def copilot_gets(self) -> list[dict[str, str]]:
        return [r for r in self.requests if r["method"] == "GET" and r["url"].endswith("/copilot_internal/v2/token")]

    def posts_to(self, suffix: str) -> list[dict[str, str]]:
        return [r for r in self.requests if r["method"] == "POST" and r["url"].endswith(suffix)]


class _UI:
    """A minimal ``AuthUI`` that records notices and returns scripted answers.

    With no scripted answers it returns ``""`` for every prompt, modelling the
    user leaving the ``enterprise_domain`` field (label: "blank for github.com")
    blank via the interactive prompt path.
    """

    def __init__(self, answers: list[str] | None = None) -> None:
        self._answers = list(answers or [])
        self.notices: list[Any] = []

    def prompt(self, prompt: Any) -> str:
        if self._answers:
            return self._answers.pop(0)
        return ""

    def notify(self, notice: Any) -> None:
        self.notices.append(notice)

    def dismiss(self, _prompt: Any) -> None:
        return None


def _auth(server: _CopilotServer, clock: _Clock) -> Auth:
    return Auth(MemoryStore(), clock=clock.wall, monotonic=clock.mono, opener=server, sleep=clock.sleep)


def _login(
    auth: Auth,
    server: _CopilotServer,
    *,
    answers: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    replace: str | None = None,
) -> Any:
    return auth.login(
        PROVIDER, "device", ui=_UI(), answers=answers, settings=settings, replace=replace, allow_unverified=True
    )


def _connection_record(
    settings: dict[str, str], *, connection_id: str = "cn_seed00000001", domain: str = "github.com", generation: int = 1
) -> dict[str, Any]:
    return {
        "generation": str(generation),
        "connection_id": connection_id,
        "revision": "1",
        "kind": "account",
        "method_id": "device",
        "instance_id": "public",
        "label": "GitHub Copilot" if domain == DEFAULT_DOMAIN else f"GitHub Copilot ({domain})",
        "created_at": "2026-01-01T00:00:00Z",
        "routes": [PROVIDER],
        "settings": dict(settings),
        "state": "ready",
        "renewal": "remint",
    }


def _copilot_material(
    *, expires_ms: int, access: str = "ghu_COP_TOKEN", refresh: str = "ghu_GHTOKEN"
) -> dict[str, Any]:
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "issued_at": 0,
        "lifetime_s": 60.0,
        "expires": expires_ms,
    }


def _seed_store(
    store: MemoryStore, *, domain: str, expires_ms: int, connection_id: str = "cn_seed00000001", generation: int = 1
) -> None:
    slot = _connection_record(
        {"enterprise_domain": domain}, connection_id=connection_id, domain=domain, generation=generation
    )
    material = _copilot_material(expires_ms=expires_ms)

    def seed(doc: dict[str, Any]) -> dict[str, Any]:
        return {META_KEY: {"version": STORE_VERSION, "slots": {PROVIDER: slot}}, PROVIDER: material}

    store.mutate(seed)


# ─── Flow-layer tests: the one-line fix, sharply ───────────────────────


def _flow_ctx(server: _CopilotServer, clock: _Clock) -> LoginContext:
    return LoginContext(
        ui=_UI(),
        deadline=clock.mono() + 300.0,
        cancel=threading.Event(),
        provider=PROVIDER,
        clock=clock.mono,
        wall_clock=clock.wall,
        opener=server,
        sleep=clock.sleep,
    )


def test_flow_login_blank_answer_emits_default_domain():
    """A blank ``enterprise_domain`` answer resolves to ``github.com`` and the
    flow now emits it in ``LoginResult.settings`` (previously ``{}``)."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    ctx = _flow_ctx(server, clock)
    result = CopilotFlow().login(ctx, METHOD_DEVICE, settings={}, answers={"enterprise_domain": ""})
    assert result.settings == {"enterprise_domain": DEFAULT_DOMAIN}
    assert result.label == "GitHub Copilot"


def test_flow_login_explicit_github_com_emits_default_domain():
    """An explicit ``github.com`` answer also emits the default domain so it
    can override a stale saved value on replacement."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    ctx = _flow_ctx(server, clock)
    result = CopilotFlow().login(ctx, METHOD_DEVICE, settings={}, answers={"enterprise_domain": "github.com"})
    assert result.settings == {"enterprise_domain": DEFAULT_DOMAIN}
    assert result.label == "GitHub Copilot"


def test_flow_login_ghe_domain_emits_ghe_domain():
    """A GHE answer still emits the resolved GHE domain unchanged — the
    intended "instance setting survives replacement" behavior for GHE re-logins."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    ctx = _flow_ctx(server, clock)
    result = CopilotFlow().login(ctx, METHOD_DEVICE, settings={}, answers={"enterprise_domain": GHE})
    assert result.settings == {"enterprise_domain": GHE}
    assert result.label == f"GitHub Copilot ({GHE})"


# ─── Manager-layer tests: the real bug scenario, end to end ───────────


def test_fresh_ghe_login_persists_domain_and_targets_ghe():
    """No regression: a fresh GHE login commits the GHE domain and the device
    flow + exchange all hit the GHE host."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    conn = _login(auth, server, answers={"enterprise_domain": GHE})
    assert conn.settings == {"enterprise_domain": GHE}
    assert conn.label == f"GitHub Copilot ({GHE})"
    assert server.posts_to("/login/device/code")[0]["host"] == GHE
    assert server.posts_to("/login/oauth/access_token")[0]["host"] == GHE
    assert server.copilot_gets()[0]["host"] == f"api.{GHE}"


def test_replacement_blank_answer_clears_stale_ghe_domain():
    """The bug scenario: switching an existing GHE connection back to github.com
    with a blank answer commits ``enterprise_domain == "github.com"`` and the
    login visibly runs against github.com. On the buggy code the stale GHE
    domain survived the commit merge, leaving the connection internally
    inconsistent (a github.com label with GHE settings)."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    first = _login(auth, server, answers={"enterprise_domain": GHE})
    assert first.settings == {"enterprise_domain": GHE}

    before = len(server.requests)
    second = _login(auth, server, answers={"enterprise_domain": ""}, replace=first.id)
    after = len(server.requests)
    # The replacement login itself ran against github.com (the flow resolves
    # its working domain from caller settings + answers, never the old slot).
    replacement_hosts = {r["host"] for r in server.requests[before:after]}
    assert GHE not in replacement_hosts
    assert "github.com" in replacement_hosts
    # And the committed connection now reflects github.com — the fix.
    assert second.settings == {"enterprise_domain": DEFAULT_DOMAIN}
    assert second.label == "GitHub Copilot"
    assert second.id != first.id


def test_replacement_explicit_github_com_clears_stale_ghe_domain():
    """An explicit ``github.com`` answer also clears the stale GHE domain (the
    bug report notes even this could not clear it before the fix)."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    first = _login(auth, server, answers={"enterprise_domain": GHE})
    second = _login(auth, server, answers={"enterprise_domain": "github.com"}, replace=first.id)
    assert second.settings == {"enterprise_domain": DEFAULT_DOMAIN}
    assert second.label == "GitHub Copilot"


def test_fresh_github_com_login_commits_explicit_default_domain():
    """A fresh login via the interactive blank-field path (no answers, so the
    manager prompts ``enterprise_domain`` and the UI returns ``""``) commits
    ``enterprise_domain == "github.com"`` explicitly; the label is plain."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    conn = _login(auth, server, answers=None)
    assert conn.settings == {"enterprise_domain": DEFAULT_DOMAIN}
    assert conn.label == "GitHub Copilot"


def test_ghe_relogin_preserves_ghe_domain():
    """No regression: a replacement login that re-answers the same GHE domain
    still commits the GHE domain (the deliberate instance-setting survival)."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    first = _login(auth, server, answers={"enterprise_domain": GHE})
    second = _login(auth, server, answers={"enterprise_domain": GHE}, replace=first.id)
    assert second.settings == {"enterprise_domain": GHE}
    assert second.label == f"GitHub Copilot ({GHE})"


def test_caller_settings_override_saved_ghe_domain():
    """The merge precedence ``slot < caller-settings < result`` is intact: a
    caller-supplied ``settings={"enterprise_domain": ...}`` overrides the saved
    GHE domain even with a blank answer."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    first = _login(auth, server, answers={"enterprise_domain": GHE})
    other = "other.ghe.com"
    second = _login(
        auth, server, answers={"enterprise_domain": ""}, settings={"enterprise_domain": other}, replace=first.id
    )
    assert second.settings == {"enterprise_domain": other}
    assert second.label == f"GitHub Copilot ({other})"


# ─── Renewal: the deferred failure the stale domain caused ────────────


def test_renewal_after_clear_targets_github_com_and_stays_ready():
    """The user-reported symptom: after a clearing replacement, fast-forwarding
    past the token's ``expires_at`` and forcing a renewal hits
    ``https://api.github.com/copilot_internal/v2/token`` (not the GHE host) and
    the connection stays ``ready``. Before the fix the stale GHE settings sent
    the renewal to the GHE endpoint with a github.com-issued token, got a 401,
    and flipped the connection to ``needs_login``."""
    clock = _Clock(start_ms=1_700_000_000_000)
    server = _CopilotServer(clock)
    auth = _auth(server, clock)
    first = _login(auth, server, answers={"enterprise_domain": GHE})
    _login(auth, server, answers={"enterprise_domain": ""}, replace=first.id)

    n_before = len(server.requests)
    clock.advance(120.0)  # past the short-lived Copilot token's expires_at
    snapshot = auth.request_auth(PROVIDER)
    copilot_gets_after = [
        r
        for r in server.requests[n_before:]
        if r["method"] == "GET" and r["url"].endswith("/copilot_internal/v2/token")
    ]
    assert copilot_gets_after, "a renewal Copilot-token GET should have been issued"
    assert copilot_gets_after[-1]["host"] == "api.github.com"
    assert GHE not in copilot_gets_after[-1]["url"]
    # The renewed request auth resolves to a github.com Copilot API base, not GHE.
    assert snapshot.base_url == DEFAULT_API_BASE
    status = auth.status(PROVIDER)
    assert isinstance(status, ConnectionStatus)
    assert status.usability == "ready"
    assert status.connection is not None
    assert status.connection.settings == {"enterprise_domain": DEFAULT_DOMAIN}


def test_renewal_with_stale_ghe_settings_is_rejected_as_credential_rejected():
    """Pins the renewal→401→``needs_login`` machinery the stale-domain bug used
    to trigger. A connection carrying a stale GHE ``enterprise_domain`` but a
    github.com-issued GitHub token cannot renew: the GHE Copilot endpoint
    rejects the foreign token with 401, and the connection is marked
    ``needs_login`` with ``reason="credential_rejected"``. The fix prevents
    reaching this state after a clearing replacement; this test locks the
    rejection contract so the symptom stays diagnosable."""
    clock = _Clock(start_ms=1_700_000_000_000)
    # GHE Copilot endpoint returns 401 for a github.com-issued token.
    server = _CopilotServer(clock, ghe_copilot_status=401)
    auth = _auth(server, clock)
    store = auth.store
    assert isinstance(store, MemoryStore)
    # Seed a connection whose saved settings still route renewals at GHE even
    # though its GitHub token was issued by github.com — exactly the internally
    # inconsistent state the buggy commit merge left behind.
    start_ms = int(clock.wall() * 1000)
    _seed_store(store, domain=GHE, expires_ms=start_ms - 1)  # already expired → renewal is due

    with pytest.raises(AuthOperationError) as exc:
        auth.request_auth(PROVIDER)
    assert exc.value.reason == "credential_rejected"
    # The renewal tried the GHE Copilot endpoint, not github.com.
    copilot_gets = server.copilot_gets()
    assert copilot_gets, "a renewal Copilot-token GET should have been issued"
    assert copilot_gets[-1]["host"] == f"api.{GHE}"
    status = auth.status(PROVIDER)
    assert isinstance(status, ConnectionStatus)
    assert status.usability == "needs_login"
