"""Regression tests for the raced manual-prompt cancellation path.

``race_callback_and_manual`` races a loopback callback listener against a
manual paste prompt that runs on a daemon thread.  The threaded wrapper must
recognize the protocol-documented ``LoginCancelled`` signal (raised by a
conforming ``AuthUI`` to cancel) as a cancellation, not as a UI bug.

See ``dspy/_vendor/lm15/login/engine.py`` and the ``AuthUI`` contract in
``dspy/_vendor/lm15/login/types.py``.
"""

import threading
import time

import pytest

from dspy._vendor.lm15.errors import AuthOperationError
from dspy._vendor.lm15.login.engine import (
    CallbackListener,
    CallbackReturn,
    LoginCancelled,
    LoginContext,
    await_return,
    parse_manual_return,
    race_callback_and_manual,
)
from dspy._vendor.lm15.login.types import ManualCodePrompt

_PATH = "/cb"
_STATE = "state-xyz"


def _far_deadline() -> float:
    return time.monotonic() + 300.0


def _make_ctx(ui: object) -> LoginContext:
    return LoginContext(
        ui=ui,  # type: ignore[arg-type]
        deadline=_far_deadline(),
        cancel=threading.Event(),
    )


def _manual_prompt() -> ManualCodePrompt:
    return ManualCodePrompt(field_id="manual_code", label="Paste the redirect URL or code")


class _ScriptedUi:
    """A minimal ``AuthUI`` that raises a configured exception from ``prompt``."""

    def __init__(self, *, prompt_exc: BaseException) -> None:
        self._prompt_exc = prompt_exc

    def prompt(self, prompt: ManualCodePrompt) -> str:
        raise self._prompt_exc

    def notify(self, notice: object) -> None:
        pass

    def dismiss(self, prompt: ManualCodePrompt) -> None:
        pass


def _listener() -> CallbackListener:
    return CallbackListener(path=_PATH, expected_state=_STATE)


def test_raced_login_cancelled_is_clean_cancellation():
    """A UI raising ``LoginCancelled`` during the race is a cancellation, not a
    UI-failure operation error (AUTH-24 conformance outcome ``cancelled``)."""
    ctx = _make_ctx(_ScriptedUi(prompt_exc=LoginCancelled("ui cancelled")))
    listener = _listener()
    listener.start()
    try:
        with pytest.raises(LoginCancelled):
            race_callback_and_manual(ctx, listener, _manual_prompt())
    finally:
        listener.stop()


def test_raced_genuine_ui_bug_is_sanitized_failure():
    """An unexpected exception from the UI is still classified as a UI failure,
    never leaked into headless mode — so broadening the cancellation tuple is
    not a silent footgun."""
    ctx = _make_ctx(_ScriptedUi(prompt_exc=ValueError("the UI is broken")))
    listener = _listener()
    listener.start()
    try:
        with pytest.raises(AuthOperationError) as exc_info:
            race_callback_and_manual(ctx, listener, _manual_prompt())
        assert exc_info.value.reason == "interaction_required"
        assert exc_info.value.recovery == "operator_action"
        assert "ValueError" in str(exc_info.value)
    finally:
        listener.stop()


def test_await_return_propagates_cancellation_from_race():
    """``await_return`` only retries ``invalid_login_state``; a raced cancellation
    propagates to the caller as a bare ``LoginCancelled``."""

    def parse(text: str) -> CallbackReturn:
        return parse_manual_return(text, expected_state=_STATE, allow_bare_code=False, registered_path=_PATH)

    ctx = _make_ctx(_ScriptedUi(prompt_exc=LoginCancelled("ui cancelled")))
    listener = _listener()
    listener.start()
    try:
        with pytest.raises(LoginCancelled):
            await_return(ctx, listener, _manual_prompt(), parse)
    finally:
        listener.stop()
