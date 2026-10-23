"""Tests for the nanosecond-expireTime fix in parse_rfc3339 / _gcp_impersonate.

Bug: on Python 3.10, ``datetime.fromisoformat`` rejects >6 fractional digits, so a
legal GCP IAM ``expireTime`` with 7-9 fractional digits crashed the credential chain
with a ``ValueError``.  The fix trims the fraction to microseconds inside
``parse_rfc3339`` and guards the ``_gcp_impersonate`` call with ``try/except ValueError``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dspy._vendor.lm15.cloud.chains import ChainContext, _gcp_impersonate
from dspy._vendor.lm15.credentials import (
    BearerToken,
    credential_from_dict,
    credential_to_dict,
    parse_rfc3339,
)
from dspy._vendor.lm15.errors import AuthError

# ─── parse_rfc3339: fractional-second normalization ──────────────────


@pytest.mark.parametrize(
    "value, expected_microsecond",
    [
        ("2026-09-03T12:00:00.123456789Z", 123456),
        ("2026-09-03T12:00:00.987654321Z", 987654),
        ("2026-09-03T12:00:00.999999999Z", 999999),
    ],
    ids=["nanosecond-9", "nanosecond-9-alt", "max-nanos"],
)
def test_parse_rfc3339_nanosecond_truncates_to_microsecond(value, expected_microsecond):
    result = parse_rfc3339(value)
    assert result == datetime(2026, 9, 3, 12, 0, 0, expected_microsecond, tzinfo=timezone.utc)


def test_parse_rfc3339_seven_digit_truncates():
    result = parse_rfc3339("2026-09-03T12:00:00.1234567Z")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_eight_digit_truncates():
    result = parse_rfc3339("2026-09-03T12:00:00.12345678Z")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_microsecond_unchanged():
    result = parse_rfc3339("2026-09-03T12:00:00.123456Z")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_millisecond_unchanged():
    result = parse_rfc3339("2026-09-03T12:00:00.123Z")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123000, tzinfo=timezone.utc)


def test_parse_rfc3339_whole_second_unchanged():
    result = parse_rfc3339("2026-09-03T12:00:00Z")
    assert result == datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_rfc3339_lowercase_z():
    result = parse_rfc3339("2026-09-03T12:00:00.123456789z")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_numeric_offset():
    result = parse_rfc3339("2026-09-03T12:00:00.123456789+00:00")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_negative_offset_converts_to_utc():
    result = parse_rfc3339("2026-09-03T14:00:00.123456789-02:00")
    assert result == datetime(2026, 9, 3, 16, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_whitespace_stripped():
    result = parse_rfc3339("  2026-09-03T12:00:00.123456789Z  ")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_parse_rfc3339_naive_gets_utc():
    result = parse_rfc3339("2026-09-03T12:00:00")
    assert result == datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_rfc3339_naive_with_nanos_gets_utc():
    result = parse_rfc3339("2026-09-03T12:00:00.123456789")
    assert result == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


# ─── _gcp_impersonate: ValueError guard ──────────────────────────────


_FAKE_URL = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@test.iam.gserviceaccount.com:generateAccessToken"


def _ctx_with_http(respond_with: dict):
    def fake_http(method, url, headers, body, timeout):
        return (200, {}, json.dumps(respond_with).encode())

    return ChainContext(
        env={},
        http=fake_http,
        now=lambda: datetime(2026, 9, 3, 11, 0, 0, tzinfo=timezone.utc),
    )


def test_gcp_impersonate_nanosecond_expire_time():
    ctx = _ctx_with_http({"accessToken": "ya29.test", "expireTime": "2026-09-03T12:00:00.123456789Z"})
    token = _gcp_impersonate(ctx, BearerToken("source-token"), _FAKE_URL, [], "test")
    assert token.value == "ya29.test"
    assert token.expires_at == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_gcp_impersonate_seven_digit_expire_time():
    ctx = _ctx_with_http({"accessToken": "ya29.test", "expireTime": "2026-09-03T12:00:00.1234567Z"})
    token = _gcp_impersonate(ctx, BearerToken("source-token"), _FAKE_URL, [], "test")
    assert token.value == "ya29.test"
    assert token.expires_at == datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_gcp_impersonate_malformed_expire_time_degrades():
    ctx = _ctx_with_http({"accessToken": "ya29.test", "expireTime": "not-a-valid-timestamp"})
    token = _gcp_impersonate(ctx, BearerToken("source-token"), _FAKE_URL, [], "test")
    assert token.value == "ya29.test"
    assert token.expires_at is None


def test_gcp_impersonate_empty_expire_time_degrades():
    ctx = _ctx_with_http({"accessToken": "ya29.test", "expireTime": ""})
    token = _gcp_impersonate(ctx, BearerToken("source-token"), _FAKE_URL, [], "test")
    assert token.value == "ya29.test"
    assert token.expires_at is None


def test_gcp_impersonate_no_expire_time():
    ctx = _ctx_with_http({"accessToken": "ya29.test"})
    token = _gcp_impersonate(ctx, BearerToken("source-token"), _FAKE_URL, [], "test")
    assert token.value == "ya29.test"
    assert token.expires_at is None


def test_gcp_impersonate_missing_access_token_raises():
    ctx = _ctx_with_http({"expireTime": "2026-09-03T12:00:00.123456789Z"})
    with pytest.raises(AuthError, match="no accessToken"):
        _gcp_impersonate(ctx, BearerToken("source-token"), _FAKE_URL, [], "test")


# ─── Credential round-trip with nanosecond timestamps ────────────────


def test_credential_roundtrip_with_nanosecond_expires_at():
    expires = parse_rfc3339("2026-09-03T12:00:00.123456789Z")
    original = BearerToken("secret-token", expires_at=expires)
    roundtripped = credential_from_dict(credential_to_dict(original))
    assert isinstance(roundtripped, BearerToken)
    assert roundtripped.value == "secret-token"
    # format_rfc3339 serializes to whole seconds, so microseconds are dropped.
    assert roundtripped.expires_at == datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_credential_to_dict_formats_expires_at_as_whole_seconds():
    expires = parse_rfc3339("2026-09-03T12:00:00.123456789Z")
    d = credential_to_dict(BearerToken("tok", expires_at=expires))
    assert d["expires_at"] == "2026-09-03T12:00:00Z"
    assert d["kind"] == "bearer_token"
