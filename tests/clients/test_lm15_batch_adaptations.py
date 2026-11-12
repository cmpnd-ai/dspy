"""Batch submission honours adaptations="refuse" on the async path.

``AsyncBaseProviderLM.batch_submit`` is documented as a perfect mirror of the
sync ``BaseProviderLM.batch_submit`` — ``await`` is the only user-visible
difference.  Both must enforce the MAP-13 adaptation policy around the inner
adapter's batch builders (``_batch_upload_request`` / ``_batch_submit_request``)
so ``adaptations="refuse"`` fires before any paid provider work.

Before the fix the async driver called the inner builders through its own
``_build()``, a thread-offload wrapper that opens no
``collecting(check_policy(...))`` scope, so ``adapt()`` inside a builder
silently degraded to the no-op ``"note"`` default: a labeled Anthropic batch
was submitted (a real POST to the paid batches queue) with the label silently
dropped, while the sync path raised ``UnsupportedFeatureError``.  These tests
pin the parity and guard the regression.
"""

import json

import pytest

from dspy._vendor.lm15.providers.base import HttpResponse
from dspy.lm15 import (
    AnthropicLM,
    AsyncAnthropicLM,
    BatchRequest,
    Message,
    Request,
    UnsupportedFeatureError,
)

MODEL = "claude-sonnet-4-5"


def _job_body(batch_id="batch_123", status="in_progress"):
    return json.dumps({"id": batch_id, "processing_status": status}).encode()


def _nested(model=MODEL):
    return Request(model=model, messages=(Message.user("hi"),))


def _batch(label, model=MODEL):
    return BatchRequest(model=model, requests=(_nested(model),), label=label)


def _ok_response():
    return HttpResponse(
        status=200,
        reason="OK",
        headers=[("content-type", "application/json")],
        body=_job_body(),
        provider="anthropic",
    )


def _sync_send(sink):
    """A sync ``_send`` replacement that records the request and replies 200."""

    def _send(request):
        sink.append(request)
        return _ok_response()

    return _send


def _async_send(sink):
    """An async ``_send`` replacement that records the request and replies 200."""

    async def _send(request):
        sink.append(request)
        return _ok_response()

    return _send


# ── The bug: "refuse" + a batch-level deviation must raise before I/O ───


@pytest.mark.asyncio
async def test_async_refuse_label_raises_before_any_io():
    alm = AsyncAnthropicLM(api_key="fake", adaptations="refuse")
    sink: list = []
    alm._send = _async_send(sink)
    try:
        with pytest.raises(UnsupportedFeatureError, match="label would be dropped") as exc:
            await alm.batch_submit(_batch(label="my-run"))
    finally:
        await alm.aclose()
    # The refusal happens at BUILD time, before any paid provider work.
    assert exc.value.feature == "label"
    assert sink == [], "no transport call must be made when refuse fires"


def test_sync_refuse_label_raises_before_any_io():
    slm = AnthropicLM(api_key="fake", adaptations="refuse")
    sink: list = []
    slm._send = _sync_send(sink)
    with pytest.raises(UnsupportedFeatureError, match="label would be dropped") as exc:
        slm.batch_submit(_batch(label="my-run"))
    assert exc.value.feature == "label"
    assert sink == [], "no transport call must be made when refuse fires"


@pytest.mark.asyncio
async def test_async_refuse_error_matches_sync_refuse_error():
    """The async path must refuse with the SAME message as the sync path."""
    sync_err: UnsupportedFeatureError | None = None
    async_err: UnsupportedFeatureError | None = None

    slm = AnthropicLM(api_key="fake", adaptations="refuse")
    slm._send = _sync_send([])
    with pytest.raises(UnsupportedFeatureError) as exc:
        slm.batch_submit(_batch(label="run-42"))
    sync_err = exc.value

    alm = AsyncAnthropicLM(api_key="fake", adaptations="refuse")
    alm._send = _async_send([])
    try:
        with pytest.raises(UnsupportedFeatureError) as exc:
            await alm.batch_submit(_batch(label="run-42"))
        async_err = exc.value
    finally:
        await alm.aclose()

    assert async_err is not None and sync_err is not None
    assert str(async_err) == str(sync_err)
    assert async_err.feature == sync_err.feature == "label"
    assert async_err.provider == sync_err.provider == "anthropic"


# ── Regression: "note" still submits with the label dropped on the wire ──


@pytest.mark.parametrize("asynchronous", [False, True])
def test_note_label_submits_and_drops_label_from_body(asynchronous):
    if asynchronous:
        alm = AsyncAnthropicLM(api_key="fake", adaptations="note")
        sink: list = []
        alm._send = _async_send(sink)

        async def run():
            try:
                return await alm.batch_submit(_batch(label="my-run"))
            finally:
                await alm.aclose()

        import asyncio

        job = asyncio.run(run())
        sent = sink
    else:
        slm = AnthropicLM(api_key="fake", adaptations="note")
        sink2: list = []
        slm._send = _sync_send(sink2)
        job = slm.batch_submit(_batch(label="my-run"))
        sent = sink2

    assert job.id == "batch_123"
    assert len(sent) == 1, "Anthropic batch is single-step (no upload)"
    body = json.loads(sent[0].body.decode())
    # The Anthropic Message Batches body has no metadata field; the label is
    # dropped (recorded under "note", never sent).
    assert "my-run" not in sent[0].body.decode()
    assert "label" not in body


# ── Controls: "refuse" with no deviation submits normally ──────────────


@pytest.mark.parametrize("asynchronous", [False, True])
def test_refuse_without_label_submits(asynchronous):
    if asynchronous:
        alm = AsyncAnthropicLM(api_key="fake", adaptations="refuse")
        sink: list = []
        alm._send = _async_send(sink)

        async def run():
            try:
                return await alm.batch_submit(_batch(label=None))
            finally:
                await alm.aclose()

        import asyncio

        job = asyncio.run(run())
        assert len(sink) == 1
    else:
        slm = AnthropicLM(api_key="fake", adaptations="refuse")
        sink2: list = []
        slm._send = _sync_send(sink2)
        job = slm.batch_submit(_batch(label=None))
        assert len(sink2) == 1
    assert job.id == "batch_123"


# ── The contextvar is copied into the asyncio.to_thread worker ─────────
#
# When the credential is callable, _build offloads the builder to a worker
# thread (asyncio.to_thread).  collecting() opens a contextvar; asyncio.to_thread
# copies the current context, so adapt() inside the offloaded builder still
# sees the "refuse" scope and raises — this is the guarantee that makes the
# fix correct for cloud-chain (callably-credentialled) adapters too.


@pytest.mark.asyncio
async def test_async_refuse_label_raises_with_callable_credential():
    alm = AsyncAnthropicLM(api_key=lambda: "fake-key", adaptations="refuse")
    assert callable(alm._inner.api_key), "fixture must exercise the to_thread branch"
    sink: list = []
    alm._send = _async_send(sink)
    try:
        with pytest.raises(UnsupportedFeatureError, match="label would be dropped") as exc:
            await alm.batch_submit(_batch(label="my-run"))
    finally:
        await alm.aclose()
    assert exc.value.feature == "label"
    assert sink == [], "refuse must fire inside the worker thread before any I/O"
