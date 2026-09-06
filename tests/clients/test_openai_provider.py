"""Regression tests for the OpenAI fine-tune cancellation race.

These tests exercise the real ``LM.finetune`` -> ``LM._run_finetune_job`` ->
``OpenAIProvider.finetune`` -> ``TrainingJobOpenAI.cancel`` code paths, mocking
only the ``openai.*`` SDK boundary so no network is required.

They pin the worker thread at deterministic points (via ``threading.Event``
pairs) to reproduce the two timing windows in which ``cancel()`` can land while
a fine-tune is in flight, and assert that the worker thread never dies with an
uncaught ``concurrent.futures.InvalidStateError`` and that ``job.result()``
surfaces ``CancelledError`` cleanly.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from contextlib import ExitStack
from unittest import mock

import openai
import pytest

import dspy
import dspy.clients.openai as oai_mod
from dspy.clients.openai import OpenAIProvider
from dspy.clients.utils_finetune import TrainDataFormat

# Real openai SDK raises ValueError on a None id (verified against openai>=1.66);
# the mocks below mirror that behavior so the worker observes the same
# exception it would see against a live account.
_NONE_JOB_MSG = "Expected a non-empty value for `fine_tuning_job_id` but received None"
_NONE_FILE_MSG = "Expected a non-empty value for `file_id` but received None"

TRAIN_DATA = [
    {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great, thank you!"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris!"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "Berlin!"},
        ]
    },
]


class _FakeFile:
    def __init__(self, file_id):
        self.id = file_id


class _FakeJob:
    def __init__(self, job_id, *, status="running", estimated_finish=None, fine_tuned_model=None):
        self.id = job_id
        self.status = status
        self.estimated_finish = estimated_finish
        self.fine_tuned_model = fine_tuned_model


class _FakePage:
    def __init__(self, data=None):
        self.data = data or []


class _Gate:
    """A pair of threading.Events used to pause/release the worker thread."""

    def __init__(self):
        self.paused = threading.Event()
        self.release = threading.Event()

    def hold(self):
        self.paused.set()
        self.release.wait(timeout=30)

    def reset(self):
        self.paused.clear()
        self.release.clear()


@pytest.fixture
def patch_sleep(monkeypatch):
    """Make the wait_for_job poll loop instantaneous so tests don't sleep 20s."""
    monkeypatch.setattr(oai_mod.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def uncaught():
    """Capture any exception that escapes a worker thread via threading.excepthook."""
    captured = {}
    original = threading.excepthook

    def hook(args):
        captured["exc"] = args.exc_value

    threading.excepthook = hook
    try:
        yield captured
    finally:
        threading.excepthook = original


def _build_openai_lm():
    return dspy.LM(model="openai/gpt-4o-mini", api_key="sk-fake", cache=False)


def _start_finetune(lm):
    """Invoke the real LM.finetune and return the live TrainingJobOpenAI."""
    return lm.finetune(
        train_data=TRAIN_DATA,
        train_data_format=TrainDataFormat.CHAT,
        train_kwargs={},
    )


def _patch_boundary(stack, patches):
    """Apply ``mock.patch.object`` on each (module, attr, side_effect) tuple."""
    for module, attr, side_effect in patches:
        stack.enter_context(mock.patch.object(module, attr, side_effect=side_effect))


def _make_boundary(gate_poll=None, gate_upload=None, terminal_status="succeeded"):
    """Build mock callables for the openai.* SDK boundary.

    Args:
        gate_poll: ``_Gate`` to hold the worker inside ``jobs.list_events``
            (after ``provider_job_id`` has been set) -- reproduces Window 1.
        gate_upload: ``_Gate`` to hold the worker inside ``files.create``
            (before ``provider_file_id`` is set) -- reproduces Window 2a.
        terminal_status: status returned by ``jobs.retrieve`` for a real id.
    """
    if gate_upload is not None:
        gate_upload.reset()
    if gate_poll is not None:
        gate_poll.reset()

    def fake_files_create(*args, **kwargs):
        if gate_upload is not None:
            gate_upload.hold()
        return _FakeFile("file-real")

    def fake_files_retrieve(file_id):
        if file_id is None:
            raise ValueError(_NONE_FILE_MSG)
        return _FakeFile(file_id)

    def fake_files_delete(file_id):
        return None

    def fake_jobs_create(*args, **kwargs):
        return _FakeJob(
            "ftjob-real",
            status="running",
            estimated_finish=int(time.time()) + 600,
            fine_tuned_model="ft:gpt-4o-mini:custom:ftjob-real",
        )

    def fake_jobs_retrieve(job_id):
        if job_id is None:
            raise ValueError(_NONE_JOB_MSG)
        return _FakeJob(
            job_id,
            status=terminal_status,
            estimated_finish=int(time.time()) + 600,
            fine_tuned_model=f"ft:gpt-4o-mini:custom:{job_id}",
        )

    def fake_jobs_list_events(*args, fine_tuning_job_id=None, **kwargs):
        if fine_tuning_job_id is None:
            raise ValueError(_NONE_JOB_MSG)
        if gate_poll is not None:
            gate_poll.hold()
        return _FakePage([])

    def fake_jobs_cancel(job_id):
        return None

    return [
        (openai.files, "create", fake_files_create),
        (openai.files, "retrieve", fake_files_retrieve),
        (openai.files, "delete", fake_files_delete),
        (openai.fine_tuning.jobs, "create", fake_jobs_create),
        (openai.fine_tuning.jobs, "retrieve", fake_jobs_retrieve),
        (openai.fine_tuning.jobs, "list_events", fake_jobs_list_events),
        (openai.fine_tuning.jobs, "cancel", fake_jobs_cancel),
    ]


def test_finetune_success_path_delivers_trained_lm(patch_sleep, uncaught):
    """Regression: with no cancel(), the trained LM is delivered via result()."""
    with ExitStack() as stack:
        _patch_boundary(stack, _make_boundary())
        lm = _build_openai_lm()
        job = _start_finetune(lm)
        job.thread.join(timeout=30)

    assert not job.thread.is_alive()
    assert "exc" not in uncaught
    result = job.result(timeout=1)
    assert isinstance(result, dspy.LM)
    assert result.model == "ft:gpt-4o-mini:custom:ftjob-real"


def test_finetune_error_path_propagates_exception_when_not_cancelled(patch_sleep, uncaught):
    """Regression: an error from the provider (not a cancel) is still stored as
    the future's result so result() surfaces it; the cancelled-guard must not
    leave the future pending when the job was never cancelled.

    Note: dspy stores the exception via ``set_result(err)`` (not
    ``set_exception``), so ``result()`` returns the exception object rather than
    re-raising it. The regression guarantee is that the future reaches FINISHED
    and the exception is delivered to the caller.
    """
    boom = RuntimeError("provider blew up")

    def evil_finetune(job, model, train_data, train_data_format, train_kwargs):
        raise boom

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(OpenAIProvider, "finetune", staticmethod(evil_finetune)))
        lm = _build_openai_lm()
        job = _start_finetune(lm)
        job.thread.join(timeout=30)

    assert not job.thread.is_alive()
    assert "exc" not in uncaught
    assert job.done()
    assert job.cancelled() is False
    # The exception is surfaced through the future (stored as the result value).
    assert job.result(timeout=1) is boom


def test_cancel_during_polling_does_not_crash_worker(patch_sleep, uncaught):
    """Window 1: cancel() lands while the worker is polling (provider_job_id set).

    Guarantees:
      - the worker thread exits cleanly (no uncaught InvalidStateError)
      - job.cancelled() is True
      - job.result() raises CancelledError (not the inner ValueError)
      - the remote cancel API was actually invoked with the live job id
    """
    gate = _Gate()
    cancel_calls = []
    patches = _make_boundary(gate_poll=gate, terminal_status="running")
    patches[-1] = (openai.fine_tuning.jobs, "cancel", lambda job_id: cancel_calls.append(job_id))

    with ExitStack() as stack:
        _patch_boundary(stack, patches)
        lm = _build_openai_lm()
        job = _start_finetune(lm)

        assert gate.paused.wait(timeout=30)
        assert job.provider_job_id == "ftjob-real"
        assert job.provider_file_id == "file-real"
        assert not job.cancelled()

        job.cancel()

        assert job.cancelled() is True
        assert job.provider_job_id is None
        assert job.provider_file_id is None

        gate.release.set()
        job.thread.join(timeout=30)

    assert not job.thread.is_alive()
    assert "exc" not in uncaught, f"worker thread died with uncaught {uncaught.get('exc')!r}"
    assert cancel_calls == ["ftjob-real"]
    with pytest.raises(CancelledError):
        job.result(timeout=1)


def test_cancel_during_upload_does_not_crash_worker(patch_sleep, uncaught):
    """Window 2a: cancel() lands during upload_data (provider_file_id NOT set).

    Guarantees:
      - the worker thread exits cleanly (no uncaught InvalidStateError) even
        though it resumes, starts the remote job, and runs it to completion
      - job.cancelled() is True
      - job.result() raises CancelledError (the trained LM is NOT delivered
        through the TrainingJob API)
    """
    gate = _Gate()
    creates = []

    def counting_create(*args, **kwargs):
        creates.append(kwargs.get("training_file"))
        return _FakeJob(
            "ftjob-real",
            status="running",
            estimated_finish=int(time.time()) + 600,
            fine_tuned_model="ft:gpt-4o-mini:custom:ftjob-real",
        )

    patches = _make_boundary(gate_upload=gate, terminal_status="succeeded")
    # Replace the jobs.create mock with a counting variant.
    patches[3] = (openai.fine_tuning.jobs, "create", counting_create)

    with ExitStack() as stack:
        _patch_boundary(stack, patches)
        lm = _build_openai_lm()
        job = _start_finetune(lm)

        assert gate.paused.wait(timeout=30)
        assert job.provider_job_id is None
        assert job.provider_file_id is None
        assert not job.cancelled()

        job.cancel()

        assert job.cancelled() is True
        # cancel() during upload_data cannot delete the file (its id is unknown
        # to cancel at this moment), so provider_file_id stays None here.
        assert job.provider_file_id is None

        gate.release.set()
        job.thread.join(timeout=30)

    assert not job.thread.is_alive()
    assert "exc" not in uncaught, f"worker thread died with uncaught {uncaught.get('exc')!r}"
    # The worker resumed from upload_data and started the remote job with the
    # just-uploaded file. The cost-prevention concern is tracked separately; this
    # test only locks down the crash fix. The trained LM must NOT be delivered.
    assert creates == ["file-real"]
    with pytest.raises(CancelledError):
        job.result(timeout=1)
