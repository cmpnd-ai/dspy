import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from dspy.utils.unbatchify import Unbatchify


def simple_batch_processor(batch):
    """A simple batch function that adds 1 to each item."""
    return [item + 1 for item in batch]


def submit(self, input_item: any) -> Future:
    """Submits an item for processing and returns a Future."""
    future = Future()
    self.input_queue.put((input_item, future))
    return future


Unbatchify.submit = submit


def test_unbatchify_batch_size_trigger():
    """Test that the batch processes exactly when max_batch_size is reached."""
    batch_fn_mock = MagicMock(wraps=simple_batch_processor)
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=2, max_wait_time=5.0)

    futures = []
    futures.append(unbatcher.submit(10))
    time.sleep(0.02)
    assert batch_fn_mock.call_count == 0

    futures.append(unbatcher.submit(20))

    results_1_2 = [f.result() for f in futures]
    assert batch_fn_mock.call_count == 1
    batch_fn_mock.assert_called_once_with([10, 20])
    assert results_1_2 == [11, 21]

    futures_3_4 = []
    futures_3_4.append(unbatcher.submit(30))
    futures_3_4.append(unbatcher.submit(40))

    results_3_4 = [f.result() for f in futures_3_4]
    time.sleep(0.01)
    assert batch_fn_mock.call_count == 2
    assert batch_fn_mock.call_args_list[1].args[0] == [30, 40]
    assert results_3_4 == [31, 41]

    unbatcher.close()


def test_unbatchify_timeout_trigger():
    """Test that the batch processes after max_wait_time."""
    batch_fn_mock = MagicMock(wraps=simple_batch_processor)
    wait_time = 0.15
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=5, max_wait_time=wait_time)

    futures = []
    futures.append(unbatcher.submit(100))
    futures.append(unbatcher.submit(200))

    time.sleep(wait_time / 2)
    assert batch_fn_mock.call_count == 0

    results = [f.result() for f in futures]

    assert batch_fn_mock.call_count == 1
    batch_fn_mock.assert_called_once_with([100, 200])
    assert results == [101, 201]

    unbatcher.close()


def test_unbatchify_honors_max_wait_time_under_trickling_input():
    """A batch must flush within max_wait_time of the FIRST item arriving, even if
    later items keep trickling in just under max_wait_time apart from each other.
    Each queue.get() must wait only the time remaining in the window, not the full
    max_wait_time again - otherwise the total wait can be a multiple of the budget."""
    batch_fn_mock = MagicMock(wraps=simple_batch_processor)
    wait_time = 0.1
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=10, max_wait_time=wait_time)

    start = time.time()
    future_1 = unbatcher.submit(1)
    time.sleep(wait_time * 0.8)  # arrives well within the window, but close to its edge
    future_2 = unbatcher.submit(2)

    results = [future_1.result(timeout=2), future_2.result(timeout=2)]
    elapsed = time.time() - start

    assert results == [2, 3]
    # Generous slack for scheduling jitter, but must stay well under 2x the budget
    # (which is what the unfixed code produces for this timing).
    assert elapsed < wait_time * 1.5

    unbatcher.close()


def test_unbatchify_raises_on_short_batch_fn_output():
    """When batch_fn returns fewer outputs than inputs, every future must receive a
    ValueError rather than a (mis-paired) result, and no future may be left pending."""

    def short_batch_fn(batch):
        # drop the second input -> len(batch) - 1 outputs
        return [item + 1 for i, item in enumerate(batch) if i != 1]

    batch_fn_mock = MagicMock(wraps=short_batch_fn)
    # max_batch_size=4 fills immediately on the 4 submits below regardless of
    # max_wait_time; a small max_wait_time just keeps close() teardown fast.
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=4, max_wait_time=0.2)

    futures = [unbatcher.submit(10), unbatcher.submit(20), unbatcher.submit(30), unbatcher.submit(40)]

    for f in futures:
        with pytest.raises(ValueError, match="batch_fn returned 3 outputs for 4 inputs"):
            f.result(timeout=2)
    # No future received a result: mis-pairing would have resolved some futures with a value
    # instead of an exception.
    for f in futures:
        assert isinstance(f.exception(), ValueError)
    assert batch_fn_mock.call_count == 1
    unbatcher.close()


def test_unbatchify_raises_on_long_batch_fn_output():
    """When batch_fn returns more outputs than inputs, every future must receive a
    ValueError rather than receiving one of the extra results out of position."""

    def long_batch_fn(batch):
        # add a spurious trailing output -> len(batch) + 1 outputs
        return [item + 1 for item in batch] + [999]

    batch_fn_mock = MagicMock(wraps=long_batch_fn)
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=3, max_wait_time=0.2)

    futures = [unbatcher.submit(10), unbatcher.submit(20), unbatcher.submit(30)]

    for f in futures:
        with pytest.raises(ValueError, match="batch_fn returned 4 outputs for 3 inputs"):
            f.result(timeout=2)
    for f in futures:
        assert isinstance(f.exception(), ValueError)
    assert batch_fn_mock.call_count == 1
    unbatcher.close()


def test_unbatchify_call_raises_and_does_not_hang_on_short_output():
    """End-to-end via __call__: when batch_fn returns short, every caller must raise
    ValueError. No caller may receive another caller's result, and no caller may hang."""

    def short_batch_fn(batch):
        return [f"out-{x}" for i, x in enumerate(batch) if i != 1]

    ub = Unbatchify(short_batch_fn, max_batch_size=4, max_wait_time=1.0)

    results = {}
    errors = {}

    def call(idx):
        try:
            results[idx] = ub(f"in-{idx}")
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    assert not any(t.is_alive() for t in threads), "a caller is still hanging"
    assert results == {}, f"callers received (mis-paired) results: {results}"
    assert set(errors) == {0, 1, 2, 3}
    for i in range(4):
        assert isinstance(errors[i], ValueError)
        assert "batch_fn returned 3 outputs for 4 inputs" in str(errors[i])

    ub.close()


def test_unbatchify_survives_after_mismatched_batch():
    """A length mismatch fails only the affected batch; the worker thread must survive
    and the same Unbatchify instance must keep servicing subsequent calls."""

    def short_batch_fn(batch):
        return [item + 1 for i, item in enumerate(batch) if i != 1]

    def good_batch_fn(batch):
        return [item + 1 for item in batch]

    ub = Unbatchify(short_batch_fn, max_batch_size=4, max_wait_time=0.2)

    futures = [ub.submit(1), ub.submit(2), ub.submit(3), ub.submit(4)]
    for f in futures:
        with pytest.raises(ValueError):
            f.result(timeout=2)

    # The worker thread must still be alive after rejecting the mismatched batch.
    assert ub.worker_thread.is_alive()

    # Swap to a well-behaved batch_fn; the same instance must service the next call.
    ub.batch_fn = good_batch_fn
    assert ub(7) == 8

    ub.close()
