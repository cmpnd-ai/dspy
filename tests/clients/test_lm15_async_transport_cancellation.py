"""Transport-level regression tests for the async pool's cancellation cleanup.

The async transport's ``except BaseException`` cleanup handler in
``_do_stream`` must release the pool slot AND discard the connection even
when a second cancellation arrives while the handler is suspended in
``await self._pool.discard(conn)`` (which awaits the pool's ``asyncio.Lock``).

These tests drive ``StdlibAsyncTransport`` directly against a local
asyncio HTTP/1.1 server so the double-cancellation timing is deterministic
(no provider keys, no network).
"""

import asyncio
import contextlib

import pytest

from dspy._vendor.lm15.transports import StdlibAsyncTransport, TransportRequest
from dspy._vendor.lm15.transports._exceptions import ReadError

# ─── Local asyncio HTTP/1.1 server ───────────────────────────────────


class _LocalHTTPServer:
    """A tiny asyncio TCP server speaking just enough HTTP/1.1.

    ``mode="park"`` (default): the first request is never answered (the
    transport parks in ``_read_head``); subsequent requests get a fixed
    ``200`` body so follow-up requests can succeed.

    ``mode="close"``: every request is answered by closing the socket
    immediately, so ``_read_head`` raises ``ReadError``.
    """

    def __init__(self, mode: str = "park") -> None:
        self.mode = mode
        self.server: asyncio.AbstractServer | None = None
        self.port: int = 0
        self.requests = 0
        self.release_first = asyncio.Event()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
            with contextlib.suppress(Exception):
                writer.close()
            return
        self.requests += 1

        if self.mode == "close":
            with contextlib.suppress(Exception):
                writer.close()
            return

        # mode == "park"
        if self.requests == 1:
            try:
                await self.release_first.wait()
            except asyncio.CancelledError:
                pass
            finally:
                with contextlib.suppress(Exception):
                    writer.close()
            return

        body = b"ok"
        head = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n"
        try:
            writer.write(head + body)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def aclose(self) -> None:
        self.release_first.set()
        if self.server is not None:
            self.server.close()
            with contextlib.suppress(Exception):
                await self.server.wait_closed()


def _request(url: str) -> TransportRequest:
    return TransportRequest(method="GET", url=url)


async def _wait_for(predicate, timeout: float = 5.0, *, msg: str = "") -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"condition not met within {timeout}s {msg}")
        await asyncio.sleep(0.005)


async def _drain(ticks: int = 20) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def _lock_waiters(lock: asyncio.Lock):
    return getattr(lock, "_waiters", None) or ()


# ─── Tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_double_cancel_under_lock_contention_releases_slot_and_closes_conn():
    """The regression: a second cancel while the cleanup handler is
    suspended in ``await self._pool.discard(conn)`` (contended pool lock)
    must not strand the semaphore slot or leave the connection open."""
    server = _LocalHTTPServer(mode="park")
    await server.start()
    transport = StdlibAsyncTransport(max_connections=1, pool_timeout=0.5, read_timeout=30.0)
    url = f"http://127.0.0.1:{server.port}/"
    release_lock = asyncio.Event()
    holder: asyncio.Task | None = None
    task: asyncio.Task | None = None
    try:
        task = asyncio.create_task(transport._do_stream(_request(url)))

        # Transport is now parked in _read_head (server never answers conn #1).
        await _wait_for(lambda: server.requests >= 1, msg="server to receive the first request")

        # Hold the pool lock so the handler's discard suspends on acquire,
        # exactly as a concurrent checkout/checkin/discard on the shared
        # transport would cause.
        async def hold_lock() -> None:
            async with transport._pool._lock:
                await release_lock.wait()

        holder = asyncio.create_task(hold_lock())
        await _wait_for(lambda: transport._pool._lock.locked(), msg="holder to acquire the pool lock")

        # First cancel enters the except handler; the discard await suspends
        # on the contended lock (a waiter appears on the lock).
        task.cancel()
        await _wait_for(
            lambda: len(_lock_waiters(transport._pool._lock)) > 0,
            msg="cleanup handler to suspend on the pool lock",
        )
        # Second cancel lands WHILE the handler is suspended in the await.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The slot MUST be free already (released synchronously before the await).
        assert transport._pool._slot._value == 1, "pool slot leaked after double cancellation"

        # The connection is still tracked / open while the lock is held...
        in_use_now = list(transport._pool._in_use)
        assert len(in_use_now) == 1
        conn = in_use_now[0]
        assert conn.closed is False

        # ...releasing the lock lets the shielded discard finish.
        release_lock.set()
        await holder
        await _drain()

        assert transport.pool_stats()["in_use"] == 0, "connection stayed tracked as in-use"
        assert conn.closed is True, "connection socket was not closed"

        # A follow-up request must succeed: the pool slot is available again.
        resp = await transport._do_stream(_request(url))
        try:
            assert resp.status == 200
            assert await resp.read() == b"ok"
        finally:
            await resp.aclose()
        assert transport._pool._slot._value == 1
        assert transport.pool_stats()["in_use"] == 0
    finally:
        release_lock.set()
        if holder is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await holder
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await transport.aclose()
        await server.aclose()


@pytest.mark.asyncio
async def test_single_cancel_no_contention_releases_pool_and_closes_conn():
    """Control: a single cancellation with an uncontended lock still cleans
    up (the pre-existing, supported behavior must not regress)."""
    server = _LocalHTTPServer(mode="park")
    await server.start()
    transport = StdlibAsyncTransport(max_connections=1, pool_timeout=0.5, read_timeout=30.0)
    url = f"http://127.0.0.1:{server.port}/"
    task: asyncio.Task | None = None
    try:
        task = asyncio.create_task(transport._do_stream(_request(url)))
        await _wait_for(lambda: server.requests >= 1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _drain()

        assert transport._pool._slot._value == 1
        assert transport.pool_stats()["in_use"] == 0

        resp = await transport._do_stream(_request(url))
        try:
            assert resp.status == 200
            assert await resp.read() == b"ok"
        finally:
            await resp.aclose()
        assert transport._pool._slot._value == 1
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await transport.aclose()
        await server.aclose()


@pytest.mark.asyncio
async def test_double_cancel_no_contention_still_releases_pool():
    """Double cancellation with an uncontended lock must not strand the
    slot, even when the second cancel arrives after cleanup has finished."""
    server = _LocalHTTPServer(mode="park")
    await server.start()
    transport = StdlibAsyncTransport(max_connections=1, pool_timeout=0.5, read_timeout=30.0)
    url = f"http://127.0.0.1:{server.port}/"
    task: asyncio.Task | None = None
    try:
        task = asyncio.create_task(transport._do_stream(_request(url)))
        await _wait_for(lambda: server.requests >= 1)

        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _drain()

        assert transport._pool._slot._value == 1
        assert transport.pool_stats()["in_use"] == 0

        resp = await transport._do_stream(_request(url))
        try:
            assert resp.status == 200
            assert await resp.read() == b"ok"
        finally:
            await resp.aclose()
        assert transport._pool._slot._value == 1
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await transport.aclose()
        await server.aclose()


@pytest.mark.asyncio
async def test_error_path_releases_slot_and_discards_conn():
    """A non-cancellation error during the stream (server closes before
    sending a response) must still release the slot and discard the
    connection — the reorder must not break the synchronous error path."""
    server = _LocalHTTPServer(mode="close")
    await server.start()
    transport = StdlibAsyncTransport(max_connections=1, pool_timeout=0.5, read_timeout=30.0)
    url = f"http://127.0.0.1:{server.port}/"
    try:
        with pytest.raises(ReadError):
            await transport._do_stream(_request(url))
        await _drain()

        assert transport._pool._slot._value == 1, "slot leaked on a plain read error"
        assert transport.pool_stats()["in_use"] == 0
    finally:
        await transport.aclose()
        await server.aclose()
