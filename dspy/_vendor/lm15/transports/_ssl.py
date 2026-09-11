"""TLS for the stdlib transports: one context, and the ways to put it on a socket.

This module imports `ssl` plainly, because it cannot work without it. Nothing imports this
module until a request actually names an https URL, so a CPython build whose stdlib omits
`ssl` — componentize-py's, for one — still carries plain HTTP through either transport.

Translating TLS failures belongs here for the same reason. An `except ssl.SSLError` clause
names the module every time an exception passes it, plain-HTTP errors included, so only a
module entitled to `ssl` can write one.

We rely on the stdlib `create_default_context`, which on Python 3.10+ loads the system trust
store correctly on Linux/macOS/Windows. No certifi bundle is shipped — set SSL_CERT_FILE if
your system store is broken, or pass an explicit ca_bundle= to the transport.
"""
from __future__ import annotations

import asyncio
import socket
import ssl

from ._exceptions import ConnectError, ConnectTimeout
from ._timeouts import wait_for


def _close(sock: socket.socket) -> None:
    try:
        sock.close()
    except Exception:
        pass


class TLS:
    """The TLS half of a transport, holding the context its connections share."""

    __slots__ = ("_ctx",)

    def __init__(self, *, verify: bool = True, ca_bundle: str | None = None) -> None:
        if not verify:
            self._ctx = ssl._create_unverified_context()
            return
        self._ctx = ssl.create_default_context()
        if ca_bundle:
            self._ctx.load_verify_locations(cafile=ca_bundle)

    def wrap(
        self, sock: socket.socket, *, server_hostname: str, timeout: float
    ) -> socket.socket:
        """Hand back `sock` with TLS on it, or close it and say why it cannot have any."""
        try:
            sock.settimeout(timeout)
            return self._ctx.wrap_socket(sock, server_hostname=server_hostname)
        except OSError as exc:  # ssl.SSLError is an OSError, so this covers the handshake
            _close(sock)
            raise ConnectError(f"TLS handshake failed: {exc}") from exc

    async def connect(
        self, *, host: str, port: int, server_hostname: str, timeout: float
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a TCP connection and run the handshake on it, in one asyncio call."""
        try:
            return await wait_for(
                asyncio.open_connection(
                    host=host, port=port, ssl=self._ctx, server_hostname=server_hostname
                ),
                timeout=timeout,
                cancel_result=lambda pair: pair[1].close(),
            )
        except asyncio.TimeoutError as exc:
            raise ConnectTimeout(f"timed out connecting to {host}:{port}") from exc
        except ssl.SSLError as exc:
            raise ConnectError(f"TLS handshake failed: {exc}") from exc
        except OSError as exc:
            raise ConnectError(f"failed to connect to {host}:{port}: {exc}") from exc

    async def connect_over(
        self, sock: socket.socket, *, server_hostname: str, timeout: float
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Run the handshake end-to-end over a socket a proxy has already tunneled."""
        try:
            return await wait_for(
                asyncio.open_connection(
                    sock=sock, ssl=self._ctx, server_hostname=server_hostname
                ),
                timeout=timeout,
                cancel_result=lambda pair: pair[1].close(),
            )
        except asyncio.TimeoutError as exc:
            _close(sock)
            raise ConnectTimeout("TLS handshake through proxy timed out") from exc
        except ssl.SSLError as exc:
            _close(sock)
            raise ConnectError(f"TLS handshake failed: {exc}") from exc
        except OSError as exc:
            _close(sock)
            raise ConnectError(f"TLS handshake through proxy failed: {exc}") from exc
