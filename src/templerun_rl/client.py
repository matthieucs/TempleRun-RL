"""TCP clients for the Frida gadget's state stream (9999) and action sink (9998).

``StateClient`` runs a background reader thread that keeps only the *latest* parsed
state frame, so the env never acts on stale buffered data. ``ActionClient`` holds a
single persistent socket (per-action connect/close drops bytes — see docs/GADGET.md).
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional, Tuple

from . import constants as C


def _conn_error(host: str, port: int, what: str) -> RuntimeError:
    return RuntimeError(
        f"Cannot reach {what} at {host}:{port}. "
        f"Make sure the phone is connected, the gadget is injected, and the ports are forwarded:\n"
        f"  python scripts/setup_device.py\n"
        f"(which runs `adb forward tcp:{port} tcp:{port}` and injects frida/temple_run_hook.js)."
    )


class StateClient:
    """Background reader that exposes the most recent state dict and a sequence counter."""

    def __init__(self, host: str = C.STATE_HOST, port: int = C.STATE_PORT,
                 connect_timeout: float = C.DEFAULT_CONNECT_TIMEOUT):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[dict] = None
        self._seq = 0
        self._last_recv = 0.0
        self._running = False

    def connect(self) -> "StateClient":
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except OSError as e:
            raise _conn_error(self.host, self.port, "state stream") from e
        self._sock.settimeout(None)
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="StateReader", daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        buf = b""
        sock = self._sock
        assert sock is not None
        while self._running:
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            # Parse every complete line; keep only the last valid one.
            *lines, buf = buf.split(b"\n")
            newest = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    newest = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
            if newest is not None:
                with self._cond:
                    self._latest = newest
                    self._seq += 1
                    self._last_recv = time.monotonic()
                    self._cond.notify_all()

    def latest(self) -> Tuple[Optional[dict], int]:
        with self._lock:
            return self._latest, self._seq

    def wait_fresh(self, after_seq: int, timeout: float) -> Tuple[dict, int]:
        """Block until a state frame newer than ``after_seq`` arrives; return (state, seq)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No new state frame within {timeout:.1f}s "
                        f"(last seq={self._seq}). Is the gadget still injected and the game running?"
                    )
                self._cond.wait(remaining)
            return self._latest, self._seq  # type: ignore[return-value]

    def close(self) -> None:
        self._running = False
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class ActionClient:
    """Single persistent socket to the action server; sends one byte per action."""

    def __init__(self, host: str = C.STATE_HOST, port: int = C.ACTION_PORT,
                 connect_timeout: float = C.DEFAULT_CONNECT_TIMEOUT):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> "ActionClient":
        self._sock = self._open()
        return self

    def _open(self) -> socket.socket:
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except OSError as e:
            raise _conn_error(self.host, self.port, "action sink") from e
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.connect_timeout)
        return s

    def send(self, action: int) -> None:
        if not (0 <= int(action) <= 7):
            raise ValueError(f"action byte out of range: {action}")
        data = bytes([int(action)])
        try:
            assert self._sock is not None
            self._sock.sendall(data)
        except (OSError, AssertionError):
            # One reconnect-and-resend attempt.
            self._sock = self._open()
            self._sock.sendall(data)

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
