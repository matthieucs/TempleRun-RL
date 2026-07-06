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
                 connect_timeout: float = C.DEFAULT_CONNECT_TIMEOUT,
                 reconnect_backoff: float = 1.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.reconnect_backoff = reconnect_backoff
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[dict] = None
        self._seq = 0
        self._last_recv = 0.0
        self._running = False

    def connect(self, required: bool = True) -> "StateClient":
        # With required=True, fail fast on the first connect (clear error if ports aren't
        # forwarded). With required=False (resilient training), tolerate a phone that's down
        # at startup: start the reader thread with no socket — it reconnects on its own when
        # frames return, so the env runs DEGRADED instead of crashing the SubprocVecEnv worker.
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            self._sock.settimeout(None)
        except OSError as e:
            if required:
                raise _conn_error(self.host, self.port, "state stream") from e
            self._sock = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="StateReader", daemon=True)
        self._thread.start()
        return self

    def _drop_sock(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None

    def _reconnect(self) -> bool:
        """Best-effort reconnect with backoff; returns True once connected. Stays responsive to close()."""
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            sock.settimeout(None)
            self._sock = sock
            return True
        except OSError:
            end = time.monotonic() + self.reconnect_backoff
            while self._running and time.monotonic() < end:
                time.sleep(0.1)
            return False

    def _reader(self) -> None:
        buf = b""
        while self._running:
            if self._sock is None:
                if not self._reconnect():
                    continue
                buf = b""
            try:
                chunk = self._sock.recv(65536)  # type: ignore[union-attr]
            except OSError:
                self._drop_sock()
                continue
            if not chunk:           # peer closed (app relaunch / gadget reload) -> reconnect
                self._drop_sock()
                continue
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

    def staleness(self) -> float:
        """Seconds since the last frame arrived (``inf`` if none yet). For health monitors."""
        with self._lock:
            return float("inf") if self._last_recv == 0.0 else time.monotonic() - self._last_recv

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
                 connect_timeout: float = C.DEFAULT_CONNECT_TIMEOUT,
                 retry_timeout: float = 30.0, retry_backoff: float = 0.5):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.retry_timeout = retry_timeout      # keep retrying a send for this long (covers re-injection)
        self.retry_backoff = retry_backoff
        self._sock: Optional[socket.socket] = None

    def connect(self, required: bool = True) -> "ActionClient":
        # required=False tolerates a phone down at startup (send() reopens later); see StateClient.
        try:
            self._sock = self._open()
        except (OSError, RuntimeError):
            if required:
                raise
            self._sock = None
        return self

    def try_send(self, action: int) -> bool:
        """One non-blocking attempt to send (no 30s retry loop). Returns True on success.

        For the degraded/recovery path where blocking would stall the lockstep vec-env.
        """
        try:
            if self._sock is None:
                self._sock = self._open()
            self._sock.sendall(bytes([int(action)]))
            return True
        except (OSError, RuntimeError):
            self._drop()
            return False

    def _open(self) -> socket.socket:
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except OSError as e:
            raise _conn_error(self.host, self.port, "action sink") from e
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.connect_timeout)
        return s

    def _drop(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None

    def send(self, action: int) -> None:
        if not (0 <= int(action) <= 9):
            raise ValueError(f"action byte out of range: {action}")
        data = bytes([int(action)])
        deadline = time.monotonic() + self.retry_timeout
        last_err: Optional[Exception] = None
        while True:
            try:
                if self._sock is None:
                    self._sock = self._open()
                self._sock.sendall(data)
                return
            except (OSError, RuntimeError) as e:  # _open wraps OSError in RuntimeError
                last_err = e
                self._drop()
                if time.monotonic() >= deadline:
                    break
                time.sleep(self.retry_backoff)
        raise _conn_error(self.host, self.port, "action sink") from last_err

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
