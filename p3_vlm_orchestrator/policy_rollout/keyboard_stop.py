"""Permission-free terminal and signal stop handling for policy rollout."""

from __future__ import annotations

from collections.abc import Callable
import select
import signal
import sys
import termios
import threading
import tty
from typing import IO, Any


STOP_KEYS = frozenset(("q", "x", "\x1b"))
POLL_INTERVAL_S = 0.10


class KeyboardStop:
    """Route terminal keys and process signals into one shared stop event.

    Construction and import are side-effect free. Signal handlers, terminal mode,
    and the daemon reader exist only while the context manager is active.
    """

    def __init__(
        self,
        *,
        event: threading.Event | None = None,
        stdin: IO[str] | None = None,
        warning_stream: IO[str] | None = None,
        signal_api: Any = None,
        termios_api: Any = None,
        tty_api: Any = None,
        select_fn: Callable[..., tuple[list[object], list[object], list[object]]] | None = None,
        thread_factory: Callable[..., object] | None = None,
        is_main_thread: Callable[[], bool] | None = None,
    ) -> None:
        self.event = event or threading.Event()
        self.stdin = stdin if stdin is not None else sys.stdin
        self.warning_stream = (
            warning_stream if warning_stream is not None else sys.stderr
        )
        self.signal_api = signal_api if signal_api is not None else signal
        self.termios_api = termios_api if termios_api is not None else termios
        self.tty_api = tty_api if tty_api is not None else tty
        self.select_fn = select_fn if select_fn is not None else select.select
        self.thread_factory = (
            thread_factory if thread_factory is not None else threading.Thread
        )
        self.is_main_thread = is_main_thread or (
            lambda: threading.current_thread() is threading.main_thread()
        )
        self._prior_handlers: dict[object, object] = {}
        self._terminal_fd: int | None = None
        self._terminal_state: object | None = None
        self._reader_thread: object | None = None
        self._entered = False
        self._closed = False
        self._closing = False

    def __enter__(self) -> KeyboardStop:
        if not self.is_main_thread():
            raise RuntimeError("KeyboardStop must be entered on the main thread")
        if self._entered and not self._closed:
            raise RuntimeError("KeyboardStop is already active")
        if self._closed:
            raise RuntimeError("KeyboardStop contexts cannot be reused")
        self._entered = True
        try:
            self._install_signal_handlers()
            self._start_terminal_reader()
        except Exception:
            self._cleanup_entry()
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._closed:
            return
        self._cleanup_entry()

    def _cleanup_entry(self) -> None:
        """Restore every entry side effect; safe after partial setup."""

        self._closed = True
        self._closing = True
        reader = self._reader_thread
        join = getattr(reader, "join", None)
        if callable(join):
            try:
                join(timeout=0.5)
            except Exception as join_error:
                self._warn(f"Keyboard-stop reader cleanup warning: {join_error}")
        self._restore_terminal()
        self._restore_signal_handlers()

    def stop(self) -> None:
        """Request a stop; repeated requests are deliberately harmless."""

        self.event.set()

    def is_set(self) -> bool:
        return self.event.is_set()

    def _install_signal_handlers(self) -> None:
        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(self.signal_api, name, None)
            if signum is None:
                continue
            previous = self.signal_api.getsignal(signum)
            self.signal_api.signal(signum, self._handle_signal)
            self._prior_handlers[signum] = previous

    def _restore_signal_handlers(self) -> None:
        if not self.is_main_thread():
            self._prior_handlers.clear()
            return
        for signum, previous in tuple(self._prior_handlers.items()):
            try:
                self.signal_api.signal(signum, previous)
            except Exception as exc:
                self._warn(f"Could not restore signal handler {signum}: {exc}")
        self._prior_handlers.clear()

    def _handle_signal(self, signum: int, frame: object) -> None:
        del signum, frame
        self.stop()

    def _start_terminal_reader(self) -> None:
        try:
            is_tty = bool(self.stdin.isatty())
        except Exception:
            is_tty = False
        if not is_tty:
            self._warn(
                "WARNING: keyboard stop is unavailable in non-TTY mode; "
                "SIGINT/SIGTERM and the physical e-stop remain available."
            )
            return

        fd = self.stdin.fileno()
        saved_state = self.termios_api.tcgetattr(fd)
        self._terminal_fd = fd
        self._terminal_state = saved_state
        self.tty_api.setcbreak(fd)
        self._reader_thread = self.thread_factory(
            target=self._read_terminal,
            name="policy-rollout-keyboard-stop",
            daemon=True,
        )
        self._reader_thread.start()

    def _read_terminal(self) -> None:
        while not self._closing and not self.event.is_set():
            try:
                readable, _writable, _errors = self.select_fn(
                    [self.stdin], [], [], POLL_INTERVAL_S
                )
                if not readable:
                    continue
                character = self.stdin.read(1)
            except Exception as exc:
                self._warn(f"WARNING: terminal keyboard reader stopped: {exc}")
                return
            if character in STOP_KEYS:
                self.stop()
                return

    def _restore_terminal(self) -> None:
        if self._terminal_fd is None or self._terminal_state is None:
            return
        fd = self._terminal_fd
        state = self._terminal_state
        self._terminal_fd = None
        self._terminal_state = None
        try:
            self.termios_api.tcsetattr(fd, self.termios_api.TCSADRAIN, state)
        except Exception as exc:
            self._warn(f"Could not restore terminal state: {exc}")

    def _warn(self, message: str) -> None:
        try:
            print(message, file=self.warning_stream, flush=True)
        except Exception:
            pass
