#!/usr/bin/env python3
"""Supervise a detached worker and stop it if the GUI owner disappears.

The GUI deliberately starts hardware workers in their own sessions so signals
cannot hit the HTTP server.  Detachment normally means a SIGKILL/OOM of the GUI
would orphan a connected robot.  This tiny supervisor owns a pipe from the GUI;
EOF forwards the configured graceful-stop signal and waits for cleanup.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-fd", required=True, type=int)
    parser.add_argument(
        "--owner-stop-signal",
        choices=("SIGHUP", "SIGINT", "SIGTERM"),
        required=True,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a worker command is required after --")
    return args


def main() -> int:
    args = parse_args()
    owner_stop_signal = int(getattr(signal, args.owner_stop_signal))
    state_lock = threading.RLock()
    child: subprocess.Popen[Any] | None = None
    queued_signals: list[int] = []

    def forward(signum: int, _frame: Any = None) -> None:
        nonlocal child
        with state_lock:
            current = child
            if current is None:
                queued_signals.append(signum)
                return
        try:
            os.killpg(current.pid, signum)
        except ProcessLookupError:
            pass

    for forwarded in (
        signal.SIGUSR1,
        signal.SIGUSR2,
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGTERM,
    ):
        signal.signal(forwarded, forward)

    def watch_owner() -> None:
        try:
            while os.read(args.owner_fd, 4096):
                pass
        except OSError:
            pass
        finally:
            try:
                os.close(args.owner_fd)
            except OSError:
                pass
        # The GUI's stdout/stderr pipe is normally broken on this exact path.
        # Stop the worker first; diagnostics must never gate robot cleanup.
        forward(owner_stop_signal)
        try:
            print(
                f"OWNER_LOST forwarding={args.owner_stop_signal}",
                file=sys.stderr,
                flush=True,
            )
        except (BrokenPipeError, OSError):
            # Prevent CPython's final stderr flush from changing an otherwise
            # clean supervisor exit to status 120 after the owner is gone.
            try:
                sys.stderr = open(os.devnull, "w")
            except OSError:
                pass

    threading.Thread(target=watch_owner, name="gui-owner-watchdog", daemon=True).start()

    worker = subprocess.Popen(
        args.command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

    def relay_worker_output() -> None:
        # Keep the worker's private pipe drained even after the GUI's log reader
        # disappears.  This makes signal-handler prints harmless and lets the
        # collector reach archive recovery and arm disconnect.
        assert worker.stdout is not None
        relay_enabled = True
        while True:
            chunk = worker.stdout.read1(64 * 1024)
            if not chunk:
                return
            if not relay_enabled:
                continue
            view = memoryview(chunk)
            while view:
                try:
                    written = os.write(1, view)
                except (BrokenPipeError, OSError):
                    relay_enabled = False
                    break
                if written <= 0:
                    relay_enabled = False
                    break
                view = view[written:]

    relay = threading.Thread(
        target=relay_worker_output,
        name="worker-log-relay",
        daemon=True,
    )
    relay.start()
    with state_lock:
        child = worker
        pending = list(queued_signals)
        queued_signals.clear()
    for pending_signal in pending:
        forward(pending_signal)

    # Do not block indefinitely in waitpid.  CPython only runs Python signal
    # handlers on the main thread between bytecode instructions, so an
    # unbounded wait can leave SIGUSR1/SIGUSR2/SIGHUP pending until the worker
    # exits—the exact opposite of a responsive recording control channel.
    # The short timeout returns to Python often enough to forward operator
    # decisions promptly while retaining the supervisor's detached ownership.
    while True:
        try:
            return_code = worker.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            continue
    relay.join()
    return return_code if return_code >= 0 else 128 + abs(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
