from __future__ import annotations

from io import StringIO
import threading
import unittest

from p3_vlm_orchestrator.policy_rollout.keyboard_stop import KeyboardStop


class FakeSignalAPI:
    SIGINT = 2
    SIGTERM = 15

    def __init__(self) -> None:
        self.current = {self.SIGINT: "old-int", self.SIGTERM: "old-term"}
        self.installs: list[tuple[int, object]] = []

    def getsignal(self, signum: int) -> object:
        return self.current[signum]

    def signal(self, signum: int, handler: object) -> object:
        previous = self.current[signum]
        self.current[signum] = handler
        self.installs.append((signum, handler))
        return previous


class FakeTermios:
    TCSADRAIN = 1

    def __init__(self) -> None:
        self.saved = ["saved-terminal-state"]
        self.get_calls: list[int] = []
        self.set_calls: list[tuple[int, int, object]] = []

    def tcgetattr(self, fd: int) -> object:
        self.get_calls.append(fd)
        return self.saved

    def tcsetattr(self, fd: int, when: int, state: object) -> None:
        self.set_calls.append((fd, when, state))


class FakeTTY:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def setcbreak(self, fd: int) -> None:
        self.calls.append(fd)


class FakeInput:
    def __init__(self, *, tty: bool, characters: str = "") -> None:
        self.tty = tty
        self.characters = list(characters)

    def isatty(self) -> bool:
        return self.tty

    def fileno(self) -> int:
        return 42

    def read(self, count: int) -> str:
        if not self.characters:
            return ""
        return self.characters.pop(0)


class ImmediateThread:
    def __init__(self, *, target, name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True
        self.target()

    def join(self, timeout: float | None = None) -> None:
        self.joined = True


class PassiveThread(ImmediateThread):
    def start(self) -> None:
        self.started = True


class KeyboardStopTest(unittest.TestCase):
    def make_stop(
        self,
        *,
        stdin: FakeInput,
        thread_factory=PassiveThread,
        is_main_thread=lambda: True,
    ) -> tuple[KeyboardStop, FakeSignalAPI, FakeTermios, FakeTTY, StringIO]:
        signals = FakeSignalAPI()
        termios = FakeTermios()
        tty = FakeTTY()
        warnings = StringIO()
        stop = KeyboardStop(
            stdin=stdin,
            warning_stream=warnings,
            signal_api=signals,
            termios_api=termios,
            tty_api=tty,
            select_fn=lambda readers, writers, errors, timeout: (readers, [], []),
            thread_factory=thread_factory,
            is_main_thread=is_main_thread,
        )
        return stop, signals, termios, tty, warnings

    def test_tty_key_sets_the_single_shared_event_and_restores_everything(self) -> None:
        for key in ("q", "x", "\x1b"):
            with self.subTest(key=repr(key)):
                stop, signals, termios, tty, _warnings = self.make_stop(
                    stdin=FakeInput(tty=True, characters=key),
                    thread_factory=ImmediateThread,
                )

                with stop as entered:
                    self.assertIs(entered, stop)
                    self.assertIsInstance(stop.event, threading.Event)
                    self.assertTrue(stop.event.is_set())

                self.assertEqual(termios.get_calls, [42])
                self.assertEqual(tty.calls, [42])
                self.assertEqual(termios.set_calls, [(42, termios.TCSADRAIN, termios.saved)])
                self.assertEqual(signals.current[signals.SIGINT], "old-int")
                self.assertEqual(signals.current[signals.SIGTERM], "old-term")

    def test_sigint_and_sigterm_handlers_set_the_same_event(self) -> None:
        for signum in (FakeSignalAPI.SIGINT, FakeSignalAPI.SIGTERM):
            with self.subTest(signum=signum):
                stop, signals, _termios, _tty, _warnings = self.make_stop(
                    stdin=FakeInput(tty=True)
                )
                with stop:
                    handler = signals.current[signum]
                    self.assertTrue(callable(handler))
                    handler(signum, None)
                    self.assertTrue(stop.event.is_set())

    def test_non_tty_keeps_signal_stop_and_warns_without_terminal_mutation(self) -> None:
        stop, signals, termios, tty, warnings = self.make_stop(
            stdin=FakeInput(tty=False)
        )

        with stop:
            signals.current[signals.SIGTERM](signals.SIGTERM, None)

        self.assertTrue(stop.event.is_set())
        self.assertIn("non-TTY", warnings.getvalue())
        self.assertEqual(termios.get_calls, [])
        self.assertEqual(tty.calls, [])

    def test_signal_handlers_are_never_installed_off_the_main_thread(self) -> None:
        stop, signals, _termios, _tty, _warnings = self.make_stop(
            stdin=FakeInput(tty=False),
            is_main_thread=lambda: False,
        )

        with stop:
            self.assertEqual(signals.installs, [])

    def test_stop_and_context_exit_are_idempotent_and_restore_on_error(self) -> None:
        stop, signals, termios, _tty, _warnings = self.make_stop(
            stdin=FakeInput(tty=True)
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with stop:
                stop.stop()
                stop.stop()
                raise RuntimeError("boom")

        stop.__exit__(None, None, None)
        self.assertTrue(stop.event.is_set())
        self.assertEqual(len(termios.set_calls), 1)
        self.assertEqual(signals.current[signals.SIGINT], "old-int")
        self.assertEqual(signals.current[signals.SIGTERM], "old-term")


if __name__ == "__main__":
    unittest.main()
