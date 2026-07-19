#!/usr/bin/env python3
"""Local-only ReBot teleoperation control panel.

The server owns the teleoperation subprocess and exposes a small same-origin
JSON API to the browser UI. It never imports the robot driver in-process, so a
failed UI request cannot take ownership of either serial bus.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

from serial.tools import list_ports

from training_workspace import (
    TrainingConfigError,
    TrainingManager,
    attempt_archive_inventory,
    attempt_artifact_path,
    camera_report,
    dataset_inventory,
    preflight as training_preflight,
    replay_attempt,
    training_profile_status,
    training_recipe,
)


KIT_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = Path(__file__).resolve().parent / "static"
OWNED_PROCESS = STATIC_ROOT.parent / "owned_process.py"
RUNTIME_ROOT = Path(
    os.environ.get("RUNTIME_ROOT", KIT_ROOT.parent / "rebot_setup" / "vendor" / "rebot_lerobot")
)
VENV = Path(os.environ.get("VENV", RUNTIME_ROOT / ".venv"))
TELEOPERATE_BIN = Path(os.environ.get("TELEOPERATE_BIN", VENV / "bin" / "lerobot-teleoperate"))
HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", RUNTIME_ROOT / "lerobot-home"))
SITE_PACKAGE_MATCHES = sorted((VENV / "lib").glob("python*/site-packages"))
SITE_PACKAGES = (
    SITE_PACKAGE_MATCHES[-1]
    if SITE_PACKAGE_MATCHES
    else VENV / "lib" / "python3.10" / "site-packages"
)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

PRESETS: dict[str, dict[str, Any]] = {
    "low": {
        "label": "Low",
        "hz": 10,
        "velocity": 5,
        "max_step": 0.5,
        "gripper_force": 0.05,
    },
    "balanced": {
        "label": "Balanced",
        "hz": 15,
        "velocity": 15,
        "max_step": 1.0,
        "gripper_force": 0.05,
    },
    "responsive": {
        "label": "Responsive",
        "hz": 30,
        "velocity": 150,
        "max_step": 5.0,
        "gripper_force": 0.05,
    },
    "fast_500": {
        "label": "Fast 500",
        "hz": 60,
        "velocity": 500,
        "max_step": 8.3,
        "gripper_force": 0.05,
    },
    "low_latency": {
        "label": "Low-latency 500",
        "hz": 120,
        "velocity": 500,
        "max_step": 4.2,
        "gripper_force": 0.05,
    },
    "hand_tracking": {
        "label": "Hand-tracking 2000",
        "hz": 240,
        "velocity": 2000,
        "max_step": 8.4,
        "gripper_force": 0.05,
    },
}

ACTUAL_HZ_RE = re.compile(r"Teleop running:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz")


class ConfigError(ValueError):
    pass


def _finite_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite")
    if not minimum <= number <= maximum:
        raise ConfigError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def validate_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigError("Request body must be a JSON object")

    hz_value = _finite_number(payload.get("hz", 240), "Hz", 1, 240)
    if not hz_value.is_integer():
        raise ConfigError("Hz must be a whole number")
    hz = int(hz_value)
    max_step = _finite_number(payload.get("max_step", 8.4), "Max step", 0.01, 45)
    gripper_force = _finite_number(
        payload.get("gripper_force", 0.05), "Gripper force ratio", 0, 1
    )
    duration_s = _finite_number(payload.get("duration_s", 0), "Duration", 0, 86400)

    supplied_velocities = payload.get("velocities")
    if supplied_velocities is None:
        global_velocity = _finite_number(
            payload.get("global_velocity", 2000), "Global velocity", 0.1, 2000
        )
        velocities = {joint: global_velocity for joint in JOINTS}
    elif isinstance(supplied_velocities, dict):
        extra = sorted(set(supplied_velocities) - set(JOINTS))
        missing = sorted(set(JOINTS) - set(supplied_velocities))
        if extra or missing:
            raise ConfigError(f"Velocity keys mismatch; missing={missing}, extra={extra}")
        velocities = {
            joint: _finite_number(
                supplied_velocities[joint], f"{joint} velocity", 0.1, 2000
            )
            for joint in JOINTS
        }
    else:
        raise ConfigError("velocities must be an object keyed by joint name")

    return {
        "hz": hz,
        "max_step": max_step,
        "gripper_force": gripper_force,
        "duration_s": duration_s,
        "velocities": velocities,
        "tracking_cap": hz * max_step,
    }


def detect_ports() -> dict[str, str]:
    ports = list(list_ports.comports())
    profile_status = training_profile_status()
    hardware = profile_status.get("profile", {}).get("hardware_identity", {})
    follower_usb = hardware.get("follower_usb", {})
    leader_usb = hardware.get("leader_usb", {})
    try:
        follower_vid_pid = (int(follower_usb["vid"]), int(follower_usb["pid"]))
        leader_vid_pid = (int(leader_usb["vid"]), int(leader_usb["pid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Training profile USB identity is unavailable") from exc

    def unique(vid: int, pid: int, label: str) -> str:
        hits = [p.device for p in ports if p.vid == vid and p.pid == pid]
        if len(hits) != 1:
            raise RuntimeError(f"{label}: expected exactly one port, found {hits}")
        return hits[0]

    return {
        "follower": unique(*follower_vid_pid, "Follower"),
        "leader": unique(*leader_vid_pid, "Leader"),
    }


def ports_in_use(ports: dict[str, str]) -> str | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/lsof", ports["follower"], ports["leader"]],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


class TeleopManager:
    def __init__(
        self,
        simulate: bool = False,
        hardware_lock: threading.Lock | None = None,
    ) -> None:
        self.simulate = simulate
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._hardware_lock = hardware_lock or threading.Lock()
        self._hardware_token: object | None = None
        self._process: subprocess.Popen[str] | None = None
        self._state = "READY"
        self._actual_hz: float | None = None
        self._requested_config: dict[str, Any] | None = None
        self._started_at: float | None = None
        self._last_exit_code: int | None = None
        self._fault: str | None = None
        self._clamp_count = 0
        self._warning_count = 0
        self._error_count = 0
        self._connected = {"follower": False, "leader": False}
        self._seq = 0
        self._logs: deque[dict[str, Any]] = deque(maxlen=1500)
        self._append_log("INFO", "Teleop GUI backend ready")

    def _claim_hardware(self) -> object:
        if not self._hardware_lock.acquire(blocking=False):
            raise RuntimeError(
                "The ReBot hardware is starting, running, or stopping in another session"
            )
        token = object()
        with self._lock:
            self._hardware_token = token
        return token

    def _release_hardware(self, token: object) -> None:
        with self._lock:
            if self._hardware_token is not token:
                return
            self._hardware_token = None
        self._hardware_lock.release()

    def _release_process_ownership_when_done(
        self,
        process: subprocess.Popen[str],
        token: object,
        owner_write_fd: int,
    ) -> None:
        try:
            process.wait()
        finally:
            try:
                os.close(owner_write_fd)
            except OSError:
                pass
            self._release_hardware(token)

    def _append_log(self, level: str, message: str) -> None:
        with self._lock:
            self._seq += 1
            self._logs.append(
                {
                    "seq": self._seq,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": level,
                    "message": message,
                }
            )

    def _devices(self) -> dict[str, Any]:
        if self.simulate:
            ports = {"follower": "/dev/sim-follower", "leader": "/dev/sim-leader"}
            busy = None
            follower_cal = True
            leader_cal = True
        else:
            ports = detect_ports()
            busy = ports_in_use(ports)
            profile_files = training_profile_status().get("calibration_files", {})
            follower_cal = bool(profile_files.get("follower", {}).get("matches"))
            leader_cal = bool(profile_files.get("leader", {}).get("matches"))
        return {
            "ports": ports,
            "ports_free": busy is None,
            "port_owner": busy,
            "calibration": {
                "follower": follower_cal,
                "leader": leader_cal,
            },
        }

    def devices(self) -> dict[str, Any]:
        try:
            return {"ok": True, **self._devices()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _real_command(self, config: dict[str, Any], ports: dict[str, str]) -> list[str]:
        velocity_list = [config["velocities"][joint] for joint in JOINTS]
        command = [
            str(TELEOPERATE_BIN),
            "--robot.type=seeed_b601_dm_follower",
            f"--robot.port={ports['follower']}",
            "--robot.id=follower1",
            "--robot.can_adapter=damiao",
            "--robot.dm_serial_baud=921600",
            f"--robot.max_relative_target={config['max_step']:g}",
            "--robot.pos_vel_velocity=" + json.dumps(velocity_list, separators=(",", ":")),
            f"--robot.force_pos_torque_ration={config['gripper_force']:g}",
            "--robot.disable_torque_on_disconnect=true",
            "--teleop.type=rebot_arm_102_leader",
            f"--teleop.port={ports['leader']}",
            "--teleop.id=rebot_arm_102_leader",
            "--teleop.baudrate=1000000",
            f"--fps={config['hz']}",
            "--display_data=false",
        ]
        if config["duration_s"] > 0:
            command.append(f"--teleop_time_s={config['duration_s']:g}")
        return command

    @staticmethod
    def _simulation_command(config: dict[str, Any]) -> list[str]:
        duration = config["duration_s"]
        code = (
            "import time\n"
            "try:\n"
            " print('leader connected.', flush=True)\n"
            " print('follower connected.', flush=True)\n"
            " start=time.monotonic()\n"
            " while True:\n"
            f"  print('Teleop running: {config['hz']:.1f} Hz (last loop 1.0ms)', flush=True)\n"
            "  time.sleep(0.25)\n"
            + (f"  if time.monotonic()-start >= {duration}: break\n" if duration > 0 else "")
            + "except KeyboardInterrupt:\n pass\n"
            "finally:\n"
            " print('follower disconnected.', flush=True)\n"
            " print('leader disconnected.', flush=True)\n"
        )
        return [sys.executable, "-u", "-c", code]

    def start(self, payload: Any) -> dict[str, Any]:
        config = validate_config(payload)
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("Another start or stop operation is in progress")
        hardware_token: object | None = None
        owner_read_fd: int | None = None
        owner_write_fd: int | None = None
        try:
            with self._lock:
                if self._process is not None:
                    if self._process.poll() is None:
                        raise RuntimeError("Teleoperation is already running")
                    raise RuntimeError("Previous teleoperation is still finalizing")
            hardware_token = self._claim_hardware()
            with self._lock:
                self._state = "STARTING"
                self._actual_hz = None
                self._requested_config = config
                self._started_at = None
                self._last_exit_code = None
                self._fault = None
                self._clamp_count = 0
                self._warning_count = 0
                self._error_count = 0
                self._connected = {"follower": False, "leader": False}

            try:
                devices = self._devices()
                if not devices["calibration"]["follower"] or not devices["calibration"]["leader"]:
                    raise RuntimeError("Both existing seven-joint calibration files are required")
                if not devices["ports_free"]:
                    raise RuntimeError("A ReBot serial port is already in use")
                if not self.simulate and not TELEOPERATE_BIN.is_file():
                    raise RuntimeError(f"Teleoperate executable is missing: {TELEOPERATE_BIN}")

                command = (
                    self._simulation_command(config)
                    if self.simulate
                    else self._real_command(config, devices["ports"])
                )
                env = os.environ.copy()
                runtime_paths = [
                    RUNTIME_ROOT / "lerobot" / "src",
                    RUNTIME_ROOT / "lerobot-robot-seeed-b601",
                    RUNTIME_ROOT / "lerobot-teleoperator-rebot-arm-102",
                    SITE_PACKAGES / "rerun_sdk",
                ]
                inherited_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(path) for path in runtime_paths]
                    + ([inherited_pythonpath] if inherited_pythonpath else [])
                )
                env["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
                env["PATH"] = str(VENV / "bin") + os.pathsep + env.get("PATH", "")
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONNOUSERSITE"] = "1"

                self._append_log(
                    "INFO",
                    (
                        f"Starting teleop: {config['hz']} Hz, "
                        f"step {config['max_step']:g} deg/cycle, "
                        f"velocities {[config['velocities'][j] for j in JOINTS]} deg/s"
                    ),
                )
                owner_read_fd, owner_write_fd = os.pipe()
                wrapped_command = [
                    sys.executable,
                    str(OWNED_PROCESS),
                    "--owner-fd",
                    str(owner_read_fd),
                    "--owner-stop-signal",
                    "SIGINT",
                    "--",
                    *command,
                ]
                process = subprocess.Popen(
                    wrapped_command,
                    cwd=str(KIT_ROOT if self.simulate else RUNTIME_ROOT),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    pass_fds=(owner_read_fd,),
                )
            except Exception as exc:
                for descriptor in (owner_read_fd, owner_write_fd):
                    if descriptor is None:
                        continue
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                if hardware_token is not None:
                    self._release_hardware(hardware_token)
                with self._lock:
                    self._state = "FAULT"
                    self._fault = str(exc)
                    self._append_log("ERROR", f"Start failed: {exc}")
                raise

            os.close(owner_read_fd)
            with self._lock:
                self._process = process
                self._started_at = time.monotonic()
            assert hardware_token is not None and owner_write_fd is not None
            threading.Thread(
                target=self._release_process_ownership_when_done,
                args=(process, hardware_token, owner_write_fd),
                name="teleop-process-owner",
                daemon=True,
            ).start()
        finally:
            self._lifecycle_lock.release()

        threading.Thread(
            target=self._read_process,
            args=(process,),
            name="teleop-log-reader",
            daemon=True,
        ).start()
        return self.status()

    def _read_process(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            level = "INFO"
            if "ERROR" in line or "Traceback" in line or "failed" in line.lower():
                level = "ERROR"
            elif "WARNING" in line or "clamped to be safe" in line:
                level = "WARN"

            with self._lock:
                if self._process is not process:
                    continue
                if level == "WARN":
                    self._warning_count += 1
                elif level == "ERROR":
                    self._error_count += 1
                if "clamped to be safe" in line:
                    self._clamp_count += 1
                match = ACTUAL_HZ_RE.search(line)
                if match:
                    self._actual_hz = float(match.group(1))
                    self._state = "RUNNING"
                if "follower1 SeeedB601DMFollower connected" in line or "follower connected" in line:
                    self._connected["follower"] = True
                if "leader connected" in line.lower():
                    self._connected["leader"] = True
                if "follower disconnected" in line.lower():
                    self._connected["follower"] = False
                if "leader disconnected" in line.lower():
                    self._connected["leader"] = False
            self._append_log(level, line)

        exit_code = process.wait()
        with self._lock:
            if self._process is not process:
                return
            self._process = None
            self._last_exit_code = exit_code
            self._actual_hz = None
            self._connected = {"follower": False, "leader": False}
            requested_stop = self._state == "STOPPING"
            if exit_code == 0 or requested_stop:
                self._state = "STOPPED"
                self._fault = None
                self._append_log("INFO", f"Teleop stopped (exit {exit_code})")
            else:
                self._state = "FAULT"
                self._fault = f"Teleop exited with code {exit_code}"
                self._append_log("ERROR", self._fault)

    def stop(self) -> dict[str, Any]:
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("Another start or stop operation is in progress")
        try:
            return self._stop_locked()
        finally:
            self._lifecycle_lock.release()

    def shutdown_cleanup(self) -> dict[str, Any]:
        """Wait for an in-flight lifecycle call and retain child ownership."""
        with self._lifecycle_lock:
            try:
                return self._stop_locked()
            except RuntimeError:
                # The normal stop path already sent SIGINT twice.  Repeatedly
                # interrupting a follower disconnect can make recovery worse,
                # so keep the GUI process alive and retain ownership until the
                # teleop child actually exits (for example after a physical
                # power cut clears a wedged bus).
                with self._lock:
                    process = self._process
                if process is None or process.poll() is not None:
                    return self.status()
                waited_s = 13
                self._append_log(
                    "ERROR",
                    "GUI shutdown is waiting for the teleop child; use the hardware "
                    "power cut if the arm bus is wedged",
                )
                while process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        waited_s += 10
                        self._append_log(
                            "ERROR",
                            f"Still waiting for teleop disconnect ({waited_s}s); "
                            "the GUI will not orphan a process holding the arms",
                        )
                return self.status()

    def _stop_locked(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                self._state = "STOPPED"
                return self.status()
            self._state = "STOPPING"
            self._append_log("INFO", "Stop requested; sending SIGINT")

        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return self.status()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._append_log("WARN", "Teleop did not stop after 10 seconds; sending SIGINT again")
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with self._lock:
                    self._state = "FAULT"
                    self._fault = "Teleop did not acknowledge SIGINT; use the hardware power cut"
                raise RuntimeError(self._fault)
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            runtime_s = (
                time.monotonic() - self._started_at
                if running and self._started_at is not None
                else 0
            )
            return {
                "state": self._state,
                "running": running,
                "pid": self._process.pid if running and self._process is not None else None,
                "actual_hz": self._actual_hz,
                "requested": self._requested_config,
                "runtime_s": runtime_s,
                "last_exit_code": self._last_exit_code,
                "fault": self._fault,
                "clamp_count": self._clamp_count,
                "warning_count": self._warning_count,
                "error_count": self._error_count,
                "connected": dict(self._connected),
                "latest_seq": self._seq,
                "simulate": self.simulate,
            }

    def logs_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            return [entry for entry in self._logs if entry["seq"] > since]


class ReBotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        manager: TeleopManager,
        training_manager: TrainingManager,
    ):
        super().__init__(address, ReBotHandler)
        self.manager = manager
        self.training_manager = training_manager
        self.control_token = secrets.token_urlsafe(24)


class ReBotHandler(BaseHTTPRequestHandler):
    server: ReBotHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def end_headers(self) -> None:
        # A framed localhost page could otherwise be clickjacked by an external
        # site while still possessing this page's control token.
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'",
        )
        super().end_headers()

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start = 0
        end = max(size - 1, 0)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range", "")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or size <= 0:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            first, last = match.groups()
            if not first:
                suffix = int(last or "0")
                if suffix <= 0:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                start = max(0, size - suffix)
            else:
                start = int(first)
            end = int(last) if first and last else end
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        content_length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _camera_snapshot(self, label: str, fallback_index: int) -> Path:
        report = camera_report()
        camera = (report.get("cameras") or {}).get(label) or {}
        candidate = Path(str(camera.get("image_path") or ""))
        camera_root = (KIT_ROOT / "camera-check").resolve()
        try:
            if candidate.is_file() and candidate.resolve().parent == camera_root:
                return candidate
        except OSError:
            pass
        return camera_root / f"{label}_index{fallback_index}.png"

    def _read_json(self) -> Any:
        if self.headers.get_content_type() != "application/json":
            raise ConfigError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ConfigError("Invalid Content-Length") from exc
        if length <= 0 or length > 65536:
            raise ConfigError("JSON body must be between 1 and 65536 bytes")
        return json.loads(self.rfile.read(length))

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == self.server.server_port

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        return host in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def _control_token_allowed(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-ReBot-Control", ""), self.server.control_token
        )

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._send_json({"error": "Host not allowed"}, HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
        elif parsed.path in {"/training", "/training.html"}:
            self._send_file(STATIC_ROOT / "training.html", "text/html; charset=utf-8")
        elif parsed.path == "/styles.css":
            self._send_file(STATIC_ROOT / "styles.css", "text/css; charset=utf-8")
        elif parsed.path == "/training.css":
            self._send_file(STATIC_ROOT / "training.css", "text/css; charset=utf-8")
        elif parsed.path == "/app.js":
            self._send_file(STATIC_ROOT / "app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/training.js":
            self._send_file(STATIC_ROOT / "training.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/camera/front.png":
            self._send_file(self._camera_snapshot("front", 0), "image/png")
        elif parsed.path == "/camera/side.png":
            self._send_file(self._camera_snapshot("side", 1), "image/png")
        elif parsed.path == "/api/status":
            self._send_json(
                {**self.server.manager.status(), "control_token": self.server.control_token}
            )
        elif parsed.path == "/api/devices":
            self._send_json(self.server.manager.devices())
        elif parsed.path == "/api/presets":
            self._send_json({"joints": JOINTS, "presets": PRESETS})
        elif parsed.path == "/api/logs":
            query = parse_qs(parsed.query)
            try:
                since = max(0, int(query.get("since", ["0"])[0]))
            except ValueError:
                since = 0
            self._send_json({"logs": self.server.manager.logs_since(since)})
        elif parsed.path == "/api/training/status":
            self._send_json(
                {
                    **self.server.training_manager.status(),
                    "control_token": self.server.control_token,
                }
            )
        elif parsed.path == "/api/training/preflight":
            self._send_json(training_preflight(self.server.manager.devices()))
        elif parsed.path == "/api/training/profile":
            self._send_json(training_profile_status())
        elif parsed.path == "/api/training/datasets":
            self._send_json({"datasets": dataset_inventory()})
        elif parsed.path == "/api/training/attempts":
            query = parse_qs(parsed.query)
            dataset_name = query.get("dataset", [""])[0]
            try:
                self._send_json(attempt_archive_inventory(dataset_name))
            except TrainingConfigError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/api/training/attempt/video":
            query = parse_qs(parsed.query)
            attempt_id = query.get("attempt_id", [""])[0]
            camera = query.get("camera", [""])[0]
            try:
                self._send_file(
                    attempt_artifact_path(attempt_id, camera),
                    "video/mp4",
                )
            except TrainingConfigError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        elif parsed.path == "/api/training/recipe":
            query = parse_qs(parsed.query)
            dataset_name = query.get("dataset", [""])[0]
            try:
                self._send_json(training_recipe(dataset_name))
            except TrainingConfigError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/api/training/logs":
            query = parse_qs(parsed.query)
            try:
                since = max(0, int(query.get("since", ["0"])[0]))
            except ValueError:
                since = 0
            self._send_json(
                {"logs": self.server.training_manager.logs_since(since)}
            )
        elif parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._host_allowed() or not self._origin_allowed():
            self._send_json({"error": "Local origin not allowed"}, HTTPStatus.FORBIDDEN)
            return
        if not self._control_token_allowed():
            self._send_json({"error": "Missing or invalid control token"}, HTTPStatus.FORBIDDEN)
            return
        try:
            if self.path == "/api/start":
                payload = self._read_json()
                self._send_json(self.server.manager.start(payload), HTTPStatus.ACCEPTED)
            elif self.path == "/api/stop":
                if self.headers.get("Content-Length", "0") not in {"", "0"}:
                    self._read_json()
                self._send_json(self.server.manager.stop())
            elif self.path == "/api/training/camera-check":
                payload = self._read_json()
                self._send_json(
                    self.server.training_manager.start_camera_check(payload),
                    HTTPStatus.ACCEPTED,
                )
            elif self.path == "/api/training/record/start":
                payload = self._read_json()
                self._send_json(
                    self.server.training_manager.start_record(
                        payload,
                        self.server.manager.devices(),
                    ),
                    HTTPStatus.ACCEPTED,
                )
            elif self.path == "/api/training/record/control":
                payload = self._read_json()
                self._send_json(
                    self.server.training_manager.record_control(payload)
                )
            elif self.path == "/api/training/attempt/replay":
                payload = self._read_json()
                self._send_json(replay_attempt(payload), HTTPStatus.ACCEPTED)
            elif self.path == "/api/training/attempt/label":
                payload = self._read_json()
                self._send_json(
                    self.server.training_manager.review_attempt(
                        {**payload, "action": "label_excluded"}
                    )
                )
            elif self.path == "/api/training/attempt/review":
                payload = self._read_json()
                self._send_json(self.server.training_manager.review_attempt(payload))
            elif self.path == "/api/training/validate":
                payload = self._read_json()
                self._send_json(
                    self.server.training_manager.start_validation(payload),
                    HTTPStatus.ACCEPTED,
                )
            elif self.path == "/api/training/job/stop":
                if self.headers.get("Content-Length", "0") not in {"", "0"}:
                    self._read_json()
                self._send_json(self.server.training_manager.stop_job())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ConfigError, TrainingConfigError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except Exception as exc:
            self.server.manager._append_log("ERROR", f"API error: {exc}")
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ReBot teleoperation GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("The teleop GUI may only bind to localhost")

    # One ownership token spans manual teleop and data collection, closing the
    # check-then-open race where two browser tabs could both observe free ports
    # before either child had opened them.
    hardware_lock = threading.Lock()
    manager = TeleopManager(simulate=args.simulate, hardware_lock=hardware_lock)
    training_manager = TrainingManager(
        simulate=args.simulate,
        hardware_lock=hardware_lock,
    )
    server = ReBotHTTPServer((args.host, args.port), manager, training_manager)
    server.timeout = 0.5
    stopping = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        manager._append_log("INFO", f"GUI server stopping on signal {signum}")
        try:
            manager.shutdown_cleanup()
        except Exception as exc:
            manager._append_log("ERROR", f"Teleop cleanup failed: {exc}")
        try:
            training_manager.shutdown_cleanup()
        except Exception as exc:
            training_manager._append_log("ERROR", f"Training workspace cleanup failed: {exc}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    atexit.register(
        lambda: manager.shutdown_cleanup() if manager.status()["running"] else None
    )
    atexit.register(
        lambda: training_manager.shutdown_cleanup()
        if training_manager.status()["running"]
        else None
    )

    url = f"http://{args.host}:{args.port}/"
    print(f"ReBot Teleop GUI: {url}", flush=True)
    if args.simulate:
        print("SIMULATION MODE: no serial devices will be opened", flush=True)
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        while not stopping.is_set():
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
