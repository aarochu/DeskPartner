#!/usr/bin/env python3
"""
RobStride RS06 电机控制测试 — 直接使用 motorbridge SDK。
注意: RobStride 使用 CAN 总线通信，不是串口。
      确保 CAN 接口已配置 (如 can0) 并正确连接电机。

RobStride RS06 motor control test — using the motorbridge SDK directly.
Note: RobStride uses CAN bus, not serial.
      Make sure the CAN interface is configured (e.g. can0) and the motor is wired correctly.

用法 / Usage:
    python example/rs06_test.py

交互命令 / Interactive commands:
    ping                                   — ping 电机获取响应 / ping motor to get a response
    mit <pos_deg> [<vel> <kp> <kd> <tau>] — MIT 模式指令 / MIT mode command
    posvel <pos_deg> [<vlim> [<loc_kp>]]  — POS_VEL 模式指令 / POS_VEL mode command
    vel <vel_rad_s>                       — 纯速度模式指令 / Pure velocity mode command
    enable                                 — 使能 / Enable
    disable                                — 去使能 / Disable
    set_zero                               — 电机零位设置 / Set motor zero position
    mode <mit|posvel|vel>                 — 切换控制模式 / Switch control mode
    state                                  — 打印当前状态 / Print current state
    clear_error                            — 清除错误 / Clear error
    read_param <param_id> [<type>]        — 读取参数 (默认 f32) / Read parameter (default f32)
    write_param <param_id> <value> [<type>] — 写入参数 (默认 f32) / Write parameter (default f32)
    loop                                   — 进入循环控制模式（stat 查看状态） / Enter loop control mode (use stat to inspect)
    q / quit                               — 退出 / Quit
"""

import math
import select
import signal
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motorbridge import Controller, Mode

CHANNEL = "can0"  # RobStride 使用 CAN 接口 / RobStride uses CAN interface
MOTOR_ID = 0x01  # 电机 ID / Motor ID
HOST_ID = 0xFD  # RobStride host/feedback ID (默认 0xFD) / RobStride host/feedback ID (default 0xFD)
MODEL = "rs-06"  # RS06 电机型号 / RS06 motor model

# RS06 电机 PID 参数配置 / RS06 motor PID parameter configuration
RS06_LOC_KP = 13.0  # 位置环 Kp (POS_VEL 模式) / Position loop Kp (POS_VEL mode)
RS06_SPD_KP = 12.0  # 速度环 Kp / Velocity loop Kp
RS06_SPD_KI = 0.1  # 速度环 Ki / Velocity loop Ki
RS06_MIT_KP = 50.0  # MIT 模式 Kp / MIT mode Kp
RS06_MIT_KD = 3.0  # MIT 模式 Kd / MIT mode Kd

PI_OVER_180 = math.pi / 180.0


def signal_handler(sig, frame):
    print("\n[ctrl+c] 退出 / exit")
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"连接到 CAN 接口 / Connecting to CAN interface {CHANNEL} ...")
    print(f"电机配置 / Motor config: id={MOTOR_ID:#04x} host_id={HOST_ID:#04x} model={MODEL}")

    ctrl = Controller(CHANNEL)
    motor = ctrl.add_robstride_motor(MOTOR_ID, HOST_ID, MODEL)
    print(f"电机已注册 / Motor registered: motor_id={MOTOR_ID:#04x} host_id={HOST_ID:#04x} model={MODEL}")

    # ---- 命令处理器 / Command handlers ------------------------------------

    def do_enable() -> None:
        ctrl.enable_all()
        time.sleep(0.3)
        print("电机已使能 / Motor enabled")

    def do_disable() -> None:
        ctrl.disable_all()
        print("电机已去使能 / Motor disabled")

    def do_set_zero() -> None:
        """设置软件零位。 / Set software zero position."""
        for _ in range(10):
            motor.request_feedback()
            st = motor.get_state()
            if st is not None and st.status_code == 0:
                break
            time.sleep(0.05)
        motor.set_zero_position()  # type-6
        time.sleep(0.1)
        print("软件零位已设置 / Software zero position set")

    def do_clear_error() -> None:
        """清除电机错误。 / Clear motor error."""
        motor.clear_error()
        time.sleep(0.1)
        print("错误已清除 / Error cleared")

    def do_ping() -> None:
        """Ping 电机。 / Ping motor."""
        try:
            device_id, responder_id = motor.robstride_ping()
            print(f"Ping 成功 / Ping OK: device_id={device_id} responder_id={responder_id:#04x}")
        except Exception as e:
            print(f"Ping 失败 / Ping failed: {e}")

    def do_mode(args: list[str]) -> None:
        if not args:
            print("用法 / Usage: mode <mit|posvel|vel>")
            return
        m = args[0].lower()
        if m == "mit":
            motor.ensure_mode(Mode.MIT, 1000)
            print("切换到 MIT 模式 / Switched to MIT mode")
        elif m == "posvel":
            motor.robstride_write_param_f32(0x701F, RS06_SPD_KP)  # 速度环 Kp / Velocity loop Kp
            time.sleep(0.01)
            motor.robstride_write_param_f32(0x7020, RS06_SPD_KI)  # 速度环 Ki / Velocity loop Ki
            time.sleep(0.01)
            loc_kp = float(args[1]) if len(args) >= 2 else RS06_LOC_KP
            motor.robstride_write_param_f32(0x701E, loc_kp)  # 位置环 Kp / Position loop Kp
            time.sleep(0.02)
            print(f"PID 参数 / PID params: 位置Kp/loc_Kp={loc_kp}, 速度Kp/spd_Kp={RS06_SPD_KP}, 速度Ki/spd_Ki={RS06_SPD_KI}")
            motor.ensure_mode(Mode.POS_VEL, 1000)
            print("切换到 POS_VEL 模式 / Switched to POS_VEL mode")
        elif m == "vel":
            motor.ensure_mode(Mode.VEL, 1000)
            print("切换到 VEL 模式 / Switched to VEL mode")
        else:
            print(f"未知模式 / Unknown mode: {m}，可用 / available: mit / posvel / vel")

    def do_state() -> None:
        print(
            "[注意 / Notice] motorbridge 内部协议配置问题：RS 电机无法像 DM 电机那样查询，"
            "可能会出现读取不到实际电机状态的情况，请以实际运动效果为准。"
            "\n"
            "motorbridge internal protocol config issue: RS motors cannot be queried like DM motors, "
            "the actual motor state may not be read; please refer to the real motion."
        )
        st = None
        for _ in range(10):
            motor.request_feedback()
            ctrl.poll_feedback_once()
            time.sleep(0.005)
            st = motor.get_state()
            if st is not None and st.status_code == 0:
                break
        if st is None:
            print("无反馈数据 / No feedback data")
            return
        print(
            f"pos={st.pos / PI_OVER_180:+.4f}deg  "
            f"vel={st.vel / PI_OVER_180:+.4f}deg/s  "
            f"torq={st.torq:+.4f}  "
            f"status={st.status_code}"
        )

    def do_mit(args: list[str]) -> None:
        if not args:
            print("用法 / Usage: mit <pos_deg> [<vel> <kp> <kd> <tau>]")
            return
        pos = float(args[0]) * PI_OVER_180
        vel = float(args[1]) if len(args) > 1 else 0.0
        kp = float(args[2]) if len(args) > 2 else RS06_MIT_KP
        kd = float(args[3]) if len(args) > 3 else RS06_MIT_KD
        tau = 0.0
        motor.send_mit(pos, vel, kp, kd, tau)

    def do_posvel(args: list[str]) -> None:
        if not args:
            print("用法 / Usage: posvel <pos_deg> [<vlim> [<loc_kp>]]")
            return
        pos = float(args[0]) * PI_OVER_180
        vlim = float(args[1]) if len(args) > 1 else 1.0
        motor.robstride_write_param_f32(0x7017, abs(vlim))  # limit_spd 速度限幅 / Velocity limit
        motor.robstride_write_param_f32(0x7016, pos)  # loc_ref 位置参考 / Position reference
        print(f"POS_VEL: pos={pos:.4f}rad vlim={vlim}")

    def do_vel(args: list[str]) -> None:
        if not args:
            print("用法 / Usage: vel <vel_rad_s>")
            return
        motor.send_vel(float(args[0]))

    def do_read_param(args: list[str]) -> None:
        """读取 RobStride 参数。 / Read a RobStride parameter."""
        if not args:
            print("用法 / Usage: read_param <param_id> [<type>]")
            print("      type 可选 / options: u8, u16, u32, f32 (默认/default f32)")
            return
        param_id = int(args[0], 0)
        param_type = args[1] if len(args) > 1 else "f32"
        timeout_ms = 1000

        try:
            if param_type == "u8":
                value = motor.robstride_get_param_u8(param_id, timeout_ms)
            elif param_type == "u16":
                value = motor.robstride_get_param_u16(param_id, timeout_ms)
            elif param_type == "u32":
                value = motor.robstride_get_param_u32(param_id, timeout_ms)
            else:
                value = motor.robstride_get_param_f32(param_id, timeout_ms)
            print(f"param 0x{param_id:04X} = {value}")
        except Exception as e:
            print(f"读取参数失败 / Read param failed: {e}")

    def do_write_param(args: list[str]) -> None:
        """写入 RobStride 参数。 / Write a RobStride parameter."""
        if len(args) < 2:
            print("用法 / Usage: write_param <param_id> <value> [<type>]")
            print("      type 可选 / options: u8, u16, u32, f32 (默认/default f32)")
            return
        param_id = int(args[0], 0)
        value = args[1]
        param_type = args[2] if len(args) > 2 else "f32"

        try:
            if param_type == "u8":
                motor.robstride_write_param_u8(param_id, int(value, 0))
            elif param_type == "u16":
                motor.robstride_write_param_u16(param_id, int(value, 0))
            elif param_type == "u32":
                motor.robstride_write_param_u32(param_id, int(value, 0))
            else:
                motor.robstride_write_param_f32(param_id, float(value))
            print(f"param 0x{param_id:04X} 已写入值 / written: {value}")
        except Exception as e:
            print(f"写入参数失败 / Write param failed: {e}")

    # loop 共享状态 / loop shared state
    loop_running = threading.Event()
    loop_lock = threading.Lock()
    _loop = {"mode": "mit", "pos": 0.0, "vel": 0.0, "vlim": 1.0}
    _hist: deque[float] = deque(maxlen=200)
    _lat_hist: deque[float] = deque(maxlen=200)

    def do_loop(_args: list[str]) -> None:
        if loop_running.is_set():
            print("loop 已在运行，q 退出 / loop already running, q to quit")
            return
        loop_running.set()

        cnt, pos, vel, torq, freq, lat, std = 0, "---", "---", "---", 0, 0.0, 0.0
        last = time.perf_counter()
        while loop_running.is_set():
            t0 = time.perf_counter()
            with loop_lock:
                m, p, v, vl = _loop["mode"], _loop["pos"], _loop["vel"], _loop["vlim"]
            if m == "mit":
                motor.send_mit(p * PI_OVER_180, 0, RS06_MIT_KP, RS06_MIT_KD, 0)
            elif m == "posvel":
                motor.robstride_write_param_f32(0x7017, abs(vl))
                motor.robstride_write_param_f32(0x7016, p * PI_OVER_180)
            else:
                motor.send_vel(v)

            dt = t0 - last
            last = t0
            _hist.append(dt)
            n = len(_hist)
            freq = 1 / dt if dt > 0 else 0
            std = statistics.stdev(_hist) * 1000 if n >= 4 else 0.0
            lat = _lat_hist[-1] if _lat_hist else 0.0
            cnt += 1

            for _ in range(20):
                ctrl.poll_feedback_once()
                st = motor.get_state()
                if st and st.status_code == 0:
                    _lat_hist.append((time.perf_counter() - t0) * 1000)
                    pos = f"{st.pos / PI_OVER_180:+.2f}"
                    vel = f"{st.vel / PI_OVER_180:+.2f}"
                    torq = f"{st.torq:+.4f}"
                    break

            # 非阻塞检查输入 / Non-blocking input check
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline()
                if not cmd:
                    break
                parts = cmd.strip().split()
                if parts:
                    cmd = parts[0].lower()
                    args = parts[1:]
                    if cmd == "q":
                        loop_running.clear()
                        break
                    elif cmd == "mit" and args:
                        _loop.update({"mode": "mit", "pos": float(args[0])})
                    elif cmd == "posvel" and args:
                        _loop.update(
                            {
                                "mode": "posvel",
                                "pos": float(args[0]),
                                "vlim": float(args[1]) if len(args) >= 2 else 1.0,
                            }
                        )
                    elif cmd == "vel" and args:
                        _loop.update({"mode": "vel", "vel": float(args[0])})
                    elif cmd == "stat":
                        print(
                            f"stat: cnt={cnt}  pos={pos}deg  vel={vel}  "
                            f"torq={torq}  {freq:.0f}Hz  lat={lat:.1f}ms  std={std:.1f}ms"
                        )
                        h, lh = list(_hist), list(_lat_hist)
                        lat_avg = statistics.mean(lh) if lh else 0
                        lat_std = statistics.stdev(lh) if len(lh) >= 4 else 0
                        avg_hz = f"{1 / statistics.mean(h):.0f}Hz" if h else "0Hz"
                        print(
                            f"avg={avg_hz}  min={min(h) * 1000:.1f}ms  "
                            f"max={max(h) * 1000:.1f}ms  "
                            f"std={statistics.stdev(h) * 1000:.1f}ms  "
                            f"lat={lat_avg:.1f}ms(std={lat_std:.1f})"
                        )
                    else:
                        print("[loop] mit/posvel/vel <参数/args>  stat  q")
            time.sleep(max(0, 0.002 - (time.perf_counter() - t0)))

        print("[loop] 退出 / exit")
        h, lh = list(_hist), list(_lat_hist)
        _hist.clear()
        _lat_hist.clear()
        if h:
            lat_avg = statistics.mean(lh) if lh else 0
            lat_std = statistics.stdev(lh) if len(lh) >= 4 else 0
            avg_hz = f"{1 / statistics.mean(h):.0f}Hz"
            print(
                f"[loop] avg={avg_hz}  min={min(h) * 1000:.1f}ms  "
                f"max={max(h) * 1000:.1f}ms  "
                f"std={statistics.stdev(h) * 1000:.1f}ms  "
                f"lat={lat_avg:.1f}ms(std={lat_std:.1f})\n"
            )

    # ---- 命令表 / Command table -------------------------------------------
    NO_ARG_COMMANDS = frozenset(
        {do_enable, do_disable, do_set_zero, do_clear_error, do_ping, do_state}
    )
    COMMANDS: dict[str, tuple] = {
        "enable": (do_enable, "使能电机 / Enable motor"),
        "disable": (do_disable, "去使能电机 / Disable motor"),
        "set_zero": (do_set_zero, "设置软件零位 / Set software zero"),
        "clear_error": (do_clear_error, "清除电机错误 / Clear motor error"),
        "ping": (do_ping, "Ping 电机 / Ping motor"),
        "state": (do_state, "打印当前状态 / Print current state"),
        "mode": (do_mode, "mode <mit|posvel|vel>"),
        "mit": (do_mit, "mit <pos_deg> [<vel> <kp> <kd> <tau>]"),
        "posvel": (do_posvel, "posvel <pos_deg> [<vlim> [<loc_kp>]]"),
        "vel": (do_vel, "vel <vel_rad_s>"),
        "read_param": (do_read_param, "read_param <param_id> [<type>]"),
        "write_param": (do_write_param, "write_param <param_id> <value> [<type>]"),
        "loop": (do_loop, "loop"),
    }

    print(
        "\n命令 / Commands: enable / disable / set_zero / clear_error / ping / state / "
        "mode / mit / posvel / vel / read_param / write_param / loop / q"
    )
    print(
        "提示 / Tip: mode 会自动在下一条控制指令前生效 / takes effect on the next control command；"
        "loop 进入实时控制循环模式 / enters real-time control loop mode\n"
    )

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("q", "quit", "exit"):
                print("退出 / Quit")
                break

            if cmd not in COMMANDS:
                print(f"未知命令 / Unknown command: {cmd}，可用 / available: {' / '.join(COMMANDS)}")
                continue

            fn, _ = COMMANDS[cmd]
            try:
                if fn in NO_ARG_COMMANDS:
                    fn()
                else:
                    fn(args)
            except Exception as e:
                print(f"错误 / Error: {e}")
    finally:
        ctrl.disable_all()
        ctrl.shutdown()
        ctrl.close()


if __name__ == "__main__":
    main()
