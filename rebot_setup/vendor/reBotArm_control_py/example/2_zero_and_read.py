#!/usr/bin/env python3
"""机械臂零点校准 + 随动模式（MIT 零刚度）。
使能后，机械臂处于随动状态（MIT，kp=0, kd=0），可手动自由移动，
同时实时打印各关节角度（度）。

Robot arm zero calibration + free-drive mode (MIT zero stiffness).
After enabling, the arm is in free-drive state (MIT, kp=0, kd=0) and can be
manually moved freely while real-time joint angles (degrees) are printed.

用法 / Usage::

    python example/2_zero_and_read.py
      自动从 config/rebotarm.yaml 的 hardware_yaml 读取配置
      Auto-load config from config/rebotarm.yaml's hardware_yaml field

    python example/2_zero_and_read.py rebotarm_dm.yaml
      强制指定硬件配置文件
      Force a specific hardware config file
"""
import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm


_hw_yaml = sys.argv[1] if len(sys.argv) > 1 else None
rebotarm = RebotArm(_hw_yaml)
print(f"[{rebotarm.hardware_yaml}] 使用配置:")
print(f"[{rebotarm.hardware_yaml}] Using config: {rebotarm.hardware_yaml}")
rebotarm.connect()
print("--- 连接成功 ---")
print("--- Connection OK ---\n")
rebotarm.set_zero()
print("--- 零点已设置 ---")
print("--- Zero set ---\n")

n_arm = rebotarm.arm.num_joints
n_total = rebotarm.num_joints

# 随动模式：MIT 零刚度，机械臂可自由手动移动
# Free-drive mode: MIT zero stiffness, arm can be freely moved by hand
_zeros_arm = np.zeros(n_arm)


def fresh_controller(r: RebotArm, dt: float) -> None:
    r.arm.send_mit(
        pos=_zeros_arm,
        vel=np.zeros(n_arm),
        kp=np.zeros(n_arm),
        kd=np.zeros(n_arm),
        tau=np.zeros(n_arm),
    )
    r.gripper.send_mit(
        pos=np.array([0.0]),
        vel=np.zeros(1),
        kp=np.zeros(1),
        kd=np.zeros(1),
        tau=np.zeros(1),
    )


print("--- 随动模式 ---")
print("实时角度（度）。按 Ctrl+C 退出。\n")
print("--- Free-drive mode ---")
print("Realtime angles (deg). Press Ctrl+C to exit.\n")
rebotarm.arm.mode_mit()
rebotarm.gripper.mode_mit()
rebotarm.enable_all()
rebotarm.start_control_loop(fresh_controller)
try:
    while True:
        positions = rebotarm.get_positions()
        row = "  ".join(f"{p*180/np.pi:+.2f}" for p in positions)
        print(f"\r{row}  ", end="", flush=True)
        time.sleep(0.002)
except (KeyboardInterrupt, EOFError):
    pass
finally:
    rebotarm.disconnect()
