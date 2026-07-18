# 适用于 reBot B601 的 reBot Arm 102 遥操作器

[English README](./README.md)

本仓库提供了一个 LeRobot 遥操作集成，用于将 reBot Arm 102 作为主手臂，并与 Seeed reBot B601 从手臂配合使用。

当前实现带有明确的预设约束：

- 关节名称按 reBot B601 对齐
- leader 侧关节限位直接取自配置文件
- follower 侧翻转方向在从手集成中配置
- 每次启动校准都会将当前舵机位置设置为零点

## 支持的硬件组合

- 主手臂：reBot Arm 102
- 从手臂：Seeed reBot B601
- 通信方式：reBot Arm 102 使用 UART，B601 使用 CAN 或达妙串口桥

## 安装

先安装 LeRobot，再以可编辑模式安装本包：

```bash
cd lerobot-teleoperator-rebot-arm-102
pip install -e .
```

本包注册了一个 teleoperator 类型：

- `rebot_arm_102_leader`

## 默认映射

- `shoulder_pan` -> 舵机 ID `0`
- `shoulder_lift` -> 舵机 ID `1`
- `elbow_flex` -> 舵机 ID `2`
- `wrist_flex` -> 舵机 ID `3`
- `wrist_yaw` -> 舵机 ID `4`
- `wrist_roll` -> 舵机 ID `5`
- `gripper` -> 舵机 ID `6`

leader 侧关节限位定义在 `lerobot_teleoperator_rebot_arm_102/config_rebot_arm_102_leader.py` 中。
follower 侧翻转方向定义在 B601 从手配置中。

## 使用方法

标准遥操作命令：

```bash
lerobot-teleoperate \
  --teleop.type=rebot_arm_102_leader \
  --teleop.id=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --robot.type=seeed_b601_dm_follower \
    --robot.id=follower1 \
    --robot.port=/dev/ttyACM4 \
    --robot.can_adapter=damiao
```

## 示例脚本

### `read_raw_angles.py`

用途：

- 直接从 SDK 读取 reBot Arm 102 原始舊机角度
- 验证舵机 ID 与关节名称的映射关系
- 检查某个关节是否真的在硬件层发生变化

用法：

```bash
cd lerobot-teleoperator-rebot-arm-102
python examples/read_raw_angles.py --port /dev/ttyUSB0
```

观察要点：

- 每次只移动一个关节
- 确认预期关节对应的列发生变化
- 如果原始值在变化，但 teleop 行为不对，问题通常出在 leader 侧量程配置或 follower 侧方向配置，而不是 SDK 读取

### `read_leader_follower_compare.py`

用途：

- 并排读取 reBot Arm 102 主手输出和 B601 从手观测值
- 逐关节比较 `leader`、`follower` 和 `delta`
- 在不发送从手控制命令的前提下安全排查方向不一致问题

前置条件：

- 需要准备好一套 reBot B601 机械臂
- 安装 reBot B601 的 Python 集成：https://github.com/Seeed-Projects/lerobot-robot-seeed-b601

行为说明：

- 连接 B601 从手后，脚本会关闭力矩，便于手动移动机械臂
- 脚本不会向从手发送动作命令

Damiao 从手接在 `can0` 时的用法：

```bash
cd lerobot-teleoperator-rebot-arm-102
python examples/read_leader_follower_compare.py --leader-port /dev/ttyUSB0 --follower-port can0 --follower-type dm
```

通过达妙串口桥连接 Damiao 从手时的用法：

```bash
cd lerobot-teleoperator-rebot-arm-102
python examples/read_leader_follower_compare.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyACM0 --follower-type dm --follower-can-adapter damiao
```

输出列说明：

脚本会显示 8 列数据帮助调试方向与限位：

- `raw` — 舵机原始角度（未经处理）
- `leader` — leader 侧经解缠与限位裁剪后的输出角度
- `f.dir` — follower 侧当前配置的方向/比例系数
- `mapped` — leader × f.dir 后、送入 follower 限位前的目标角度
- `range` — 配置文件中该关节的限位范围
- `follower` — B601 从手当前观测角度
- `delta` — follower - mapped 的差值

观察要点：

- 在主手上移动一个关节，观察 `raw` 列是否随之变化
- 对比 `mapped` 与 `follower` 的变化方向是否一致
- 若方向相反，修改 follower 配置中的 `joint_directions` 对应关节
- 若 `leader` 与预期差异大，检查 `joint_ranges` 是否覆盖实际运动范围

## 说明

- 按当前实现，启动校准会把每个 reBot Arm 102 舊机的当前位置重设为零点。
- `joint_ranges` 取自配置文件，而不是校准数据。
- 如果某个关节看起来总是卡在某个限位附近，优先检查 `joint_ranges`。
