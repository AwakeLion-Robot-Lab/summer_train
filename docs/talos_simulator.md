# Talos 仿真接入与联调手册

本文是 `/home/rm/atooooooom/version` 接入
`/home/rm/rm_summmer/bevy_robomaster_simulator` 的正式运行、验收和排障手册，
适用于当前 Talos IPC v2 共享内存接口。硬件串口和 Talos 仿真共用同一套 L2–L5
算法；仿真适配只位于 L1 `RuntimeIo`，不会复制一套自瞄或打符算法。

## 1. 接入边界

```text
Talos 仿真器
  ├─ RGB 图像、camera_info、world<-barrel 姿态
  └─ ground truth（只读）
             ↓
          TalosIo ─ L2 ─ L3 ─ L4 ─ L5
                                      ↓
                     绝对 yaw/pitch + fire → Talos 共享内存
```

Talos 后端负责读取图像/姿态、编码项目命令、平滑执行绝对姿态、覆盖启动旧命令，
并对异常大角跳转执行“保持当前位置、`fire=0`”安全门。它不负责修改 L2–L5
坐标、不把世界角转换成底盘相对角、不累加增量，也不替代真实电控的模式、弹速和
开火反馈。

## 2. 坐标和 pitch 约定（最重要）

本工程和 `summer_train` 使用同一坐标定义：

| 量 | 正方向 | 说明 |
|---|---|---|
| yaw | 项目约定正方向 | 世界系绝对角 |
| pitch | 低头 | **正值低头，负值抬头** |
| barrel 前向 | `x` | 枪管坐标为 `x 前、y 左、z 上` |

目标在机器人上方时，算法输出通常是负 pitch；Talos 适配层不能再次取反。

### 2.1 线协议编码

共享内存中的 `GimbalCmd.pitch_deg` 是带水平偏置的协议值：

```text
wire_pitch_deg = -project_pitch_deg - 90
project_pitch_deg = -wire_pitch_deg - 90
```

因此 `project=0°` 对应水平 `wire=-90°`，`project=-10°` 对应抬头 10°，
`project=+10°` 对应低头 10°。`-90°` 只是协议偏置，不是算法坐标转换。
Rust 解码后必须直接使用项目 pitch，不能再写成 `-project_pitch`。

### 2.2 姿态和相机

仿真器 `capture.rs` 已把 Bevy 姿态转换为项目统一的 `world<-barrel` 四元数，
C++ 端直接使用：`rotationToRpy(q).z()` 是 yaw，`rotationToRpy(q).y()` 是
项目 pitch，不再乘固定轴矩阵，也不再对 pitch 二次取反。

相机图像仍是 OpenCV 光学系（`z 前、x 右、y 下`）。`T_barrel_camera` 只描述
真实光学系到枪管系的外参。Buff 的 PnP 已在 L3 转到 `x 前/y 左/z 上`，L1
只抵消光学轴转换；再次手动换轴会造成约 90° 的世界 yaw 偏差。

## 3. 配置

主配置为 [`config/talos.yaml`](../config/talos.yaml)：

```yaml
camera_name: talos
talos:
  shm_dir: /tmp
  enemy: blue
  mode: small_buff
  bullet_speed: 25.0
  shoot_enable: false
  timeout_ms: 50
  default_distance_m: 1.0
  max_yaw_jump_deg: 30.0
  max_pitch_jump_deg: 45.0
```

- `camera_name: talos` 选择 Talos 后端，否则仍走硬件相机/串口。
- `enemy` 为 `red` 或 `blue`。
- `mode` 支持 `auto_aim`、`outpost`、`small_buff`、`big_buff`、`idle`。
- `bullet_speed` 是注入 `RobotState` 的仿真弹速，单位 m/s，不是算法内写死值。
- `shoot_enable` 是 Talos 最终开火许可，首次联调必须为 `false`。
- `max_yaw_jump_deg`/`max_pitch_jump_deg` 只保护 Talos 仿真 I/O，超限时保持当前位置
  并清除 `fire`，不会改动 L2–L5 计划。

真实机器人上工作模式通常来自电控串口状态；Talos 没有真实串口模式回传，所以由
`talos.mode` 注入 `RobotState.mode`。这只是仿真输入替代，不能当作实车模式握手。
打符参数和安全策略见 [`docs/buff.md`](buff.md) 与 [`config/buff.yaml`](../config/buff.yaml)。

## 4. 编译

```bash
cd /home/rm/atooooooom/version
xmake f -m release --use_system_deps=y --use_openvino=y
xmake build talos_auto_aim talos_gimbal_check talos_contract_smoke buff_video_replay

cd /home/rm/rm_summmer/bevy_robomaster_simulator
cargo fmt --check
cargo test talos::plugin::tests
cargo build --release
```

仿真器测试中的 `negative_project_pitch_raises_muzzle` 必须通过，它直接验证负
pitch 的实际枪口方向指向项目坐标 `+Z`。

## 5. 标准启动和停止

### 5.1 启动

终端 A，进入机器人视角后先不要按 F5：

```bash
cd /home/rm/rm_summmer/bevy_robomaster_simulator
cargo run --release
```

终端 B：

```bash
cd /home/rm/atooooooom/version
xmake run talos_auto_aim -- config/talos.yaml
```

看到 `runtime I/O backend started talos` 后，回仿真器按 F5。视觉第一帧会先写入
当前位置保持、`fire=0`，用于覆盖共享内存旧命令，避免 F5 启动瞬间消费旧目标。

不要把参数路径拆成两条 shell 命令；换行必须使用反斜杠：

```bash
xmake run talos_auto_aim -- /home/rm/atooooooom/version/config/talos.yaml
```

停止时先在视觉终端 `Ctrl-C`，确认 `talos_auto_aim` 退出后再关闭仿真器。
Talos 是单消费者接口，`talos_auto_aim`、`talos_gimbal_check`、`talos_shm_smoke`
不能同时运行。

### 5.2 打符

先使用 `mode: small_buff` 或 `mode: big_buff`、`shoot_enable: false` 观察检测、
跟踪、规划和跟随；确认方向和目标切换正确后再考虑开火。离线回放没有云台姿态、
电控模式和实时弹速，只能验收检测到弹道预测，且始终强制 `fire=false`。

## 6. 人工方向验收

### 6.1 协议验收（不需要仿真器）

```bash
cd /home/rm/atooooooom/version
xmake run talos_contract_smoke
```

预期：`talos_contract_smoke: ok`。

### 6.2 零偏移和物理方向

仿真器运行、尚未按 F5 时执行一次：

```bash
xmake run talos_gimbal_check -- 0 0 120
```

返回码 2 并提示按 F5 是预期行为。按 F5 后再次执行零偏移，再执行：

```bash
# 相对当前姿态 pitch=-5°，必须抬头
xmake run talos_gimbal_check -- 0 -5 180

# 相对当前姿态 pitch=+5°，必须低头
xmake run talos_gimbal_check -- 0 5 180
```

工具始终写 `fire=0`，末帧 yaw/pitch 误差应不超过 1°。完成后可执行原三条验收：

```bash
xmake run talos_gimbal_check -- 0 0 120
xmake run talos_gimbal_check -- 10 -5 300
xmake run talos_gimbal_check -- -10 5 300
```

这些参数是相对当前姿态的小偏移。`xmake` 对工具非零退出可能额外打印 `execv`
错误，以工具本身返回码和诊断文字为准。

### 6.3 自动瞄准验收

方向验收通过后：关闭 F5 → 确认 `shoot_enable: false` → 启动视觉 → 确认 Talos
后端日志 → 再按 F5。面对目标观察云台平滑追踪，不能出现 90°/180° 猛转。
仿真器按 `vehicle.gimbal_rotation_speed` 沿最短四元数路径追踪绝对角，不会单帧瞬移。

能量机关模型中的 `R` 标识（`FACE_*_R_PADDING/POWERED/UNPOWERED`）已从旋转面
脱离，并使用 glTF 网格包围盒中心对齐到机关中心；同时补偿模型自带的 45° 滚转，
固定显示 powered 版本，并每帧锁定其世界姿态。五片符面和目标仍按原规则旋转，R
标不会随机关转动或熄灭。

## 7. 离线视频回放

`buff_video_replay` 不初始化相机、串口或 Talos：

```bash
cd /home/rm/atooooooom/version
xmake run buff_video_replay \
  --calibration=/home/rm/FosuVision-main/src/ros2_hik_camera-main/config/camera_info.yaml \
  --enemy=red \
  /home/rm/FosuVision-main/src/rm_auto_aim/armor_detector/test/buffVideo.avi
```

常用选项：`--show-original=true`（同步原视频，默认开）、`--sync-playback=true`、
`--wait=0`、`--start-index=N --end-index=M`、`--output=/tmp/buff_replay.avi`、
`--headless=true`。交互键：空格暂停，`N` 单步，`A/D` 前后 1 秒，`J/L` 前后 5 秒，
`R` 回起点，`Q`/`Esc` 退出。原视频和标注窗口使用同一解码帧；算法过慢时降速而不丢帧。

## 8. 常见问题

### F5 后猛低头

停止视觉并关闭 F5，确认仿真器已经重新 `cargo build --release` 后重启；检查
`talos_contract_smoke` 和 `talos_gimbal_check -- 0 -5 180`。代码中不能出现
`desired_muzzle_world_rotation(..., -pitch_f32)`，C++ 反馈也不能写成
`state.rpy.pitch = -rpy.y()`。

### F5 后猛转 90°/180°

检查四元数是否二次换轴、是否把世界绝对角当底盘相对角、是否把绝对命令累加为
增量，以及是否有旧视觉进程继续写共享内存。安全门日志中的
`actual/desired/delta` 可定位首次跳变。

### 读帧超时

确认仿真器已运行、`/tmp` 共享内存存在、没有其他 Talos 消费者；仿真器重启后建议
完整重启视觉。连续超时后 Talos 后端会尝试重新打开共享内存。

### 回放没有攻击框

攻击框只在有效物理叶片、PnP、跟踪、计划和未来命中时刻均成立时绘制。离线回放
不能验收真实云台方向，也不能证明实车开火条件成立。

## 9. 上实车前清单

- 实测相机内参、畸变和 `T_barrel_camera`；
- 确认云台 yaw/pitch 零点和正方向；
- 确认电控串口模式回传，不能继续依赖 Talos 配置注入模式；
- 验证实时弹速回传并标定回退值；
- 标定云台速度、控制周期、命令延迟和开火延迟；
- `shoot_enable=false` 下完成静态、动态、丢目标、模式切换测试；
- 确认无目标、无姿态、失联、超时都会清除 `fire`；
- 先空仓，再低能量，最后正式开火。

Talos 安全门只保护仿真 I/O，不能替代实车电控限位、模式和急停保护。

## 10. 代码位置和提交检查

| 功能 | 文件 |
|---|---|
| Talos 配置 | `src/l1_sensor/talos/talos_config.cpp` |
| 命令编码/解码 | `src/l1_sensor/talos/gimbal_codec.cpp` |
| 共享内存 | `src/l1_sensor/talos/talos_reader.cpp` |
| 仿真 Runtime I/O | `src/runtime/runtime_io.cpp` |
| 方向验收 | `tools/talos/talos_gimbal_check.cpp`、`tests/talos_contract_smoke.cpp` |
| 仿真执行 | `bevy_robomaster_simulator/src/talos/plugin.rs` |
| 仿真姿态发布 | `bevy_robomaster_simulator/src/talos/capture.rs` |
| 打符回放 | `tools/replay/buff_video_replay.cpp`、`docs/buff.md` |

提交前运行：

```bash
cd /home/rm/atooooooom/version
git diff --check
xmake run talos_contract_smoke
xmake run buff_math_smoke
xmake build talos_auto_aim talos_gimbal_check buff_video_replay

cd /home/rm/rm_summmer/bevy_robomaster_simulator
git diff --check
cargo fmt --check
cargo test talos::plugin::tests
cargo build --release
```

编译通过后仍必须完成一次第 6 节窗口方向验收；窗口物理验收不能由编译结果替代。
