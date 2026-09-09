# Daedalus 五次多项式测试与 PlotJuggler 遥测

测试入口是 `daedalus_quintic_probe`，复用本目录 `plan/quintic-blend` 的
`AimSmoother` 和 `Planner`。实时模式直接使用此分支已有的
`ArmorDetector → Tracker（PnP + 整车 EKF）→ Planner`。
本次没有移植另一分支的 UVL/IESKF。

通过与 runtime 相同的 `L6Telemetry::UdpJsonSender`，每次规划后向
`127.0.0.1:9870` 发送一包 JSON。默认另外保存 CSV，便于事后复查。
此测试验证参考轨迹与指令序列，不评估受限电机的实际响应。

## 构建和启动

```bash
cd /home/rm/nv-merge
xmake f -m release --use_openvino=y
xmake build daedalus_quintic_probe
```

先启动模拟器（需要其 Rust/Bevy 构建依赖）：

```bash
cd /home/rm/bevy_robomaster_simulator-master
cargo run --release
```

然后运行测试：

```bash
cd /home/rm/nv-merge
xmake run daedalus_quintic_probe --show
```

默认仅取图、识别、规划和发布曲线。移动/旋转模拟器内目标，可观察切板变化。
模拟器需要处于 **Robot 第一人称视角**，F3 循环切换视角。IPC 没有提供自由
相机的朝向，因此不能用第三人称或自由视角做这条图像自瞄测试。
程序逐帧更新相机外参，避免把启动远景的相机位置永久用于后续帧；
相机离枪口超过 1 m 时暂停规划，终端显示 `Waiting for Robot camera view`。
这个距离检查只能排除明显的远景，不能识别所有靠近枪口的自由视角。
`--follow` 会把规划角度发送给模拟器；在模拟器中按 F5 启用 AutoAim 后生效。
本程序所有命令的 `fire_advice` 固定为 false。
`--enemy=red|blue|any` 控制检测目标颜色，默认 any。
按 Ctrl+C 或图像窗口的 q/Esc 退出。

模拟器 Talos 图像通道是 SPSC，只能同时运行一个消费图像的客户端。
当前移植的客户端匹配 Talos IPC v2，会验证版本、图像大小和同步帧号。

## 直接接收 PlotJuggler 曲线

1. 在 PlotJuggler 选择 **UDP Server**，端口 **9870**，解析格式 **JSON**。
2. 启用消息字段作为时间戳，字段名填 **t**，单位是秒。
3. 启动接收，把下面的字段拖入图表。无需额外插件或布局文件。

| 图表 | 建议同时绘制的字段 |
|---|---|
| yaw 角度 | `/yaw/raw_rad`、`/yaw/planned_rad`、`/yaw/measured_rad` |
| yaw 速度 | `/yaw/raw_velocity_rad_s`、`/yaw/planned_velocity_rad_s`、`/yaw/max_speed_rad_s`、`/yaw/min_speed_rad_s` |
| yaw 加速度 | `/yaw/raw_acceleration_rad_s2`、`/yaw/planned_acceleration_rad_s2`、`/yaw/max_acc_rad_s2`、`/yaw/min_acc_rad_s2` |
| 切板状态 | `/armor_id`、`/next_armor_id`、`/blend/active`、`/blend/progress`、`/blend/late` |
| 整段峰值 | `/yaw/segment_peak_speed_rad_s`、`/yaw/segment_peak_acc_rad_s2` |
| 求解结果 | `/blend/search_attempted`、`/blend/search_valid`、`/blend/search_acc_feasible` |
| 规划代价 | `/timing/plan_us`、`/blend/duration_ms`、`/coverage/blend_fraction` |

pitch 对应字段将路径中的 `yaw` 换成 `pitch`。
角度为 rad，速度为 rad/s，加速度为 rad/s²；不要和度混画。
需要换端口时使用 `--port=9871`，PlotJuggler 同时修改。
runtime 和测试入口使用同一发送方式，采集时选择一个发送源，避免同端口混流。

## 先用确定性输入检查数值

不启动模拟器、不加载模型，也可以直接发布曲线：

```bash
cd /home/rm/nv-merge
xmake run daedalus_quintic_probe --synthetic --realtime --duration=30
```

此时 `/source/synthetic=1`，输入为 4 m、半径 0.26 m、6 rad/s 的合成四板车，
直接进入同一个 Planner。实测云台角为空，不会伪造实际跟随曲线。
可通过 `--omega=-10 --distance=3 --radius=0.26` 改变旋转方向和构型。
`--hz=200` 是合成采样率；实时图像模式按到达的图像帧规划。

不可行诊断用例：

```bash
xmake run daedalus_quintic_probe --synthetic --realtime --duration=10 --yaw-acc=2 --pitch-acc=2
```

不平滑的对照：

```bash
xmake run daedalus_quintic_probe --synthetic --realtime --duration=10 --no-blend
```

每次退出打印规划时间 p50/p95/p99 和 UDP 发送失败数。
规划耗时包含本测试启用的诊断采样，不包含推理、JSON 序列化或画图；
`/timing/perception_us` 单独记录检测和跟踪耗时。

## 怎样判读连续性与超限

- `/boundary/start/*`、`/boundary/end/*`：拟合时六个边界条件的残差，
  仅 `/boundary/valid=1` 时有效，应接近浮点数值误差。
- `/join/end_event=1`：刚完成一段过渡。
  `/join/end_valid=1` 时，`/join/end/*` 比较**同一终点时刻**的旧多项式
  与最新目标快照的射击轨迹。它可能因预测修正、选板变化而不连续，
  即使多项式自身的边界残差接近零。
- `/replan/*`：上一帧参考模型外推至当前时刻后，与当前输出比较。
  这是跨帧预测/参考改变的指标，不是直接把两个时刻的角度相减。
- `/raw_sampler_error/*`：固定飞行时间的边界采样与本帧迭代弹道解之间的角度差。
  边界采样沿用现有 Planner 的固定飞行时间近似，不等于再次完整迭代求解。
- `*_velocity_rad_s` / `*_acceleration_rad_s2`：过渡段为多项式直接求导；
  跟随段为同一目标快照、同一物理板的中心差分。
- `*_fd_velocity_rad_s` / `*_fd_acceleration_rad_s2`：真正发出的角度序列
  按实际、不等间隔时间差分，可以看到跨帧接缝和原始切板的尖峰。
  它们是离散诊断量，不是切板阶跃处存在的连续导数。

加速度整段峰值来自端点和 jerk 的内部零点；速度整段峰值来自端点和加速度
的内部零点。后者先用 jerk 零点隔离三次函数的单调区间，再求根到浮点精度，
不靠固定网格估计峰值。

`search_valid=1` 只代表数值解存在，`search_acc_feasible=1` 才代表该解满足
当前加速度限制。原有算法在找不到满足加速度限制的解时仍可能提交超限段；
本次保留其行为，用 `/blend/segment_acc_feasible=0` 和
`/counts/acc_infeasible_commits` 明确暴露。
时长二分依赖峰值随时长下降的近似，失败表示**现有搜索没有找到可行解**，
不构成整个时长区间均不可行的数学证明。

速度阈值 `--yaw-speed=10 --pitch-speed=10` 仅用于检查，默认值是测试参考值。
现有求解器只约束加速度，`/blend/speed_limit_enforced=0` 表明速度上限没有参与求解。
同样，加速度连续不意味着 jerk 连续。

未定义/丢目标字段通过 UDP 发为 null，CSV 留空；重新跟踪会断开差分历史。
如果没有 `/yaw/raw_rad` 或 `/yaw/planned_rad`，先看 `/raw_valid`、`/valid`：
它们为 0 时没有有效的原始/规划角度，PlotJuggler 不会凭 null 创建数值曲线。
同时检查 `/source/camera_ready`（1 表示相机通过距离检查）、
`/source/tracker_resets` 和 `/reason`（1=没有目标，4=弹道失败，5=选板窗口外）。
`/camera/*_barrel_m` 是当前帧使用的相机平移，单位 m。
`/coverage/blend_fraction` 是实际观察到的有效规划时段中处于过渡段的时间比例，
按时间积分，不按帧数粗算。

## 仿真参数与数据保存

测试入口读取 `config/auto_aim.yaml` 的模型、跟踪和切板参数，并在进程内设置：
开启 blend（除非 `--no-blend`）、发射前固定延迟为 0、重力 9.81、阻力 0。
这些与当前模拟器默认的真空弹道对应，不改现场配置文件。
`--bullet-speed=25` 要与模拟器 `projectile.speed` 一致；
启用了模拟器空气阻力时，需要另外对齐弹道模型。
真实图像的曝光到规划延迟仍按时间戳计算。

CSV 默认在 `logs/quintic_时间戳.csv`，也可指定 `--csv=/tmp/quintic.csv`。
CSV 时间列为 `t`，其他列名与 PlotJuggler UDP 字段路径一致。
`--no-udp` 可只记录文件，`--save=/tmp/quintic-overlay.png` 可保存最后一帧叠加图。

## 验证

```bash
xmake run aim_smoother_smoke
xmake run planner_smoke
xmake run quintic_trace_smoke
xmake run daedalus_client_smoke
python3 tests/quintic_udp_smoke.py
```

新增测试覆盖速度峰值与稠密采样交叉检查、跨帧同时间比较、终点预测漂移、
不可行状态、丢目标与差分重置，以及诊断开关不改变现有规划指令。
UDP 测试启动真实测试程序，检查 JSON、时间戳与 CSV 数据一致。
