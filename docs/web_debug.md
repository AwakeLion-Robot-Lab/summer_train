# FSVISION 远程 Web 调试台

> **一句话**：车端无头跑算法，笔记本浏览器看视频 / 曲线 / 健康，在线调参并回放。
>
> **旁路设计**：整体删除不影响瞄准链路。不进火控环。
>
> 代码：`modules/vision_tools/web_debug/` · `modules/vision_tools/src/web/`；`web.py` 和 `include/l6_telemetry/web_{sink,cmd}.hpp` 保留兼容入口。
>
> 设计吸收：WUST awakening（shm+MJPEG）、SJTU CVRM（在线调参）、Seeker2026（调参+空间视图）、JNU-SHARK（健康诊断/录制）。

---

## 目录

1. [解决什么问题](#1-解决什么问题)
2. [架构与数据契约](#2-架构与数据契约)
3. [快速开始（两个终端）](#3-快速开始两个终端)
4. [页面功能](#4-页面功能)
5. [在线调参](#5-在线调参)
6. [录制与回放](#6-录制与回放)
7. [配置项一览](#7-配置项一览)
8. [代码入口](#8-代码入口)
9. [设计约束](#9-设计约束)
10. [故障排查](#10-故障排查)
11. [与各开源方案对比](#11-与各开源方案对比)
12. [已知限制](#12-已知限制)
13. [验收清单](#13-验收清单)

---

## 1. 解决什么问题

| 痛点 | 原来 | 现在 |
|---|---|---|
| 比赛机无显示器，`cv::imshow` 直接抛 | overlay 默认关 | Web 远程看标注视频 |
| imshow 耗时计入 `image_to_plan` | 不敢开 overlay | 绘制与显示分离，写帧在延迟测量之后 |
| 曲线和图像不在一个视图 | PlotJuggler 只有曲线 | 同一页：视频 + 曲线 + 健康 + 三视图 |
| 改 YAML 要重启 | 调参周期长 | 滑条热更 ~0.2s，另可写回 YAML |
| 赛后难复盘 | 只有 track_diag CSV | series.jsonl + frames 回放页 |

**不是** ROS/Foxglove 替代品。PlotJuggler（UDP 9870）继续可用，两边消费**同一份** `telemetryFrame` 字段。

---

## 2. 架构与数据契约

### 2.1 总览

```
┌──────────────── auto_aim_runtime / buff_runtime ────────────────┐
│  drawAimOverlay / Buff::drawOverlay                             │
│  telemetryFrame / buffTelemetryFrame                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │  L6 旁路（可删）
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   WebSink 写          WebCmd 读          UdpJsonSender
        │                   ▲                   │
        ▼                   │                   ▼
 /dev/shm/newvision_*  newvision_cmd.json   UDP 9870
        │                   │                   │
        └────────►  web.py :9000  ◄──────────────┘（并存）
                        │
                        ▼
              浏览器 http://<车端IP>:9000
```

### 2.2 两个进程，缺一不可

| 进程 | 命令 | 职责 |
|---|---|---|
| 算法 | `xmake run talos_auto_aim` / `auto_aim` | 写 shm |
| Web | `.venv/bin/python web.py` | 读 shm，提供 HTTP |

C++ 日志里的 `web debug enabled http://0.0.0.0:9000` **只表示数据在写 shm**，不代表 HTTP 已监听。  
网页打不开九成是 **`web.py` 没起**。

### 2.3 共享内存契约

| 路径 | 格式 | 写 | 读 |
|---|---|---|---|
| `/dev/shm/newvision_frame` | `uint32 LE 长度 + JPEG`，预分配 2 MB | C++ | web.py |
| `/dev/shm/newvision_data.json` | `{time:[], "gimbal/yaw":[], ...}` | C++ | web.py |
| `/dev/shm/newvision_log.json` | `{key:…, events:[…]}` 诊断快照 | C++ | web.py |
| `/dev/shm/newvision_cmd.json` | 嵌套 JSON 命令树 | web.py | C++ |

普通程序日志不再复制到 shm。`aw_logger::WebsocketAppender` 使用 MessagePack
发送到 `web.py` 的 `:1234` 中继，再以 JSON 广播到浏览器；Console/File Appender
仍然并行保留。网页修改日志等级时，中继把 `SET_LEVEL` 原样转发给 logger。

- JSON 一律 **tmp 文件 + rename** 原子替换。
- 命名 `newvision_*`，与 Talos `talos_ipc_*` 不冲突。
- 帧头 4 字节是 JPEG 长度；内容以 `FF D8` 开头。

### 2.4 时序键名（与 PlotJuggler 同源）

`telemetryFrame()` / `buffTelemetryFrame()` 的嵌套 JSON 拍平成斜杠路径：

```
t                          # 曝光时刻，秒
gimbal/yaw, gimbal/pitch, gimbal/bullet_speed, gimbal/heat
serial/rx, serial/rx_dropped, serial/rx_skipped_bytes,
serial/pose_wait, serial/pose_ready, serial/tx, serial/tx_failed, …
track/state, track/n_obs, track/resets, aim/rejects
obs/x, obs/y, obs/z, obs/distance, obs/yaw, obs/yaw_raw, obs/reproj_err
ekf/x, ekf/vx, … ekf/dz2          # 13 维整车状态
ekf/nis, ekf/nees, ekf/res_yaw, …  # 一致性
aim/yaw, aim/pitch, aim/shoot_yaw, aim/status, aim/blending, …
delay/image_to_plan, delay/plan_to_send, delay/send_to_control,
delay/control_to_fire, delay/fire_to_hit, delay/before_fire
fire/feasible, fire/shoot, fire/sent, fire/yaw_err, fire/pitch_err, fire/reason
buff/x, buff/y, buff/z, buff/rot_speed, buff/rot_angle, buff/face_yaw,
buff/sine_valid, buff/sine_C, buff/sine_v, buff/sine_a, buff/sine_omega,
buff/plan_valid, buff/degraded, buff/blade, buff/fly_time, …
```

**单位**：角度 degree，延迟 ms，`t` 为秒。  
**分节点是硬要求**：`gimbal/` 与 `aim/`、`obs/` 与 `ekf/` 不得混层，否则跟随误差没法读。

---

## 3. 快速开始（两个终端）

### 3.1 依赖

```bash
# C++（启用 aw_logger → Web）
xmake f --web_logger=y -y
xmake -y

# Python（只需一次）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt    # flask + pyyaml
```

### 3.2 打开 Web 旁路

```yaml
# config/auto_aim.yaml
debug:
  web: true                 # 总开关，默认 false
  web_frame_every: 1        # 每 N 帧写一张 JPEG
  web_record_dir: ""        # 非空则录制
  web_record_max_frames: 3000
```

### 3.3 启动

```bash
# ── 终端 1：算法（无显示器也可以）──
xmake run talos_auto_aim -- config/talos.yaml   # 仿真
# 或
xmake run auto_aim                              # 实机
# 或
./build/linux/x86_64/release/auto_aim

# ── 终端 2：Web ──
cd ~/atooooooom/version
.venv/bin/python web.py
# → fsvision web debugger: http://10.201.107.226:9000
```

浏览器打开日志打印的地址。本机可 `http://127.0.0.1:9000`，局域网用 `http://<车端IP>:9000`。

查看本机 IP：

```bash
hostname -I
# 或
ip -4 route get 1.1.1.1 | awk '{print $7; exit}'
```

### 3.4 比赛前检查

- [ ] `debug.web: false`
- [ ] `debug.web_record_dir: ""`
- [ ] `debug.force_work_mode: ""`（清空强制模式）
- [ ] `fire.shoot_enable` 按验收要求
- [ ] `planning.trust_fallback_bullet_speed` 按电控弹速情况
- [ ] `/dev/shm/newvision_cmd.json` 已清空或不再应用旧热更

清空热更命令：

```bash
echo '{}' > /dev/shm/newvision_cmd.json
# 或页面「清空 cmd」
```

---

## 4. 页面功能

布局（大屏两列，小屏单列）：

```
┌─────────────┬──────────┐
│ 视频流       │ Log      │
├─────────────┼──────────┤
│ 健康         │ 三视图   │
├──────────────┴─────────┤
│ 调参                   │
├────────────────────────┤
│ 回放                   │
├────────────────────────┤
│ 时序（总图）            │
├────────────────────────┤
│ 独立图表                │
└────────────────────────┘
```

### 4.1 视频流

- MJPEG `/video`，来自 `newvision_frame`。
- 画面底部帧率仪表：`CAM` = 相机驱动实际采集，`ALG` = 主算法循环实际消费，
  `WEB` = 后台 JPEG 编码输出。`CAM` 明显高于 `ALG` 时，算法处理不过来；
  `WEB` 只反映调试旁路，不代表算法帧率。
- JPEG 缩放和压缩在单槽后台线程完成；来不及编码时覆盖旧待处理帧，不排队、
  不反压瞄准主循环。输出最大为 960×720、质量 75。
- 标注与 `debug.overlay` 相同：
  - 蓝框 = L2 网络角点
  - 绿框 + 绿箭头 = 送入滤波器的单板 PnP
  - 绿色整车 = EKF 展开的物理板
  - 红框 = Plan 选中的命中板
  - 红圆 = 瞄准点，黄箭头 = 控制速度
  - 顶部：时间戳 / 延迟 / FSM / 攻击目标
  - 底部：yaw/pitch 指令 p/v/a/enable
  - `Fire!` = 开火指令
- 全屏按钮；`web_frame_every` 控制提交给 WEB 的频率。

### 4.2 Log

“运行日志”通过 `aw_logger` WebSocket 实时显示，包含时间、等级、线程、源码位置
与正文，并可在网页动态设置日志等级。默认地址为 `ws://127.0.0.1:1234`；算法与
Web 服务不在同一机器时，用 `NEWVISION_LOG_WS_URL` 指定算法端连接地址。

“诊断快照”仍轮询 `/log`（500 ms）。`events[]` 为关键事件时间线（环形 50 条）：

| type | 含义 |
|---|---|
| `mode_change` | 下位机 WorkMode 变化 |
| `tracker_reset` | EKF 重建 |
| `fire_reject` | 开火拒绝原因变化 |
| `tuning_applied` | Web 热更生效键列表 |

### 4.3 健康

状态灯（绿 / 黄 / 红）：

| 组 | 字段 | 含义 |
|---|---|---|
| 串口 | `rx_dropped` / `rx_skipped_bytes` / `pose_wait` / `pose_ready` | 丢包、跳字节、姿态等待 |
| 跟踪 | `track/state` / `resets` / `n_obs` | 状态机、EKF 重建、观测数 |
| 延迟 | `image_to_plan` / `plan_to_send` / `before_fire` | 延迟链关键段 |
| 开火 | `shoot` / `reason` / `ekf/nis` / `aim/rejects` | 拒绝原因、一致性 |
| Buff | `rot_speed` / `plan_valid` / `n_icon` / `n_pred` / `sine_*` / `degraded` / `update_count` | 打符链路 |

阈值：`pose_wait > 20 ms` 红、`image_to_plan > 30 ms` 红、`nis > 13.8` 红（卡方 95% 约 7.8 黄）。

### 4.4 三视图

俯视 x-y / 侧视 x-z / 前视 y-z：

- 原点 = 枪口
- 蓝点 = `ekf`（滤波），绿点 = `obs`（PnP 观测）
- 下方文字：`gimbal yaw/pitch` + `obs/distance`

### 4.5 时序 / 独立图表

- 默认勾选：`aim/yaw` vs `gimbal/yaw`、`aim/pitch` vs `gimbal/pitch`、`delay/before_fire`、`fire/*_err`、`ekf/nis`、`buff/*`
- 跟随 / 延迟 / 开火 / 打符四组预设；字段可搜索，也可勾选任意拍平键
- 「窗口点数」裁剪横轴（10–1000）；关闭总图只隐藏画布，不会丢失重新开启的入口
- 曲线使用仓库内的原生 Canvas 绘制，不依赖公网 CDN，车端断网仍可用
- **深度调参（NIS 分解、延迟五段、blend）仍建议 PlotJuggler**，数据同源

---

## 5. 在线调参

### 5.1 两条通道

| 动作 | 通道 | 落盘？ | 生效 |
|---|---|---|---|
| **热更到运行时** | POST `/params` → `newvision_cmd.json` → runtime 每 10 帧 poll | 否 | ~0.2 s |
| **写入 YAML** | POST `/params/save`，按行外科替换 | 是 | 重启后 |
| 清空 cmd | POST `/params/clear` | 否 | 停止继续应用 |
| 强制 WorkMode | 下拉 + **确认框** | 否 | 下一帧 |

> 热更**不**自动回退：清空 cmd 只是不再下发新值，已应用的保持到重启。

### 5.2 白名单

`fire.shoot_enable` **故意不支持**在线改，只能改 YAML。

| Web 键 | YAML 键 | 单位 | 说明 |
|---|---|---|---|
| `plan.send_to_control_ms` | `send_to_control_ms` | ms | 延迟链标定 |
| `plan.yaw_offset_deg` | `yaw_offset_deg` | deg | |
| `plan.pitch_offset_deg` | `pitch_offset_deg` | deg | |
| `plan.fallback_bullet_speed` | `fallback_bullet_speed_mps` | m/s | |
| `target.q_translation` | `q_translation` | — | EKF 过程噪声 |
| `target.q_rotation` | `q_rotation` | — | |
| `target.angle_variance` | `angle_variance` | — | 观测噪声 |
| `target.distance_variance_factor` | `distance_variance_factor` | — | |
| `target.armor_yaw_variance_base` | `armor_yaw_variance_base` | — | |
| `fire.hit_margin_ratio` | `hit_margin_ratio` | — | (0,1] |
| `fire.min_yaw_tolerance_deg` | `min_yaw_tolerance_deg` | deg | |
| `fire.min_pitch_tolerance_deg` | `min_pitch_tolerance_deg` | deg | |
| `debug.force_work_mode` | `force_work_mode` | str | 危险，UI 确认 |
| `debug.overlay` | `overlay` | bool | |
| `debug.web_frame_every` | `web_frame_every` | int | |
| `camera.exposure_ms` | （热更 only） | ms | 仅 Hik |
| `camera.gain` | （热更 only） | dB | 仅 Hik |

### 5.3 热更写入哪些活对象

| 对象 | 接口 | 生效 |
|---|---|---|
| 整车 EKF | `Tracker::setTargetConfig` → `TrackedTarget::setConfig` | 下一帧 |
| 规划 | `Planner::setImpactConfig` | 下一帧 |
| 火控容差 | `Controller::setFireConfig` | 下一帧 |
| 相机 | `RuntimeIo::setExposureGain` | Hik 立即；其他驱动 WARN |

**需重启**：`ballistic.drag_coefficient`（阻力模型在构造时定死）、推理后端/模型。

### 5.4 日志

- `logWarn("web tuning applied", "a,b,c")`
- `events[]` 一条 `tuning_applied`

---

## 6. 录制与回放

### 6.1 录制

```yaml
debug:
  web: true
  web_record_dir: "records/run_01"
  web_record_max_frames: 3000   # ≈ 2 分钟 @ web_frame_every=1
```

目录结构：

```
records/<session>/
  series.jsonl           # 每行一个嵌套 JSON 样本（telemetryFrame + t）
  frames/index.jsonl     # 每张 JPEG 的源帧号和曝光时刻
  frames/frame_000000.jpg
  frames/frame_000001.jpg
  …
```

- 帧达到 `web_record_max_frames` 后**停帧不停时序**；`<=0` 不限。
- 与 `tests/track_diag` CSV、`tools/replay/buff_video_replay` 并列，不互相替代。

### 6.2 回放

1. 跑完一轮后打开「回放」面板（或点「刷新列表」）
2. 点会话 → 加载帧 + 曲线
3. 拖 scrub / 播放（按 `index.jsonl` 的真实帧间隔；旧录像按 50 ms/帧）

HTTP：

| 路由 | 说明 |
|---|---|
| `GET /replays` | 会话列表 |
| `GET /replays/<name>/series` | 拍平时序（与 `/data` 同构） |
| `GET /replays/<name>/frames/<i>` | 第 i 张 JPEG |

根目录可用 `NEWVISION_RECORD_ROOT` 改（默认 `records`）。

---

## 7. 配置项一览

### 7.1 `config/auto_aim.yaml` → `debug:`

| 键 | 默认 | 说明 |
|---|---|---|
| `overlay` | false | OpenCV 窗口（需显示器，会计入 image_to_plan） |
| `overlay_every` | 1 | 每 N 帧绘制 |
| `plot` | true | PlotJuggler UDP |
| `plot_host` | 127.0.0.1 | |
| `plot_port` | 9870 | |
| `web` | false | Web 旁路总开关 |
| `web_frame_every` | 1 | 写 JPEG 频率 |
| `web_record_dir` | "" | 录制目录，空 = 不录 |
| `web_record_max_frames` | 3000 | 帧上限，≤0 不限 |
| `force_work_mode` | "" | 强制模式，**比赛必须清空** |

### 7.2 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `NEWVISION_YAML` | `config/auto_aim.yaml` | 「写入 YAML」目标路径 |
| `NEWVISION_RECORD_ROOT` | `records` | 回放扫描根 |
| `VISION_TOOLS_IPC_PREFIX` | `/dev/shm/newvision` | 独立模块 Web 的 IPC 文件前缀，需与 C++ 一致 |

### 7.3 端口

| 端口 | 进程 | 用途 |
|---|---|---|
| 9000 | web.py | HTTP 调试台 |
| 9870 | （接收）PlotJuggler | UDP JSON 曲线 |

---

## 8. 代码入口

### 8.1 C++（L6 旁路）

| 路径 | 职责 |
|---|---|
| `include/l6_telemetry/web_sink.hpp` | 帧 / 时序 / 日志 / 事件 / 录制 |
| `src/l6_telemetry/web_sink.cpp` | 实现 |
| `include/l6_telemetry/web_cmd.hpp` | 调参命令读取 |
| `src/l6_telemetry/web_cmd.cpp` | 实现 |
| `runtime::applyWebTuning()` in `auto_aim_config.cpp` | 白名单热更 |
| `src/runtime/auto_aim_runtime.cpp` debug 段 | 挂钩：绘制、写帧、写曲线、事件、热更 |
| `src/runtime/buff/buff_runtime.cpp` | `Result.state` 供 buff 遥测 |

### 8.2 热更 setter

| 类 | 方法 |
|---|---|
| `L3Estimation::Tracker` | `setTargetConfig` |
| `L3Estimation::TrackedTarget` | `setConfig` |
| `L4Planning::Planner` | `setImpactConfig` |
| `L5Control::Controller` / `FireDecider` | `setFireConfig` / `setConfig` |
| `L1Sensor::Camera` / `RuntimeIo` / `HikRobot` | `setExposureGain` |

### 8.3 Web

| 路径 | 职责 |
|---|---|
| `web.py` | 兼容启动入口，调用模块中的 HTTP 服务 |
| `modules/vision_tools/web_debug/app.py` | HTTP 路由 |
| `modules/vision_tools/web_debug/templates/index.html` | 页面骨架 |
| `modules/vision_tools/web_debug/static/js/` | 视频、曲线、健康、调参及回放 |
| `modules/vision_tools/src/web/` | 共享内存帧、时序、日志和命令通道 |

### 8.4 测试

```bash
xmake run web_sink_smoke      # 帧/时序/日志/事件/录制
xmake run web_cmd_smoke       # 命令路径查找
xmake run web_tuning_smoke    # 白名单、单位转换与重复命令幂等性
xmake run auto_aim_config_smoke  # 含白名单校验
.venv/bin/python -m unittest discover -s tests -p test_web_app.py -v  # Flask 路由/校验/回放抽样
```

---

## 9. 设计约束

1. **旁路可删**：`debug.web: false` 时 WebSink/WebCmd 不构造，主流程零差异。
2. **不进延迟测量**：写帧、热更排在 `measured_plan_to_send` 之后。
3. **绘制共用**：runtime 与离线工具同一 `drawAimOverlay` / `Buff::drawOverlay`。
4. **Schema 同源**：Web 时序 = PlotJuggler JSON 拍平，不另起字段。
5. **安全闸门**：`fire.shoot_enable` 不进 Web 白名单；`force_work_mode` UI 需确认。
6. **YAML 写回保注释**：行级替换，禁止 `yaml.safe_dump` 整树重写。
7. **画与显分离**：无头机器只画不 `imshow`。

---

## 10. 故障排查

| 现象 | 原因 / 排查 |
|---|---|
| **网页打不开** | ① `web.py` 是否在跑（终端 2）② `ss -ltnp \| grep 9000` ③ 防火墙 ④ 是否用对 IP（`hostname -I`） |
| C++ 打印 web enabled 但无网页 | 正常：那只代表 shm 在写；**必须另起 web.py** |
| DATA / LOG 红灯 | 算法是否在跑、`debug.web: true`、`/dev/shm/newvision_*` 是否存在 |
| 视频黑屏 | `xxd /dev/shm/newvision_frame \| head` 看是否 `ff d8`；`web_frame_every` 是否过大 |
| 曲线不动 | `/data` 是否有 `time`；是否 Lost 无观测 |
| 热更无效 | 日志 `web tuning applied`；键是否白名单；Hik 曝光是否驱动支持 |
| 日志里莫名 `web tuning applied` | 残留 `/dev/shm/newvision_cmd.json`，`echo '{}' > …` 清掉 |
| 写入 YAML 无效果 | 键是否被注释（会尝试取消注释）；对照表确认 `yaml_key` |
| 回放列表空 | `records/` 是否有会话；`web_record_dir` 是否开过 |
| 带宽不够 / 卡 | 调大 `web_frame_every`；或后续换硬件编码/RTP |
| 端口占用 | 杀掉旧 web.py：`pkill -f 'python.*web.py'` |

---

## 11. 与各开源方案对比

| 来源 | 采用 | 未采用 |
|---|---|---|
| **awakening** | shm、MJPEG、JSON 树 | 其前端风格与公网图表依赖 |
| **SJTU CVRM** | 在线滑条、运行时/持久化分离 | 同进程 ObjManager |
| **Seeker2026** | 调参、空间视图、曲线勾选 | roslib/ROS、真 3D Three.js |
| **JNU-SHARK** | 健康面板、事件、录制回放 | 进程管理、雷达频谱 |
| **Foxglove** | 空间 + 时序双视图理念 | 不引入 ROS |
| **仲恺奇点** | — | 硬件编码/RTP（带宽不够时再做） |
| **本仓库** | telemetryFrame、PlotJuggler、overlay、AimTrace | — |

相对只抄 awakening 的增量：调参反向通道、健康/延迟链诊断、三视图、Buff 补线、录制回放、PlotJuggler 并存。

---

## 12. 已知限制

- Flask 开发服务器；长时间运行建议 gunicorn/uvicorn（接口不变）
- 尚无浏览器端 E2E；JS 有语法检查，Flask 路由与数据契约有 unittest
- MindVision / Talos 曝光热更未实现（返回 false）
- `ballistic.drag_coefficient` 改后需重启
- 回放帧和曲线都有时间戳；前端尚未把曲线光标自动定位到当前帧
- 单路视频流
- 热更不自动回退（清空 cmd 后需重载 YAML 或重启）

---

## 13. 验收清单

```bash
# 构建与冒烟
xmake -y
xmake run web_sink_smoke
xmake run web_cmd_smoke
xmake run web_tuning_smoke
.venv/bin/python -m unittest tests.test_web_app -v
xmake run auto_aim_config_smoke

# 无显示器联调
#   config: debug.web: true
xmake run talos_auto_aim -- config/talos.yaml   # 终端 1
.venv/bin/python web.py                          # 终端 2
# 浏览器 http://<ip>:9000
#   [ ] 视频有标注
#   [ ] 健康有数值
#   [ ] 三视图有点
#   [ ] 时序在动
#   [ ] 调参热更后曲线/健康变化
#   [ ] web_record_dir 开启后回放可见帧+曲线

# 旁路可删
#   debug.web: false 后重启，行为与改前一致
```

---

*文档对应代码状态：Phase 1–3 全部合入（视频/曲线/健康/三视图/Buff/录制/调参/回放）。*
