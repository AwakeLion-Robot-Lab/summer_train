# 打符（Buff）链路

打符跟踪、规划沿用 `/home/rm/rmcs_auto_aim_v2` 最后提交版本（`4f4340956a989b0d3ecd4fbd94b1e72a301293cd`）的 Rune 算法，独立放在各层的 `buff/` 子目录中。检测支持模型与传统两种路径，不复用装甲板的检测、跟踪或开火状态。

## 数据流

`L1 相机/姿态 → L2 靶面检测 + 中心 R/十字端点 → L3 PnP + EKF + 能量拟合 → L4 弹道与飞行时间迭代 → L5 安全开火命令`。

核心常量保持源实现：符面半径 `0.7 m`、图标偏移 `0.1 m`、PnP EPNP 后 ITERATIVE、能量拟合窗口 `6 s`、线性/正弦模型选择规则、弹道步长 `5 ms`、最大迭代 `5` 次，以及 `0.4 s/0.2 s` 空闲/射击调度。

## 启用方式

`config/buff.yaml` 的 `detector.method: model` 启用 `model/buff_model/buff.onnx`，
`traditional` 恢复旧版完整颜色/轮廓检测；缺少 method 的旧配置仍使用传统路径。
模型使用 OpenVINO，需要构建配置 `--use_openvino=y`。目前默认 CPU；配置 GPU
加载失败会告警并回退 CPU，CPU 也失败则报告 Buff runtime unavailable，不静默改用传统检测。
修改检测配置需要重启程序。目标颜色跟随下位机，或由 `auto_aim.yaml` 中的
`debug.force_enemy_color` 指定；离线回放使用 `--enemy=red/blue`。

模型输入为 RGB、除以 255、居中补 114 至 640×640。输出 `[1,16,8400]`：
`xywh + BR/BW/RR/RW 四类置信度 + 四个二维关键点`，没有独立 objectness 或关键点置信度。
先按四类最大分数和目标颜色筛选，再执行 NMS 并还原原图坐标。
BR/RR 是待打靶，BW/RW 是已激活靶；模型元数据中的类别顺序为准。

**几何适配：**该模型不检测中心 R，待打靶网络点位于内圈，不能直接当作
现有 PnP 的 0.15 m 十字外端点。因此模型选靶面 ROI 和激活状态后，仍用传统
方法在 ROI 内提取十字端点，中心 R 也用传统检测。网络框可见但端点提取失败时，
只显示网络框，不把不匹配的几何送入跟踪。`active_threshold`、`match_threshold`
和中心 R 的距离筛选仍然生效；网络阈值在 `detector.inference` 下单独设置。
叠加层青色框/点为网络输出，红/绿几何点为实际送入跟踪的待打/已激活靶。

模型来源核对：本仓库 ONNX 与 FosuVision 的同名模型 SHA-256 均为
`52acb9f484d4ac4839958f02187d4912f63839848d3bb09074d1c3df2eb16b85`。
其原检测代码使用 BGR 并交换了类别顺序；这里通过红色录像及交换红蓝通道验证，
采用 RGB 与模型自带的 BR/BW/RR/RW 顺序。原始四点仅用于中心定位和显示。

运行时读取 [`config/buff.yaml`](../config/buff.yaml)。当前 `control.shoot_enable: false`，只调试跟随。相机必须同时提供内参与 `T_barrel_camera`；缺少枪管外参时链路保持安全，不使用单位阵猜测。

验证命令：

```bash
xmake build auto_aim
xmake build buff_decoder_smoke
xmake run buff_decoder_smoke
xmake build openvino_buff_smoke
xmake run openvino_buff_smoke
# 可选：提供含红色待打靶和中心 R 的原始图，额外验证几何提取和红蓝切换。
xmake run openvino_buff_smoke /path/to/red_buff.png
```

## 安全边界

- 未知敌方颜色、无姿态、无有效观测或观测超过 100 ms 时只保持角度并清除 `shoot`。
- 弹道优先使用电控随 `RobotState` 上报的实时弹速；低于
  `min_valid_bullet_speed_mps` 时使用配置回退值继续跟随，但默认禁止开火。
  只有显式打开 `trust_fallback_bullet_speed` 才允许按回退弹速进入开火节拍。
- 目标模型 1.5 s 无有效修正或协方差/几何发散时销毁并重新初始化。
- 收敛预热期间以及退化瞄准中心时不自动开火。

调试叠加和 CSV 样本位于 `include/l6_telemetry/buff/`，只读结果，不改变控制链路。

## 离线视频回放

`buff_video_replay` 直接读取视频并运行 Buff 的 L2～L5 链路，不初始化相机和串口，
且无条件关闭控制层的实际开火输出。画面中的 `SHOOT READY` 只表示算法判定，
不会向任何设备发送命令。

L4 选中命中时刻的未来叶片后，会额外输出只读的 `attack_box` 调试几何。回放中
橙框表示仍在预瞄，绿框表示算法已进入可开火节拍；没有选中物理叶片、退化瞄准
符中心或计划无效时不画框。该几何不参与选叶、弹道和开火判定。

```bash
xmake build buff_video_replay
xmake run buff_video_replay \
  --calibration=/path/to/camera_info.yaml \
  --enemy=red \
  --playback-mode=realtime \
  /path/to/buffVideo.avi
```

支持本工程 `camera_config.yaml` 的 `calibration:` 格式，也支持 ROS
`camera_info.yaml` 的 `{rows, cols, data}` 格式。视频分辨率必须与标定一致。

常用选项：

- `--wait=0`：从首帧开始单步检查；
- `--start-index=1000 --end-index=2000`：只回放指定区间；
- `--output=/tmp/buff_replay.avi`：导出带标注视频；
- `--headless=true`：不创建窗口，适合批量统计或服务器导出。
- `--show-original=true`：同步显示未加标注的原始画面（默认关闭）；
- `--playback-mode=all`：逐帧处理，不丢帧；
- `--playback-mode=realtime`：按录像时间播放，处理跟不上时跳过落后帧（默认）；
- `--timing-source=auto`：优先使用录制工具生成的同名 `.txt` 逐帧时间戳，
  没有时才使用视频文件的 FPS（默认）；`pose` 强制读取 `.txt`，`video` 强制使用视频 FPS；
- `--sync-playback=true`：在 `all` 模式中，处理足够快时按录像 FPS 对时（默认开启）。

交互按键：空格暂停/继续，`N` 单步，`A/D` 前后跳 1 秒，`J/L` 前后跳
5 秒，`R` 回到起始帧，`Q` 或 `Esc` 退出。由于录像没有云台姿态，工具使用
固定相机位姿；它适合验收检测、跟踪连续性和规划时序，不替代实车外参验收。
离线回放没有电控状态，因此显式用 `fallback_bullet_speed_mps` 模拟一条有效实测
弹速；该工具仍会强制关闭控制输出中的 `shoot`。
原始窗口和叠加窗口始终使用同一解码帧。`all` 模式下若算法耗时超过单帧周期，
实际播放速度会低于录像标称速度。`realtime` 模式会跳过来不及处理的帧，
画面时间轴跟随源视频，但算法检测和跟踪只更新实际处理的帧。导出视频时，
跳帧位置沿用上一张标注画面以保持视频时长；编码本身也可能拖慢播放。
结束统计中的 `wall_fps` 是实际处理帧数除以总运行时间（包括暂停），
`skipped` 是跳过的源帧数，`avg_detect_ms`、`avg_decode_ms`、`avg_overlay_ms`
可用于定位处理瓶颈。纯 `--headless=true` 且不导出时会跳过无用的画面绘制，
此时 `avg_overlay_ms=0`。

`record_capture` 的 `--fps` 是最大录制帧率，AVI 也以这个固定 FPS 保存；如果采集、
姿态等待或丢帧使实际采集间隔变长，只按 AVI FPS 播放就会加速。将同名 `.avi` 和
`.txt` 放在同一目录，回放会显示 `timing=pose` 和由时间戳计算的 `capture_fps`，
并按每帧真实采集时间播放、跳转和更新跟踪。缺少 `.txt` 时显示 `timing=video`。
导出视频仍是固定帧率，使用本次采集的平均 FPS，不能完全保留不均匀的帧间隔。
