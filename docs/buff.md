# 打符（Buff）链路

打符沿用 `/home/rm/rmcs_auto_aim_v2` 最后提交版本（`4f4340956a989b0d3ecd4fbd94b1e72a301293cd`）的传统 Rune 算法，独立放在各层的 `buff/` 子目录中，不复用装甲板的检测、跟踪或开火状态。

## 数据流

`L1 相机/姿态 → L2 颜色差分与轮廓 → L3 PnP + EKF + 能量拟合 → L4 弹道与飞行时间迭代 → L5 安全开火命令`。

核心常量保持源实现：符面半径 `0.7 m`、图标偏移 `0.1 m`、PnP EPNP 后 ITERATIVE、能量拟合窗口 `6 s`、线性/正弦模型选择规则、弹道步长 `5 ms`、最大迭代 `5` 次，以及 `0.4 s/0.2 s` 空闲/射击调度。

## 启用方式

运行时读取 [`config/buff.yaml`](../config/buff.yaml)。`control.shoot_enable` 默认保持源 Rune 的自动开火语义；调试跟随时可显式关闭。相机必须同时提供内参与 `T_barrel_camera`；缺少枪管外参时链路保持安全，不使用单位阵猜测。

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
  /path/to/buffVideo.avi
```

支持本工程 `camera_config.yaml` 的 `calibration:` 格式，也支持 ROS
`camera_info.yaml` 的 `{rows, cols, data}` 格式。视频分辨率必须与标定一致。

常用选项：

- `--wait=0`：从首帧开始单步检查；
- `--start-index=1000 --end-index=2000`：只回放指定区间；
- `--output=/tmp/buff_replay.avi`：导出带标注视频；
- `--headless=true`：不创建窗口，适合批量统计或服务器导出。
- `--show-original=true`：同步显示未加标注的原始画面（默认开启）；
- `--sync-playback=true`：检测速度足够时按录像 FPS 对时（默认开启）。

交互按键：空格暂停/继续，`N` 单步，`A/D` 前后跳 1 秒，`J/L` 前后跳
5 秒，`R` 回到起始帧，`Q` 或 `Esc` 退出。由于录像没有云台姿态，工具使用
固定相机位姿；它适合验收检测、跟踪连续性和规划时序，不替代实车外参验收。
离线回放没有电控状态，因此显式用 `fallback_bullet_speed_mps` 模拟一条有效实测
弹速；该工具仍会强制关闭控制输出中的 `shoot`。
原始窗口和叠加窗口始终使用同一解码帧；若算法耗时超过单帧周期，工具不会丢帧
追赶，因此两窗仍同步，但实际播放速度会低于录像标称速度。
