# 项目专用工具

构建目标名称保持不变，在仓库根目录运行 `xmake build <目标>`、`xmake run <目标>`。

| 目录 | 内容 | 构建目标 |
|---|---|---|
| `capture/` | 当前相机与串口的录像、棋盘格采图适配 | `record_capture`、`camera_capture` |
| `replay/` | 打符算法视频回放适配 | `buff_video_replay` |
| `diagnostics/` | 串口 RTT、云台阶跃与延迟采集示例 | `serial_rtt`、`gimbal_step_csv` |
| `talos/` | Talos 仿真入口与云台方向检查 | `talos_auto_aim`、`talos_gimbal_check` |

内参标定命令 `camera_calibrator` 直接使用 `modules/vision_tools/examples/calibrate.cpp`，通用库与 Web 前端均在 [`modules/vision_tools/`](../modules/vision_tools/README.md)。`camera_sdk/`、`serial/`、`logger/`、`config_set/`、`LatestBuffer/` 是现有驱动和依赖源码，仍由主工程构建脚本引用。
