# 录像、标定与回放工具

通用实现在 [`modules/vision_tools/README.md`](../modules/vision_tools/README.md)，可以独立复制构建。此处只记录本项目的相机、串口和算法适配入口。

| 工具 | 构建 | 运行示例 |
|---|---|---|
| 原始视频及姿态采集 | `xmake build record_capture` | `xmake run record_capture --preview=false --fps=90 --output-dir=records` |
| 棋盘格采图 | `xmake build camera_capture` | `xmake run camera_capture --fps=1 --output-dir=calibration_images` |
| 相机内参标定 | `xmake build camera_calibrator` | `xmake run camera_calibrator calibration_images/<会话目录> --cols=10 --rows=7 --square-size=15` |
| 打符视频回放 | `xmake build buff_video_replay` | `xmake run buff_video_replay <视频.avi> --playback-mode=realtime --timing-source=auto` |
| Web 调试与回放 | 无需 C++ 单独构建 | `.venv/bin/python web.py` |

`record_capture` 只接收串口姿态，不发送云台控制帧；对不齐姿态的相机帧会跳过。它使用本项目的 `config/camera_config.yaml`、`config/serial_config.yaml`，录像是同名 AVI/TXT。也可以在 `config/auto_aim.yaml` 的顶层 `recorder` 节启用主程序原图录像；默认关闭。主程序录制时在叠加绘制前复制原图，处理完当前帧后写入，避免录像里混进调试标记。

标定默认使用 10×7 **内角点**，对应 11×8 方格；单格边长按实际标定板填写。`camera_calibrator` 输出 `calibration.yaml`、质量报告和诊断图。将结果文件的 `calibration` 节合入 `config/camera_config.yaml` 的同名节，再检查图像尺寸及内参；现有枪管外参字段不要用内参标定结果覆盖。手眼和云台外参计算不在本次迁移范围。

`buff_video_replay` 的 `--playback-mode=realtime` 按实际时间轴播放，处理跟不上时跳帧；`--playback-mode=all` 保留逐帧处理。`--timing-source=auto` 优先读取 AVI 同名 TXT 的实际采集时间。若只有 AVI，无法从容器的固定 FPS 还原采集掉帧造成的时间空洞。Web 调试回放使用 `frames/index.jsonl` 中的逐帧采集时刻，旧的无索引录制仍按固定间隔播放。
