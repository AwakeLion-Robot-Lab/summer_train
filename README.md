# NewVision

主程序和算法代码在 `src/`、`include/`；运行配置在 `config/`，模型在 `model/`。项目专用命令行工具按用途放在 [`tools/`](tools/README.md)，可复制的录像、标定、回放与 Web 能力放在 [`modules/vision_tools/`](modules/vision_tools/README.md)，设计与操作说明放在 `docs/`。

Buff 学习手册和界面预览分别见 [`docs/bufflearn.md`](docs/bufflearn.md)、[`docs/ui-preview.html`](docs/ui-preview.html)。

常用入口：

| 任务 | 命令 |
|---|---|
| 构建主程序 | `xmake build auto_aim` |
| 采集原始视频与姿态 | `xmake build record_capture && xmake run record_capture` |
| 棋盘格采图、内参标定 | `xmake build camera_capture`、`xmake build camera_calibrator`；具体命令见 [`docs/vision_tools.md`](docs/vision_tools.md) |
| 打符视频回放 | `xmake build buff_video_replay && xmake run buff_video_replay --help` |
| Web 调试 | `.venv/bin/python web.py`；说明见 [`docs/web_debug.md`](docs/web_debug.md) |

`build/`、`logs/`、`records/`、`tmp/` 是本地生成数据；现有内容没有随目录整理移动或删除。
