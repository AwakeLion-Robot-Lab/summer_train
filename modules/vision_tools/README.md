# 可复制的视觉工具模块

本目录可单独复制到另一个 C++ 项目。`include/vision_tools/` 是公共接口，`src/` 是实现，`examples/` 是可运行的接入示例，`web_debug/` 是 Python Web 调试台。相机 SDK、串口协议和具体算法留在宿主项目的适配层。

## 构建与依赖

需要 C++20、OpenCV 4（core、imgproc、imgcodecs、videoio、calib3d、highgui）、Eigen3、yaml-cpp；Web C++ 通道还需要 nlohmann/json 和 Linux POSIX 共享内存。Ubuntu 可安装 `libopencv-dev libeigen3-dev libyaml-cpp-dev nlohmann-json3-dev`。CMake 用 `find_package` 查找依赖，xmake 示例使用系统头文件路径。

```bash
cmake -S modules/vision_tools -B /tmp/vision_tools-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vision_tools-build -j
ctest --test-dir /tmp/vision_tools-build --output-on-failure
```

复制后在模块目录直接执行 `cmake -S . -B build` 即可。xmake 可执行 `xmake -P modules/vision_tools vision_tools_example_replay_info`；嵌入现有 xmake 项目时 `includes("path/to/vision_tools/targets.lua")`，然后 `add_deps("vision_tools_recording")` 等。CMake 宿主可 `add_subdirectory(path/to/vision_tools)`，并链接相应 target。

## 模块接口

| Target | 接口 | 用途 |
|---|---|---|
| `vision_tools_recording` | `recording/recorder.hpp`、`calibration/timed_image_saver.hpp` | 原图 AVI + 同名 TXT 姿态时间戳；定时保存标定图 |
| `vision_tools_calibration` | `calibration/high_precision_calibrator.hpp` | 棋盘格内参标定、质量筛选、报告和预览 |
| `vision_tools_replay` | `replay/frame_timeline.hpp` | 用录像 TXT 的实际采集时刻建立回放时间轴 |
| `vision_tools_web_ipc` | `web/web_sink.hpp`、`web/web_cmd.hpp` | JPEG、时序、日志和调参命令的 IPC 通道 |

相机/串口接入时，把驱动取出的 `cv::Mat`、`std::chrono::steady_clock::time_point` 和曝光时刻的 `Eigen::Quaterniond` 传给 `Recorder::record`。串口上报姿态必须先按宿主的 IMU→枪管外参转换成世界系枪管姿态；没有对齐姿态时跳过帧。模块不会控制相机或向云台发命令。

```cpp
vision_tools::RecorderConfig config;
config.enabled = true;
config.fps = 90;
config.output_dir = "records";
vision_tools::Recorder recorder(config);
recorder.record(raw_frame, q_world_barrel, exposure_time);
recorder.stop();
```

`examples/record_synthetic.cpp` 可独立生成 AVI/TXT；`examples/calibrate.cpp` 接受采图目录；`examples/replay_info.cpp` 显示 AVI 的实际时间源及有效帧率。录像 TXT 每行依次是采集秒数、`w x y z`；AVI 容器只写固定 FPS，实时回放应优先使用 TXT，避免采集掉帧时播放加速。`FrameTimeline::loadFrameTimeline(..., TimingSource::Auto)` 自动选择同名 TXT，缺失时回退 AVI FPS。

## Web 调试台

```bash
python3 -m venv .venv
.venv/bin/pip install -r modules/vision_tools/requirements.txt
VISION_TOOLS_IPC_PREFIX=/dev/shm/newvision \
  .venv/bin/python modules/vision_tools/web.py
```

WebSink 构造函数第四个参数是 IPC 前缀，WebCmd 可传入命令文件路径；Python 的 `VISION_TOOLS_IPC_PREFIX` 必须与 C++ 一致。`VISION_TOOLS_RECORD_ROOT` 指定 Web 回放会话目录，`VISION_TOOLS_YAML` 指定调参 YAML；旧的 `NEWVISION_*` 环境变量仍可用。复制到其他项目时可用 `VISION_TOOLS_PARAM_CATALOG` 指向自己的 JSON 参数目录，并用 `VISION_TOOLS_FORCE_MODES` 设置逗号分隔的模式值。参数目录条目至少含 `key`、`yaml_key`，`examples/web_params.json` 给出了最小示例。Web 调试录像的 `frames/index.jsonl` 记录每张 JPEG 的源帧号与采集秒数；页面据此按真实间隔播放，旧录像仍按 50 ms 回放。

当前项目的硬件入口在 `tools/capture/`，根目录 `web.py` 是兼容包装器，`buff_video_replay` 复用本模块时间轴。相机标定结果中的 `calibration` 节点可放进 `config/camera_config.yaml`；手眼与云台外参仍需单独标定。
