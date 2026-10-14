# 能量机关代码框架

检测关键点 → 筛选轮廓 → 重建三维旋转平面 → 估计叶片相位与运动 → 预测弹丸到达时的靶心 → 输出云台角度和开火许可。

本模块借鉴 [RP-26Rune](https://github.com/SZURPVision/RP-26Rune) 的分层数据流与失败关闭原则，按本工程的 xmake、Eigen、OpenCV 约定实现独立 `Rune::Processor`。代码位于 `include/rune/` 与 `src/rune/`，已由 `newvision` 静态库自动收录。

| 层 | 当前职责 | 后续精化点 |
| --- | --- | --- |
| f1-observation | `OpenVinoKeypointDetector` 推理 RP-26Rune 五点模型并解码；`DetectionStage` 验证全图坐标、挑选未激活叶片并裁剪 ROI；`ObservationStage` 做红蓝差分、5×5 高斯滤波、Otsu 阈值和轮廓提取，识别装甲板、灯臂、R 标 | 跨帧目标关联、轮廓质量检查 |
| f2-building | 使用五个语义点、**实测尺寸**与相机标定做 PnP 初值，检查重投影误差，转换符心、靶心、平面法向到枪管坐标系 | 五叶完整模型、距离场和 Chamfer 轮廓优化 |
| f3-estimator | 小符连续相位差与速度平滑，拒绝时间断层、跳变和位姿突变；`IBigMotionModel` 提供大符拟合器接入口 | 小符滤波参数标定；大符时变角速度拟合与诊断 |
| f4-planning | 按采集到下发延迟和飞行时间迭代预测靶心，用无空气阻力抛物线求角度；实测云台到位后仅发一次开火脉冲 | 空气阻力、弹速标定、目标切换与下位机确认 |

## 接口约定

公共类型见 `include/rune/types.hpp`，入口见 `include/rune/processor.hpp`。`Frame::detections` 由模型适配器填入，五点顺序为 `top / left / right / bottom / center_r`，坐标必须是 **原图像素**；`class_id=0` 表示未激活，`1` 表示已激活。`Frame::captured_at` 是图像采集时间，不能用推理完成时间替代。`T_barrel_camera` 把相机光学系变换到枪管系，枪管系为 x 向前、y 向左、z 向上。`ControlInput::command_at` 应是当前发送周期时刻；角度均为弧度，长度为米，时间为秒。

```cpp
Rune::PlanningConfig planning;
planning.shoot_enable = false; // 标定及回放时保持关闭；实机验证后显式启用
Rune::Processor rune({.radius_m = measured_radius,
                      .armor_half_width_m = measured_half_width,
                      .armor_half_height_m = measured_half_height},
                     {}, planning);

Rune::Frame frame;
frame.bgr = image;
frame.captured_at = capture_time;
frame.mode = Rune::Mode::Small;
frame.color = Rune::Color::Red;
frame.detections = detector.detect(image, frame.mode);
frame.camera_matrix = camera_matrix;
frame.distortion = distortion;
frame.T_barrel_camera = T_barrel_camera;

Rune::ControlInput control;
control.command_at = std::chrono::steady_clock::now();
control.bullet_speed_m_s = measured_bullet_speed;
control.actual_yaw_rad = measured_yaw;
control.actual_pitch_rad = measured_pitch;
const Rune::Command command = rune.process(frame, control);
// 只有 command.found 为 true 时才使用 yaw_rad / pitch_rad。
// 只有 command.fire 为 true 时才下发单发开火请求。
```

`Geometry` 的默认值为零，必须填入实测值；`PlanningConfig::shoot_enable` 默认关闭，实机验证后才显式启用。缺少标定、轮廓或位姿时关闭目标发现和开火。运动尚未稳定或观测过期时也不开火。默认未提供大符拟合器，故大符返回 `MotionNotReady`；可通过 `Processor` 构造函数的第四个参数注入 `IBigMotionModel`，由模型返回拟合质量和未来相位变化函数。该函数必须持有拟合参数的值副本。大符模型与几何、弹道参数需完成实测标定后才能用于实弹开火。单发闩锁需要在外部确认上一片已激活并切换到另一片后调用 `Processor::rearm()`；切换大小符模式时会自动重置。此接口应由单个处理线程持有，不能并发调用同一实例。

`xmake build newvision` 构建主库；`xmake build rune_pipeline_smoke && xmake run rune_pipeline_smoke` 检查基础数据流和开火门控。

## 五关键点模型部署

模型位于 `model/rune_model/model-0624.onnx`，来源、SHA-256 与许可见该目录的 `README.md`。推理复用工程现有 OpenVINO 后端，模型只在构造时加载一次；适配器独立于相机和串口线程。

```bash
xmake f --use_openvino=y
xmake build rune_detect
xmake build rune_detector_smoke
xmake run rune_detector_smoke
# 图片或视频；--output 保存最后一帧标注图，数字 0～4 分别为 top/left/right/bottom/R。
xmake run rune_detect --input /path/to/rune.mp4 --mode big --frames 30 --output build/rune_result.png
```

检测入口不需要相机标定，可独立回放图像。默认使用 CPU，阈值与设备可在 `config/rune_detector.yaml` 中修改，模型相对路径以配置文件所在目录为基准。工具不下发云台或开火指令。

```cpp
#include "rune/f1-observation/keypoint_detector.hpp"

Rune::OpenVinoKeypointDetector detector(
    Rune::KeypointDetectorConfig::load("config/rune_detector.yaml"));
// 每帧在调用 rune.process(frame, control) 前：
frame.detections = detector.detect(frame.bgr, frame.mode);
```

输入必须是 `CV_8UC3` BGR 原图，输出点为原图坐标。`detect(image)` 缺省使用小符模式；大小符切换应使用 `detect(image, frame.mode)`。模型原始顺序是 `top/left/R/right/bottom`，适配器已经换成工程顺序，并按模式过滤另一类已激活叶片。五个点均需达到置信度阈值。错误输入或推理失败会抛异常，调用方应清空本帧检测并处理错误。每个实例由一个线程持有，返回的关键点拥有自己的数据，可以跨帧保存。

2026-10-13 本机 OpenVINO 2024.6 CPU 验证：构建、解码测试与真实模型空图推理通过；上游 `big_rune_demo.mp4` 前 30 帧全部检出目标，共 60 片，平均检测耗时约 13.4 ms（含预处理、推理与解码，不含启动编译）。这段回放用于确认模型部署与关键点输出，不代表实机精度或完整控制链路验证。
