# 能量机关代码框架

检测关键点 → 筛选轮廓 → 重建三维旋转平面 → 估计叶片相位与运动 → f4 选叶/弹道/轨迹优化 → f5 对准/射击节拍/单发输出。

本模块借鉴 [RP-26Rune](https://github.com/SZURPVision/RP-26Rune) 的分层数据流与失败关闭原则，按本工程的 xmake、Eigen、OpenCV 约定实现独立 `Rune::Processor`。代码位于 `include/rune/` 与 `src/rune/`，已由 `newvision` 静态库自动收录。

| 层 | 当前职责 | 后续精化点 |
| --- | --- | --- |
| f1-observation | `OpenVinoKeypointDetector` 解码五点模型；`DetectionStage::runAll` 同时保留未激活及激活叶片；各 ROI 提取装甲板、灯臂、R 标，大符激活叶片允许仅灯臂与 R 标 | 更复杂的轮廓遮挡恢复 |
| f2-building | 透视校正的五叶扇区匹配、PCA 锚框、联合 IPPE 与 Huber Chamfer；曝光时刻重力约束；完整公共坐标位姿下的平面滑窗；统一输出枪口系模型变换 | 实车标定、定位数据接入与录像评估 |
| f3-estimator | 符心/法向/自转相位与速度 EKF；马氏门限与匈牙利逐特征更新；小符独立恒速预测，大符独立恒速/正弦双拟合并按代价自动切换 | 真实录像标定与评估 |
| f4-planning | 固定身份选叶、大小符预热、延迟/飞行时间迭代；复用现有真空/线性阻力弹道与 TinyMPC | 移动车体的平移外推与实车标定 |
| f5-control | 实测对准与稳定门限、idle/shoot 时序、每窗口单发、冷却、可选手动闩锁、现有串口命令适配 | 实车发射周期与反馈时序验证 |

## 接口约定

主程序 `auto_aim` 已接入独立的 `runtime::RuneRuntime`。在 `config/auto_aim.yaml` 顶层设置 `mode: 1` 为装甲板、`2` 为小能量机关、`3` 为大能量机关，修改后重启。该配置覆盖下位机 WorkMode；串口仍提供曝光姿态、当前姿态、敌方颜色与弹速。只加载选中模式的检测器，大小符的 f3 模型保持独立。

能量机关参数位于同一文件的 `rune` 节点：`detector` 配置检测模型、设备及阈值，模型路径相对于该 YAML 目录；`geometry` 三项尺寸必须填写实测米制值，默认 0 时仅检测；`fire.shoot_enable` 独立于装甲板开关，默认关闭。弹道、弹速回退、出膛延迟及 MPC 调参复用 `config/planner_config.yaml`，可用 `rune.mpc_enable` 单独覆盖 MPC 开关。相机标定复用 `sensor/camera/camera_calibration.hpp`，反馈过期、颜色未知或姿态缺失时中断跟踪并保持最后方向、关火。当前没有定位平移输入，不启用依赖完整公共坐标位姿的平面滑窗。

运行时在关键点推理完成后取得规划时刻及最新反馈。能量机关命令通过 `SerialWorker::updateCommand(command, true)` 下发，单发位在一次完整串口写入后清除，角度仍周期发送；装甲板继续使用原有持续开火语义。`auto_aim_mode_smoke` 覆盖配置与传感器适配，`serial_single_shot_smoke` 用伪串口验证单发与超时。

公共类型见 `include/rune/types.hpp`，入口见 `include/rune/processor.hpp`。`Frame::detections` 由模型适配器填入，五点顺序为 `top / left / right / bottom / center_r`，坐标必须是 **原图像素**；`class_id=0` 表示未激活，`1` 表示已激活。`Frame::captured_at` 是图像采集时间，不能用推理完成时间替代。`T_barrel_camera` 把相机光学系变换到枪管系，枪管系为 x 向前、y 向左、z 向上。`ControlInput::command_at` 应是当前发送周期时刻；角度均为弧度，长度为米，时间为秒。f2 会将曝光时刻姿态带入规划；规划和反馈角均使用重力对齐的绝对坐标，pitch 向上为正，转为串口命令时统一取反。

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
frame.q_world_barrel = measured_exposure_orientation;

Rune::ControlInput control;
control.command_at = std::chrono::steady_clock::now();
control.bullet_speed_m_s = measured_bullet_speed;
control.actual_yaw_rad = measured_yaw;
control.actual_pitch_rad = measured_pitch;
const Rune::Command command = rune.process(frame, control);
// 只有 command.found 为 true 时才使用 yaw_rad / pitch_rad。
// 只有 command.fire 为 true 时才下发单发开火请求。
```

`Geometry` 的默认值为零，必须填入实测值；`PlanningConfig::shoot_enable` 默认关闭，实机验证后才显式启用。缺少标定、轮廓或位姿时关闭目标发现和开火。运动尚未稳定或观测过期时也不开火。大小符按模式分别预测：小符恒速模型位于 `f3-estimator/motion/small_rune_motion.cpp`，大符恒速/正弦双模型位于 `f3-estimator/motion/big_rune_motion.cpp`，仅在大符路径中按拟合代价自动选择模型；均需至少 1.5 秒连续观测预热。原第四参数 `IBigMotionModel` 仍可显式覆盖大符模型。f4 从本次跟踪开始默认等待小符 3 s、大符 6 s，与 f3 拟合预热同时进行；预热期间有有效位姿时输出符心跟随且不开火。状态、关联、参数和诊断见 [f3-estimator/README.md](f3-estimator/README.md)。大符模型与几何、弹道参数需完成实测标定后才能用于实弹开火。f5 默认按 idle/shoot 窗口输出单发脉冲，支持未激活叶片重试；仅在 `FireControlConfig::require_rearm=true` 时需要外部确认后 `Processor::rearm()`。失去目标或切换大小符会重新预热并保持射击冷却。此接口应由单个处理线程持有，不能并发调用同一实例。

f4/f5 的接口、MPC、坐标约定与射击时序见 [f4-planning](f4-planning/README.md) 和 [f5-control](f5-control/README.md)。`Processor::lastPlan()` 可查看目标身份、MPC 状态与规划轨迹。

`xmake build newvision` 构建主库；`xmake build rune_pipeline_smoke && xmake run rune_pipeline_smoke` 检查基础数据流和开火门控。三维求解配置、实测轮廓模型接入与专项验证见 [f2-building/README.md](f2-building/README.md)。

## 五关键点模型部署

新导入的 `video/rune/` 录像，小符使用 `xmake run auto_aim_test_small_tune`，
大符使用 `xmake run auto_aim_test_big_tune`。两个入口复用实机 `RuneRuntime`，
默认分别播放 `2026-10-28_00-05-41.avi` 和 `2026-10-28_00-03-18.avi`，
也可在命令后传入其他录像路径。
窗口、绘制间隔及 UDP 遥测统一读取 `config/auto_aim.yaml` 的 `debug` 节点。
有配套姿态和标定时运行完整链路，
缺少姿态时仅检测。装甲板回放入口为 `auto_aim_test_armor`。
参数、视频选择与无窗口运行见 [离线回放说明](../../docs/replay_tests.md)。

模型采用 [SZURPVision/RuneDetectionModel](https://github.com/SZURPVision/RuneDetectionModel) 的 `model/model-0624.onnx`，本地位于 `model/rune_model/model-0624.onnx`，来源、SHA-256 与许可见该目录的 `README.md`。推理复用工程现有 OpenVINO 后端，模型只在构造时加载一次；适配器独立于相机和串口线程。

```bash
xmake f --use_openvino=y
xmake build rune_detect
xmake build rune_detector_smoke
xmake run rune_detector_smoke
# 图片或视频；--output 保存最后一帧标注图，数字 0～4 分别为 top/left/right/bottom/R。
xmake run rune_detect --input /path/to/rune.mp4 --mode big --frames 30 --output build/rune_result.png
```

检测入口不需要相机标定，可独立回放图像。默认使用 CPU，阈值与设备可在 `config/auto_aim.yaml` 的 `rune.detector` 节点中修改，模型相对路径以配置文件所在目录为基准。工具不下发云台或开火指令。

实时 `debug.overlay: true` 窗口与 `rune_detect --output` 共用 `Rune::drawDetectionOverlay`：未激活靶板画绿色轮廓，已激活画橙色；五点沿用 RuneDetectionModel 的颜色（上绿、左黄、R 蓝、右紫、下红）及白色描边，黑底白字显示 `inactive / small_activated / big_activated` 和置信度百分比。只显示过阈值关键点；上、左、下、右依次连线，仅两端都可靠时画边，不把 R 点连入靶板框。显示的是当前帧网络检测结果，标定或姿态缺失时仍可显示检测。

关键点筛选采用 SZU 官方规则，`min_valid_keypoints: 3` 表示至少三点过阈值；
NMS 的中心和平均置信度也只使用这些点。解码仍保留全部五点坐标、各点置信度
和可靠性标记，供轮廓搜索使用。三维 IPPE 初始化需要至少四个可靠锚点，
弱点不参与 PnP 或滤波。配置与回放结果见 [后处理对齐记录](../../docs/rune_szu_alignment.md)。

无框排查：窗口第二行显示 `det`（最终检测数）、`score_pass`（过类别阈值的候选数）、`kpt_drop`（关键点低分或坐标无效）、`nms_drop`（重复候选）、`mode_drop`（当前大小符模式不匹配）；`max_cls` 为未激活/小符激活/大符激活三类的全帧最高分，`min_kpt` 为类别分数最高候选的五点最低分。启用 overlay 或 plot 时，相同诊断每秒写入一次日志。加载失败会在窗口明确显示 `Rune detector unavailable`。检测统计独立于后续跟踪状态，`status=1` 不代表网络一定没有检出。

`valid_kpt=N/5` 为类别分数最高候选过关键点门限的数量，`partial=N` 为最终
检测中不足五个可靠点的数量；`rune_detect` 还输出本地顺序的 `kpt_conf` 与
`valid` 五位标记，便于区分类别漏检与关键点漏检。

在实时窗口按 `s` 保存下一帧未叠加标注的原图到 `logs/rune_input.png`（再次按下会覆盖）。可用 `xmake run rune_detect --input logs/rune_input.png --mode small --output build/rune_result.png` 对同一图像复查；大符使用 `--mode big`。

```cpp
#include "rune/f1-observation/keypoint_detector.hpp"

Rune::OpenVinoKeypointDetector detector(
    Rune::KeypointDetectorConfig::load("config/auto_aim.yaml"));
// 每帧在调用 rune.process(frame, control) 前：
frame.detections = detector.detect(frame.bgr, frame.mode);
```

输入必须是 `CV_8UC3` BGR 原图，输出点为原图坐标。`detect(image)` 缺省使用小符模式；大小符切换应使用 `detect(image, frame.mode)`。模型原始顺序是 `top/left/R/right/bottom`，适配器已经换成工程顺序，并按模式过滤另一类已激活叶片。五个点均需达到置信度阈值。错误输入或推理失败会抛异常，调用方应清空本帧检测并处理错误。每个实例由一个线程持有，返回的关键点拥有自己的数据，可以跨帧保存。

近距离录像的目标占比过大时，该模型的类别分数可能明显偏低。当前
`rune.detector.input_scale: 0.75` 将画面缩小后居中补边，配合面积采样减少 LED
点阵缩小时的失真；关键点仍还原到原图，保持现有标定与几何配置。
`input_scale: 1.0` 恢复上游满幅输入。详情见
[尺度诊断与验证](../../docs/rune_detection_scale.md)。

工程配置已开启 `rune.detector.fuse_keypoints`：先保留原有 NMS 分组和类别选择，
再对同组、同模型类别且关键点几何一致的候选做单帧加权融合，降低获胜候选切换
带来的跳点。质量和几何门限附近的权重连续衰减，弱点不参与，获胜候选的类别、
置信度和可靠性标记保持原值。该操作没有跨帧缓存，也不对运动点做时间低通。
设为 `false` 可恢复原 NMS 坐标；未配置时默认关闭。参数、回放对照和限制见
[关键点抖动优化](../../docs/rune_keypoint_fusion.md)。诊断中的 `fused=N` 表示本帧
实际贡献坐标的被抑制候选数。

实机及有姿态的回放还会叠加 `filtered R`（青色十字圆环）和
`filtered target`（白色十字圆环），分别为当前曝光时刻滤波后的符心与靶心投影。
五点颜色仍表示检测输出；没有标定、滤波位姿或当前帧跟踪失败时不画滤波投影。

2026-10-13 本机 OpenVINO 2024.6 CPU 验证：构建、解码测试与真实模型空图推理通过；上游 `big_rune_demo.mp4` 前 30 帧全部检出目标，共 60 片，平均检测耗时约 13.4 ms（含预处理、推理与解码，不含启动编译）。这段回放用于确认模型部署与关键点输出，不代表实机精度或完整控制链路验证。
