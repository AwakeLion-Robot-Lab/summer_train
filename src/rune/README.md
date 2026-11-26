# 能量机关代码框架

实时入口、`rune_detect` 与大小符回放均使用 SCUT 轮廓检测靶圈、灯臂和 R 标。
SCUT 的配置、接口和回放方法见 [SCUT 识别适配](../../docs/rune_scut_detection.md)。

关键点持续抖动的图像运动补偿、光流回退与完整录像对照见
[关键点防抖修正](../../docs/rune_keypoint_stabilization.md)。

检测关键点 → 筛选轮廓 → 重建三维旋转平面 → 估计叶片相位与运动 → f4 选叶/弹道/轨迹优化 → f5 对准/射击节拍/单发输出。

本模块借鉴 [RP-26Rune](https://github.com/SZURPVision/RP-26Rune) 的分层数据流与失败关闭原则，按本工程的 xmake、Eigen、OpenCV 约定实现独立 `Rune::Processor`。代码位于 `include/rune/` 与 `src/rune/`，已由 `newvision` 静态库自动收录。

| 层 | 当前职责 | 后续精化点 |
| --- | --- | --- |
| f1-observation | `ScutContourDetector` 提取颜色轮廓和多角点，保留未激活及已激活叶片 | 更复杂的轮廓遮挡恢复 |
| f2-building | 透视校正的五叶扇区匹配、PCA 锚框、联合 IPPE 与 Huber Chamfer；曝光时刻重力约束；完整公共坐标位姿下的平面滑窗；统一输出枪口系模型变换 | 实车标定、定位数据接入与录像评估 |
| f3-estimator | 符心/法向/自转相位与速度 EKF；马氏门限与匈牙利逐特征更新；小符独立恒速预测，大符独立恒速/正弦双拟合并按代价自动切换 | 真实录像标定与评估 |
| f4-planning | 固定身份选叶、大小符预热、延迟/飞行时间迭代；复用现有真空/线性阻力弹道与 TinyMPC | 移动车体的平移外推与实车标定 |
| f5-control | 实测对准与稳定门限、idle/shoot 时序、每窗口单发、冷却、可选手动闩锁、现有串口命令适配 | 实车发射周期与反馈时序验证 |

## 接口约定

主程序 `auto_aim` 已接入独立的 `runtime::RuneRuntime`。在 `config/auto_aim.yaml` 顶层设置 `mode: 1` 为装甲板、`2` 为小能量机关、`3` 为大能量机关，修改后重启。该配置覆盖下位机 WorkMode；串口仍提供曝光姿态、当前姿态、敌方颜色与弹速。只加载选中模式的检测器，大小符的 f3 模型保持独立。

能量机关参数位于同一文件的 `rune` 节点。`detector.scut` 配置颜色、轮廓及几何匹配阈值，`geometry` 三项尺寸必须填写实测米制值，默认 0 时仅检测。装甲板与大小符统一使用顶层 `fire.shoot_enable` 开火开关，缺省关闭。`rune.fire` 配置能量机关的对准门限和射击节拍。弹道、弹速回退、出膛延迟及 MPC 调参复用 `config/planner_config.yaml`，可用 `rune.mpc_enable` 单独覆盖 MPC 开关。相机标定复用 `sensor/camera/camera_calibration.hpp`，反馈过期、颜色未知或姿态缺失时中断跟踪并保持最后方向、关火。当前没有定位平移输入，不启用依赖完整公共坐标位姿的平面滑窗。

运行时在轮廓检测完成后取得规划时刻及最新反馈。能量机关命令通过 `SerialWorker::updateCommand(command, true)` 下发，单发位在一次完整串口写入后清除，角度仍周期发送；装甲板继续使用原有持续开火语义。`rune_backend_smoke` 覆盖配置与检测入口。

公共类型见 `include/rune/types.hpp`，入口见 `include/rune/processor.hpp`。`Frame::detections` 由轮廓检测器填入，五点顺序为 `top / left / right / bottom / center_r`，坐标必须是 **原图像素**。`class_id=0` 表示未激活，`1` 表示已激活。`Frame::captured_at` 是图像采集时间。`T_barrel_camera` 把相机光学系变换到枪管系，枪管系为 x 向前、y 向左、z 向上。`ControlInput::command_at` 应是当前发送周期时刻。角度均为弧度，长度为米，时间为秒。f2 会将曝光时刻姿态带入规划。规划和反馈角均使用重力对齐的绝对坐标，pitch 向上为正，转为串口命令时统一取反。

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

## 检测与回放

新导入的 `video/rune/` 录像，小符使用 `xmake run auto_aim_test_small_rune`，
大符使用 `xmake run auto_aim_test_big_rune`。两个入口复用实机 `RuneRuntime`，
默认分别播放 `2026-11-15_15-04-07.mkv` 和 `2026-11-15_15-05-03.mkv`，
也可在命令后传入其他录像路径。
窗口、绘制间隔及 UDP 遥测统一读取 `config/auto_aim.yaml` 的 `debug` 节点。
两段默认录像均有同名姿态文本；有配套姿态和标定时运行完整链路，
缺少姿态但标定有效时运行相机相对视觉预测。装甲板回放入口为 `auto_aim_test_armor`。
参数、视频选择与无窗口运行见 [离线回放说明](../../docs/replay_tests.md)。

```bash
xmake build rune_detect
xmake build rune_scut_detector_smoke
xmake run rune_scut_detector_smoke
# 图片或视频；--output 保存最后一帧标注图，数字 0～4 分别为 top/left/right/bottom/R。
xmake run rune_detect --input /path/to/rune.mp4 --mode big --frames 30 --output build/rune_result.png
```

检测入口可独立回放图像，阈值在 `config/auto_aim.yaml` 的 `rune.detector.scut` 节点配置。

实时调试窗口、大小符回放和 `rune_detect --output` 共用 `Rune::drawDetectionOverlay`。
SCUT 绘制测得的靶圈椭圆、缺口点、灯臂端点和 R，`ImageRefiner` 对这些特征做图像运动补偿。

实时窗口和大小符回放不绘制未来位置预测框。运动估计、弹道规划及控制计算继续运行；`drawTrackingOverlay` 保留当前曝光时刻的滤波 R/靶心标记。

调试窗口显示检测数、靶圈和 R 标候选数、灯臂匹配数及轮廓检测次数。启用 `overlay` 或 `plot` 时，相同诊断每秒写入一次日志。`rune_detect` 还输出 `kpt_conf` 及五位有效标记 `valid`。

在实时窗口按 `s` 保存下一帧未叠加标注的原图到 `logs/rune_input.png`（再次按下会覆盖）。可用 `xmake run rune_detect --input logs/rune_input.png --mode small --output build/rune_result.png` 对同一图像复查；大符使用 `--mode big`。

```cpp
#include "rune/f1-observation/scut_contour_detector.hpp"

Rune::ScutContourDetector detector;
// 每帧在调用 rune.process(frame, control) 前：
frame.detections = detector.detect(frame.bgr, frame.mode, frame.color);
```

输入为 `CV_8UC3` BGR 原图，输出点和轮廓均使用原图坐标。每个实例由一个线程持有，返回结果可以跨帧保存。

实机及有姿态的回放还会叠加 `filtered R`（青色十字圆环）和
`filtered target`（白色十字圆环），分别为当前曝光时刻滤波后的符心与靶心投影。
五点颜色仍表示检测输出；没有标定、滤波位姿或当前帧跟踪失败时不画滤波投影。
