# newvision 纯 C++ 自瞄技术路线

本文基于当前仓库内多个开源自瞄实现的本地代码阅读整理，目标是为 `newvision` 规划一条**无 ROS2、纯 C++、高执行性**的自瞄路线。参考对象包括 `sp_vision_25-main`、`Climber_Vision_26-main`、`FYT2024_vision-main`、`jlu_vision_26-master`、`SHtech_auto_aim-ax650-dev-2026`、`rm.cv.fans-main`、`rmcs_auto_aim-main`、`HUST_HeroAim_2024-main`、`awakening-main`、`talos_2026-master` 等本地项目，并吸收了 `AUTO_AIM_SURVEY.md` 的横向调研结论。

## 1. 总体判断

`newvision` 现在的分层是正确的：

```text
l1_sensor      传感器、时间戳、机器人状态
l2_perception  识别器与检测结果
l3_estimation  PnP、重投影、目标估计、EKF
l4_planning    延迟补偿、预测、弹道、轨迹规划
l5_control     火控、串口命令、控制输出
l6_telemetry   日志、trace、调试观测
runtime        主循环和系统装配
```

但当前核心算法还未完成：`EkfTracker::reset()` 为空实现，`TargetEstimator::update()` 直接返回 `nullopt`，`Planner::plan()` 只判断目标是否存在，`BallisticSolver::solvePitch()` 还未解弹道。因此路线必须先补“可运行闭环”，再引入高级估计和 MPC。

最终推荐路线：

```text
相机/串口时间同步
  -> YOLO 四点检测 + 传统灯条 fallback
  -> 角点顺序统一 + PCA/几何角点修正
  -> IPPE PnP + 重投影 yaw 优化
  -> 整车 EKF + 跟踪状态机
  -> 延迟补偿 + 选板死区
  -> 真空弹道，随后升级线性空气阻力
  -> yaw/pitch setpoint，第一版不实现 MPC，但保留 MPC planner 接口
  -> 火控窗口 + 跳变抑制
  -> 日志、回放、参数扫描
```

## 2. 各开源项目可吸收的技术点

### sp_vision_25

同济方案是 `newvision` 最适合的主参考。它不依赖 ROS，采用 CMake + C++ 模块化结构，包含相机、串口、配置、日志、测试、标定、检测、EKF、规划和火控。自瞄流程为：

```text
Camera + RobotState
  -> Detector
  -> Solver/PnP
  -> Tracker/Target EKF
  -> Planner/TinyMPC
  -> Shooter
  -> Command
```

关键技术：

- 神经网络四点检测替代纯传统识别，提高召回。
- `Tracker` 使用 `lost/detecting/tracking/temp_lost/switching` 状态机。
- `Target` 使用整车状态 EKF，而不是只滤单块装甲板。
- `Planner` 先预测飞行时间，再生成未来 yaw/pitch 参考轨迹，使用 TinyMPC 约束云台角加速度。
- 火控不只看当前角度误差，而是看规划轨迹与目标轨迹误差。
- README 中还提出了五次多项式替代 TinyMPC 的构思：只在小陀螺切板导致的轨迹断点附近做显式过渡，跟随段仍贴合原射击轨迹。五次多项式用两端的位置、速度、加速度 6 个边界条件确定，逐步增大过渡时间，直到最大加速度低于云台极限。该方案仿真验证过，但未作为国赛上车主方案。

### Climber_Vision_26

这是同济路线的强化版，适合吸收专项优化：

- 前哨站 3 装甲板模型：维护 3 个槽位，使用代价矩阵和 DFS 做观测到槽位的匹配。
- 前哨站主打击面选择：根据历史主槽、角速度方向、观测数量和可见性规则切换，避免反向跳槽。
- 弹道从真空模型升级为线性空气阻力模型 `dv/dt = -k*v - g`，`k=0` 时退化为真空弹道。
- 火控增加“命令突变抑制”：上一帧命令变化过大、云台尚未跟上、目标无效时禁止开火。

### FYT2024

FYT 是 ROS2 项目，通信框架不采用，但算法细节很有价值：

- 传统灯条检测：灰度二值化为主，颜色判断放到后处理，避免灯条过曝导致 R/B 不稳定。
- 灯条筛选：轮廓 `minAreaRect`，按长宽比、倾斜角、左右灯条长度比、中心距、装甲板角度过滤。
- 数字分类：透视变换装甲 ROI，Otsu 二值化，LeNet/MLP 分类。
- PCA 角点修正：不用旋转矩形顶点直接做 PnP，而是沿灯条主轴寻找亮度变化最大的上下端点，提高不同曝光下角点一致性。
- EKF 状态建议：`[xc, yc, z, yaw, vxc, vyc, vz, vyaw, r]`。
- Tracker 状态：`DETECTING/TRACKING/TEMP_LOST/LOST`，按位置距离和 yaw 差关联观测。

### jlu_vision_26

JLU 的运行框架较重，但估计和选板思想值得后期吸收：

- 检测节点组合了深圳大学 YOLO 权重、同济并发框架、中南 PCA 角点矫正。
- 明确指出不同项目的关键点顺序不同，PnP 前必须统一角点顺序。
- 整车观测器使用重投影误差 + 因子图优化，通过四个角点重投影因子约束整车位置和 yaw。
- 选板策略：先用“目标中心距离减半径”的距离估算飞行时间，再选最面对的装甲板；选定后迭代飞行时间时不反复换板，避免不收敛。
- 45 度附近切板加入死区，防止观测噪声导致左右反复横跳。

因子图数值敏感，不建议作为第一版。它应作为 EKF 稳定后的增强观测器。

### SHtech_auto_aim

SHtech 的价值在工程框架：

- 纯 C++ `PipelineTask + SubModule + Bridge`，没有 ROS2。
- 有界缓冲队列和零缓冲握手管道都实现了，适合控制延迟和背压。
- 主程序严格定义生命周期：资源分配、静态连线、线程创建、统一启动、安全停机、资源释放。
- Tracker 采用多模型策略：装甲板模型、整车模型、模型信任度、装甲板跳变检测、半径限制、前哨站特判。
- Planner 有多策略：无模型直瞄、装甲板模型预测、整车模型预测、车中心策略，并可接 TinyMPC。

`newvision` 可以先单线程闭环，后续再采用类似 Bridge 的纯 C++ 多线程流水线。

### rm.cv.fans

该项目不是普通的“检测 + EKF + 弹道”自瞄，而是围绕 LMTD 反陀螺模型、精细延迟匹配、枪口坐标弹道、运行时热更新参数和弹道自校正组织。它对 `newvision` 的价值不在于直接复制结构，而在于吸收以下工程和算法原则：

- PnP 测距噪声与距离关系：若像素浮动近似恒定，距离方差近似随距离四次方增长。因此 EKF 的观测噪声 `R` 应随距离增大。
- 延迟必须拆分：图像曝光中点、预测开始、发送、电控响应、发射、命中。视觉算法不应只加一个固定 delay，而应至少区分处理延迟、通信/控制延迟和飞行时间。
- 延迟拆分不是“每段都天然精确知道”，而是给每段定义可见性：`img_to_predict` 可由本帧图像时间和预测开始时间直接测量；`predict_to_send` 可由预测开始和发送时间测量后滤波估计；`send_to_control` 依赖电控/通信标定或配置；`control_to_fire` 可由电控反馈标定，当前实现主要用配置；`fire_delay` 由枪口坐标距离和弹速计算。
- 坐标语义必须清晰：它区分相机坐标、相机光心下 IMU 坐标、枪口中心下 IMU 坐标、枪口中心相机朝向坐标、枪口发射坐标。`newvision` 不必照搬命名，但必须明确 target 坐标和 aim 坐标分别以相机还是枪口为原点。
- 弹道解算在枪口坐标中做，而不是在相机坐标中硬加 yaw/pitch 偏置。相机到枪口的平移先转换到目标点，再解重力/阻力补偿。
- 运行时参数热更新很有价值：调延迟、火控阈值、空气阻力、目标类型和滤波噪声时不应每次重启程序。`newvision` 可先做启动加载，后期加热更新。
- 装甲板模型不是只维护一个目标，而是按装甲板 id 建多个短生命周期滤波线程，消失后按 `credit_time` 淘汰。这个适合 `newvision` 后期处理多块同车装甲板。
- 目标状态先按敌人编号聚合，再按兵种类型解释：普通步兵、平衡步兵、前哨站、基地/水晶使用不同装甲板数量、默认半径、装甲板 pitch 和火控误差参数。
- “双板 PnP”更准确地说是双板重投影朝向拟合：先对每块板做普通 IPPE PnP 得到位置，再假设两块可见板是同车相邻装甲板，法向夹角固定为 `2π / armor_count`，只优化整车朝向角，并用两块板共 8 个角点的重投影代价约束。它可行，但不应作为第一版主位姿来源。
- `ShootMode` 的分层很实用：`IDLE`、`TRACKING`、`SHOOT_NOW` 比单个 `fire` bool 更利于表达“是否控制”和“是否开火”。
- LMTD 的反陀螺模型可作为后期参考：状态包含车中心位置/速度、主装甲板 yaw/yaw_rate、半径；可直接瞄当前正对装甲板，也可在高速旋转时等待下一块装甲板进入可击打角度。
- 自主弹道校正思路先进：记录每次 aim id、图像时间和瞄准参数，通过电控回传发射 id，反推发射时刻的控制命令，并用图像中检测到的弹丸轨迹估计误差。这一部分实现复杂，适合作为后期 telemetry/校准工具。

不建议照搬的部分：`CoordConverter` 过于集中，承担了坐标、延迟、弹道、火控和比较逻辑。`newvision` 应把这些拆到 `LatencyCompensator`、`BallisticSolver`、`FireDecision`、`TargetSelector` 中，保留它的时间和坐标语义。

### HUST_HeroAim_2024

HUST 是较传统的多线程工程：

- 串口线程、相机线程、检测线程分开。
- 传统灯条匹配条件清晰：角度差、灯条高度差、长度匹配、装甲板宽高比。
- PnP 使用大/小装甲板不同尺寸，坐标系通过 Sophus SE3 串联：camera -> gimbal -> world。
- 代码中有整车 EKF 思路，但部分实现被注释，适合参考思想，不适合作为主实现。

### awakening

awakening 的 README 给出了非常完整的自瞄/能量机关流程。自瞄侧值得吸收：

- ROI 检测：目标稳定时只在预测 ROI 内跑网络，丢失时回到全图。
- 网络检测和传统 CV 检测并行作为候选。
- Tracker 使用 ESEKF，状态包含整车中心、速度、旋转、半径、高度和姿态。
- 匹配不仅使用装甲板，也使用灯条观测更新。
- Aimer 生成未来控制点序列，再用轨迹限制器检查最大加速度。

实现较复杂，适合作为后期优化目标。

### talos_2026

Talos 的架构很适合作为长期方向：

- 五级火控结构：L1 Sensor、L2 Perception、L3 Estimation、L4 Planning、L5 Weapon。
- 通过资源读写声明构建调度系统，不依赖 ROS2。
- L3 tracker 以固定频率运行，输出所有 tracker，由 L4 选择目标。
- L4 构建参考轨迹，采样未来和过去窗口，用有限差分得到 yaw/pitch 速度。
- L5 火控把物理可击打窗口转换成角度阈值，距离越近窗口越大，并设置最小阈值。

`newvision` 当前分层和 Talos 很接近，但不必直接引入复杂 ECS 调度。

### AUTO_AIM_SURVEY.md 横向补充

`AUTO_AIM_SURVEY.md` 的价值是把各项目的共同路线压缩成一套 2026 可持续架构。对 `newvision` 的补充结论如下：

- 高命中率首先来自稳定数据闭环，而不是单点更复杂的检测器或滤波器。每层必须有明确输入、输出、时间戳、坐标系和可回放日志。
- ROS2、Foxglove、PlotJuggler、Web 调试界面只适合作为旁路适配器，不进入高频火控主循环。主循环保持纯 C++、直接内存传递和有界队列。
- L2 识别主线应是 YOLO/YOLO11 四角点装甲板检测，传统灯条 + 分类器作为低置信度复核、过曝兜底、模型故障兜底，而不是长期与网络结果投票。
- L3 不应停留在“PnP 点 + EKF”。推荐演进为整车状态滤波：即使只看到一块装甲板，也用整车中心、yaw、半径和高度解释观测。
- 跟踪层拆成三层：检测关联、目标生命周期、模型管理。模型管理可采用轻量 IMM 思路，在单装甲板、多装甲板、整车、小陀螺、前哨站、能量机关等模型间切换。
- L4/L5 的瞄准目标必须是 `t_hit_est` 时刻、枪口坐标系下的目标点，而不是当前相机坐标下的目标点。延迟、控制响应和飞行时间要分别记录。
- 调试系统必须记录可复盘字段：图像、检测、PnP、tracker、预测、弹道、火控、系统耗时。否则 EKF、选板和弹道参数无法可靠调优。

## 3. 纯 C++ newvision 框架

第一版建议使用单进程、少线程：

```text
Main Runtime
  ├── Camera/Video source
  ├── Serial receiver
  ├── Detector
  ├── TargetEstimator
  ├── Planner
  ├── Controller
  └── TelemetryLogger
```

最小闭环可先单线程：

```cpp
while (running) {
  FramePacket frame = camera.read();
  RobotState robot = serial.latestState();
  auto detections = detector.detect(frame.image);
  auto target = estimator.update(detections, robot, frame.timestamp);
  auto plan = planner.plan(target, robot);
  auto command = controller.makeCommand(plan, robot);
  serial.write(command);
  telemetry.record(frame, detections, target, plan, command);
}
```

稳定后再拆线程：

```text
CameraThread -> LatestBuffer<FramePacket>
SerialThread -> LatestBuffer<RobotState>
VisionThread -> LatestBuffer<AimPlan>
ControlThread -> SerialCommand
```

不要使用 ROS2 topic。线程间通信使用 `LatestBuffer` 或有界队列，过期帧直接丢弃，保证实时性。

建议把每层输出做成稳定数据契约：

```text
L1: FramePacket + RobotState + timestamp
L2: Detection[] + backend/model/version + infer_ms
L3: TargetState + covariance + innovation + tracker_state
L4: AimPlan + prediction_time + selected_armor + ballistic_result
L5: SerialCommand + fire_decision + reject_reason
L6: FrameTrace + replay index
```

这样后续可以替换检测后端、滤波模型或 planner，而不破坏主循环和回放系统。

### 3.1 具体文件框架

当前 `newvision` 已经有 `include/`、`src/`、`tests/`、`tools/` 和 `docs/`。这里先只列**需要的框架文件**，不要求立即填充算法内容；第一版可以先建空壳、最小声明和可编译 stub。

推荐目录：

```text
newvision/
  xmake.lua
  configs/
    camera.yaml
    armor.yaml
    ballistic.yaml
    runtime.yaml
  include/
    core/
      geometry.hpp
      time.hpp
      result.hpp
    l1_sensor/
      camera.hpp
      frame_packet.hpp
      robot_state.hpp
      serial_receiver.hpp
    l2_perception/
      detection.hpp
      detector.hpp
      yolo_detector.hpp
      lightbar_detector.hpp
      corner_refiner.hpp
    l3_estimation/
      armor_observation.hpp
      armor_pose.hpp
      pnp_solver.hpp
      reprojection_error.hpp
      ekf_tracker.hpp
      target_estimator.hpp
      target_state.hpp
    l4_planning/
      planner_interface.hpp
      aim_plan.hpp
      setpoint_planner.hpp
      predictor.hpp
      latency_compensator.hpp
      target_selector.hpp
      ballistic_solver.hpp
    l5_control/
      controller.hpp
      fire_decision.hpp
      serial_command.hpp
      reject_reason.hpp
    l6_telemetry/
      frame_trace.hpp
      telemetry_logger.hpp
      replay_reader.hpp
      trace_scope.hpp
    runtime/
      auto_aim_runtime.hpp
      app_config.hpp
      module_factory.hpp
  src/
    ... 与 include 同名 .cpp
  tests/
    pnp_solver_test.cpp
    ballistic_solver_test.cpp
    planner_interface_test.cpp
    latest_buffer_smoke.cpp
```

第一版新增文件清单：

```text
include/core/geometry.hpp
include/core/time.hpp
include/core/result.hpp

include/l2_perception/detection.hpp
include/l2_perception/corner_refiner.hpp
src/l2_perception/corner_refiner.cpp

include/l3_estimation/armor_pose.hpp
include/l3_estimation/armor_observation.hpp
include/l3_estimation/target_state.hpp

include/l4_planning/aim_plan.hpp
include/l4_planning/planner_interface.hpp
include/l4_planning/setpoint_planner.hpp
include/l4_planning/target_selector.hpp
src/l4_planning/setpoint_planner.cpp
src/l4_planning/target_selector.cpp

include/l6_telemetry/replay_reader.hpp
src/l6_telemetry/replay_reader.cpp

include/runtime/app_config.hpp
include/runtime/module_factory.hpp
src/runtime/app_config.cpp
src/runtime/module_factory.cpp

configs/camera.yaml
configs/armor.yaml
configs/ballistic.yaml
configs/runtime.yaml
```

现有文件需要保留并后续整理：

```text
include/l1_sensor/camera/camera.hpp
include/l1_sensor/camera/frame_packet.hpp
include/l1_sensor/serial/robot_state.hpp
include/l2_perception/detector.hpp
include/l2_perception/yolo_detector.hpp
include/l2_perception/lightbar_detector.hpp
include/l3_estimation/pnp_solver.hpp
include/l3_estimation/reprojection_error.hpp
include/l3_estimation/ekf_tracker.hpp
include/l3_estimation/target_estimator.hpp
include/l4_planning/predictor.hpp
include/l4_planning/latency_compensator.hpp
include/l4_planning/ballistic_solver.hpp
include/l5_control/controller.hpp
include/l5_control/fire_decision.hpp
include/l5_control/serial_command.hpp
include/l6_telemetry/frame_trace.hpp
include/l6_telemetry/telemetry_logger.hpp
include/runtime/auto_aim_runtime.hpp
```

后期预留文件，不在第一版填充：

```text
include/l4_planning/mpc_planner.hpp
src/l4_planning/mpc_planner.cpp
include/l4_planning/quintic_switch_planner.hpp
src/l4_planning/quintic_switch_planner.cpp
include/l3_estimation/double_armor_yaw_observer.hpp
src/l3_estimation/double_armor_yaw_observer.cpp
include/l6_telemetry/ballistic_calibrator.hpp
src/l6_telemetry/ballistic_calibrator.cpp
```

## 4. 模块级算法设计

### 4.1 L1 Sensor

目标：

- 相机帧必须带时间戳，时间戳尽量表示曝光中点。
- 串口状态必须带接收时间，包括 yaw/pitch、四元数或欧拉角、子弹速度、敌方颜色、模式。
- 保留视频/图片回放入口，方便无硬件调试。

建议数据：

```cpp
struct FramePacket {
  cv::Mat image;
  TimePoint exposure_time;
  uint64_t frame_id;
};

struct RobotState {
  Eigen::Quaterniond q_imu;
  double yaw;
  double pitch;
  double bullet_speed;
  int enemy_color;
  int mode;
  TimePoint timestamp;
};
```

### 4.2 L2 Perception

主检测器采用 YOLO 四点模型，传统灯条作为 fallback 或调试模式。

统一输出：

```cpp
struct Detection {
  std::array<cv::Point2f, 4> corners; // 固定为左上、右上、右下、左下
  int class_id;
  ArmorSize size;
  EnemyColor color;
  float confidence;
};
```

关键要求：

- 所有检测器输出前必须统一角点顺序。
- 网络 ROI 坐标必须还原到原图。
- 低置信度、颜色错误、非法类别直接过滤。
- 保留 `LightbarDetector`，用于无模型、模型故障或近距离过曝场景。
- PCA 角点修正作为传统检测的必要步骤，也可用于网络四点后处理。
- 检测结果记录 `backend/model/model_version/confidence/infer_ms`，便于比较 OpenVINO、TensorRT、ONNXRuntime 或不同权重。
- 传统检测的输出进入同一 `Detection` 结构，作为 fallback 或低置信度复核，不单独开一条后续链路。

传统检测流程：

```text
灰度二值化
  -> findContours
  -> minAreaRect
  -> 灯条长宽比/角度过滤
  -> 轮廓内 R/B 求和判色
  -> 左右灯条配对
  -> PCA 修正角点
  -> 数字 ROI 透视变换 + 分类
```

### 4.3 L3 Estimation

第一版做 PnP + EKF，不直接上因子图。

PnP：

- 大装甲板和小装甲板使用不同 3D 点。
- 使用 `cv::SOLVEPNP_IPPE`。
- 输出相机系、云台系、世界系位置。
- 对普通装甲板做重投影 yaw 优化；平衡步兵/特殊 pitch 不确定目标先跳过。
- 双板可见时，不建议直接把 8 个角点拼成一个通用 `solvePnP` 问题。更稳的方式是沿用 rm.cv.fans：单板 PnP 提供每块板位置，双板只作为整车 yaw/orientation 的重投影观测，并给它更小但非零的观测噪声。

坐标链：

```text
armor -> camera -> gimbal -> world
```

整车 EKF 状态：

```text
x = [xc, vxc, yc, vyc, z, vz, yaw, vyaw, r, dz1, dz2]
```

第一版可以先用 9 维：

```text
x = [xc, yc, z, yaw, vxc, vyc, vz, vyaw, r]
```

观测：

```text
z = [armor_x, armor_y, armor_z, armor_yaw]
```

由装甲板反推车中心：

```text
xc = armor_x + r * cos(armor_yaw)
yc = armor_y + r * sin(armor_yaw)
```

Tracker 状态机：

```text
LOST
  -> DETECTING     连续命中 N 帧
  -> TRACKING      正常输出
  -> TEMP_LOST     短时无观测，靠预测维持
  -> LOST          超过丢失阈值
```

关联门限：

- 类别相同。
- 位置 L2 距离小于阈值。
- yaw 差小于阈值。
- NIS 或重投影误差不过大。
- dt 过大时直接 reset。

远距离时增大观测噪声：

```text
R_position = base_R * max(1, distance^4 / reference_distance^4)
```

后续模型管理建议采用轻量 IMM 思路，但不必第一版实现完整 IMM 数学框架：

```text
same TargetState semantics
  + different transition/observation/noise models
  + model confidence
  + explicit switch reason
```

推荐模型包括：单装甲板、普通整车、多装甲板整车、小陀螺 LMTD、前哨站 3 板、基地/水晶、能量机关。第一版只实现普通整车，日志字段先预留 `model_name/model_confidence/switch_reason`。

双板朝向拟合的可行条件：

- 两块检测结果必须属于同一目标，且 id/排序能判断左右相邻关系。
- 已知目标装甲板数量，普通车 4 板相邻夹角为 90 度，前哨站 3 板为 120 度。
- 两块板都要有可靠角点、类别、尺寸和去畸变后的图像点。
- 只把它作为 `orientation_yaw` 观测，不直接覆盖车中心、距离和半径。
- 单板/双板使用不同 `R`，双板 `R` 可小一些，但必须根据重投影误差、面积比例、视角和 id 稳定性动态放大。

不满足这些条件时退回单板 PnP + 整车 EKF，避免错误双板约束把整车 yaw 拉偏。

### 4.4 L4 Planning

先实现可调试的 setpoint planner。第一版不要实现 MPC，也不要引入 TinyMPC 依赖；但 `Planner` 的输入输出必须按可替换 planner 设计，后续能直接挂 `MPCPlanner` 或 `QuinticSwitchBridge`。

延迟模型：

```text
total_prediction_time =
  image_to_now_processing_delay
  + communication_delay
  + controller_response_delay
  + bullet_fly_time
```

注意：不同火控定义可能会忽略 `control_to_fire`，但必须在文档和配置里固定语义。

参考 `rm.cv.fans` 后，`newvision` 应把延迟分成“可测、可估、配置、计算”四类，而不是只维护一个总 delay：

| 段 | 含义 | 来源 | newvision 第一版做法 |
| --- | --- | --- | --- |
| `img_to_predict` | 曝光中点到开始运动/目标解算 | 本帧图像时间与解算开始时间相减 | 直接测量，异常时 clamp 到默认值 |
| `predict_to_send` | 解算开始到串口准备发送 | 解算开始时间与发送时间相减 | 每帧测量，用一阶滤波或滑动平均 |
| `send_to_control` | 串口发送到电控开始执行 | 通信、电控调度、控制周期 | 配置标定，若有 ack 再在线估计 |
| `control_to_fire` | 电控执行到弹丸真实发射 | 机械传动、拨弹、加速过程 | 配置标定；有发射 id 回传后用于回放校正 |
| `fire_delay` | 发射到击中 | 弹速、距离、弹道模型 | 由枪口坐标目标点和弹道解算得到 |

这里要区分两个时间：

```text
prediction_time = img
  + img_to_predict
  + predict_to_send
  + send_to_control
  + fire_delay

hit_time = img
  + img_to_predict
  + predict_to_send
  + send_to_control
  + control_to_fire
  + fire_delay
```

`prediction_time` 对应 rm.cv.fans 的 water-gun 假设：视觉认为电控能在 `control` 时刻达到命令角度，因此瞄准的是“control 时刻发出的弹流能击中”的目标。`hit_time` 对应真实单发弹丸回放和落点校正，要额外加入 `control_to_fire`。第一版 `newvision` 可以先统一使用 `hit_time` 做预测，但必须把这两个语义留在日志里，避免后续调火控时混淆。

必须记录并尽量估计以下时间点：

```text
t_exposure
t_frame_arrive
t_l2_start / t_l2_end
t_l3_start / t_l3_end
t_l4_start / t_l4_end
t_cmd_send
t_cmd_ack
t_fire_est
t_hit_est
```

Planner 解算目标必须写清时间语义：若按单发弹丸建模，目标是 `hit_time` 时刻相对枪口的位置；若采用 rm.cv.fans 的 water-gun 控制补偿，目标是 `prediction_time` 时刻相对枪口的位置。日志中同时保留 `prediction_time_est`、`hit_time_est` 和 `latency_breakdown`，后续才能判断火控误差来自控制响应、发射延迟还是弹道飞行时间。

弹道第一版用真空模型：

```text
x = v cos(theta) t
y = v sin(theta) t - 0.5 g t^2
```

后续升级线性空气阻力：

```text
dv/dt = -k v - g
```

选板策略：

1. 先用目标中心距离估算粗略飞行时间。
2. 根据未来 yaw 和半径生成各装甲板位置。
3. 选择最面对、可击打、距离合理的装甲板。
4. 锁定装甲板后迭代飞行时间，不在迭代中反复换板。
5. 45 度附近加入切板死区。
6. 如果预测后装甲板背对，放弃该帧，而不是强行换板。

第一版输出：

```cpp
enum class PlannerKind {
  Setpoint,
  TinyMpc,
  QuinticSwitch
};

struct TrajectoryPoint {
  double t;
  double yaw;
  double pitch;
  double yaw_rate;
  double pitch_rate;
  double yaw_acc;
  double pitch_acc;
};

struct AimPlan {
  double yaw;
  double pitch;
  double yaw_rate;
  double pitch_rate;
  double yaw_acc;
  double pitch_acc;
  double fly_time;
  double prediction_time;
  int selected_armor_id;
  PlannerKind planner_kind;
  std::vector<TrajectoryPoint> reference_trajectory;
  std::vector<TrajectoryPoint> planned_trajectory;
  bool valid;
  bool fire;
  std::string reject_reason;
};
```

第一版 `SetpointPlanner` 可以只填当前命令点：`reference_trajectory/planned_trajectory` 各放 1 个点，`yaw_acc/pitch_acc = 0`，`planner_kind = Setpoint`。这样控制层、串口层、日志层不需要知道后续是否启用 MPC。

建议接口：

```cpp
class PlannerInterface {
public:
  virtual ~PlannerInterface() = default;
  virtual AimPlan plan(const TargetState& target,
                       const RobotState& robot,
                       const PlannerContext& ctx) = 0;
};
```

后续新增 `MPCPlanner` 时，只替换 `PlannerInterface` 实现，不改 `Controller`、`SerialCommand` 和 telemetry schema。

MPC 升级：

- 构建 `HORIZON` 个采样点。
- 每个采样点调用 `aim(target.predict(t))` 得到 yaw/pitch。
- 用有限差分得到 yaw_vel/pitch_vel。
- yaw/pitch 分轴 TinyMPC，状态 `[angle, angular_velocity]`，控制量为角加速度。
- 约束最大 yaw/pitch 角加速度。

五次多项式过渡可作为 TinyMPC 的轻量替代或切板专项优化：

```text
shooting trajectory:
  normal follow segment
  -> switch discontinuity detected
  -> quintic bridge segment
  -> next normal follow segment
```

核心不是用多项式拟合整条目标轨迹，而是只桥接切板断点。对 yaw/pitch 分轴构造：

```text
q(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5

q(0) = q0, q'(0) = v0, q''(0) = acc0
q(T) = q1, q'(T) = v1, q''(T) = acc1
```

然后从最小 `T` 开始搜索，计算 `max |q''(t)|`，直到满足 `max_yaw_acc / max_pitch_acc`。这样正常跟随段和射击轨迹完全重合，只在必须换板的窗口偏离，火控可直接按“规划轨迹与射击轨迹误差”关火。

落地建议：

- 第一版不做 MPC，先输出 yaw/pitch/yaw_rate/pitch_rate，并保留 `yaw_acc/pitch_acc/reference_trajectory/planned_trajectory` 字段。
- TinyMPC 阶段先复现 sp_vision 当前实现，验证日志和火控。
- 小陀螺切板稳定后，再实现 `QuinticSwitchBridge`，作为 TinyMPC 的轻量分支或高转速专项策略。
- 五次多项式必须基于已锁定的切板时刻和前后两块装甲板轨迹；切板预测不稳定时不要启用。

### 4.5 L5 Control

视觉端只发送目标角度、角速度、开火位，不直接做电机 PID。

火控条件：

- target 有效且 tracker 为 `TRACKING`。
- 当前 yaw/pitch 误差小于阈值。
- 阈值应由物理窗口换算为角度：

```text
yaw_thresh = atan(shooting_width / 2 / distance)
pitch_thresh = atan(shooting_height / 2 / distance)
```

- 设置最小阈值，避免远距离阈值过小。
- 切板跳变、命令突变、目标背对、估计发散时禁止开火。
- 连发可用更严格阈值，首发可略宽。

## 5. newvision 执行计划

### 第 0 阶段：接口定型

目标：先把数据结构定死，避免后续反复改接口。

需要完成：

- `Detection` 改为固定 4 点数组，并记录角点顺序。
- 增加 `ArmorPose`、`ArmorObservation`、`TrackerState`。
- `RobotState` 补充子弹速度、敌方颜色、模式、云台 yaw/pitch。
- 增加 `PlannerInterface`、`AimPlan`、`TrajectoryPoint`、`PlannerKind`。第一版实现 `SetpointPlanner`，但接口字段包含 `yaw_acc/pitch_acc/reference_trajectory/planned_trajectory`，为 MPC 保留位置。
- 配置文件定义相机内参、畸变、外参、装甲板尺寸、弹速默认值。

### 第 1 阶段：可运行闭环

文件重点：

- `l3_estimation/pnp_solver.*`
- `l3_estimation/target_estimator.*`
- `l4_planning/ballistic_solver.*`
- `l4_planning/planner.*`
- `runtime/auto_aim_runtime.*`

闭环要求：

```text
读图
  -> 检测
  -> PnP
  -> 选择最近/最高优先级装甲板
  -> 真空弹道
  -> SetpointPlanner 输出 yaw/pitch/yaw_rate/pitch_rate
```

此阶段可以没有 EKF，也不实现 MPC，但必须能在视频或图片上输出稳定调试信息。`AimPlan` 中的 trajectory 字段可以只填单点，保证 telemetry 和控制接口先稳定。

### 第 2 阶段：EKF 与状态机

实现：

- `EkfTracker` 状态、预测、更新、reset。
- `TargetEstimator` 维护 tracker 状态机。
- 观测关联和丢失恢复。
- 距离相关观测噪声。
- dt 异常 reset。

验收：

- 目标短时丢失不立刻断输出。
- 目标跳变能 reset 或切换。
- 日志中能看到状态机变化、残差、NIS、预测时间。

### 第 3 阶段：选板与打陀螺

实现：

- 普通车 4 装甲板模型。
- 前哨站 3 装甲板模型。
- 选板死区。
- 迭代飞行时间。
- 背对拒绝。

验收：

- 低速旋转不频繁切板。
- 高速小陀螺能稳定选车中心或最面对装甲板。
- 前哨站不会槽位乱跳。

### 第 4 阶段：弹道与火控增强

实现：

- 线性空气阻力弹道。
- 距离分段/物理窗口火控。
- 命令突变抑制。
- 延迟配置拆分。

验收：

- 同一距离多次命中点可通过 `pitch_offset` 和 `k` 标定收敛。
- 连发时不会在切板瞬间误开火。

### 第 5 阶段：TinyMPC 或五次多项式 Planner

实现：

- 在 `PlannerInterface` 下新增 `MPCPlanner` 或 `QuinticSwitchPlanner`，不改上层调用。
- 引入 TinyMPC、轻量自写 QP 包装，或五次多项式切板桥接。
- 构建 yaw/pitch 参考轨迹。
- 输出角度、角速度、角加速度。
- 火控基于未来轨迹误差。

验收：

- 高速转目标时云台命令连续。
- 切板时能提前减速。
- 最大角加速度可由配置限制。

### 第 6 阶段：工程化

实现：

- 回放系统：视频 + 时间戳 + robot_state。
- 每帧 trace：检测数量、PnP、tracker 状态、选板、弹道、火控。
- 参数热加载或启动加载。
- 多线程 `LatestBuffer` 流水线。
- 崩溃/卡死日志和自动重启脚本。

trace 字段至少包含：

```text
image: frame_id, exposure_time, camera_param_version
detection: backend, model, corners, confidence, infer_ms
pnp: tvec, rvec, reproj_error, view_angle, covariance_hint
tracking: target_id, state, model_name, x, P, innovation, gate_result
prediction: prediction_time, target_pos_hit, target_vel_hit, selected_armor
ballistics: bullet_speed, pitch_solve, fly_time, drag_k
fire: yaw_cmd, pitch_cmd, fire, reject_reason, serial_seq
system: per-layer timing, queue size, drop count, CPU/GPU/NPU load
```

## 6. 不建议第一版做的事

- 不要引入 ROS2。
- 不要第一版上因子图。
- 不要一开始做复杂 ECS 调度。
- 不要同时维护多个推理后端。
- 不要一开始同时支持所有兵种。
- 不要只追单块装甲板长期作主方案。
- 不要没有回放就调 EKF/MPC。

## 7. 第一版最小目标

第一版成功标准：

```text
纯 C++
无 ROS2
xmake 可编译
视频/相机均可输入
能检测装甲板
能解 PnP
能输出 yaw/pitch
有基础火控
有日志
可回放调试
```

第一版不追求最高命中率，追求可验证、可复现、可调参。等这条链路稳定，再把 EKF、选板、空气阻力、TinyMPC、前哨站模型逐个加进去。

## 8. 推荐最终形态

`newvision` 最终应是一套纯 C++ 火控流水线：

```text
L1 Sensor
  Camera / Video / Serial / Timestamp

L2 Perception
  YOLODetector / LightbarDetector / CornerRefiner

L3 Estimation
  PnPSolver / ReprojectionYawOptimizer / EkfTracker / TargetEstimator

L4 Planning
  LatencyCompensator / ArmorSelector / BallisticSolver / Predictor / MPCPlanner

L5 Control
  FireDecision / Controller / SerialCommand

L6 Telemetry
  Logger / FrameTrace / Replay / Plot Data
```

核心原则：

- 检测结果统一。
- 坐标系统一。
- 时间语义统一。
- 状态机可观测。
- 参数可调。
- 每一步都能单测或回放验证。

这条路线吸收了同济的非 ROS 工程闭环、FYT 的传统视觉细节、Climber 的前哨站和空气阻力、JLU 的选板和重投影思想、SHtech 的纯 C++ 流水线、rm.cv.fans 的延迟/PnP误差分析、awakening 和 Talos 的长期架构方向，同时保持 `newvision` 自己的简洁分层。

## 9. 技术细节代码索引

本节按 `newvision` 后续实现时会遇到的技术点，列出最值得参考的本地代码段落。路径均相对仓库根目录。

### 9.1 纯 C++ 框架与线程流水线

- `SHtech_auto_aim-ax650-dev-2026/common/pipeline.hpp:50`  
  参考 `BasicTask / PipelineTask / SubModule / Bridge` 的纯 C++ 分层。它证明不使用 ROS2 也可以做清晰的任务生命周期和模块组合。

- `SHtech_auto_aim-ax650-dev-2026/common/pipeline.hpp:85`  
  参考 `BufferedPipeline::get/put` 的有界队列实现，可用于 `CameraThread -> VisionThread -> ControlThread`。

### 9.2 传统灯条检测

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_detector.cpp:43`  
  传统检测完整流程：预处理、找灯条、匹配装甲板、数字分类、角点修正、过滤忽略类别。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_detector.cpp:72`  
  灰度二值化预处理。该实现刻意不先依赖颜色二值化，适合灯条过曝场景。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_detector.cpp:81`  
  `findContours` 提取灯条，并在轮廓内统计 R/B 值判断颜色。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_detector.cpp:115`  
  灯条几何过滤：短长边比例、倾斜角。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_detector.cpp:135`  
  双灯条配对、过滤中间干扰灯条。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_detector.cpp:190`  
  装甲板几何判断：左右灯条长度比、中心距、连线角度、大/小装甲板分类。

### 9.3 角点修正

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/light_corner_corrector.cpp:22`  
  PCA 灯条角点修正入口。可作为 `newvision::LightbarDetector` 的后处理。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/light_corner_corrector.cpp:55`  
  通过 ROI 亮度分布和 PCA 求灯条对称轴。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/light_corner_corrector.cpp:113`  
  沿主轴方向搜索最大亮度突变点，得到更稳定的上下端点。

### 9.4 关键点顺序

- `jlu_vision_26-master/src/auto_aim/armor_detector/README.md:18`  
  总结了不同项目和模型的四点顺序差异。`newvision` 必须在进入 PnP 前统一顺序，建议固定为左上、右上、右下、左下。

### 9.5 PnP、坐标变换与重投影

- `Climber_Vision_26-main/tasks/auto_aim/solver.cpp:64`  
  YAML 读取相机内参、畸变、相机到云台外参、云台到 IMU body 旋转。

- `Climber_Vision_26-main/tasks/auto_aim/solver.cpp:99`  
  `solvePnP`、相机系到云台系、世界系转换，以及 ypr/ypd 计算。

- `Climber_Vision_26-main/tasks/auto_aim/solver.cpp:146`  
  给定世界系装甲板位置和 yaw，重新投影四点到图像。

- `Climber_Vision_26-main/tasks/auto_aim/solver.cpp:185`  
  前哨站不同 pitch 假设下的重投影误差计算，可用于区分高低板或异常姿态。

- `Climber_Vision_26-main/tasks/auto_aim/solver.cpp:252`  
  yaw 搜索优化入口。普通目标可用重投影误差修正 PnP yaw。

- `FYT2024_vision-main/rm_auto_aim/armor_detector/src/armor_pose_estimator.cpp:107`  
  IPPE PnP 双解排序：结合重投影误差、roll 角和灯条倾斜方向选择正确解。

- `rm.cv.fans-main/aimer/auto_aim/predictor/pnp/pnp.cpp:13`  
  大/小装甲板使用不同 3D 点，调用 `cv::SOLVEPNP_IPPE` 的简洁实现。

- `rm.cv.fans-main/aimer/auto_aim/predictor/pnp/pnp.cpp:56`  
  PnP 后距离修正、去畸变点、相机系到惯性系转换和装甲板法向计算。

### 9.6 Tracker 状态机与 EKF

- `sp_vision_25-main/tasks/auto_aim/tracker.cpp:31`  
  目标跟踪主流程：计算 dt、过滤敌方颜色、按图像中心和优先级排序、选择 set/update。

- `sp_vision_25-main/tasks/auto_aim/tracker.cpp:75`  
  发散检测和 NIS 收敛质量检测。`newvision` 的 EKF 也应输出类似健康状态。

- `sp_vision_25-main/tasks/auto_aim/tracker.cpp:180`  
  `lost/detecting/tracking/temp_lost/switching` 状态机。

- `sp_vision_25-main/tasks/auto_aim/tracker.cpp:231`  
  针对平衡步兵、前哨站、基地、普通车使用不同半径、装甲板数量和初始协方差。

- `FYT2024_vision-main/rm_auto_aim/armor_solver/src/armor_tracker.cpp:47`  
  EKF 初始化：选择离图像中心最近的装甲板作为初始目标。

- `FYT2024_vision-main/rm_auto_aim/armor_solver/src/armor_tracker.cpp:78`  
  观测关联：同 id、预测位置距离、yaw 差门限。

- `FYT2024_vision-main/rm_auto_aim/armor_solver/src/armor_tracker.cpp:151`  
  `DETECTING/TRACKING/TEMP_LOST/LOST` 状态机。

- `FYT2024_vision-main/rm_auto_aim/armor_solver/src/armor_tracker.cpp:187`  
  由单块装甲板反推整车中心和半径，初始化 EKF 状态。

- `FYT2024_vision-main/rm_auto_aim/armor_solver/src/armor_tracker.cpp:206`  
  装甲板跳变处理，包括 yaw 跳变、半径切换、高度差修正和状态纠偏。

- `SHtech_auto_aim-ax650-dev-2026/predict/Tracker.cpp:106`  
  多模型 tracker：装甲板模型、整车模型、yaw KF、跳变检测、模型信任度和前哨站特判。

### 9.7 选板与小陀螺

- `jlu_vision_26-master/src/auto_aim/armor_tracker/README.md:13`  
  选板核心思想：先用目标中心距离估算飞行时间，再选择最面对装甲板；锁定后迭代飞行时间，不在迭代中反复换板。

- `jlu_vision_26-master/src/auto_aim/armor_tracker/README.md:19`  
  45 度附近切板死区，避免观测噪声导致帧间反复切板。

### 9.8 弹道、延迟与 Planner

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:27`  
  同济 planner 主流程：检查弹速、估算飞行时间、预测 target、生成参考轨迹、TinyMPC 求解、输出控制与开火。

- `sp_vision_25-main/readme.md:300`  
  轨迹规划器设计说明：小陀螺切板会让射击轨迹不连续，因此需要提前减速，使规划后轨迹满足云台最大加速度限制。

- `sp_vision_25-main/readme.md:305`  
  对比两种实现：TinyMPC 隐式搜索加速度序列；五次多项式显式搜索过渡时间，只生成切板过渡段。README 明确方案二仅仿真验证，未上车。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:96`  
  按目标角速度选择高/低速延迟时间，再预测到未来时刻。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:110`  
  yaw 轴 TinyMPC 设置：`A/B` 矩阵、`Q/R`、最大角加速度约束。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:133`  
  pitch 轴 TinyMPC 设置。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:156`  
  `aim()`：选择最近装甲板，计算 yaw，调用弹道模型得到 pitch。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:179`  
  未来参考轨迹生成：前后窗口采样、目标预测、有限差分得到 yaw/pitch 速度。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:58`  
  当前上车实现把参考射击轨迹送入 yaw/pitch 两个 TinyMPC solver，得到加速度受限的规划轨迹；五次多项式构思应替换这里的求解器，而不是替换 `aim()` 或弹道。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:87`  
  火控依据未来偏移：比较射击轨迹和规划轨迹在 `shoot_offset` 后的误差，而不是只看当前角度误差。五次多项式方案也应沿用这个 fire decision。

- `sp_vision_25-main/tasks/auto_aim/planner/planner.cpp:117`  
  TinyMPC 的状态模型是 `[angle, angular_velocity]`，控制量为角加速度，`max_yaw_acc/max_pitch_acc` 是核心约束。五次多项式搜索 `T` 时也应检查同一约束。

- `SHtech_auto_aim-ax650-dev-2026/planner/Planner.cpp:118`  
  多策略 planner 入口：根据目标旋转速度和模型状态选择不同瞄准策略。

- `SHtech_auto_aim-ax650-dev-2026/planner/Planner.cpp:166`  
  装甲板模型预测：处理延迟、通信延迟、飞行时间相加后预测装甲板位置。

- `SHtech_auto_aim-ax650-dev-2026/planner/Planner.cpp:213`  
  整车模型预测：估算平均半径、预测最近装甲板、构建 MPC 参考轨迹。

- `talos_2026-master/src/fcs/L4_planning/trajectory_builder.cpp:38`  
  对角度序列做归一化有限差分，得到 yaw 速度。

- `talos_2026-master/src/fcs/L4_planning/trajectory_builder.cpp:79`  
  构建参考轨迹：采样未来/过去窗口，保存 yaw、pitch、distance、time_of_flight 和选中装甲板。

### 9.9 火控

- `Climber_Vision_26-main/tasks/auto_aim/shooter.cpp:19`  
  火控门控：必须有控制命令、有目标、允许自动开火；距离分段阈值；命令突变和云台未跟上时禁止射击。

- `talos_2026-master/src/fcs/L5_weapon/fire_decision.hpp:43`  
  把物理可击打窗口转换成 yaw/pitch 角度阈值，并设置最小阈值。适合 `newvision::FireDecision`。

### 9.10 因子图与高级估计

- `jlu_vision_26-master/src/auto_aim/armor_tracker/README.md:7`  
  重投影误差约束的因子图整车观测器说明。该方案能降低 PnP 抖动，但数值敏感，建议在 EKF、回放和日志成熟后再作为高级增强。

### 9.11 rm.cv.fans 专项参考

- `rm.cv.fans-main/README.md`  
  项目总览，列出了 LMTD 运动方程、Water-gun 火控、精细延迟匹配、运行时热更新参数、自主弹道校正、通用 EKF 等亮点。

- `rm.cv.fans-main/aimer/auto_aim/README.md`  
  更新日志里记录了大量实战问题：远距离 PnP 方差、半径估计不稳、陀螺仪 pitch 漂移、控制和发射延迟分离、运行时参数、距离相关观测噪声等。适合作为排雷清单。

- `rm.cv.fans-main/docs/auto_aim/pnp.md`  
  PnP 距离方差随距离增长的推导。`newvision` 的 EKF 观测噪声可参考它按距离放大。

- `rm.cv.fans-main/docs/auto_aim/latency.md`  
  延迟拆分文档：`img/predict/send/control/fire/hit`。这是 `newvision::LatencyCompensator` 最值得参考的语义定义。

- `rm.cv.fans-main/docs/auto_aim/latency.md:7`  
  延迟时间点定义表。`img` 是曝光中点，`predict` 是神经网络和预测器预处理完成后开始运动解算的时间点，`send` 是准备发信号的时间点，`control/fire/hit` 分别对应电控执行、弹丸发射和击中。

- `rm.cv.fans-main/docs/auto_aim/latency.md:34`  
  water-gun 假设：`control_to_fire` 存在几十毫秒，但瞄准预测中希望 `control` 时刻发出的弹流命中目标，因此 `prediction_time` 忽略 `control_to_fire`。

- `rm.cv.fans-main/docs/auto_aim/latency.md:42`  
  理想弹道重现：电控回传发射 id 后，用信号对应的图像时间和 `get_img_to_fire_latency()` 推真实发射时间，再找当时对应的控制命令。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:22`  
  线性空气阻力弹道残差 `ResistanceFuncLinear`。它用 Ceres 求解发射角，后期可作为 `BallisticSolver` 的阻力模型参考。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:109`  
  `img_to_predict` 的可测实现：`predict_timestamp_binder[frame] - img_t`，并用最小/最大阈值 clamp，异常时退回默认值。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:119`  
  `predict_to_send` 的估计实现：从滤波器按当前图像时间预测该段延迟，而不是直接取单帧抖动值。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:123`  
  `send_to_control` 和 `control_to_fire` 当前来自参数配置。代码里保留了从 `robot_status.latency_cmd_to_fire` 获取真实发射延迟的注释，说明这段最好由电控反馈标定。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:137`  
  `fire_delay` 当前按 `aim_xyz_i_barrel.norm() / bullet_speed` 计算；后续可替换成带空气阻力弹道飞行时间。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:143`  
  `get_img_to_prediction_latency()` 组合 `img_to_predict + predict_to_send + send_to_control + fire_delay`，刻意不含 `control_to_fire`。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:155`  
  `get_img_to_hit_latency()` 组合真实单发击中时间，额外包含 `control_to_fire`。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:209`  
  记录 predict 时间戳；`catch_send_timestamp()` 用发送时间减预测时间，滤波估计 `predict_to_send_latency`。`newvision` 后期可以把这类统计写入 telemetry。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:240`  
  火控误差从角度转换为目标平面偏差，并考虑装甲板宽高和倾斜角。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:319`  
  `aim_cmp`：优先非 `IDLE`，再按云台摆动成本选择目标。适合做 `TargetSelector` 的 tie-breaker。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:327`  
  相机坐标和 IMU 坐标互转，明确每帧图像和四元数绑定。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:474`  
  相机目标坐标转换到枪口中心坐标。这里的思想是：先处理相机到枪口平移，再解弹道，不要直接用 yaw/pitch 补偿替代几何关系。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:500`  
  `target_pos_to_shoot_param()`：在枪口坐标中用空气阻力模型求发射角，输出 `ShootParam`。

- `rm.cv.fans-main/aimer/base/robot/coord_converter.cpp:539`  
  迭代预测命中时间：用预测位置反复更新飞行时间，得到更一致的 hit/prediction time。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/armor_model.cpp:19`  
  单装甲板滤波更新，观测噪声中的 distance 项按 `distance^2` 放大。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/armor_model.cpp:82`  
  按装甲板 id 管理多个 `FilterThread`，并用 `credit_time` 淘汰消失目标。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/armor_model.cpp:123`  
  正向瞄准策略：选择面积足够大的活跃装甲板，输出 `TRACKING` 或 `SHOOT_NOW`。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/armor_model.cpp:171`  
  守株待兔/被动瞄准策略：分离跟随范围和开火范围。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/lmtd_top_model.cpp:11`  
  LMTD 观测模型：由整车状态推导装甲板 ypd 和 yaw。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/lmtd_top_model.cpp:35`  
  由 LMTD 状态计算装甲板位置和速度，状态包含中心、速度、yaw、yaw_rate、半径。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/lmtd_top_model.cpp:134`  
  直接可击打装甲板筛选：按装甲板法向和相机方向夹角判断是否正对。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/lmtd_top_model.cpp:173`  
  间接瞄准：高速旋转时等待下一块装甲板进入可击打角度。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/top_model.cpp:56`  
  `radial_armor_corners()`：给定装甲板 PnP 位置、装甲板类型、pitch 和候选朝向角，按整车径向几何生成该板 3D 角点。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/top_model.cpp:92`  
  `radial_double_pts()`：双板可见时分别投影左板和右板，右板朝向使用 `z_to_l + π/2`。这说明它默认普通 4 板车相邻装甲板夹角为 90 度；泛化到 3 板/不同兵种时应改为 `2π / armor_count`。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/top_model.cpp:136`  
  `DoubleCost`：把两块板各自 4 点重投影代价相加，用 8 个角点共同约束一个整车朝向变量。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/top_model.cpp:190`  
  `fit_double_z_to_l()`：用三分法在可行角度范围内寻找最小重投影代价，而不是做无约束 6DoF PnP。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/lmtd_top_model.cpp:786`  
  LMTD 观测构造：单板时调用 `fit_single_z_to_v()`，双板时调用 `fit_double_z_to_l()`，并根据装甲板数量设置相邻夹角。

- `rm.cv.fans-main/aimer/auto_aim/predictor/motion/lmtd_top_model.cpp:819`  
  双板朝向观测使用 `r.orientation-yaw.double`，单板使用 `r.orientation-yaw.single`。参数中双板噪声更小，表示双板朝向更可信但仍是带噪观测。

- `rm.cv.fans-main/assets/base.param.toml:187`  
  LMTD 参数给出 `r.orientation-yaw.single = 30.0`、`r.orientation-yaw.double = 15.0`，可作为 `newvision` 初始观测噪声比例参考。

- `rm.cv.fans-main/aimer/auto_aim/predictor/enemy/enemy_state.cpp:10`  
  按敌方类型定义装甲板尺寸、pitch、装甲板数量、默认半径和火控误差参数。

- `rm.cv.fans-main/aimer/auto_aim/predictor/enemy/enemy_state.cpp:79`  
  `EnemyState::update()`：过滤不可信装甲板、前哨站修正、平衡步兵判断、最终排序。

- `rm.cv.fans-main/aimer/auto_aim/predictor/enemy/enemy_state.cpp:175`  
  过滤新旧装甲板：面积比例、上一帧最近距离、最大有效距离。

- `rm.cv.fans-main/base/param/parameter.cpp:67`  
  参数文件每秒重载，发现变化后更新共享参数对象并打印变化。可作为 `newvision` 后期热更新参考。

- `rm.cv.fans-main/aimer/auto_aim/predictor/aim/aim_corrector.cpp:35`  
  `AimHistory`：记录 aim id、图像时间、延迟和瞄准参数。

- `rm.cv.fans-main/aimer/auto_aim/predictor/aim/aim_corrector.cpp:90`  
  `ProjectileSimulator`：按发射时间模拟带空气阻力的弹丸位置和图像投影圆。

- `rm.cv.fans-main/aimer/auto_aim/predictor/aim/aim_corrector.cpp:199`  
  通过电控回传的发射 id 匹配历史 aim，反推发射时刻并创建弹丸模拟器。适合后期做自动落点校正。

- `rm.cv.fans-main/aimer/auto_aim/predictor/aim/aim_corrector.cpp:226`  
  发射 id 回放时用 `origin_aim.img_t + control_to_fire_latency` 找真实发射时刻对应的控制 aim，体现 `prediction_time` 与真实 `hit_time` 的区别。

### 9.12 AUTO_AIM_SURVEY.md 横向调研参考

- `AUTO_AIM_SURVEY.md:3`  
  总览结论：长期方案应以稳定数据闭环为核心，主循环保持纯 C++，ROS2/Foxglove/Web/PlotJuggler 只做旁路调试和可视化。

- `AUTO_AIM_SURVEY.md:13`  
  各项目架构特点横向总结。适合在选型时快速判断 Talos、awakening、sp_vision、Climber、HUST、rm.cv.fans、JLU、SHtech、FYT、rm_vision_core 分别该参考哪一部分。

- `AUTO_AIM_SURVEY.md:145`  
  识别技术横向对比：网络关键点检测作为主线，传统灯条作为 ROI 细化、过曝兜底、低置信度复核和模型故障兜底。

- `AUTO_AIM_SURVEY.md:166`  
  跟踪、滤波与预测对比：推荐从 PnP + adaptive EKF 演进到整车状态、重投影/灯条观测更新，再考虑因子图或复杂模型。

- `AUTO_AIM_SURVEY.md:204`  
  弹道、延迟与火控对比：强调 `t_exposure` 到 `t_hit_est` 的分段记录，以及命中时刻枪口坐标目标点的语义。

- `AUTO_AIM_SURVEY.md:225`  
  推荐方案总览：L1-L5 数据契约、多后端识别、整车滤波、轻量 IMM、延迟分段和 replay/debug 字段。

- `AUTO_AIM_SURVEY.md:338`  
  调试与回放方案：列出了 image、detection、PnP、tracking、prediction、ballistics、fire、system 的必备日志字段。

- `AUTO_AIM_SURVEY.md:355`  
  分阶段路线：可回放基础闭环、多后端识别与传统兜底、整车模型与反陀螺、延迟分段与弹道在线校正、MPC 和系统可视化。

## 10. 代码索引的推荐阅读顺序

如果目标是尽快把 `newvision` 写成可运行纯 C++ 自瞄，建议按以下顺序阅读和移植：

1. `AUTO_AIM_SURVEY.md` 总览和推荐方案：先明确整体路线、模块边界和日志字段。
2. FYT 检测与角点修正：`armor_detector.cpp`、`light_corner_corrector.cpp`。
3. Climber PnP 与重投影：`solver.cpp`。
4. FYT/sp_vision Tracker：状态机、EKF 初始化、观测关联、跳变处理。
5. sp_vision Planner：弹道、延迟、TinyMPC 参考轨迹。
6. Talos/Climber 火控：物理窗口阈值、命令突变抑制。
7. JLU 选板：锁定装甲板、切板死区、反复换板抑制。
8. rm.cv.fans 延迟和枪口坐标弹道：先吸收语义，再拆成 `LatencyCompensator` 和 `BallisticSolver`。
9. SHtech Pipeline：在单线程闭环稳定后，再拆成纯 C++ 多线程流水线。
