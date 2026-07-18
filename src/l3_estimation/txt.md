可以。最适合的是让 TargetEstimator 成为整个 L3 的唯一门面，Runtime 不直接接触 PnpSolver、坐标转换、重投影 yaw 或 EKF。

  Runtime 最终只保留：

  auto armors = armor_detector.detect(frame);

  std::erase_if(armors, [&state](const auto& armor) {
    return !isEnemyArmor(armor.color, state.enemy_color);
  });

  auto vehicles = target_estimator.update(armors, timestamp);
  auto plan = planner.plan(vehicles, state);

  整体结构：

  Runtime
    → TargetEstimator::update(armors, timestamp)
        → PnP
        → 坐标转换
        → 重投影 yaw
        → 装甲面关联
        → Tracker/EKF
    ← VehicleState[]
    → L4 Planner

  ## 1. L3 对外只暴露一个入口

  建议接口：

  class TargetEstimator {
  public:
    TargetEstimator(
      L1Sensor::CameraCalibration calibration,
      CameraExtrinsics extrinsics,
      GimbalPoseProvider pose_provider);

    [[nodiscard]] std::vector<VehicleState> update(
      const std::vector<L2Perception::ArmorDetection>& armors,
      std::chrono::steady_clock::time_point timestamp);

  private:
    PnpSolver pnp_solver_;
    CoordinateTransformer coordinate_transformer_;
    YawOptimizer yaw_optimizer_;

    std::unordered_map<int, VehicleTracker> trackers_;

    GimbalPoseProvider pose_provider_;
  };

  Runtime 只知道：

  输入 ArmorDetection[]
  输出 VehicleState[]

  其他全部封装在 L3。

  ## 2. 为什么只传 armors 和 timestamp

  L3 做世界坐标转换还需要图像时刻的云台姿态。

  为了让 Runtime 接口保持：

  update(armors, timestamp)

  可以在构造 TargetEstimator 时注入一个姿态查询函数：

  using TimePoint = std::chrono::steady_clock::time_point;

  using GimbalPoseProvider =
    std::function<std::optional<Eigen::Quaterniond>(TimePoint)>;

  Runtime 初始化：

  L3Estimation::TargetEstimator target_estimator(
    *camera->calibration(),
    camera_extrinsics,
    [&serial](auto timestamp) {
      return serial.gimbalPoseAt(timestamp);
    });

  这样 L3 内部可以：

  const auto gimbal_pose = pose_provider_(timestamp);

  Runtime 每帧不需要自己处理姿态，只传图像时间戳。

  这也避免让 TargetEstimator 直接依赖整个 SerialWorker 类。

  ## 3. L3 输出给 L4 的消息

  建议使用 VehicleState：

  #pragma once

  #include <Eigen/Core>

  #include <chrono>

  namespace L3Estimation {

  enum class TrackerState {
    Lost,
    Detecting,
    Tracking,
    TemporaryLost
  };

  struct VehicleState {
    int robot_id = -1;

    Eigen::Vector3d center_world{};
    Eigen::Vector3d velocity_world{};

    double yaw = 0.0;
    double yaw_rate = 0.0;
    double radius = 0.0;

    int armor_count = 4;

    TrackerState tracker_state = TrackerState::Lost;
    double tracking_quality = 0.0;

    std::chrono::steady_clock::time_point timestamp{};
  };

  }  // namespace L3Estimation

  返回：

  std::vector<VehicleState>

  而不是：

  std::optional<TargetState>

  因为：

  - L3 负责维护所有车辆
  - L4 负责决定最终打哪辆车
  - L3 不应该提前进行战术目标选择

  ## 4. TargetEstimator::update() 内部逻辑

  std::vector<VehicleState> TargetEstimator::update(
    const std::vector<L2Perception::ArmorDetection>& armors,
    TimePoint timestamp)
  {
    const auto gimbal_pose = pose_provider_(timestamp);
    if (!gimbal_pose) {
      return {};
    }

    std::vector<ArmorObservation> observations;
    observations.reserve(armors.size());

    for (const auto& armor : armors) {
      const auto size = armorSizeFromClass(armor.class_id);

      const auto pose = pnp_solver_.solve(armor, size);
      if (!pose) {
        continue;
      }

      const auto world_pose =
        coordinate_transformer_.transform(*pose, *gimbal_pose);

      const auto yaw_result = yaw_optimizer_.optimize(
        armor,
        *pose,
        world_pose,
        size);

      observations.push_back({
        .robot_id = robotIdFromClass(armor.class_id),
        .armor_class =
          L2Perception::armorClassFromId(armor.class_id),
        .position_world = world_pose.position,
        .yaw_raw_world = yaw_result.raw_yaw,
        .yaw_world = yaw_result.optimized_yaw,
        .confidence = armor.confidence,
        .reprojection_error_px =
          yaw_result.reprojection_error_px,
        .timestamp = timestamp
      });
    }

    predictAllTrackers(timestamp);
    updateTrackers(observations, timestamp);
    removeExpiredTrackers(timestamp);

    return collectVehicleStates(timestamp);
  }

  Runtime 完全不关心这些步骤。

  ## 5. 当前阶段的简化实现

  现在还没有坐标转换、yaw 优化和 Tracker，可以让 TargetEstimator 先只跑到 PnP：

  std::vector<ArmorPose> TargetEstimator::update(
    const std::vector<ArmorDetection>& armors,
    TimePoint timestamp)
  {
    (void)timestamp;

    std::vector<ArmorPose> poses;

    for (const auto& armor : armors) {
      const auto size = armorSizeFromClass(armor.class_id);
      const auto pose = pnp_solver_.solve(armor, size);

      if (pose) {
        poses.push_back(*pose);
      }
    }

    return poses;
  }

  但这个返回类型以后会变化。更推荐从一开始就固定最终接口：

  std::vector<VehicleState> update(...);

  尚未实现 Tracker 时，可以暂时构造一个简化 VehicleState，或者内部逐阶段开发、等 L3 能提供车辆状态后再正式接入 Runtime。

  ## 6. 推荐的文件结构

  推荐压缩成 4 个头文件、3 个实现文件。

  include/l3_estimation/
  ├── types.hpp
  ├── pnp_solver.hpp
  ├── target_estimator.hpp
  └── ekf_tracker.hpp

  src/l3_estimation/
  ├── pnp_solver.cpp
  ├── target_estimator.cpp
  └── ekf_tracker.cpp

  ## 1. types.hpp

  集中放 L3 消息类型，不再分别创建多个消息头文件：

  #pragma once

  #include "l2_perception/armor.hpp"

  #include <Eigen/Core>
  #include <opencv2/core.hpp>

  #include <chrono>

  namespace L3Estimation {

  using TimePoint = std::chrono::steady_clock::time_point;

  enum class ArmorSize {
    Small,
    Large
  };

  struct ArmorPose {
    cv::Vec3d rvec{};
    cv::Vec3d tvec{};
  };

  struct ArmorObservation {
    int robot_id = -1;
    L2Perception::ArmorClass armor_class =
      L2Perception::ArmorClass::Unknown;

    Eigen::Vector3d position_world{};

    double yaw_raw_world = 0.0;
    double yaw_world = 0.0;

    float confidence = 0.0F;
    double reprojection_error_px = 0.0;

    TimePoint timestamp{};
  };

  enum class TrackerState {
    Lost,
    Detecting,
    Tracking,
    TemporaryLost
  };

  struct VehicleState {
    int robot_id = -1;

    Eigen::Vector3d center_world{};
    Eigen::Vector3d velocity_world{};

    double yaw = 0.0;
    double yaw_rate = 0.0;
    double radius = 0.0;

    TrackerState tracker_state = TrackerState::Lost;
    TimePoint timestamp{};
  };

  }  // namespace L3Estimation

  这样所有 L3 模块统一引用：

  #include "l3_estimation/types.hpp"

  ## 2. pnp_solver

  只负责：

  ArmorDetection → ArmorPose

  class PnpSolver {
  public:
    std::optional<ArmorPose> solve(
      const L2Perception::ArmorDetection& armor,
      ArmorSize size) const;
  };

  文件：

  pnp_solver.hpp
  pnp_solver.cpp

  ## 3. target_estimator

  它是 Runtime 唯一使用的 L3 入口。

  内部负责：

  大小装甲板判断
  → 调用 PnpSolver
  → camera → world
  → 重投影 yaw
  → 生成 ArmorObservation
  → 调用 EkfTracker
  → 输出 VehicleState

  class TargetEstimator {
  public:
    std::vector<VehicleState> update(
      const std::vector<L2Perception::ArmorDetection>& armors,
      TimePoint timestamp);

  private:
    ArmorSize armorSizeFromClass(int class_id) const;

    std::optional<ArmorObservation> makeObservation(
      const L2Perception::ArmorDetection& armor,
      TimePoint timestamp);

    double optimizeYaw(
      const L2Perception::ArmorDetection& armor,
      const ArmorPose& pose) const;

    PnpSolver pnp_solver_;
    std::unordered_map<int, EkfTracker> trackers_;
  };

  坐标转换和重投影 yaw 先写成 TargetEstimator 的私有函数，不单独创建：

  coordinate_transformer.hpp/.cpp
  yaw_optimizer.hpp/.cpp
  reprojection_error.hpp/.cpp

  等代码真的变复杂或需要复用时再拆。

  ## 4. ekf_tracker

  封装单辆车的滤波和生命周期：

  class EkfTracker {
  public:
    void predict(TimePoint timestamp);
    void update(const std::vector<ArmorObservation>& observations);

    [[nodiscard]] VehicleState state() const;
    [[nodiscard]] bool expired(TimePoint now) const;

  private:
    VehicleState state_;
  };

  第一版甚至可以先不实现完整 EKF，只完成：

  - 保存最新观测
  - 简单速度计算
  - 丢失计数
  - 输出 VehicleState

  以后再替换内部实现，外部接口不变。

  ## Runtime 最终代码

  Runtime 只需要包含：

  #include "l3_estimation/target_estimator.hpp"

  每帧：

  auto armors = armor_detector.detect(frame);

  std::erase_if(armors, [&state](const auto& armor) {
    return !isEnemyArmor(armor.color, state.enemy_color);
  });

  const auto vehicles =
    target_estimator.update(armors, timestamp);

  const auto plan =
    planner.plan(vehicles, state);

  ## 推荐开发顺序

  1. types.hpp
  2. PnpSolver
  3. TargetEstimator 内先接通 PnP
  4. TargetEstimator 内加入坐标转换
  5. TargetEstimator 内加入重投影 yaw
  6. EkfTracker
  7. 多车辆 Tracker 管理

  这个方案的核心是：

  types.hpp          放消息
  pnp_solver         做几何求解
  ekf_tracker        做连续状态估计
  target_estimator   做 L3 总编排

  只有当 target_estimator.cpp 明显过长时，再拆坐标转换或 yaw 优化模块。