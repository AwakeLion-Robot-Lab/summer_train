#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Geometry>

#include <memory>
#include <optional>
#include <vector>

namespace L3Estimation {

// L2 -> L3 的显式转换入口：只复制检测字段，位姿由各后端内部的 PnpSolver 补充。
// 两个后端走的是同一条 L2 -> L3 转换，所以它留在共享层。
[[nodiscard]] Armor toArmorObservation(
  const L2Perception::Armor& detection,
  TimePoint timestamp);

// L3 的入口接口。一帧的完整流程是
//   L2 检测 -> 逐板 PnP -> 候选排序 -> 关联/初始化 -> 整车状态估计 -> 状态机
// 中间那一段"怎么估"由具体后端决定（filter_est 的整车 EKF，或 gtsam_est 的
// 因子图），但两端的契约是共享的：输入 L2 检测，输出 TrackedTarget 的副本。
//
// 之所以做成虚接口而不是编译期二选一：回放工具要能靠一个命令行开关切后端来
// 对比同一段数据，运行期切换比重新编译一次快得多。虚调用的代价是每帧一次，
// 相对于一次 PnP yaw 搜索可以忽略。
class ITracker {
public:
  virtual ~ITracker() = default;

  ITracker() = default;
  ITracker(const ITracker&) = delete;
  ITracker& operator=(const ITracker&) = delete;
  ITracker(ITracker&&) = delete;
  ITracker& operator=(ITracker&&) = delete;

  // 相机标定无效时对象仍可构造，但 ready() 为 false，track() 只返回空。
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual TrackState state() const noexcept = 0;
  // 实际生效的后端，供遥测标注这一段数据是谁估出来的。
  [[nodiscard]] virtual EstimatorBackend backend() const noexcept = 0;

  // 处理一帧检测。枪管姿态必须对应 timestamp 所表示的图像曝光时刻。
  // Lost 或初始化失败时返回空；TempLost 时返回纯预测状态。
  [[nodiscard]] virtual std::optional<TrackedTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp) = 0;

  // 本帧全部 PnP 结果（含被丢弃的），只读调试快照。
  [[nodiscard]] virtual const std::vector<Armor>& observations() const noexcept = 0;

  // 当前整车模型展开的装甲板 [x, y, z, yaw]，Lost 时为空。
  [[nodiscard]] virtual std::vector<Eigen::Vector4d> targetArmorPoses() const = 0;

  // 清空状态机、时间连续性记录和本帧观测。
  virtual void reset() noexcept = 0;
};

// 按后端枚举构造 Tracker。标定无效不是错误（返回的对象 ready() 为 false），
// 但**请求一个没编进来的后端是错误**：选了 gtsam 却没开 xmake 的 use_gtsam 时
// 直接抛 std::runtime_error，而不是悄悄回退到 EKF——回退会让你以为在验证因子图，
// 其实读的是卡尔曼的曲线。
[[nodiscard]] std::unique_ptr<ITracker> makeTracker(
  EstimatorBackend backend,
  const L1Sensor::CameraCalibration& calibration,
  ArmorConfig armor_config = {},
  TrackerConfig tracker_config = {});

// 本次构建实际编进来的后端。gtsam 是否可用取决于 xmake f --use_gtsam=y。
[[nodiscard]] bool estimatorBackendAvailable(EstimatorBackend backend) noexcept;

}  // namespace L3Estimation
