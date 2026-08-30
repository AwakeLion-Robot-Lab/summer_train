#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/armor/association.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/uvl_measure.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/error_state_ekf.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/types.hpp>

#include <optional>
#include <utility>
#include <vector>

// 误差状态整车目标。照搬 awakening 的 ArmorTarget，是 TrackedTarget 的 ESEKF
// 替代品，对外保持同一组 L3 → L4 契约方法。
namespace L3Estimation {

// 整车 ESEKF 的全部旋钮。噪声部分复用 VehicleModel::NoiseConfig。
struct EskfTargetConfig
{
  VehicleModel::NoiseConfig noise{};
  // 装甲板物理尺寸，决定灯条端点的三维坐标。
  ArmorConfig armor{};

  // 迭代 ESEKF 的高斯牛顿步数。观测模型强非线性（透视除法 + 畸变 + atan2），
  // 迭代收益明显。
  int iteration_num{5};

  // 观测噪声。位置与长度的 sigma **正比于灯条像素长度**：灯条长度与距离成
  // 反比，取 σ ∝ L 相当于让"物理尺度上的观测噪声"近似恒定，距离权重不用
  // 手调。角度的 sigma 取常数。
  double sigma_pixel_by_length{0.2};
  double sigma_length_by_length{0.5};
  double sigma_angle{0.1};

  // 装甲板关联的四边形代价权重与门限。见 matchArmor。
  double match_gate{200.0};
  // 还没见过 0 号以外的板时，整车 yaw、第二组半径、高度差几乎不可观测，
  // 其余板的位置全靠初值猜，所以门限要放宽让第二块板更容易进来。
  double match_gate_not_all_init{1000.0};
  double weight_center_error{5.0};
  double weight_angle_error{10.0};
  double weight_side_length_error{1.0};

  // 初始半径先验，按目标类型取。
  double initial_radius{0.26};
  double initial_radius_outpost{0.2765};
  double initial_radius_base{0.3205};
};

class EskfTarget
{
public:
  using Filter = ErrorStateEkf<VehicleModel::kStateSize, VehicleModel::Motion>;
  using State = Eigen::Matrix<double, VehicleModel::kStateSize, 1>;

  EskfTarget() = default;

  // 用首个装甲板观测反推旋转中心并初始化整个状态。
  //
  // camera_in_world 必须是**该帧曝光时刻**的相机光学系位姿。
  void reset(
    const Armor & armor, const EskfTargetConfig & config, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world);

  // 把滤波器推进到指定时刻。dt 取实际帧间隔而非标称值，掉帧与耗时抖动自然吸收。
  void predictEkf(TimePoint timestamp);

  // 关联候选装甲板。返回 (物理板编号, 观测) 对。
  std::vector<std::pair<int, Armor>> matchArmor(
    const std::vector<Armor> & armors, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // 把关联好的板拆成灯条 UVL 观测并执行一次多观测更新。返回观测条数。
  int update(
    const std::vector<std::pair<int, Armor>> & matched, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world);

  // 由状态预测某块板某条灯条的上下端点像素坐标。关联、ROI 与叠加层都走这个。
  std::pair<cv::Point2f, cv::Point2f> predictLight(
    int id, bool is_left, const Eigen::VectorXd & state,
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // --- L3 → L4 契约，与 TrackedTarget 同名同义 ---------------------------

  // 注意第 8、9 维对外吐的是**线性**半径而非对数：L4 的 planner.cpp:240 按
  // abs(x[8]) <= 2.0 判物理半径，内部的 log 表示不能泄漏出去。
  Eigen::VectorXd ekf_x() const;
  const State & rawState() const noexcept { return x_; }
  TimePoint t() const noexcept { return t_; }
  int armor_num() const noexcept;

  // 在副本上外推，不影响滤波器状态。
  void predict(double dt);
  void predict(TimePoint timestamp);

  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // 任一候选半径离开物理范围时认为发散。
  bool diverged() const;
  bool converged() const noexcept { return converged_; }

  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  // 是否关联到过 0 号以外的板。**粘滞**：一旦为真不再复位。为 false 时整车
  // yaw、第二组半径与高度差几乎不可观测。
  bool jumped{false};
  int last_id{0};

  bool initialized() const noexcept { return initialized_; }

  // 下游拿到的是不含滤波器的轻量副本：外推可以随便做，不会污染滤波器状态。
  EskfTarget snapshot() const;

private:
  UvlContext makeContext(
    int id, bool is_left, const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  EskfTargetConfig config_{};
  ArmorConfig armor_config_{};

  State x_{State::Zero()};
  TimePoint t_{};

  std::optional<Filter> filter_;
  bool initialized_{false};
  bool converged_{false};
  int update_count_{0};
};

}  // namespace L3Estimation
