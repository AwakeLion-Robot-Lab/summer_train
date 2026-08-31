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
#include <tuple>
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

  // 灯条中心误差是强各向异性的：沿灯条方向端点是亮度渐变、定位差（σ∥，正比
  // 于灯条长度）；垂直方向是陡峭边缘、定位好（σ⊥，由边缘锐度决定，**不随
  // 灯条长度缩放**）。
  //
  // 把两者当成同一个值会同时犯两个错：垂直方向被高估（扔掉本来很准的横向
  // 信息），沿灯条方向被低估。而横向恰恰承载左右灯条间距——UVL 里深度的主要
  // 线索。3 m 处实测：各向同性时间距 sigma 4.52 px（深度不确定 8.3%），
  // 取 σ⊥=1.5 px 后降到 1.50 px（2.8%），深度精度差三倍。
  //
  // sigma_perp_px <= 0 时退回各向同性，与 awakening 原始写法一致。
  double sigma_perp_px{-1.0};

  // 独立灯条（未构成完整装甲板、纯几何关联）的噪声放大系数。它们没有编号和
  // 颜色证据支撑，理应比从装甲板拆出来的灯条更不可信。
  double isolated_light_sigma_scale{1.0};

  // 单完整板的左右灯条中心深度差 sigma，以及独立灯条关联参数。数值与
  // Awakening test.yaml 的 armor_tracker 节点一致。
  double armor_lights_depth_diff_sigma{0.1};
  bool enable_lights_measure{true};
  double light_match_length_ratio_gate{0.2};
  double light_match_angle_gate{0.2};
  double light_match_pos_gate_by_length_ratio{5.0};

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
  using MatchedLight = std::tuple<int, bool, L2Perception::Light>;

  // 状态维度。下游遥测与单测按这个数读状态。
  static constexpr int kStateSize = VehicleModel::kStateSize;

  EskfTarget() = default;

  // 构造指定构型的目标，用于无观测的确定性初始化（离线回放和单测）。
  // 板数由 name 推出，而不是再单独传一个——否则 name 和板数可以各说各话。
  // 旋转中心落在 (x, 0, 0)，yaw 放开成可选参数，否则测不到"整车转到某个
  // 角度"的构型。
  //
  // 这个入口不建滤波器：它只填名义状态，供 L4/L5/L6 在合成目标上做单测。
  // 需要真正跑滤波的场景走 reset()。
  EskfTarget(
    ArmorName name, double x, double vyaw, double radius, double yaw = 0.0,
    double height_offset = 0.0,
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero(),
    EskfTargetConfig config = {});

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

  // 将 ROI 内独立检出的灯条与当前最可能可见的物理灯条做严格几何关联。
  // 与 Awakening 一样，只有本帧至少关联到一块完整装甲板时才启用。
  std::vector<MatchedLight> matchLight(
    const std::vector<L2Perception::Light>& lights,
    const std::vector<std::pair<int, Armor>>& matched_armors,
    TimePoint timestamp, const L1Sensor::CameraCalibration& calibration,
    const Eigen::Isometry3d& camera_in_world) const;

  // 把关联好的板拆成灯条 UVL 观测并执行一次多观测更新。返回观测条数。
  int update(
    const std::vector<std::pair<int, Armor>> & matched, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world);

  // 完整 Awakening 更新入口：完整板拆成两条 UVL，独立灯条各加一条 UVL；当
  // matched 只有一块且 PnP 成功时，再加入一维 depth_difference。
  int update(
    const std::vector<std::pair<int, Armor>>& matched,
    const std::vector<MatchedLight>& matched_lights,
    const std::optional<double>& armor_lights_depth_difference,
    TimePoint timestamp, const L1Sensor::CameraCalibration& calibration,
    const Eigen::Isometry3d& camera_in_world);

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

  // 最近一次更新的归一化创新平方（NIS）及其自由度。UVL 的观测维数随本帧关联
  // 到的灯条条数变化（每条 4 维），所以自由度必须一并给出，否则没法和卡方
  // 门限比。滤波器一致时 NIS 期望值等于自由度。
  double lastNis() const noexcept { return last_nis_; }
  int lastNisDof() const noexcept { return last_nis_dof_; }

  // 最近一次更新的 UVL 残差，按物理含义分组聚合。四个观测分量量纲不同
  // （角度是 rad、中心和长度是 px），混在一个范数里没有意义，所以分开给。
  //
  // 诊断用途：中心残差大说明整车位置估计偏了，长度残差大说明深度偏了，
  // 角度残差大说明姿态偏了——三者能把"预测被什么带偏"分开。
  struct UvlResidual
  {
    double angle_rms_deg{0.0};
    double center_rms_px{0.0};
    double length_rms_px{0.0};
    // 单板深度差观测的残差，单位米。本帧没有该观测时为 0。
    double depth_diff_m{0.0};
    // 参与本次更新的灯条条数（完整板拆出的 + 独立的）。
    int light_count{0};
  };
  const UvlResidual & lastUvlResidual() const noexcept { return last_uvl_residual_; }

  // 由相机标定与枪管姿态算出相机光学系在世界系的位姿。
  // 世界系原点取枪管原点，与 PnpSolver 的约定一致。
  static Eigen::Isometry3d cameraInWorld(
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Quaterniond & q_world_barrel);
  bool converged() const noexcept { return converged_; }

  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  // 是否关联到过 0 号以外的板。**粘滞**：一旦为真不再复位。为 false 时整车
  // yaw、第二组半径与高度差几乎不可观测。
  bool jumped{false};
  int last_id{0};

  bool initialized() const noexcept { return initialized_; }

  // 前哨站转向投票器的当前判定，供遥测与调试观察。
  VehicleModel::Voter::Direction outpostDirection() const noexcept
  {
    return voter_.direction;
  }
  bool lightMeasurementsEnabled() const noexcept
  {
    return config_.enable_lights_measure;
  }

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
  double last_nis_{0.0};
  int last_nis_dof_{0};
  UvlResidual last_uvl_residual_{};
  // 前哨站转向投票。非前哨目标上它一直停在 Collecting，不影响推进。
  VehicleModel::Voter voter_{};
};

}  // namespace L3Estimation
