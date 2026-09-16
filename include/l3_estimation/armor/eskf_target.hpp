#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/tracking/association.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/uvl_measure.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/filter/error_state_ekf.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/types.hpp>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

// 误差状态整车目标：一辆车的状态、它的迭代 ESEKF，以及把观测关联到这辆车上
// 的那几步。
namespace L3Estimation {

// 整车 ESEKF 的全部旋钮：过程噪声在 VehicleModel::NoiseConfig 里，这里是观测
// 噪声、关联门限和初值。
struct EskfTargetConfig
{
  VehicleModel::NoiseConfig noise{};
  // 装甲板物理尺寸，决定灯条端点的三维坐标。
  ArmorConfig armor{};

  // 每次更新做几步高斯牛顿。观测模型强非线性（透视除法 + 畸变 + atan2），
  // 迭代比单次线性化明显更稳。
  int iteration_num{5};

  // UVL 观测噪声。位置和长度的 sigma 按灯条像素长度成比例给（乘上这两个
  // 系数），角度的 sigma 取常数。灯条长度与距离成反比，σ 正比于长度相当于让
  // 物理尺度上的观测噪声近似恒定，不用按距离另外调权。
  double sigma_pixel_by_length{0.2};
  double sigma_length_by_length{0.5};
  double sigma_angle{0.1};

  // 灯条中心在垂直于灯条方向上的 sigma，单位 px，不随灯条长度缩放；取 <= 0
  // 时这一维退回用上面的各向同性写法。
  //
  // 分开给是因为中心误差本身是各向异性的：沿灯条方向端点是亮度渐变、定位差
  // 且误差随长度缩放；垂直方向是陡峭边缘、定位好。合成一个值会同时高估垂直
  // 方向、低估沿灯条方向，而垂直方向恰恰承载左右灯条间距，是 UVL 里深度的
  // 主要线索。3 m 处实测：各向同性时间距 sigma 4.52 px（深度不确定 8.3%），
  // 取 1.5 px 后降到 1.50 px（2.8%）。
  double sigma_perp_px{-1.0};

  // 独立灯条那几条观测的 sigma 统一乘上它。这些灯条没配成完整板、只靠几何
  // 关联，没有数字和板型证据，应当比从装甲板拆出来的灯条更不可信。
  double isolated_light_sigma_scale{1.0};

  // 单块完整板时那一维深度差观测的 sigma（米），以及独立灯条关联的三道门：
  // 长度比、角度差、中心距离（按灯条长度归一）。
  double armor_lights_depth_diff_sigma{0.1};
  bool enable_lights_measure{true};
  double light_match_length_ratio_gate{0.2};
  double light_match_angle_gate{0.2};
  double light_match_pos_gate_by_length_ratio{5.0};

  // 装甲板关联的代价权重与门限，代价怎么算见 matchArmor。
  double match_gate{200.0};
  // 还没见过 0 号以外的板时改用这个更宽的门限：那时整车 yaw、第二组半径和
  // 高度差几乎不可观测，其余板的位置全是初值猜的，门限太紧第二块板进不来。
  double match_gate_not_all_init{1000.0};
  double weight_center_error{5.0};
  double weight_angle_error{10.0};
  double weight_side_length_error{1.0};

  // reset() 用的初始半径先验，按目标类型取。
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

  // 状态维度，下游遥测和单测按这个数读状态。
  static constexpr int kStateSize = VehicleModel::kStateSize;

  EskfTarget() = default;

  // 直接按给定构型填名义状态：旋转中心放在 (x, 0, 0)，半径、yaw、角速度、
  // 高度差和平移速度由参数给，板数从 name 推出。
  //
  // 这个入口不建滤波器，只供 L4/L5/L6 在合成目标上做单测和离线回放；要真跑
  // 滤波走 reset()。
  EskfTarget(
    ArmorName name, double x, double vyaw, double radius, double yaw = 0.0,
    double height_offset = 0.0,
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero(),
    EskfTargetConfig config = {});

  // 用第一块观测到的板初始化：由板的世界系位姿沿板法向退一个半径反推旋转
  // 中心，其余状态取先验，然后建立滤波器。
  //
  // camera_in_world 必须是该帧曝光时刻的相机光学系位姿。
  void reset(
    const Armor & armor, const EskfTargetConfig & config, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world);

  // 把滤波器推进到 timestamp，dt 取与上一帧的实际间隔，掉帧和耗时抖动自然
  // 被吸收。
  void predictEkf(TimePoint timestamp);

  // 把本帧的候选板关联到整车的各块物理板上：对每个 (观测, 板编号) 组合，把
  // 该板按当前状态投影出四个角点，与观测角点比中心、角度和边长，加权成一个
  // 代价，再用 greedyMatch 在门限内贪心配对。返回 (物理板编号, 观测) 对。
  std::vector<std::pair<int, Armor>> matchArmor(
    const std::vector<Armor> & armors, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // 把 ROI 里单独检出的灯条关联到整车的某根物理灯条上：预测各灯条的端点，
  // 按长度比、角度差、中心距离三道门筛，通过的记下 (板编号, 左右)。
  // 本帧一块完整板都没关联上时直接返回空。
  std::vector<MatchedLight> matchLight(
    const std::vector<L2Perception::Light>& lights,
    const std::vector<std::pair<int, Armor>>& matched_armors,
    TimePoint timestamp, const L1Sensor::CameraCalibration& calibration,
    const Eigen::Isometry3d& camera_in_world) const;

  // 把关联好的板各拆成左右两条 UVL 观测，做一次多观测更新，返回观测条数。
  int update(
    const std::vector<std::pair<int, Armor>> & matched, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world);

  // 完整更新入口：完整板各拆成两条 UVL，独立灯条各加一条（sigma 乘上放大
  // 系数），matched 恰好一块且传入了深度差时再加一维 depth_diff 观测，最后
  // 一起送进迭代更新。返回观测条数。
  int update(
    const std::vector<std::pair<int, Armor>>& matched,
    const std::vector<MatchedLight>& matched_lights,
    const std::optional<double>& lights_depth_diff,
    TimePoint timestamp, const L1Sensor::CameraCalibration& calibration,
    const Eigen::Isometry3d& camera_in_world);

  // 由状态投影出某块板某条灯条的上下端点像素坐标。关联、ROI 和叠加层都用它。
  std::pair<cv::Point2f, cv::Point2f> predictLight(
    int id, bool is_left, const Eigen::VectorXd & state,
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // --- L3 → L4 契约 -------------------------------------------------------

  // 对外的状态向量。第 8、9 维吐的是线性半径，不是内部的对数表示：L4 按
  // 半径的物理量纲做判断，log 表示不能泄漏出去。
  Eigen::VectorXd ekf_x() const;
  const State & rawState() const noexcept { return x_; }
  TimePoint t() const noexcept { return t_; }
  int armor_num() const noexcept;

  // 在副本上按运动模型外推，不动滤波器状态。
  void predict(double dt);
  void predict(TimePoint timestamp);

  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // 任一半径跑出物理范围就算发散，调用方据此丢弃目标。
  bool diverged() const;

  // 最近一次更新的归一化创新平方（NIS）和它的自由度。UVL 的观测维数随本帧
  // 关联到的灯条条数变（每条 4 维），所以自由度要一并给出才能和卡方门限比。
  // 滤波器一致时 NIS 的期望值等于自由度。
  double lastNis() const noexcept { return last_nis_; }
  int lastNisDof() const noexcept { return last_nis_dof_; }

  // 最近一次更新的 UVL 残差，按分量分组取 RMS。四个分量量纲不同（角度是
  // rad、中心和长度是 px），混进一个范数没有意义，所以分开给。
  //
  // 读法：中心残差大是整车位置偏了，长度残差大是深度偏了，角度残差大是姿态
  // 偏了。
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

  // 由相机标定的 camera -> barrel 外参和当帧枪管姿态合成相机光学系在世界系
  // 的位姿。世界系原点取枪管原点，与 PnpSolver 的约定一致。
  static Eigen::Isometry3d cameraInWorld(
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Quaterniond & q_world_barrel);
  bool converged() const noexcept { return converged_; }

  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  // 是否关联到过 0 号以外的板。粘滞，一旦为真就不再复位。为 false 时整车
  // yaw、第二组半径和高度差几乎不可观测，L4 只瞄正在观测的那块板。
  bool jumped{false};
  int last_id{0};

  bool initialized() const noexcept { return initialized_; }

  // 前哨转向投票器的当前判定，给遥测和调试看。
  VehicleModel::Voter::Direction outpostDirection() const noexcept
  {
    return voter_.direction;
  }
  bool lightsEnabled() const noexcept
  {
    return config_.enable_lights_measure;
  }

  // 不含滤波器的轻量副本，下游随便外推都不会污染滤波器状态。
  EskfTarget snapshot() const;

private:
  // 拼一个 UvlContext：板编号、左右、板数、板几何加当帧相机位姿与内参。
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
  // 前哨转向投票器。非前哨目标上它一直停在 Collecting，不影响推进。
  VehicleModel::Voter voter_{};
};

}  // namespace L3Estimation
