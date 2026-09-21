#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/armor/armor_observation.hpp"
#include "l3_estimation/armor/light_residual.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/types.hpp>

#include <optional>
#include <vector>

// 误差状态整车目标：一辆车的状态，加它的迭代 ESEKF。
//
// 这个类只负责"状态怎么走、观测怎么吸收"。把观测挂到哪块板上是 armor_matcher
// 的事，观测怎么投影是 armor_observation 的事，残差怎么读是 light_residual 的
// 事——分开之后这里从头到尾只有一条线：reset → predictEkf → update。
namespace L3Estimation {

// 整车 ESEKF 的全部旋钮：过程噪声在 VehicleModel::NoiseConfig 里，这里是观测
// 噪声、关联门限和初值。
struct EskfTargetConfig
{
  VehicleModel::NoiseConfig noise{};
  // 装甲板物理尺寸，决定灯条端点的三维坐标。
  ArmorConfig armor{};

  // 每次更新做几步高斯牛顿。观测模型强非线性（透视除法 + 畸变），迭代比单次
  // 线性化明显更稳。
  int iteration_num{8};

  // 端点观测噪声，单位 px。每个端点的误差拆成沿灯条、垂直灯条两个方向，
  // sigma 取「系数 × 灯条像素长度」与 sigma_min_px 的较大值。垂直方向决定
  // 灯条倾角（σ_角 = √2·σ⊥ / 长度），近正对时整车 yaw 主要靠它观测。
  //
  // 默认两个系数为 0、下限 √40 ≈ 6.32，即 rmcs_auto_aim_v2 的各向同性常数
  // R = 40 px²。四段 3 m 录像（灯条 25–34 px）上它比按旧 UVL 边缘方差换算的
  // 0.16 / 0.10 × 长度更好，在 3m_high、3m_run_mid、fast_run 上后者反而不如
  // 改动前的 UVL；垂直系数再减半（更信倾角）四段都更差。
  //
  // 按长度缩放的依据是长度与距离成反比，σ 正比于长度相当于物理尺度上的噪声
  // 近似恒定；沿灯条取大是因为端点落在亮度渐变的灯条两头。录像都在 3 m，
  // 常数与按长度缩放在这个距离上分不出来，换距离后要重新比。
  double sigma_along_by_length{0.0};
  double sigma_perp_by_length{0.0};
  double sigma_min_px{6.32};


  // 独立灯条的两个 sigma 再乘上它。这些灯条没配成完整板、只靠几何关联，
  // 没有数字和板型证据，应当比从装甲板拆出来的灯条更不可信。
  double isolated_light_sigma_scale{1.4};

  // 单块完整板时那一维深度差观测的 sigma（米）。
  double armor_lights_depth_diff_sigma{0.1};

  // 侧边灯条（L2 在跟踪 ROI 里找到、不属于任何检出装甲板的灯条）的关联门限。
  // 这些灯条没有类别证据，错配比漏配代价高得多，所以几道门都是硬拒绝：
  //   长度比、角度差  与预测灯条比；
  //   卡方门限        先验点上的马氏距离 rᵀS⁻¹r，S = H·P·Hᵀ + R，4 自由度；
  //                   13.28 是 99% 分位。它随距离和滤波器不确定度自动缩放，
  //                   比固定像素门限严：预测越准，放进来的范围越小；
  //   require_jumped  见过 0 号以外的板之后才启用，在那之前整车 yaw、第二组
  //                   半径和高度差几乎不可观测，邻板灯条的预测位置是猜的。
  bool enable_lights_measure{true};
  double light_match_length_ratio_gate{0.2};
  double light_match_angle_gate{0.2};
  double light_match_chi2_gate{13.28};
  bool light_match_require_jumped{true};


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
  using Filter = VehicleFilter;
  using State = Eigen::Matrix<double, VehicleModel::kStateSize, 1>;
  // 诊断量已经搬到 light_residual.hpp，别名留着是因为 track_diag 按
  // EskfTarget::LightResidual::Channel 写死了列。
  using LightResidual = L3Estimation::LightResidual;

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

  // --- 一帧的三步：reset 或 predictEkf，然后 update ----------------------

  // 用第一块观测到的板初始化：由板的世界系位姿沿板法向退一个半径反推旋转
  // 中心，其余状态取先验，然后建立滤波器。
  void reset(const Armor & armor, const EskfTargetConfig & config, TimePoint timestamp);

  // 把滤波器推进到 timestamp，dt 取与上一帧的实际间隔，掉帧和耗时抖动自然
  // 被吸收。
  //
  // 给了 hold_from 时，运动只积分到这一刻，之后位置和姿态原地不动，协方差仍按
  // 完整的 dt 增长。长时间没有观测时，匀速外推出来的位置比"停在最后看到的
  // 地方"更不可信：速度和角速度只要有一点偏差（错关联一次就够），外推一秒
  // 就能转出去半圈。
  void predictEkf(TimePoint timestamp, std::optional<TimePoint> hold_from = std::nullopt);

  // 吸收本帧的全部观测：完整板各拆成左右两根灯条，独立灯条各加一根（sigma
  // 乘上放大系数），每根一个四维端点观测；matched 恰好一块且传入了深度差时
  // 再加一维 depth_diff 观测，最后一起送进迭代更新。返回观测块数。
  int update(
    const std::vector<MatchedArmor> & matched,
    const std::vector<MatchedLight> & matched_lights,
    const std::optional<double> & lights_depth_diff, TimePoint timestamp,
    const ObsContext & ctx);

  // 只有完整板可用时的简写。
  int update(
    const std::vector<MatchedArmor> & matched, TimePoint timestamp, const ObsContext & ctx)
  {
    return update(matched, {}, std::nullopt, timestamp, ctx);
  }

  // --- 关联要用的只读接口 -------------------------------------------------

  // 本帧的观测上下文：这辆车的几何配上当帧的相机位姿和内参。
  ObsContext obsContext(
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // 按运动模型外推到 timestamp 的状态副本，滤波器不动。
  //
  // 一帧里关联的每一步都该用同一份，不要各自再外推一次：Motion 在 dt=0 时也会
  // 跑 clamp（前哨半径钉死、超界的高度差和角速度归零），反复外推不是恒等。
  Eigen::VectorXd stateAt(TimePoint timestamp) const;

  // 把一根灯条的上下端点做成一个四维观测，sigma 按本目标的配置算。关联门限和
  // 更新共用它，门限看到的 S 就是更新时真正用的。
  VehicleObs lightObs(
    const ObsContext & ctx, const cv::Point2f & top, const cv::Point2f & bottom, int id,
    bool is_left, bool isolated) const;

  // 这个观测落在滤波器先验上的马氏距离平方。没有滤波器或 S 不正定时返回空。
  std::optional<double> mahalanobis(const VehicleObs & obs) const;

  const EskfTargetConfig & config() const noexcept { return config_; }
  // 合成目标不带滤波器，只能外推不能更新。
  bool hasFilter() const noexcept { return filter_.has_value(); }

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

  // --- 诊断量，取自最近一次 update -----------------------------------------

  // 归一化创新平方（NIS）和它的自由度，取先验点上的创新量。观测维数随本帧
  // 关联到的灯条根数变（每根 4 维），所以自由度要一并给出才能和卡方门限比。
  // 滤波器一致时 NIS 的期望值等于自由度。
  double lastNis() const noexcept { return last_nis_; }
  int lastNisDof() const noexcept { return last_nis_dof_; }
  // 端点创新按物理通道的分解，含义见 LightResidual。
  const LightResidual & lastLightResidual() const noexcept { return last_light_residual_; }

  // --- 身份与状态标志 -----------------------------------------------------

  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  // 是否关联到过 0 号以外的板。粘滞，一旦为真就不再复位。为 false 时整车
  // yaw、第二组半径和高度差几乎不可观测，L4 只瞄正在观测的那块板。
  bool jumped{false};
  int last_id{0};

  bool initialized() const noexcept { return initialized_; }
  bool converged() const noexcept { return converged_; }

  // 前哨转向投票器的当前判定，给遥测和调试看。
  VehicleModel::Voter::Direction outpostDirection() const noexcept { return voter_.direction; }
  bool lightsEnabled() const noexcept { return config_.enable_lights_measure; }

  // 不含滤波器的轻量副本，下游随便外推都不会污染滤波器状态。
  EskfTarget snapshot() const;

private:
  // 一根灯条的两个 sigma（沿灯条、垂直灯条），单位 px。
  std::pair<double, double> lightSigma(double length, bool isolated) const;

  EskfTargetConfig config_{};

  State x_{State::Zero()};
  TimePoint t_{};

  std::optional<Filter> filter_;
  bool initialized_{false};
  bool converged_{false};
  int update_count_{0};
  double last_nis_{0.0};
  int last_nis_dof_{0};
  LightResidual last_light_residual_{};
  // 前哨转向投票器。非前哨目标上它一直停在 Collecting，不影响推进。
  VehicleModel::Voter voter_{};
};

}  // namespace L3Estimation
