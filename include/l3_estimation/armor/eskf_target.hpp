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

namespace L3Estimation {

/**
 * @brief 整车 ESKF 的全部旋钮
 *
 * 过程噪声在 VehicleModel::NoiseConfig 里，这里是观测噪声、关联门限和初值。
 */
struct EskfTargetConfig
{
  /// 过程噪声
  VehicleModel::NoiseConfig noise{};

  /// 装甲板物理尺寸，决定灯条端点的三维坐标。
  ArmorConfig armor{};

  /// 每次更新做几步高斯牛顿。观测模型强非线性（透视除法 + 畸变），迭代比单次
  /// 线性化明显更稳。
  int iteration_num{8};

  /// 端点观测噪声沿灯条方向的系数，单位 px / px。
  ///
  /// 每个端点的误差拆成沿灯条、垂直灯条两个方向，sigma 取「系数 × 灯条像素长度」
  /// 与 sigma_min_px 的较大值。垂直方向决定灯条倾角（σ_角 = √2·σ⊥ / 长度），
  /// 近正对时整车 yaw 主要靠它观测。
  ///
  /// 默认两个系数为 0、下限 √40 ≈ 6.32，即 rmcs_auto_aim_v2 的各向同性常数
  /// R = 40 px²。四段 3 m 录像（灯条 25–34 px）上它比按旧 UVL 边缘方差换算的
  /// 0.16 / 0.10 × 长度更好，在 3m_high、3m_run_mid、fast_run 上后者反而不如
  /// 改动前的 UVL；垂直系数再减半（更信倾角）四段都更差。
  ///
  /// 按长度缩放的依据是长度与距离成反比，σ 正比于长度相当于物理尺度上的噪声
  /// 近似恒定；沿灯条取大是因为端点落在亮度渐变的灯条两头。录像都在 3 m，
  /// 常数与按长度缩放在这个距离上分不出来，换距离后要重新比。
  double sigma_along_by_length{0.0};

  /// 端点观测噪声垂直灯条方向的系数，见 sigma_along_by_length
  double sigma_perp_by_length{0.0};

  /// 端点观测噪声的下限，px，见 sigma_along_by_length
  double sigma_min_px{6.32};


  /// 独立灯条的两个 sigma 再乘上它。这些灯条没配成完整板、只靠几何关联，
  /// 没有数字和板型证据，应当比从装甲板拆出来的灯条更不可信。
  double isolated_light_sigma_scale{1.4};

  /// 单块完整板时那一维深度差观测的 sigma（米）。
  double armor_lights_depth_diff_sigma{0.1};

  /// 是否启用侧边灯条观测。
  ///
  /// 侧边灯条（L2 在跟踪 ROI 里找到、不属于任何检出装甲板的灯条）的关联门限。
  /// 这些灯条没有类别证据，错配比漏配代价高得多，所以几道门都是硬拒绝：
  ///   长度比、角度差  与预测灯条比；
  ///   卡方门限        先验点上的马氏距离 rᵀS⁻¹r，S = H·P·Hᵀ + R，4 自由度；
  ///                   13.28 是 99% 分位。它随距离和滤波器不确定度自动缩放，
  ///                   比固定像素门限严：预测越准，放进来的范围越小；
  ///   require_jumped  见过 0 号以外的板之后才启用，在那之前整车 yaw、第二组
  ///                   半径和高度差几乎不可观测，邻板灯条的预测位置是猜的。
  bool enable_lights_measure{true};

  /// 侧边灯条长度门限：|实测长度 − 预测长度| 超过预测长度的这么多倍就拒绝
  double light_match_length_ratio_gate{0.2};

  /// 侧边灯条与预测灯条的角度差门限，rad
  double light_match_angle_gate{0.2};

  /// 侧边灯条的卡方门限，4 自由度
  double light_match_chi2_gate{13.28};

  /// 是否要求 jumped 之后才关联侧边灯条
  bool light_match_require_jumped{true};


  /// 装甲板关联的代价门限，代价怎么算见 matchArmor。
  double match_gate{200.0};

  /// 还没见过 0 号以外的板时改用这个更宽的门限：那时整车 yaw、第二组半径和
  /// 高度差几乎不可观测，其余板的位置全是初值猜的，门限太紧第二块板进不来。
  double match_gate_not_all_init{1000.0};

  /// 像素代价过不了门时的第二条路：两根灯条在滤波器先验上的马氏距离之和
  /// 低于它也放行，0 表示不启用。像素门限不看不确定度——车体倾斜估偏 10°，
  /// 1.6 m 处刚转进视野的邻板就偏 50 px 被拒，目标随之进 TempLost、被备用槽
  /// 用单板新建的目标顶掉。20.09 是 8 自由度的 99% 分位；两根灯条共享状态、
  /// 误差正相关，直接相加偏大，是偏严的近似。
  double match_chi2_gate{20.09};

  /// 装甲板关联代价里中心距离的权重
  double weight_center_error{5.0};

  /// 装甲板关联代价里角度差的权重
  double weight_angle_error{10.0};

  /// 装甲板关联代价里边长差的权重
  double weight_side_length_error{1.0};

  /// reset() 用的初始半径先验，按目标类型取。这是普通车的，meter。
  double initial_radius{0.26};

  /// 前哨站的初始半径，meter
  double initial_radius_outpost{0.2765};

  /// 基地的初始半径，meter
  double initial_radius_base{0.3205};
};

/**
 * @brief 误差状态整车目标：一辆车的状态，加它的迭代 ESEKF
 *
 * 这个类只负责"状态怎么走、观测怎么吸收"。把观测挂到哪块板上是 armor_matcher
 * 的事，观测怎么投影是 armor_observation 的事，残差怎么读是 light_residual 的
 * 事——分开之后这里从头到尾只有一条线：reset → predictEkf → update。
 */
class EskfTarget
{
public:
  /// 整车滤波器
  using Filter = VehicleFilter;

  /// 名义状态向量
  using State = Eigen::Matrix<double, VehicleModel::kStateSize, 1>;

  /// 诊断量已经搬到 light_residual.hpp，别名留着是因为 track_diag 按
  /// EskfTarget::LightResidual::Channel 写死了列。
  using LightResidual = L3Estimation::LightResidual;

  /// 状态维度，下游遥测和单测按这个数读状态。
  static constexpr int kStateSize = VehicleModel::kStateSize;

  /**
   * @brief 空目标，未初始化、不带滤波器
   */
  EskfTarget() = default;

  /**
   * @brief 直接按给定构型填名义状态
   *
   * 旋转中心放在 (x, 0, 0)，板数从 name 推出。这个入口不建滤波器，只供
   * L4/L5/L6 在合成目标上做单测和离线回放；要真跑滤波走 reset()。构造出的目标
   * 直接视为已初始化、已收敛、已 jumped。
   *
   * @param name 车辆类别
   * @param x 旋转中心的 x 坐标，meter
   * @param vyaw 绕车体 z 轴的角速度，rad/s
   * @param radius 半径，饱和到物理范围；非前哨两组半径都取它，meter
   * @param yaw 整车 yaw，rad
   * @param height_offset 奇偶板的高度差，meter；前哨不用
   * @param velocity 车心平移速度，m/s
   * @param config 滤波器配置
   */
  EskfTarget(
    ArmorName name, double x, double vyaw, double radius, double yaw = 0.0,
    double height_offset = 0.0,
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero(),
    EskfTargetConfig config = {});

  // --- 一帧的三步：reset 或 predictEkf，然后 update ----------------------

  /**
   * @brief 用第一块观测到的板初始化
   *
   * 由板的世界系位姿沿板法向退一个半径反推旋转中心，其余状态取先验，然后建立
   * 滤波器。
   *
   * @param armor 已解出世界系位姿的装甲板
   * @param config 滤波器配置
   * @param timestamp 这块板的曝光时刻
   */
  void reset(const Armor & armor, const EskfTargetConfig & config, TimePoint timestamp);

  /**
   * @brief 把滤波器推进到 timestamp
   *
   * dt 取与上一帧的实际间隔，掉帧和耗时抖动自然被吸收。
   *
   * 给了 hold_from 时，运动只积分到这一刻，之后位置和姿态原地不动，协方差仍按
   * 完整的 dt 增长。长时间没有观测时，匀速外推出来的位置比"停在最后看到的
   * 地方"更不可信：速度和角速度只要有一点偏差（错关联一次就够），外推一秒
   * 就能转出去半圈。
   *
   * @param timestamp 本帧曝光时刻
   * @param hold_from 运动外推的截止时刻，空表示一直外推
   */
  void predictEkf(TimePoint timestamp, std::optional<TimePoint> hold_from = std::nullopt);

  /**
   * @brief 吸收本帧的全部观测
   *
   * 完整板各拆成左右两根灯条，独立灯条各加一根（sigma 乘上放大系数），每根一个
   * 四维端点观测；matched 恰好一块且传入了深度差时再加一维 depth_diff 观测，
   * 最后一起送进迭代更新。
   *
   * @param matched 关联上的完整装甲板
   * @param matched_lights 关联上的侧边灯条
   * @param lights_depth_diff 单块完整板时的左右灯条深度差，meter
   * @param timestamp 本帧曝光时刻
   * @param ctx 本帧的观测上下文
   * @return 观测块数；0 表示没有更新
   */
  int update(
    const std::vector<MatchedArmor> & matched,
    const std::vector<MatchedLight> & matched_lights,
    const std::optional<double> & lights_depth_diff, TimePoint timestamp,
    const ObsContext & ctx);

  /**
   * @brief 只有完整板可用时的简写
   * @param matched 关联上的完整装甲板
   * @param timestamp 本帧曝光时刻
   * @param ctx 本帧的观测上下文
   * @return 观测块数
   */
  int update(
    const std::vector<MatchedArmor> & matched, TimePoint timestamp, const ObsContext & ctx)
  {
    return update(matched, {}, std::nullopt, timestamp, ctx);
  }

  // --- 关联要用的只读接口 -------------------------------------------------

  /**
   * @brief 本帧的观测上下文：这辆车的几何配上当帧的相机位姿和内参
   * @param calibration 相机标定，取内参与畸变
   * @param camera_in_world 曝光时刻相机光学系在世界系的位姿
   * @return 观测上下文
   */
  ObsContext obsContext(
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  /**
   * @brief 按运动模型外推到 timestamp 的状态副本，滤波器不动
   *
   * 一帧里关联的每一步都该用同一份，不要各自再外推一次：Motion 在 dt=0 时也会
   * 跑 clamp（前哨半径钉死、超界的高度差和角速度归零），反复外推不是恒等。
   *
   * @param timestamp 外推到的时刻
   * @return 外推后的状态
   */
  Eigen::VectorXd stateAt(TimePoint timestamp) const;

  /**
   * @brief 把一根灯条的上下端点做成一个四维观测，sigma 按本目标的配置算
   *
   * 关联门限和更新共用它，门限看到的 S 就是更新时真正用的。
   *
   * @param ctx 本帧的观测上下文
   * @param top 检测到的上端点
   * @param bottom 检测到的下端点
   * @param id 物理板编号
   * @param is_left true 为左灯条
   * @param isolated true 为独立灯条，sigma 乘上 isolated_light_sigma_scale
   * @return 观测块
   */
  VehicleObs lightObs(
    const ObsContext & ctx, const cv::Point2f & top, const cv::Point2f & bottom, int id,
    bool is_left, bool isolated) const;

  /**
   * @brief 这个观测落在滤波器先验上的马氏距离平方
   * @param obs 观测块
   * @return rᵀS⁻¹r；没有滤波器或 S 不正定时返回空
   */
  std::optional<double> mahalanobis(const VehicleObs & obs) const;

  /**
   * @brief 滤波器配置
   * @return 配置
   */
  const EskfTargetConfig & config() const noexcept { return config_; }

  /**
   * @brief 是否带滤波器；合成目标不带滤波器，只能外推不能更新
   * @return 带滤波器返回 true
   */
  bool hasFilter() const noexcept { return filter_.has_value(); }

  // --- L3 → L4 契约 -------------------------------------------------------

  /**
   * @brief 对外的状态向量
   *
   * 第 8、9 维吐的是线性半径，不是内部的对数表示：L4 按半径的物理量纲做判断，
   * log 表示不能泄漏出去。
   *
   * @return 十三维状态，下标见 VehicleModel::idx
   */
  Eigen::VectorXd ekf_x() const;

  /**
   * @brief 内部名义状态，半径为对数表示
   * @return 名义状态
   */
  const State & rawState() const noexcept { return x_; }

  /**
   * @brief 状态对应的时刻
   * @return 时刻
   */
  TimePoint t() const noexcept { return t_; }

  /**
   * @brief 物理装甲板数量
   * @return 板数；类别未知时按 4
   */
  int armor_num() const noexcept;

  /**
   * @brief 在副本上按运动模型外推，不动滤波器状态
   * @param dt 外推时长，s
   */
  void predict(double dt);

  /**
   * @brief 在副本上按运动模型外推到 timestamp，不动滤波器状态
   * @param timestamp 外推到的时刻
   */
  void predict(TimePoint timestamp);

  /**
   * @brief 各块装甲板在世界系的位姿
   * @return 每块板一个 [x, y, z, yaw]，下标即物理板编号；未初始化时为空
   */
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  /**
   * @brief 任一半径跑出物理范围或状态非有限就算发散，调用方据此丢弃目标
   * @return 发散返回 true
   */
  bool diverged() const;

  // --- 诊断量，取自最近一次 update -----------------------------------------

  /**
   * @brief 归一化创新平方（NIS），取先验点上的创新量
   *
   * 滤波器一致时 NIS 的期望值等于自由度。
   *
   * @return NIS
   */
  double lastNis() const noexcept { return last_nis_; }

  /**
   * @brief NIS 的自由度
   *
   * 观测维数随本帧关联到的灯条根数变（每根 4 维），所以自由度要一并给出才能和
   * 卡方门限比。
   *
   * @return 自由度
   */
  int lastNisDof() const noexcept { return last_nis_dof_; }

  /**
   * @brief 端点创新按物理通道的分解，含义见 LightResidual
   * @return 通道分解
   */
  const LightResidual & lastLightResidual() const noexcept { return last_light_residual_; }

  // --- 身份与状态标志 -----------------------------------------------------

  /// 车辆类别
  ArmorName name{ArmorName::Unknown};

  /// 是否关联到过 0 号以外的板。粘滞，一旦为真就不再复位。为 false 时整车
  /// yaw、第二组半径和高度差几乎不可观测，L4 只瞄正在观测的那块板。
  bool jumped{false};

  /// 最近一次关联上的物理板编号
  int last_id{0};

  /**
   * @brief 是否已经由 reset() 或合成构造填过状态
   * @return 已初始化返回 true
   */
  bool initialized() const noexcept { return initialized_; }

  /**
   * @brief 滤波器是否已收敛
   * @return 累计吸收的观测块超过 20 且未发散后返回 true
   */
  bool converged() const noexcept { return converged_; }

  /**
   * @brief 是否启用侧边灯条观测
   * @return config.enable_lights_measure
   */
  bool lightsEnabled() const noexcept { return config_.enable_lights_measure; }

  /**
   * @brief 不含滤波器的轻量副本，下游随便外推都不会污染滤波器状态
   * @return 状态与诊断量的副本
   */
  EskfTarget snapshot() const;

private:
  /**
   * @brief 一根灯条的两个 sigma
   * @param length 灯条像素长度
   * @param isolated true 为独立灯条
   * @return (沿灯条, 垂直灯条)，px
   */
  std::pair<double, double> lightSigma(double length, bool isolated) const;

  /// 滤波器配置
  EskfTargetConfig config_{};

  /// 名义状态，半径为对数表示
  State x_{State::Zero()};

  /// 状态对应的时刻
  TimePoint t_{};

  /// 迭代 ESEKF；合成目标和快照不带
  std::optional<Filter> filter_;

  /// 是否已填过状态
  bool initialized_{false};

  /// 是否已收敛
  bool converged_{false};

  /// 累计吸收的观测块数，用于判断收敛
  int update_count_{0};

  /// 最近一次更新的 NIS
  double last_nis_{0.0};

  /// 最近一次更新的 NIS 自由度
  int last_nis_dof_{0};

  /// 最近一次更新的端点创新通道分解
  LightResidual last_light_residual_{};

  /// 前哨转向投票器。非前哨目标上它一直停在 Collecting，不影响推进。
  VehicleModel::Voter voter_{};
};

}  // namespace L3Estimation
