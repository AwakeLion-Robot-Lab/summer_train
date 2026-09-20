#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/tracking/association.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/light_measure.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/filter/error_state_ekf.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/types.hpp>

#include <memory>
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

// matchLight 各道门毙掉了多少根侧边灯条，累计值。门限只看最终采纳数是调不
// 动的：采纳数为零时，不知道是候选板槽位根本没开出来，还是某一道门收太紧。
struct LightMatchStats
{
  // 整帧没进关联：开关关着、目标是基地、本帧没关联上完整板，或 jumped 未满足。
  std::size_t frames_skipped{0};
  // 进了关联但一个候选灯条槽位都没开出来：能看见的板本帧都已配成完整板，
  // 或邻板背对相机。这时侧边灯条本来就无处可去，不算被门毙掉。
  std::size_t frames_no_candidate{0};
  // 开出来的候选灯条槽位总数，每帧 0~4 个。除以"进了关联且有槽位的帧数"就是
  // 每帧平均有几个位置能接侧边灯条，也就是采纳数的天花板——一个槽位最多收一
  // 根。参与率低的时候先看它：槽位本来就只有一个的话，再松门限也多不出来。
  std::size_t slots{0};
  // 逐 (灯条, 候选槽位) 对的计数，下面几项按门的先后顺序互斥累加。
  std::size_t considered{0};
  std::size_t reject_length{0};
  std::size_t reject_angle{0};
  std::size_t reject_chi2{0};
  std::size_t passed{0};
  // 贪心配对之后真正返回的根数，必然不大于 passed。
  std::size_t matched{0};
  // 至少采纳了一根侧边灯条的帧数。
  std::size_t frames_matched{0};
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
  //
  // 给了 hold_from 时，运动只积分到这一刻，之后位置和姿态原地不动，协方差仍按
  // 完整的 dt 增长。长时间没有观测时，匀速外推出来的位置比"停在最后看到的
  // 地方"更不可信：速度和角速度只要有一点偏差（错关联一次就够），外推一秒
  // 就能转出去半圈。
  void predictEkf(TimePoint timestamp, std::optional<TimePoint> hold_from = std::nullopt);

  // 把本帧的候选板关联到整车的各块物理板上：对每个 (观测, 板编号) 组合，把
  // 该板按当前状态投影出四个角点，与观测角点比中心、角度和边长，加权成一个
  // 代价，再用 greedyMatch 在门限内贪心配对。返回 (物理板编号, 观测) 对。
  std::vector<std::pair<int, Armor>> matchArmor(
    const std::vector<Armor> & armors, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // 把侧边灯条关联到整车的某根物理灯条上。候选只取最正对的那块板及其两块
  // 邻板靠近它的那根灯条，已配成完整板的板和背对相机的板不参与；按长度比、
  // 角度差、卡方三道门筛，通过的按马氏距离贪心配对，记下 (板编号, 左右)。
  // 本帧一块完整板都没关联上、或 require_jumped 时还没见过别的板，直接返回空。
  //
  // 卡方门限用滤波器当前的先验协方差，调用前应已 predictEkf(timestamp)。
  //
  // stats 非空时逐道门累加拒绝数，供 track_diag 打印；不影响关联结果。
  std::vector<MatchedLight> matchLight(
    const std::vector<L2Perception::Light>& lights,
    const std::vector<std::pair<int, Armor>>& matched_armors,
    TimePoint timestamp, const L1Sensor::CameraCalibration& calibration,
    const Eigen::Isometry3d& camera_in_world,
    LightMatchStats* stats = nullptr) const;

  // 把关联好的板各拆成左右两根灯条的端点观测，做一次多观测更新，返回观测
  // 块数。
  int update(
    const std::vector<std::pair<int, Armor>> & matched, TimePoint timestamp,
    const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world);

  // 完整更新入口：完整板各拆成两根灯条，独立灯条各加一根（sigma 乘上放大
  // 系数），每根一个四维端点观测；matched 恰好一块且传入了深度差时再加一维
  // depth_diff 观测，最后一起送进迭代更新。返回观测块数。
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

  // 最近一次更新的归一化创新平方（NIS）和它的自由度，取先验点上的创新量。
  // 观测维数随本帧关联到的灯条根数变（每根 4 维），所以自由度要一并给出才能
  // 和卡方门限比。滤波器一致时 NIS 的期望值等于自由度。
  double lastNis() const noexcept { return last_nis_; }
  int lastNisDof() const noexcept { return last_nis_dof_; }

  // 最近一次更新的端点创新量（先验点上），投到每根灯条自己的坐标系里分方向
  // 单位 px（倾角为 rad）。
  //
  // 读法：沿灯条分量大，多半是深度（灯条长度）或高度偏了；垂直分量大，是
  // 横向位置或灯条倾角（姿态）偏了。
  struct LightResidual
  {
    // 一个通道的统计量。mean 有符号，看的是系统偏差；rms 看的是噪声。两者
    // 必须分开：只看 RMS 分不出「检测器有偏」和「检测器抖」，而这两种病的
    // 治法完全不同——前者要在检测侧修，后者才该动 R。
    struct Channel
    {
      double mean{0.0};
      double rms{0.0};
    };

    // 把每根灯条两个端点的残差 r_top / r_bot 投到灯条方向 e 和法向 n 上，
    // 再折成四个互相正交的物理通道。L 为该灯条的像素长度：
    //   shift_perp  = (r_top·n + r_bot·n) / 2   整根灯条横向平移，px
    //   shift_along = (r_top·e + r_bot·e) / 2   整根灯条沿自身平移，px
    //   tilt        = (r_top·n − r_bot·n) / L   灯条倾角，rad
    //   length      = (r_bot·e − r_top·e)       灯条长度，px
    // 拆成这四维是因为它们互相正交、各自有物理意义：哪一维偏了直接对应
    // 哪个环节有问题。端点级的平方和把符号吃掉，看不出偏差。
    Channel shift_perp{};
    Channel shift_along{};
    Channel tilt{};
    Channel length{};

    // 端点级的 RMS，单位 px。等价于上面四个通道的重新组合（沿灯条方向有
    // along_rms² = shift_along.rms² + length.rms²/4），保留是因为历史 A/B
    // 记录用的就是这两个数。
    double along_rms_px{0.0};
    double perp_rms_px{0.0};
    // 单板深度差观测的残差，单位米。本帧没有该观测时为 0。
    double depth_diff_m{0.0};
    // 参与本次更新的灯条根数（完整板拆出的 + 独立的）。
    int light_count{0};
  };
  const LightResidual & lastLightResidual() const noexcept { return last_light_residual_; }

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
  // 拼一个 LightContext：板编号、左右、板数、板几何加当帧相机位姿与内参。
  LightContext makeContext(
    int id, bool is_left, const L1Sensor::CameraCalibration & calibration,
    const Eigen::Isometry3d & camera_in_world) const;

  // 把一根灯条的上下端点做成一个四维观测，R 由 lightCov 按灯条方向写出。
  // isolated 表示侧边灯条，两个 sigma 再乘 isolated_light_sigma_scale。关联门限
  // 和更新共用它，门限看到的 S 就是更新时真正用的。
  std::shared_ptr<Filter::ObsBase> lightObs(
    const cv::Point2f & top, const cv::Point2f & bottom, int id, bool is_left,
    bool isolated, const L1Sensor::CameraCalibration & calibration,
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
  LightResidual last_light_residual_{};
  // 前哨转向投票器。非前哨目标上它一直停在 Collecting，不影响推进。
  VehicleModel::Voter voter_{};
};

}  // namespace L3Estimation
