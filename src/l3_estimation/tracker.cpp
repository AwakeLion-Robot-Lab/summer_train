#include "l3_estimation/tracker.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace L3Estimation {
namespace {

[[nodiscard]] bool validTrackerConfig(const TrackerConfig& config) noexcept
{
  return config.min_detect_count > 0 &&
         config.max_frame_interval > std::chrono::milliseconds::zero() &&
         config.max_temp_lost_count > 0 &&
         config.outpost_max_temp_lost_count > 0 &&
         config.outpost_max_temp_lost_count >= config.max_temp_lost_count;
}

[[nodiscard]] bool isBase(ArmorName name) noexcept
{
  return name == ArmorName::BaseSmall || name == ArmorName::BaseLarge;
}

[[nodiscard]] bool badRecentNis(const TrackedTarget& target)
{
  // 最近窗口中至少 40% 的更新超过 NIS 门限时认为滤波持续异常。
  const auto& ekf = target.ekf();
  const int failures = std::accumulate(
    ekf.recent_nis_failures.begin(),
    ekf.recent_nis_failures.end(),
    0);
  return static_cast<double>(failures) >=
    0.4 * static_cast<double>(ekf.window_size);
}

}  // namespace

Armor toArmorObservation(
  const L2Perception::Armor& detection,
  TimePoint timestamp)
{
  // 这里只搬运检测元数据；三维位姿由当前帧的 PnpSolver 计算。
  Armor observation;
  observation.class_id = detection.class_id;
  observation.points = detection.corners;
  observation.center = detection.center;
  observation.confidence = static_cast<double>(detection.confidence);
  observation.area = L6Telemetry::polygonArea(detection.corners);
  observation.timestamp = timestamp;
  return observation;
}

Tracker::Tracker(
  const L1Sensor::CameraCalibration& calibration,
  ArmorConfig armor_config,
  TrackerConfig tracker_config)
  : armor_config_(armor_config),
    tracker_config_(tracker_config),
    pnp_solver_(calibration, armor_config_),
    image_center_{
      static_cast<float>(calibration.image_size.width) * 0.5F,
      static_cast<float>(calibration.image_size.height) * 0.5F},
    ready_(pnp_solver_.ready() && validTrackerConfig(tracker_config_))
{
}

bool Tracker::ready() const noexcept
{
  return ready_;
}

TrackState Tracker::state() const noexcept
{
  return state_;
}

std::optional<TrackedTarget> Tracker::track(
  const std::vector<L2Perception::Armor>& detections,
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  TimePoint timestamp)
{
  observations_.clear();
  if (!ready_) {
    resetTracking();
    return std::nullopt;
  }

  // 每个检测独立生成观测，失败结果也保留在 observations_ 中供诊断。
  pnp_solver_.set_R_world_barrel(q_world_barrel);
  observations_.reserve(detections.size());
  for (const auto& detection : detections) {
    observations_.push_back(toArmorObservation(detection, timestamp));
    pnp_solver_.single_pnp(observations_.back());
  }

  // 单板结果全部就位后再做一遍双板联合 yaw。必须放在这里而不是循环内：配对
  // 需要同帧另一块板的世界系位置，而位置本身来自各自的单板 PnP。
  pnp_solver_.refine_double_armor(observations_);


  // 时间戳倒退或跟踪期间帧间隔过大时，旧运动状态不再可信。
  if (last_timestamp_) {
    const auto elapsed = timestamp - *last_timestamp_;
    if (elapsed <= TimePoint::duration::zero() ||
        (state_ != TrackState::Lost &&
         elapsed > tracker_config_.max_frame_interval)) {
      L6Telemetry::logWarn("Tracker reset after invalid frame interval");
      resetTracking();
    }
  }
  last_timestamp_ = timestamp;

  // 逐行对应 sp_vision 的主相机 Tracker::track：Lost 时从已排序观测
  // 中初始化，其余状态只关联当前车辆类别。sp 只在带全向感知队列的
  // 另一个重载中才会在 Tracking 期间按优先级强制切目标；本管线没有
  // 那路输入，不能把单帧分类抖动当成切车信号。
  auto armors = usableObservations();
  bool found = false;
  if (state_ == TrackState::Lost) {
    found = initializeTarget(armors, timestamp);
  } else {
    found = updateTarget(armors, timestamp);
  }

  updateState(found);

  // 状态机更新后统一检查物理半径和近期创新统计量。
  if (state_ != TrackState::Lost && target_ && target_->diverged()) {
    L6Telemetry::logDebug("[Tracker] Target diverged");
    resetTracking();
    return std::nullopt;
  }
  if (state_ != TrackState::Lost && target_ && badRecentNis(*target_)) {
    L6Telemetry::logDebug("[Target] Bad Converge Found");
    resetTracking();
    return std::nullopt;
  }

  if (state_ == TrackState::Lost || !target_) {
    return std::nullopt;
  }
  if (!target_->ekf_x().allFinite() || !target_->ekf().P.allFinite()) {
    L6Telemetry::logWarn("Tracker reset after non-finite EKF state");
    resetTracking();
    return std::nullopt;
  }

  return *target_;
}

const std::vector<Armor>& Tracker::observations() const noexcept
{
  return observations_;
}

std::vector<Eigen::Vector4d> Tracker::targetArmorPoses() const
{
  if (!target_ || state_ == TrackState::Lost) {
    return {};
  }
  return target_->armor_xyza_list();
}

void Tracker::reset() noexcept
{
  resetTracking();
  last_timestamp_.reset();
  observations_.clear();
}

bool Tracker::observationUsable(const Armor& armor) const noexcept
{
  // 唯一的门限是"single_pnp 是否成功提交了位姿"：name 只在提交的那一步才被
  // 赋值，任何失败路径上都保持 Unknown。
  //
  // sp_vision 连这一条都没有——它的 Solver::solve() 无条件写位姿，没有失败
  // 路径可言。这里保留它不是额外的质量门，而是因为 single_pnp 在标定缺失、
  // 曝光时刻姿态缺失、角点非有限时会提前返回，不筛掉就会把零位姿喂进 EKF。
  return armor.name != ArmorName::Unknown && armor.xyz_in_world.allFinite();
}

std::vector<const Armor*> Tracker::usableObservations() const
{
  std::vector<const Armor*> usable;
  usable.reserve(observations_.size());
  for (const auto& observation : observations_) {
    if (observationUsable(observation)) {
      usable.push_back(&observation);
    }
  }

  // SP 主相机 Tracker 先按图像中心距离排序。其第二次 priority 排序依赖外围
  // Decider 预先写入字段；单 Tracker/离线入口没有 Decider，不能在 L3 内硬塞
  // 一张外围优先级表。完整哨兵多目标路径应由独立 Decider 显式提供该字段。
  std::stable_sort(
    usable.begin(),
    usable.end(),
    [this](const Armor* lhs, const Armor* rhs) {
      return L6Telemetry::squaredDistance(lhs->center, image_center_) <
             L6Telemetry::squaredDistance(rhs->center, image_center_);
    });
  return usable;
}

bool Tracker::initializeTarget(
  const std::vector<const Armor*>& armors,
  TimePoint timestamp)
{
  if (armors.empty()) {
    return false;
  }

  const Armor& armor = *armors.front();
  double radius = 0.2;
  // 板数取 types.hpp 的共用映射，与 PnpSolver 的双板配对同源；类别未知时按四板。
  int armor_count = armorCountOf(armor.name).value_or(4);
  Eigen::VectorXd covariance_diagonal(11);

  // 不同车辆结构使用对应的初始半径和协方差。
  // 平衡步兵已退出赛场，普通车辆一律按四板整车模型初始化。
  if (armor.name == ArmorName::Outpost) {
    radius = 0.2765;
    covariance_diagonal << 1.0, 64.0, 1.0, 64.0, 1.0, 81.0,
      0.4, 100.0, 1e-4, 0.0, 0.0;
  } else if (isBase(armor.name)) {
    radius = 0.3205;
    covariance_diagonal << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0,
      0.4, 100.0, 1e-4, 0.0, 0.0;
  } else {
    covariance_diagonal << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0,
      0.4, 100.0, 1.0, 1.0, 1.0;
  }

  target_.emplace(
    armor, timestamp, radius, armor_count, std::move(covariance_diagonal));
  return true;
}

bool Tracker::updateTarget(
  const std::vector<const Armor*>& armors,
  TimePoint timestamp)
{
  if (!target_) {
    return false;
  }

  // 即使本帧没有匹配观测，也先把目标预测到当前曝光时刻。
  target_->predict(timestamp);
  bool found = false;
  for (const Armor* armor : armors) {
    if (armor->name == target_->name && armor->type == target_->armor_type) {
      target_->update(*armor);
      found = true;
    }
  }
  return found;
}

void Tracker::updateState(bool found)
{
  switch (state_) {
  case TrackState::Lost:
    // 首次发现目标后进入连续确认阶段。
    if (!found) {
      return;
    }
    state_ = TrackState::Detecting;
    detect_count_ = 1;
    return;

  case TrackState::Detecting:
    // 确认阶段任何一次丢失都会放弃当前初始化结果。
    if (found) {
      ++detect_count_;
      if (detect_count_ >= tracker_config_.min_detect_count) {
        state_ = TrackState::Tracking;
      }
    } else {
      resetTracking();
    }
    return;

  case TrackState::Tracking:
    // 稳定跟踪首次丢失时保留预测状态并进入 TempLost。
    if (!found) {
      temp_lost_count_ = 1;
      state_ = TrackState::TempLost;
    }
    return;

  case TrackState::TempLost:
    // 重新关联后立即恢复 Tracking，否则累计允许的预测帧数。
    if (found) {
      state_ = TrackState::Tracking;
      return;
    }

    ++temp_lost_count_;
    if (target_) {
      const int max_temp_lost_count = target_->name == ArmorName::Outpost
        ? tracker_config_.outpost_max_temp_lost_count
        : tracker_config_.max_temp_lost_count;
      if (temp_lost_count_ > max_temp_lost_count) {
        resetTracking();
      }
    } else {
      resetTracking();
    }
    return;
  }
}

void Tracker::resetTracking() noexcept
{
  // 保留 last_timestamp_ 和 observations_，由完整 reset() 或下一帧管理。
  state_ = TrackState::Lost;
  detect_count_ = 0;
  temp_lost_count_ = 0;
  target_.reset();
}

}  // namespace L3Estimation
