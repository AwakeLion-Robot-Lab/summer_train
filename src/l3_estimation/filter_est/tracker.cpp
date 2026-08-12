#include "l3_estimation/filter_est/tracker.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

namespace L3Estimation::FilterEst {
namespace {

// 参数来自 YAML，填成 0 或负数会让状态机永远停在某一档，所以在这里挡住。
[[nodiscard]] bool validConfig(const TrackerConfig& config) noexcept
{
  return config.min_detect_count > 0 &&
         config.max_frame_interval > std::chrono::milliseconds::zero() &&
         config.max_temp_lost_count > 0 &&
         config.outpost_max_temp_lost_count >= config.max_temp_lost_count;
}

}  // namespace

Tracker::Tracker(
  const L1Sensor::CameraCalibration& calibration,
  ArmorConfig armor_config,
  TrackerConfig tracker_config,
  L3Estimation::TargetConfig target_config,
  TargetConfig filter_config)
: armor_config_(armor_config),
  tracker_config_(tracker_config),
  target_config_(target_config),
  filter_config_(filter_config),
  pnp_solver_(calibration, armor_config_),
  image_center_{
    static_cast<float>(calibration.image_size.width) * 0.5F,
    static_cast<float>(calibration.image_size.height) * 0.5F},
  ready_(pnp_solver_.ready() && validConfig(tracker_config_))
{
}

bool Tracker::ready() const noexcept { return ready_; }

TrackState Tracker::state() const noexcept { return state_; }

EstimatorBackend Tracker::backend() const noexcept { return EstimatorBackend::Filter; }

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

  // 逐个检测独立解算，失败的结果也留在 observations_ 里供调试查看。
  // PnP 在目标筛选**之前**做完：选哪辆车要看三维位置，而不是像素框。
  pnp_solver_.set_R_world_barrel(q_world_barrel);
  observations_.reserve(detections.size());
  for (const auto& detection : detections) {
    observations_.push_back(toArmorObservation(detection, timestamp));
    pnp_solver_.single_pnp(observations_.back());
  }

  // 时间戳倒退，或跟踪期间帧间隔过大（相机掉线）时，旧运动状态不再可信。
  if (last_timestamp_) {
    const auto elapsed = timestamp - *last_timestamp_;
    if (elapsed <= TimePoint::duration::zero() ||
        (state_ != TrackState::Lost && elapsed > tracker_config_.max_frame_interval)) {
      L6Telemetry::logWarn("Tracker reset after invalid frame interval");
      resetTracking();
    }
  }
  last_timestamp_ = timestamp;

  // Lost 时从候选里挑一辆车建目标，其余状态只关联同一辆车的观测。
  const std::vector<const Armor*> armors = usableObservations();
  const bool found = state_ == TrackState::Lost ? initializeTarget(armors, timestamp)
                                                : updateTarget(armors, timestamp);
  updateState(found);

  if (state_ == TrackState::Lost || !target_) {
    return std::nullopt;
  }

  // 状态机推进之后再统一体检：半径跑出物理范围，或最近的 NIS 窗口持续失败，
  // 都说明这个整车假设已经和观测对不上了，整个目标作废重来。
  if (target_->diverged()) {
    L6Telemetry::logDebug("Tracker reset: target diverged");
    resetTracking();
    return std::nullopt;
  }
  if (nisPersistentlyBad(*target_)) {
    L6Telemetry::logDebug("Tracker reset: NIS window keeps failing");
    resetTracking();
    return std::nullopt;
  }

  return target_->snapshot();
}

const std::vector<Armor>& Tracker::observations() const noexcept { return observations_; }

std::vector<Eigen::Vector4d> Tracker::targetArmorPoses() const
{
  if (!target_ || state_ == TrackState::Lost) {
    return {};
  }
  return target_->armorPoses();
}

void Tracker::reset() noexcept
{
  resetTracking();
  last_timestamp_.reset();
  observations_.clear();
}

bool Tracker::nisPersistentlyBad(const Target& target) const
{
  const auto& ekf = target.filter();
  const int failures =
    std::accumulate(ekf.recent_nis_failures.begin(), ekf.recent_nis_failures.end(), 0);
  return static_cast<double>(failures) >=
         filter_config_.max_nis_failure_ratio * static_cast<double>(ekf.window_size);
}

std::vector<const Armor*> Tracker::usableObservations() const
{
  std::vector<const Armor*> usable;
  usable.reserve(observations_.size());
  for (const auto& observation : observations_) {
    // name 只在 single_pnp 成功提交位姿的那一步才被赋值，任何失败路径上都
    // 保持 Unknown。这是唯一的观测门限。
    if (observation.name != ArmorName::Unknown) {
      usable.push_back(&observation);
    }
  }

  // 靠近画面中心的目标更可能是当前正在瞄的那辆车。多目标优先级（工程车 <
  // 步兵 < 英雄这类战术排序）属于更上层的决策，不在 L3 里硬编码。
  std::stable_sort(
    usable.begin(), usable.end(), [this](const Armor* lhs, const Armor* rhs) {
      return L6Telemetry::squaredDistance(lhs->center, image_center_) <
             L6Telemetry::squaredDistance(rhs->center, image_center_);
    });
  return usable;
}

bool Tracker::initializeTarget(const std::vector<const Armor*>& armors, TimePoint timestamp)
{
  if (armors.empty()) {
    return false;
  }

  const Armor& armor = *armors.front();
  double radius = target_config_.radius;
  int armor_count = 4;
  Eigen::VectorXd covariance_diagonal(11);

  // 初始协方差按"哪些量已知"给：前哨站和基地的半径是固定的（1e-4 表示几乎
  // 确定），第二组半径和高度差不存在（0 表示不更新）；普通四板车三者都要估。
  if (armor.name == ArmorName::Outpost) {
    radius = target_config_.outpost_radius;
    armor_count = 3;
    covariance_diagonal << 1.0, 64.0, 1.0, 64.0, 1.0, 81.0, 0.4, 100.0, 1e-4, 0.0, 0.0;
  } else if (armor.name == ArmorName::BaseSmall || armor.name == ArmorName::BaseLarge) {
    radius = target_config_.base_radius;
    armor_count = 3;
    covariance_diagonal << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, 1e-4, 0.0, 0.0;
  } else {
    covariance_diagonal << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, 1.0, 1.0, 1.0;
  }

  target_.emplace(
    armor, timestamp, radius, armor_count, std::move(covariance_diagonal),
    target_config_, filter_config_);
  return true;
}

bool Tracker::updateTarget(const std::vector<const Armor*>& armors, TimePoint timestamp)
{
  if (!target_) {
    return false;
  }

  // 即使本帧没有匹配观测，也先把目标预测到当前曝光时刻。
  target_->predict(timestamp);

  bool found = false;
  for (const Armor* armor : armors) {
    if (armor->name == target_->name() && armor->type == target_->armorType()) {
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
      if (found) {
        state_ = TrackState::Detecting;
        detect_count_ = 1;
      }
      return;

    case TrackState::Detecting:
      // 确认阶段任何一次丢失都放弃当前初始化结果。
      if (!found) {
        resetTracking();
      } else if (++detect_count_ >= tracker_config_.min_detect_count) {
        state_ = TrackState::Tracking;
      }
      return;

    case TrackState::Tracking:
      // 稳定跟踪首次丢失时保留滤波状态，转入纯预测。
      if (!found) {
        temp_lost_count_ = 1;
        state_ = TrackState::TempLost;
      }
      return;

    case TrackState::TempLost:
      // 重新关联立即恢复 Tracking，否则累计允许的纯预测帧数。
      if (found) {
        state_ = TrackState::Tracking;
        return;
      }
      ++temp_lost_count_;
      if (temp_lost_count_ > (target_ && target_->name() == ArmorName::Outpost
                                ? tracker_config_.outpost_max_temp_lost_count
                                : tracker_config_.max_temp_lost_count)) {
        resetTracking();
      }
      return;
  }
}

void Tracker::resetTracking() noexcept
{
  // 只重置跟踪生命周期；last_timestamp_ 和 observations_ 由 reset() 或下一帧管理。
  state_ = TrackState::Lost;
  detect_count_ = 0;
  temp_lost_count_ = 0;
  target_.reset();
}

}  // namespace L3Estimation::FilterEst
