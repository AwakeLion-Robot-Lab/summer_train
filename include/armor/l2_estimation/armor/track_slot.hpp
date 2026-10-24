#pragma once

#include "l2_estimation/armor/armor_observation.hpp"
#include "l2_estimation/armor/eskf_target.hpp"
#include "l2_estimation/tracking/association.hpp"

#include <optional>
#include <vector>

namespace L3Estimation {

// 本帧真正用于滤波更新的灯条；随目标槽交换，不能由备用槽覆盖当前槽的诊断。
struct UsedLight {
  cv::Point2f top{}, bottom{};
  int armor_id{-1};
  bool is_left{false};
  bool isolated{false};
  std::size_t light_id{0};
};

// 一个整车目标及其生命周期、更新时刻和采纳清单。
class TrackSlot {
public:
  TrackState state() const noexcept { return life_.state; }
  bool alive() const noexcept { return state() != TrackState::Lost; }
  bool confirmed() const noexcept { return life_.isTracking(); }
  const EskfTarget& target() const noexcept { return target_; }
  ArmorName name() const noexcept { return target_.name; }
  double idle(TimePoint now) const noexcept { return elapsedSeconds(last_update_, now); }
  TimePoint holdFrom(double hold) const noexcept;
  const std::vector<UsedLight>& usedLights() const noexcept { return used_lights_; }

  void start(const Armor& armor, const EskfTargetConfig& config, TimePoint now);
  void predict(TimePoint now, double hold);
  bool update(const std::vector<MatchedArmor>& matched,
              const std::vector<MatchedLight>& matched_lights,
              const std::optional<double>& depth_diff, TimePoint now, const ObsContext& ctx);
  void step(bool found, TimePoint now, int tracking_thres, double lost_thres);
  void drop() noexcept;
  void clearFrame() noexcept { used_lights_.clear(); }

private:
  EskfTarget target_;
  TrackLifecycle life_;
  TimePoint last_update_{};
  std::vector<UsedLight> used_lights_;
};

}  // namespace L3Estimation
