#include "l3_estimation/armor/track_slot.hpp"

#include "l6_telemetry/logger.hpp"

#include <chrono>

namespace L3Estimation {

TimePoint TrackSlot::holdFrom(double hold) const noexcept
{
  return last_update_ + std::chrono::duration_cast<TimePoint::duration>(
                          std::chrono::duration<double>(hold));
}

void TrackSlot::start(const Armor & armor, const EskfTargetConfig & config, TimePoint now)
{
  target_.reset(armor, config, now);
  life_.start();
  last_update_ = now;
  used_lights_.clear();
}

void TrackSlot::predict(TimePoint now, double hold)
{
  target_.predictEkf(now, holdFrom(hold));
}

bool TrackSlot::update(
  const std::vector<MatchedArmor> & matched, const std::vector<MatchedLight> & matched_lights,
  const std::optional<double> & depth_diff, TimePoint now, const ObsContext & ctx)
{
  used_lights_.clear();
  if (target_.update(matched, matched_lights, depth_diff, now, ctx) <= 0) {
    return false;
  }

  // update() 成功之后才填这份清单，它因此严格等于本帧真正进了滤波更新的那些
  // 灯条观测，而不是 matchLight 的候选或关联中间结果。
  used_lights_.reserve(matched.size() * 2 + matched_lights.size());
  for (const auto & [id, armor] : matched) {
    used_lights_.push_back({armor.points[0], armor.points[3], id, true, false});
    used_lights_.push_back({armor.points[1], armor.points[2], id, false, false});
  }
  for (const auto & [id, is_left, light] : matched_lights) {
    used_lights_.push_back({light.top, light.bottom, id, is_left, true, light.id});
  }
  last_update_ = now;
  return true;
}

void TrackSlot::step(bool found, TimePoint now, int tracking_thres, double lost_thres)
{
  life_.step(found, tracking_thres, idle(now), lost_thres);
  if (life_.confirmed() && target_.diverged()) {
    drop();
    LOG_WARN("EskfTracker: 目标发散，已复位");
  }
}

void TrackSlot::drop() noexcept
{
  life_.drop();
  used_lights_.clear();
}

}  // namespace L3Estimation
