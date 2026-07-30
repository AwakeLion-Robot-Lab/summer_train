#include "l4_planning/state.hpp"

#include "l4_planning/planner.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace L4Planning {

namespace {

constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] const ArmorCandidate* findCandidate(
  const std::vector<ArmorCandidate>& candidates,
  std::optional<int> armor_id,
  bool require_valid = false) noexcept
{
  if (!armor_id.has_value()) {
    return nullptr;
  }
  const auto candidate = std::find_if(
    candidates.begin(),
    candidates.end(),
    [armor_id](const ArmorCandidate& item) {
      return item.armor.armor_id == *armor_id;
    });
  if (candidate == candidates.end() || (require_valid && !candidate->valid)) {
    return nullptr;
  }
  return &*candidate;
}

[[nodiscard]] const ArmorCandidate* bestValidCandidate(
  const std::vector<ArmorCandidate>& candidates,
  std::optional<int> excluded_armor_id,
  bool prefer_entering) noexcept
{
  const auto is_eligible =
    [excluded_armor_id](const ArmorCandidate& candidate) {
      return candidate.valid
             && (!excluded_armor_id.has_value()
                 || candidate.armor.armor_id != *excluded_armor_id);
    };
  const auto choose_best =
    [&candidates, &is_eligible](bool entering_only) {
      const ArmorCandidate* best = nullptr;
      for (const auto& candidate : candidates) {
        if (!is_eligible(candidate)
            || (entering_only && !candidate.entering_firing_window)) {
          continue;
        }
        if (best == nullptr
            || candidate.score.quality > best->score.quality) {
          best = &candidate;
        }
      }
      return best;
    };

  if (prefer_entering) {
    if (const ArmorCandidate* entering = choose_best(true)) {
      return entering;
    }
  }
  return choose_best(false);
}

[[nodiscard]] const ArmorCandidate* bestScoredCandidate(
  const std::vector<ArmorCandidate>& candidates,
  std::optional<int> excluded_armor_id) noexcept
{
  const ArmorCandidate* best = nullptr;
  for (const auto& candidate : candidates) {
    if (!candidate.valid
        || (excluded_armor_id.has_value()
            && candidate.armor.armor_id == *excluded_armor_id)) {
      continue;
    }
    if (best == nullptr
        || candidate.score.quality > best->score.quality) {
      best = &candidate;
    }
  }
  return best;
}

[[nodiscard]] bool aimSettled(
  const ArmorCandidate& candidate,
  const PlannerConfig& config) noexcept
{
  const double threshold = config.switch_dead_zone * kPi / 180.0;
  return std::isfinite(candidate.aim_angle_error)
         && candidate.aim_angle_error <= threshold;
}

}  // namespace

SelectionResult Planner::selectArmor(
  const SelectionRequest& request,
  TimePoint selection_time,
  const PlannerConfig& config)
{
  const auto set_phase =
    [this, selection_time](ArmorTrackingPhase phase) {
      if (tracking_state_.phase != phase) {
        tracking_state_.phase = phase;
        tracking_state_.phase_started_at = selection_time;
      }
    };
  const auto finish =
    [this](const ArmorCandidate* candidate,
           bool tracking_ready,
           SelectionReason reason) {
      SelectionResult result;
      result.phase = tracking_state_.phase;
      result.tracking_ready = tracking_ready;
      result.reason = reason;
      if (candidate != nullptr) {
        result.selected = *candidate;
        result.valid = candidate->valid;
      }
      return result;
    };
  const auto choose_next =
    [&request](std::optional<int> excluded_armor_id) {
      return bestValidCandidate(
        request.candidates, excluded_armor_id, true);
    };
  const auto begin_switch =
    [this, &set_phase, &finish, &request, &config](
      const ArmorCandidate* candidate,
      SelectionReason reason) {
      if (candidate == nullptr) {
        return finish(nullptr, false, SelectionReason::NoCandidate);
      }
      tracking_state_.next_armor_id = candidate->armor.armor_id;
      tracking_state_.current_lost_frames = 0;
      tracking_state_.next_stable_frames = 0;
      set_phase(ArmorTrackingPhase::Switching);
      if (aimSettled(*candidate, config)) {
        set_phase(ArmorTrackingPhase::Stabilizing);
        tracking_state_.next_stable_frames =
          request.observation_fresh ? 1 : 0;
      }
      return finish(candidate, false, reason);
    };

  // 一次调用最多连续跨越两个瞬时状态，例如 Switching -> Stabilizing
  // -> Tracking（lock_stable_frames == 1），循环上限用于防止错误状态自旋。
  for (int transition = 0; transition < 4; ++transition) {
    switch (tracking_state_.phase) {
      case ArmorTrackingPhase::Unlocked: {
        const ArmorCandidate* selected = findCandidate(
          request.candidates, request.preferred_armor_id, true);
        if (selected == nullptr) {
          selected = choose_next(std::nullopt);
        }
        if (selected == nullptr) {
          return finish(nullptr, false, SelectionReason::NoCandidate);
        }

        tracking_state_.next_armor_id = selected->armor.armor_id;
        tracking_state_.next_stable_frames = 0;
        if (aimSettled(*selected, config)) {
          set_phase(ArmorTrackingPhase::Stabilizing);
          tracking_state_.next_stable_frames =
            request.observation_fresh ? 1 : 0;
          if (tracking_state_.next_stable_frames
              >= config.lock_stable_frames) {
            tracking_state_.current_armor_id =
              tracking_state_.next_armor_id;
            tracking_state_.next_armor_id.reset();
            tracking_state_.current_lost_frames = 0;
            set_phase(ArmorTrackingPhase::Tracking);
            return finish(
              selected,
              request.observation_fresh,
              SelectionReason::LockConfirmed);
          }
          return finish(
            selected, false, SelectionReason::Stabilizing);
        }

        set_phase(ArmorTrackingPhase::Switching);
        return finish(
          selected, false, SelectionReason::InitialLock);
      }

      case ArmorTrackingPhase::Tracking: {
        const ArmorCandidate* current = findCandidate(
          request.candidates,
          tracking_state_.current_armor_id,
          true);
        if (current == nullptr) {
          tracking_state_.score_candidate_id.reset();
          tracking_state_.score_stable_frames = 0;
          ++tracking_state_.current_lost_frames;
          // 当前装甲板短暂失效时保留锁定 ID，避免单帧预测、弹道或
          // 迭代失败导致立即换板；宽限期内不返回候选，因此禁止开火。
          if (tracking_state_.current_lost_frames
              <= config.max_lost_frames) {
            return finish(nullptr, false, SelectionReason::NoCandidate);
          }

          const ArmorCandidate* next =
            choose_next(tracking_state_.current_armor_id);
          if (next != nullptr) {
            return begin_switch(
              next, SelectionReason::SwitchToCandidate);
          }
          tracking_state_ = {};
          return finish(nullptr, false, SelectionReason::NoCandidate);
        }

        tracking_state_.current_lost_frames = 0;
        const ArmorCandidate* best = bestScoredCandidate(
          request.candidates, tracking_state_.current_armor_id);
        if (best != nullptr
            && best->score.quality
                 > current->score.quality + config.score_switch_threshold
            && request.observation_fresh) {
          if (tracking_state_.score_candidate_id
              == best->armor.armor_id) {
            ++tracking_state_.score_stable_frames;
          } else {
            tracking_state_.score_candidate_id = best->armor.armor_id;
            tracking_state_.score_stable_frames = 1;
          }
          if (tracking_state_.score_stable_frames
              >= config.score_switch_stable_frames) {
            tracking_state_.score_candidate_id.reset();
            tracking_state_.score_stable_frames = 0;
            return begin_switch(
              best, SelectionReason::SwitchToCandidate);
          }
        } else {
          tracking_state_.score_candidate_id.reset();
          tracking_state_.score_stable_frames = 0;
        }
        return finish(
          current,
          request.observation_fresh,
          SelectionReason::KeepCurrent);
      }

      case ArmorTrackingPhase::Switching: {
        const ArmorCandidate* next = findCandidate(
          request.candidates,
          tracking_state_.next_armor_id,
          true);
        if (next == nullptr) {
          const ArmorCandidate* current = findCandidate(
            request.candidates,
            tracking_state_.current_armor_id,
            true);
          if (current != nullptr) {
            tracking_state_.next_armor_id.reset();
            tracking_state_.current_lost_frames = 0;
            tracking_state_.next_stable_frames = 0;
            set_phase(ArmorTrackingPhase::Tracking);
            return finish(
              current,
              request.observation_fresh,
              SelectionReason::KeepCurrent);
          }

          next = choose_next(tracking_state_.current_armor_id);
          if (next == nullptr) {
            ++tracking_state_.current_lost_frames;
            if (tracking_state_.current_lost_frames
                > config.max_lost_frames) {
              tracking_state_ = {};
            }
            return finish(nullptr, false, SelectionReason::NoCandidate);
          }
          tracking_state_.next_armor_id = next->armor.armor_id;
          tracking_state_.next_stable_frames = 0;
        }
        tracking_state_.current_lost_frames = 0;

        if (!aimSettled(*next, config)) {
          return finish(
            next, false, SelectionReason::SwitchToCandidate);
        }
        set_phase(ArmorTrackingPhase::Stabilizing);
        tracking_state_.next_stable_frames =
          request.observation_fresh ? 1 : 0;
        if (tracking_state_.next_stable_frames
            < config.lock_stable_frames) {
          return finish(
            next, false, SelectionReason::Stabilizing);
        }
        // lock_stable_frames == 1 时在同一周期继续完成锁定。
        continue;
      }

      case ArmorTrackingPhase::Stabilizing: {
        const ArmorCandidate* next = findCandidate(
          request.candidates,
          tracking_state_.next_armor_id,
          true);
        if (next == nullptr) {
          const ArmorCandidate* current = findCandidate(
            request.candidates,
            tracking_state_.current_armor_id,
            true);
          if (current != nullptr) {
            tracking_state_.next_armor_id.reset();
            tracking_state_.current_lost_frames = 0;
            tracking_state_.next_stable_frames = 0;
            set_phase(ArmorTrackingPhase::Tracking);
            return finish(
              current,
              request.observation_fresh,
              SelectionReason::KeepCurrent);
          }

          next = choose_next(tracking_state_.current_armor_id);
          if (next == nullptr) {
            ++tracking_state_.current_lost_frames;
            if (tracking_state_.current_lost_frames
                > config.max_lost_frames) {
              tracking_state_ = {};
            }
            return finish(nullptr, false, SelectionReason::NoCandidate);
          }
          tracking_state_.next_armor_id = next->armor.armor_id;
          tracking_state_.current_lost_frames = 0;
          tracking_state_.next_stable_frames = 0;
          set_phase(ArmorTrackingPhase::Switching);
          return finish(
            next, false, SelectionReason::SwitchToCandidate);
        }
        tracking_state_.current_lost_frames = 0;

        if (!aimSettled(*next, config)) {
          tracking_state_.next_stable_frames = 0;
          set_phase(ArmorTrackingPhase::Switching);
          return finish(
            next, false, SelectionReason::SwitchToCandidate);
        }
        if (request.observation_fresh) {
          ++tracking_state_.next_stable_frames;
        } else {
          tracking_state_.next_stable_frames = 0;
        }
        if (tracking_state_.next_stable_frames
            < config.lock_stable_frames) {
          return finish(
            next, false, SelectionReason::Stabilizing);
        }

        tracking_state_.current_armor_id =
          tracking_state_.next_armor_id;
        tracking_state_.next_armor_id.reset();
        tracking_state_.current_lost_frames = 0;
        tracking_state_.next_stable_frames = 0;
        set_phase(ArmorTrackingPhase::Tracking);
        return finish(
          next,
          request.observation_fresh,
          SelectionReason::LockConfirmed);
      }
    }
  }

  tracking_state_ = {};
  return finish(nullptr, false, SelectionReason::NoCandidate);
}

}  // namespace L4Planning
