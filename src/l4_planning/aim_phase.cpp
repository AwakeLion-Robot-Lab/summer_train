#include "l4_planning/aim_phase.hpp"

#include <cmath>
#include <cstdlib>

namespace L4Planning {

void AimPhaseTracker::update(double abs_v_yaw, bool geometry_observed) noexcept
{
  // 几何不可观测，或者角速度本身是非法值时，一律退回最保守的档位。
  if (!geometry_observed || !std::isfinite(abs_v_yaw)) {
    reset();
    return;
  }

  switch (phase_) {
    case AimPhase::SingleArmor:
      counter_ = abs_v_yaw > config_.single_to_whole_up ? counter_ + 1 : 0;
      if (counter_ > config_.transfer_count) {
        phase_ = AimPhase::WholeCarArmor;
        counter_ = 0;
      }
      break;

    case AimPhase::WholeCarArmor:
      // 上行看 whole_to_center_up，下行看 single_to_whole_down——下行阈值比
      // 进来时的 single_to_whole_up 低，中间那段死区就是施密特触发的回差。
      if (abs_v_yaw > config_.whole_to_center_up) {
        ++counter_;
      } else if (abs_v_yaw < config_.single_to_whole_down) {
        --counter_;
      } else {
        counter_ = 0;
      }
      if (std::abs(counter_) > config_.transfer_count) {
        phase_ = counter_ > 0 ? AimPhase::WholeCarCenter : AimPhase::SingleArmor;
        counter_ = 0;
      }
      break;

    case AimPhase::WholeCarCenter:
      counter_ = abs_v_yaw < config_.whole_to_center_down ? counter_ + 1 : 0;
      if (counter_ > config_.transfer_count) {
        phase_ = AimPhase::WholeCarArmor;
        counter_ = 0;
      }
      break;
  }
}

}  // namespace L4Planning
