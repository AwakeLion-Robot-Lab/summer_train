#include "runtime/l4_target_adapter.hpp"

namespace runtime {

std::optional<L3Estimation::TargetState> toL4TargetState(
  const std::optional<L3Estimation::TrackedTarget>& target) noexcept
{
  if (!target) {
    return std::nullopt;
  }

  const Eigen::VectorXd state = target->ekf_x();
  const Eigen::MatrixXd& covariance = target->ekf().P;
  if (state.size() < L3Estimation::STATE_DIM ||
      covariance.rows() < L3Estimation::STATE_DIM ||
      covariance.cols() < L3Estimation::STATE_DIM) {
    return std::nullopt;
  }

  L3Estimation::TargetState snapshot;
  snapshot.robot_id = static_cast<int>(target->name);
  snapshot.armor_count = target->armor_num();
  snapshot.center = {
    state[L3Estimation::XC],
    state[L3Estimation::YC],
    state[L3Estimation::ZC]};
  snapshot.velocity = {
    state[L3Estimation::VX],
    state[L3Estimation::VY],
    state[L3Estimation::VZ]};
  snapshot.yaw = state[L3Estimation::YAW];
  snapshot.yaw_rate = state[L3Estimation::YAW_RATE];
  snapshot.radius = state[L3Estimation::RADIUS];
  snapshot.radius_offset = state[L3Estimation::RADIUS_OFFSET];
  snapshot.height_offset = state[L3Estimation::HEIGHT_OFFSET];
  if (snapshot.armor_count == 3) {
    if (state.size() < L3Estimation::TrackedTarget::kStateSize) {
      return std::nullopt;
    }
    snapshot.three_armor_height_offsets = {0.0, state[11], state[12]};
  }
  snapshot.covariance = covariance.topLeftCorner<
    L3Estimation::STATE_DIM, L3Estimation::STATE_DIM>();
  snapshot.timestamp = target->t();

  if (snapshot.robot_id < 0 ||
      (snapshot.armor_count != 3 && snapshot.armor_count != 4) ||
      !snapshot.center.allFinite() ||
      !snapshot.velocity.allFinite() || !snapshot.covariance.allFinite()) {
    return std::nullopt;
  }
  return snapshot;
}

}  // namespace runtime
