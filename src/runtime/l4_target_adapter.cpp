#include "runtime/l4_target_adapter.hpp"

namespace runtime {

std::optional<L3Estimation::TargetState> toL4TargetState(
  const std::optional<L3Estimation::EskfTarget>& target) noexcept
{
  if (!target) {
    return std::nullopt;
  }

  const Eigen::VectorXd state = target->ekf_x();
  if (state.size() < L3Estimation::EskfTarget::kStateSize) {
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
  // EskfTarget 对外第 9 维是第二组绝对半径（四板车）或 1 号板高度
  // （三板目标）；Bruce0178 的 L4 则使用 r2-r1 和独立三板高度数组。
  if (snapshot.armor_count == 4) {
    snapshot.radius_offset = state[9] - state[8];
    snapshot.height_offset = state[10];
  } else {
    snapshot.radius_offset = 0.0;
    snapshot.height_offset = 0.0;
    snapshot.three_armor_height_offsets = {0.0, state[9], state[10]};
  }

  // 把 ESKF 的误差状态协方差投影到 L4 的十一维接口。半径在 ESKF 内部为
  // log(r)，L4 使用线性半径及 r2-r1，因此这里显式乘雅可比。
  Eigen::Matrix<double, L3Estimation::STATE_DIM,
                L3Estimation::EskfTarget::kStateSize> jacobian =
    Eigen::Matrix<double, L3Estimation::STATE_DIM,
                  L3Estimation::EskfTarget::kStateSize>::Zero();
  for (int index = 0; index < 8; ++index) {
    jacobian(index, index) = 1.0;
  }
  jacobian(L3Estimation::RADIUS, 8) = state[8];
  if (snapshot.armor_count == 4) {
    jacobian(L3Estimation::RADIUS_OFFSET, 8) = -state[8];
    jacobian(L3Estimation::RADIUS_OFFSET, 9) = state[9];
    jacobian(L3Estimation::HEIGHT_OFFSET, 10) = 1.0;
  }
  snapshot.covariance =
    jacobian * target->covariance() * jacobian.transpose();
  snapshot.timestamp = target->t();
  snapshot.filter_state = target->snapshot();

  if (snapshot.robot_id < 0 ||
      (snapshot.armor_count != 3 && snapshot.armor_count != 4) ||
      !snapshot.center.allFinite() ||
      !snapshot.velocity.allFinite() || !snapshot.covariance.allFinite()) {
    return std::nullopt;
  }
  return snapshot;
}

}  // namespace runtime
