#include "l4_planning/predictor.hpp"

#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/target_state.hpp"
#include "l4_planning/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

namespace L4Planning {

namespace {

constexpr double kPi = 3.14159265358979323846;

// 将角度限制到 (-pi, pi]，避免装甲板跨越 ±pi 时出现跳变。
[[nodiscard]] double normalizeAngle(double angle) noexcept
{
  angle = std::remainder(angle, 2.0 * kPi);
  return angle <= -kPi ? angle + 2.0 * kPi : angle;
}

// 检查 L3 状态能否安全用于车辆和装甲板几何预测。
[[nodiscard]] bool validTargetState(const L3Estimation::TargetState& target) noexcept
{
  const double second_radius = target.radius + target.radius_offset;
  const bool armor_count_valid =
    target.armor_count == 3 || target.armor_count == 4;
  const bool heights_valid = std::all_of(
    target.three_armor_height_offsets.begin(),
    target.three_armor_height_offsets.end(),
    [](double value) { return std::isfinite(value); });
  return target.robot_id >= 0
         && armor_count_valid
         && target.center.allFinite()
         && target.velocity.allFinite()
         && std::isfinite(target.yaw)
         && std::isfinite(target.yaw_rate)
         && std::isfinite(target.radius)
         && std::isfinite(target.radius_offset)
         && std::isfinite(target.height_offset)
         && target.radius > 0.0
         && (target.armor_count != 4 || second_radius > 0.0)
         && heights_valid
         && target.covariance.allFinite();
}

// 根据 Fosu 识别类别确定整车使用的大/小装甲板类型。后期考虑大小装甲板走不同的锁定条件，暂时保留
[[nodiscard]] ArmorType armorTypeForRobot(int robot_id) noexcept
{
  // The merged L3 geometry maps only Hero (class 1) to a physical big plate.
  return robot_id == 1
           ? ArmorType::Large
           : ArmorType::Small;
}

[[nodiscard]] L3Estimation::TargetState predictFilter(
  const L3Estimation::TargetState& target, TimePoint prediction_time)
{
  auto predicted = target;
  auto filter = *target.filter_state;
  filter.predict(prediction_time);
  const Eigen::VectorXd state = filter.ekf_x();
  if (state.size() != L3Estimation::EskfTarget::kStateSize) {
    predicted.center.setConstant(std::numeric_limits<double>::quiet_NaN());
    return predicted;
  }
  predicted.center = {state[L3Estimation::XC], state[L3Estimation::YC],
                      state[L3Estimation::ZC]};
  predicted.velocity = {state[L3Estimation::VX], state[L3Estimation::VY],
                        state[L3Estimation::VZ]};
  predicted.yaw = state[L3Estimation::YAW];
  predicted.yaw_rate = state[L3Estimation::YAW_RATE];
  predicted.radius = state[L3Estimation::RADIUS];
  if (predicted.armor_count == 4) {
    predicted.radius_offset = state[9] - state[8];
    predicted.height_offset = state[10];
  } else {
    predicted.radius_offset = 0.0;
    predicted.height_offset = 0.0;
    predicted.three_armor_height_offsets = {0.0, state[9], state[10]};
  }
  Eigen::Matrix<double, L3Estimation::STATE_DIM,
                L3Estimation::EskfTarget::kStateSize> jacobian =
    Eigen::Matrix<double, L3Estimation::STATE_DIM,
                  L3Estimation::EskfTarget::kStateSize>::Zero();
  for (int index = 0; index < 8; ++index) {
    jacobian(index, index) = 1.0;
  }
  jacobian(L3Estimation::RADIUS, 8) = state[8];
  if (predicted.armor_count == 4) {
    jacobian(L3Estimation::RADIUS_OFFSET, 8) = -state[8];
    jacobian(L3Estimation::RADIUS_OFFSET, 9) = state[9];
    jacobian(L3Estimation::HEIGHT_OFFSET, 10) = 1.0;
  }
  predicted.covariance =
    jacobian * filter.covariance() * jacobian.transpose();
  predicted.timestamp = prediction_time;
  predicted.filter_state = std::move(filter);
  return predicted;
}

}  // namespace

L3Estimation::TargetState Predictor::predict(const L3Estimation::TargetState& target, double dt) const
{
  auto predicted = target;
  if (!std::isfinite(dt)) {
    return predicted;
  }

  if (predicted.filter_state) {
    const auto prediction_time = target.timestamp +
      std::chrono::duration_cast<TimePoint::duration>(
        std::chrono::duration<double>(dt));
    return predictFilter(target, prediction_time);
  }

  // 与 L3 EKF 相同的匀速、匀角速度状态转移模型。
  predicted.center += target.velocity * dt;
  predicted.yaw = normalizeAngle(target.yaw + target.yaw_rate * dt);

  // 11维状态转移矩阵只耦合位置-速度和 yaw-yaw_rate。
  L3Estimation::StateCovariance transition =
    L3Estimation::StateCovariance::Identity();
  transition(L3Estimation::XC, L3Estimation::VX) = dt;
  transition(L3Estimation::YC, L3Estimation::VY) = dt;
  transition(L3Estimation::ZC, L3Estimation::VZ) = dt;
  transition(L3Estimation::YAW, L3Estimation::YAW_RATE) = dt;
  predicted.covariance =
    transition * target.covariance * transition.transpose();
  predicted.covariance =
    0.5 * (predicted.covariance + predicted.covariance.transpose());

  predicted.timestamp += std::chrono::duration_cast<TimePoint::duration>(
    std::chrono::duration<double>(dt));
  return predicted;
}

PredictionResult Predictor::predict(const PredictionRequest& request) const
{
  PredictionResult result;
  if (!validTargetState(request.target)
      || request.target_time < request.target.timestamp) {
    result.predicted_vehicle = request.target;
    return result;
  }

  const double dt = std::chrono::duration<double>(
    request.target_time - request.target.timestamp).count();
  if (!std::isfinite(dt)) {
    result.predicted_vehicle = request.target;
    return result;
  }

  result.predicted_vehicle = request.target.filter_state
    ? predictFilter(request.target, request.target_time)
    : predict(request.target, dt);
  // 对外结果精确标记为调用者请求的绝对命中时刻。
  result.predicted_vehicle.timestamp = request.target_time;
  if (!validTargetState(result.predicted_vehicle)) {
    return result;
  }

  result.armor_candidates.reserve(
    static_cast<std::size_t>(result.predicted_vehicle.armor_count));

  // The new L3 filter models the complete SO(3) vehicle pose.  Keep the
  // Bruce0178 candidate selection/window policy, but feed it armor poses from
  // that model instead of flattening roll/pitch back into the legacy yaw-only
  // geometry.  A short deterministic look-ahead supplies the per-armor
  // velocity used by the existing selector.
  std::vector<Eigen::Vector4d> filter_armors;
  std::vector<Eigen::Vector4d> filter_armors_next;
  constexpr double kVelocitySampleSeconds = 1.0e-3;
  if (result.predicted_vehicle.filter_state) {
    filter_armors =
      result.predicted_vehicle.filter_state->armor_xyza_list();
    auto next_filter = *result.predicted_vehicle.filter_state;
    next_filter.predict(kVelocitySampleSeconds);
    filter_armors_next = next_filter.armor_xyza_list();
  }
  const bool use_filter_geometry =
    filter_armors.size() ==
      static_cast<std::size_t>(result.predicted_vehicle.armor_count)
    && filter_armors_next.size() == filter_armors.size();

  for (int armor_id = 0;
       armor_id < result.predicted_vehicle.armor_count;
       ++armor_id) {
    // 四装甲模型中 0/2 与 1/3 分别使用两组半径和高度。
    const bool second_group =
      result.predicted_vehicle.armor_count == 4 && armor_id % 2 != 0;
    const double radius =
      result.predicted_vehicle.radius
      + (second_group ? result.predicted_vehicle.radius_offset : 0.0);
    const double armor_yaw = normalizeAngle(
      result.predicted_vehicle.yaw
      + static_cast<double>(armor_id) * 2.0 * kPi /
          static_cast<double>(result.predicted_vehicle.armor_count));

    ArmorPose armor;
    armor.robot_id = result.predicted_vehicle.robot_id;
    armor.armor_id = armor_id;
    armor.armor_type = armorTypeForRobot(armor.robot_id);

    if (use_filter_geometry) {
      const auto index = static_cast<std::size_t>(armor_id);
      armor.position_world = filter_armors[index].head<3>();
      armor.velocity_world =
        (filter_armors_next[index].head<3>() - armor.position_world)
        / kVelocitySampleSeconds;
      armor.yaw_world = filter_armors[index].w();
      armor.timestamp = request.target_time;
      armor.valid = armor.position_world.allFinite()
                    && armor.velocity_world.allFinite()
                    && std::isfinite(armor.yaw_world);
      result.armor_candidates.push_back(armor);
      continue;
    }

    // L3 的 yaw 指向“装甲板到车辆中心”，因此装甲板位置为 center-r*n。
    const double height_offset = result.predicted_vehicle.armor_count == 3
      ? result.predicted_vehicle.three_armor_height_offsets[
          static_cast<std::size_t>(armor_id)]
      : (second_group ? result.predicted_vehicle.height_offset : 0.0);
    armor.position_world = {
      result.predicted_vehicle.center.x() - radius * std::cos(armor_yaw),
      result.predicted_vehicle.center.y() - radius * std::sin(armor_yaw),
      result.predicted_vehicle.center.z() + height_offset};

    // 对 center-r*[cos(yaw), sin(yaw)] 求导，得到旋转产生的切向速度。
    armor.velocity_world = result.predicted_vehicle.velocity;
    armor.velocity_world.x() +=
      radius * result.predicted_vehicle.yaw_rate * std::sin(armor_yaw);
    armor.velocity_world.y() -=
      radius * result.predicted_vehicle.yaw_rate * std::cos(armor_yaw);
    armor.yaw_world = armor_yaw;
    armor.timestamp = request.target_time;
    armor.valid =
      armor.position_world.allFinite() && armor.velocity_world.allFinite();
    result.armor_candidates.push_back(armor);
  }

  result.valid = result.armor_candidates.size() ==
    static_cast<std::size_t>(result.predicted_vehicle.armor_count);
  for (const auto& armor : result.armor_candidates) {
    result.valid = result.valid && armor.valid;
  }
  return result;
}

}  // namespace L4Planning
