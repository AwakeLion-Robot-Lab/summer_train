#include "l4_planning/predictor.hpp"

#include "l3_estimation/armor/target_estimator.hpp"
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
  auto filter = std::make_shared<L3Estimation::TrackedTarget>(*target.filter_state);
  filter->predict(prediction_time);
  const auto& state = filter->ekf().x;
  const auto& covariance = filter->ekf().P;
  if (state.size() != L3Estimation::TrackedTarget::kStateSize ||
      covariance.rows() != L3Estimation::TrackedTarget::kStateSize ||
      covariance.cols() != L3Estimation::TrackedTarget::kStateSize) {
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
  predicted.radius_offset = state[L3Estimation::RADIUS_OFFSET];
  predicted.height_offset = state[L3Estimation::HEIGHT_OFFSET];
  if (predicted.armor_count == 3) {
    predicted.three_armor_height_offsets = {0.0, state[11], state[12]};
  }
  predicted.covariance = covariance.topLeftCorner<
    L3Estimation::STATE_DIM, L3Estimation::STATE_DIM>();
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

  if (predicted.filter_state && dt >= 0.0) {
    const auto prediction_time = target.timestamp +
      std::chrono::duration_cast<TimePoint::duration>(
        std::chrono::duration<double>(dt));
    return predictFilter(target, prediction_time);
  }

  // TinyMPC 的居中轨迹需要回推到观测之前。回推不使用 EKF 过程噪声，
  // 否则负 dt 会生成负的协方差项；随后沿用数值状态向前采样。
  if (dt < 0.0) {
    predicted.filter_state.reset();
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
