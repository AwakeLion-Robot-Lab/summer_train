#include "l4_planning/predictor.hpp"

#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"

#include <chrono>
#include <cmath>

namespace L4Planning {

namespace {

constexpr int kArmorCount = 4;
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
  return target.robot_id >= 0
         && target.center.allFinite()
         && target.velocity.allFinite()
         && std::isfinite(target.yaw)
         && std::isfinite(target.yaw_rate)
         && std::isfinite(target.radius)
         && std::isfinite(target.radius_offset)
         && std::isfinite(target.height_offset)
         && target.radius > 0.0
         && second_radius > 0.0
         && target.covariance.allFinite();
}

// 根据 Fosu 识别类别确定整车使用的大/小装甲板类型。
[[nodiscard]] ArmorType armorTypeForRobot(int robot_id) noexcept
{
  // Fosu 类别编号中 1 为英雄大装甲，8 为基地大装甲，其余车辆使用小装甲。
  return robot_id == 1 || robot_id == 8
           ? ArmorType::Large
           : ArmorType::Small;
}

}  // namespace

L3Estimation::TargetState Predictor::predict(const L3Estimation::TargetState& target, double dt) const
{
  auto predicted = target;
  if (!std::isfinite(dt)) {
    return predicted;
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
  result.predicted_vehicle = request.target;
  if (!validTargetState(request.target)
      || request.target_time < request.target.timestamp) {
    return result;
  }

  const double dt = std::chrono::duration<double>(
    request.target_time - request.target.timestamp).count();
  if (!std::isfinite(dt)) {
    return result;
  }

  result.predicted_vehicle = predict(request.target, dt);
  // duration<double> 到 steady_clock::duration 的转换可能有舍入；对外结果应
  // 精确标记为调用者请求的绝对命中时刻。
  result.predicted_vehicle.timestamp = request.target_time;
  if (!validTargetState(result.predicted_vehicle)) {
    return result;
  }

  result.armor_candidates.reserve(kArmorCount);
  for (int armor_id = 0; armor_id < kArmorCount; ++armor_id) {
    // 四装甲模型中 0/2 与 1/3 分别使用两组半径和高度。
    const bool second_group = armor_id % 2 != 0;
    const double radius =
      result.predicted_vehicle.radius
      + (second_group ? result.predicted_vehicle.radius_offset : 0.0);
    const double armor_yaw = normalizeAngle(
      result.predicted_vehicle.yaw
      + static_cast<double>(armor_id) * kPi / 2.0);

    ArmorPose armor;
    armor.robot_id = result.predicted_vehicle.robot_id;
    armor.armor_id = armor_id;
    armor.armor_type = armorTypeForRobot(armor.robot_id);
    // L3 的 yaw 指向“装甲板到车辆中心”，因此装甲板位置为 center-r*n。
    armor.position_world = {
      result.predicted_vehicle.center.x() - radius * std::cos(armor_yaw),
      result.predicted_vehicle.center.y() - radius * std::sin(armor_yaw),
      result.predicted_vehicle.center.z()
        + (second_group ? result.predicted_vehicle.height_offset : 0.0)};

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

  result.valid = result.armor_candidates.size() == kArmorCount;
  for (const auto& armor : result.armor_candidates) {
    result.valid = result.valid && armor.valid;
  }
  return result;
}

}  // namespace L4Planning
