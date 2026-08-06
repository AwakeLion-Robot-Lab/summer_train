#include "l4_planning/predictor.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>
#include <numbers>

namespace L4Planning {

L3Estimation::TargetState Predictor::predict(
  const L3Estimation::TargetState& target, double dt) const
{
  auto predicted = target;
  if (!std::isfinite(dt)) {
    return predicted;
  }

  // 与 L3 的 F 矩阵同构：位置和 yaw 各自恒速度推进，半径和高度差不变。
  predicted.position += target.velocity * dt;
  predicted.yaw = L6Telemetry::limit_rad(target.yaw + target.v_yaw * dt);

  // 时间戳跟着一起走，L5 才能判断这个快照对应哪个时刻。
  predicted.timestamp =
    target.timestamp + std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
                         std::chrono::duration<double>(dt));
  return predicted;
}

std::vector<Eigen::Vector4d> Predictor::armorPoses(
  const L3Estimation::TargetState& target) const
{
  std::vector<Eigen::Vector4d> armors;
  if (target.armor_num < 1) {
    return armors;
  }

  armors.reserve(static_cast<std::size_t>(target.armor_num));
  for (int id = 0; id < target.armor_num; ++id) {
    const double angle = L6Telemetry::limit_rad(
      target.yaw + id * 2.0 * std::numbers::pi / target.armor_num);

    // 四板车的奇数板使用第二组半径和高度，与 L3 的 h_armor_xyz 一致。
    const bool use_alternate = target.armor_num == 4 && (id == 1 || id == 3);
    const double radius = use_alternate ? target.second_radius : target.radius;
    const double height = use_alternate ? target.height_diff : 0.0;

    armors.emplace_back(
      target.position.x() - radius * std::cos(angle),
      target.position.y() - radius * std::sin(angle),
      target.position.z() + height,
      angle);
  }
  return armors;
}

std::vector<Eigen::Vector4d> Predictor::armorPosesAt(
  const L3Estimation::TargetState& target, double dt) const
{
  return armorPoses(predict(target, dt));
}

}  // namespace L4Planning
