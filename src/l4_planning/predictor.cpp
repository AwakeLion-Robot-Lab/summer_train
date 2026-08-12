#include "l4_planning/predictor.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>

namespace L4Planning {

L3Estimation::TrackedTarget Predictor::predict(
  const L3Estimation::TrackedTarget& target, double dt) const
{
  auto predicted = target;
  if (!std::isfinite(dt)) {
    return predicted;
  }

  // 走绝对时间入口，滤波器时刻随之推进，返回的副本自带正确的 t()。
  predicted.predict(target.timestamp() + L6Telemetry::toDuration(dt));
  return predicted;
}

std::vector<Eigen::Vector4d> Predictor::armorPoses(
  const L3Estimation::TrackedTarget& target) const
{
  return target.armorPoses();
}

std::vector<Eigen::Vector4d> Predictor::armorPosesAt(
  const L3Estimation::TrackedTarget& target, double dt) const
{
  return armorPoses(predict(target, dt));
}

}  // namespace L4Planning
