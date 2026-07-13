#include "l6_telemetry/math.hpp"

#include <algorithm>

namespace L6Telemetry {


double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b)
{
  std::chrono::duration<double> c = a - b;
  return c.count();
}

Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw)
{
  //标准的右手坐标系
  Eigen::Quaterniond q =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  q.normalize();
  return q;
}

Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k)
{
  const double ratio = std::clamp(k, 0.0, 1.0);
  return a.normalized().slerp(ratio, b.normalized()).normalized();
}

}
