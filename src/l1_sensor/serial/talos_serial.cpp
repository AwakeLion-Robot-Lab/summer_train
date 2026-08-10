#include "l1_sensor/serial/talos_serial.hpp"

#include "l6_telemetry/math.hpp"

#include <numbers>

namespace L1Sensor {
namespace {

// 仿真器发布的云台四元数是 ROS 相机系（X 前、Y 左、Z 上）到世界系的旋转；
// L3 的 PnP 输出在 OpenCV 相机系（X 右、Y 下、Z 前）。这里做固定轴转换：
//   R_world_barrel = R_ros_camera * T_ros_to_opencv
// 其中 T 的列是 ROS 轴在 OpenCV 系中的坐标，T_ros_to_opencv = [[0,0,1],[-1,0,0],[0,-1,0]]，
// 其逆（转置）为 [[0,0,1],[-1,0,0],[0,-1,0]] 的转置。
// 实测若发现方位错 90°/镜像，只改这里即可。
const Eigen::Matrix3d kRosToOpencvInverse =
  (Eigen::Matrix3d() <<
   0.0, 0.0, 1.0,
   -1.0, 0.0, 0.0,
   0.0, -1.0, 0.0).finished();

}  // namespace

TalosSerial::TalosSerial(
  std::shared_ptr<talos::TalosReader> reader,
  talos::TalosSimConfig config)
  : reader_(std::move(reader)), config_(std::move(config))
{
}

std::optional<RobotState> TalosSerial::latestState() const
{
  if (!reader_ || !reader_->isOpen()) {
    return std::nullopt;
  }
  const auto gimbal = reader_->pose(talos::PoseIndex::Gimbal);
  if (!gimbal || gimbal->timestamp_ns == 0) {
    return std::nullopt;
  }
  const talos::ChassisObservation chassis = reader_->chassisObservation();

  RobotState state;
  const Eigen::Quaterniond q(
    gimbal->quaternion[0], gimbal->quaternion[1],
    gimbal->quaternion[2], gimbal->quaternion[3]);
  const Eigen::Vector3d rpy =
    (q.norm() > 1e-9 && q.coeffs().allFinite())
      ? L6Telemetry::rotationToRpy(q.normalized().toRotationMatrix())
      : Eigen::Vector3d::Zero();
  state.rpy.roll = rpy.x();
  state.rpy.pitch = rpy.y();
  state.rpy.yaw = rpy.z();
  state.yaw_rate = chassis.wz_radps;
  state.pitch_rate = 0.0;
  state.bullet_speed = config_.bullet_speed;
  state.heat = 0.0;
  state.enemy_color = config_.enemy_color;
  state.mode = config_.mode;
  state.timestamp = reader_->toSteadyTime(gimbal->timestamp_ns);
  return state;
}

std::optional<Eigen::Quaterniond> TalosSerial::gimbalPoseAt(
  std::chrono::steady_clock::time_point /*timestamp*/) const
{
  if (!reader_ || !reader_->isOpen()) {
    return std::nullopt;
  }
  const auto gimbal = reader_->pose(talos::PoseIndex::Gimbal);
  if (!gimbal || gimbal->timestamp_ns == 0) {
    return std::nullopt;
  }
  const Eigen::Quaterniond q_ros(
    gimbal->quaternion[0], gimbal->quaternion[1],
    gimbal->quaternion[2], gimbal->quaternion[3]);
  if (q_ros.norm() < 1e-9 || !q_ros.coeffs().allFinite()) {
    return std::nullopt;
  }
  const Eigen::Quaterniond q_world_barrel(
    (q_ros.normalized().toRotationMatrix() * kRosToOpencvInverse));
  return q_world_barrel.normalized();
}

void TalosSerial::updateCommand(
  const L5Control::SerialCommand& command, double distance_m)
{
  if (!reader_ || !reader_->isOpen()) {
    return;
  }
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  talos::GimbalCmd cmd;
  cmd.timestamp_ns = talos::TalosReader::realtimeNowNs();
  // 仿真器约定：local_yaw = yaw_deg.to_radians()，
  // pitch = (-pitch_deg - 90°).to_radians()，即 -90° 为水平。
  cmd.yaw_deg = static_cast<float>(command.yaw * kRadToDeg);
  cmd.pitch_deg =
    static_cast<float>(-command.pitch * kRadToDeg - 90.0);
  cmd.distance_m = static_cast<float>(
    distance_m > 0.0 ? distance_m : config_.default_distance_m);
  cmd.fire_advice = command.shoot ? 1 : 0;
  reader_->writeGimbalCmd(cmd);
}

}  // namespace L1Sensor
