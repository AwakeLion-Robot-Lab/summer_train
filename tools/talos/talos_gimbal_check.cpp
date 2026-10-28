#include "l1_sensor/talos/gimbal_codec.hpp"
#include "l1_sensor/talos/talos_reader.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>

int main(int argc, char** argv)
{
  const int first = argc > 1 && std::string{argv[1]} == "--" ? 2 : 1;
  const double yaw_delta_deg = argc > first ? std::strtod(argv[first], nullptr) : 10.0;
  const double pitch_delta_deg = argc > first + 1 ? std::strtod(argv[first + 1], nullptr) : -5.0;
  const int frame_count = argc > first + 2 ? std::max(1, std::atoi(argv[first + 2])) : 300;
  L1Sensor::talos::TalosReader reader{"/tmp"};
  if (!reader.open()) {
    std::cerr << "talos_gimbal_check: simulator not running\n";
    return 1;
  }
  constexpr double kDegToRad = std::numbers::pi / 180.0;
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  L1Sensor::talos::TalosFrame first_frame;
  if (!reader.readFrame(first_frame, std::chrono::milliseconds{500})) {
    std::cerr << "talos_gimbal_check: initial frame timeout\n";
    return 1;
  }
  const auto first_pose = reader.pose(L1Sensor::talos::PoseIndex::Gimbal);
  if (!first_pose) return 1;
  Eigen::Quaterniond first_quaternion(
    first_pose->quaternion[0], first_pose->quaternion[1],
    first_pose->quaternion[2], first_pose->quaternion[3]);
  const Eigen::Vector3d first_rpy = L6Telemetry::rotationToRpy(
    first_quaternion.normalized().toRotationMatrix());
  const double initial_yaw_deg = first_rpy.z() * kRadToDeg;
  const double initial_pitch_deg = first_rpy.y() * kRadToDeg;
  const double target_yaw_deg = initial_yaw_deg + yaw_delta_deg;
  const double target_pitch_deg = initial_pitch_deg + pitch_delta_deg;
  const L5Control::SerialCommand hold{
    initial_yaw_deg * kDegToRad, initial_pitch_deg * kDegToRad, false};
  const L5Control::SerialCommand requested{
    target_yaw_deg * kDegToRad, target_pitch_deg * kDegToRad, false};

  // 先写入当前绝对角，再让操作者开启消费，杜绝 F5 读到旧槽位。
  reader.writeGimbalCmd(L1Sensor::talos::encodeGimbalCommand(hold, 1.0, false));
  if (reader.runtimeState().following == 0) {
    std::cerr << "talos_gimbal_check: current yaw=" << initial_yaw_deg
              << " pitch=" << initial_pitch_deg
              << "; now press F5, then run this command again\n";
    return 2;
  }
  std::cout << "initial yaw=" << initial_yaw_deg
            << " pitch=" << initial_pitch_deg
            << "; target yaw=" << target_yaw_deg
            << " pitch=" << target_pitch_deg << '\n';
  double last_yaw = 0.0;
  double last_pitch = 0.0;
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    L1Sensor::talos::TalosFrame frame;
    if (!reader.readFrame(frame, std::chrono::milliseconds{500})) {
      std::cerr << "talos_gimbal_check: frame timeout\n";
      return 1;
    }
    const auto& command = frame_index < 60 ? hold : requested;
    reader.writeGimbalCmd(L1Sensor::talos::encodeGimbalCommand(command, 1.0, false));
    const auto pose = reader.pose(L1Sensor::talos::PoseIndex::Gimbal);
    if (!pose) return 1;
    Eigen::Quaterniond quaternion(
      pose->quaternion[0], pose->quaternion[1],
      pose->quaternion[2], pose->quaternion[3]);
    const Eigen::Vector3d rpy = L6Telemetry::rotationToRpy(
      quaternion.normalized().toRotationMatrix());
    last_yaw = rpy.z() * kRadToDeg;
    last_pitch = rpy.y() * kRadToDeg; // 项目约定：正值低头，负值抬头
    if (frame_index % 30 == 0 || frame_index + 1 == frame_count) {
      std::cout << "frame=" << frame_index
                << " actual_yaw=" << last_yaw
                << " actual_pitch=" << last_pitch << '\n';
    }
  }
  const double yaw_error = std::remainder(last_yaw - target_yaw_deg, 360.0);
  const double pitch_error = last_pitch - target_pitch_deg;
  std::cout << "talos_gimbal_check: error yaw=" << yaw_error
            << " deg pitch=" << pitch_error << " deg, fire=0\n";
  return std::abs(yaw_error) <= 1.0 && std::abs(pitch_error) <= 1.0 ? 0 : 1;
}
