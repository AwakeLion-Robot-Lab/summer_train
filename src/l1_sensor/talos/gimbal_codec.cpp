#include "l1_sensor/talos/gimbal_codec.hpp"

#include "l1_sensor/talos/talos_reader.hpp"

#include <numbers>

namespace L1Sensor::talos {

GimbalCmd encodeGimbalCommand(
  const L5Control::SerialCommand& command,
  double distance_m,
  bool shoot_enable) noexcept
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  GimbalCmd output;
  output.timestamp_ns = TalosReader::realtimeNowNs();
  output.yaw_deg = static_cast<float>(command.yaw * kRadToDeg);
  output.pitch_deg = static_cast<float>(-command.pitch * kRadToDeg - 90.0);
  output.distance_m = static_cast<float>(distance_m);
  output.fire_advice = command.shoot && shoot_enable ? 1 : 0;
  return output;
}

SimulatorSetpoint decodeSimulatorSetpoint(const GimbalCmd& command) noexcept
{
  constexpr double kDegToRad = std::numbers::pi / 180.0;
  // Talos wire pitch=-project_pitch-90deg；Rust 解码后直接按项目语义执行。
  // 项目 pitch 遵循右手绕 Y：正值低头，负值抬头。
  return {command.yaw_deg * kDegToRad,
          (-command.pitch_deg - 90.0) * kDegToRad};
}

} // namespace L1Sensor::talos
