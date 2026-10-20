#include "l1_sensor/talos/gimbal_codec.hpp"
#include "l1_sensor/talos/shm_layout.hpp"
#include "l6_telemetry/math.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>

int main()
{
  static_assert(sizeof(L1Sensor::talos::ShmMetaRegion) == 3712);
  constexpr double kTolerance = 2e-6;
  const std::array<L5Control::SerialCommand, 5> commands{{
    {0.0, 0.0, false},
    {20.0 * std::numbers::pi / 180.0, 8.0 * std::numbers::pi / 180.0, true},
    {-35.0 * std::numbers::pi / 180.0, -12.0 * std::numbers::pi / 180.0, true},
    {0.2, -0.1, false},
    {-0.2, 0.1, false},
  }};
  for (const auto& input : commands) {
    const auto encoded = L1Sensor::talos::encodeGimbalCommand(input, 3.0, false);
    const auto decoded = L1Sensor::talos::decodeSimulatorSetpoint(encoded);
    // Talos 解码后必须保持项目语义：pitch 正值低头，负值抬头。
    if (std::abs(decoded.yaw_ros - input.yaw) > kTolerance ||
        std::abs(decoded.pitch_ros - input.pitch) > kTolerance ||
        encoded.fire_advice != 0 || encoded.distance_m != 3.0F) {
      std::printf("talos_contract_smoke: FAIL\n");
      return 1;
    }
    const auto repeated = L1Sensor::talos::encodeGimbalCommand(input, 3.0, false);
    if (repeated.yaw_deg != encoded.yaw_deg || repeated.pitch_deg != encoded.pitch_deg) {
      std::printf("talos_contract_smoke: FAIL non-idempotent setpoint\n");
      return 1;
    }
  }

  // yaw 穿过 0 时必须保持同一欧拉分支，不能跳成 yaw≈180°、pitch>90°。
  constexpr double kPitch = 33.6901 * std::numbers::pi / 180.0;
  for (const double yaw : {-1e-6, 0.0, 1e-6}) {
    const Eigen::Matrix3d rotation = L6Telemetry::rpyToRotation({0.0, kPitch, yaw});
    const Eigen::Vector3d rpy = L6Telemetry::rotationToRpy(rotation);
    if (std::abs(rpy.x()) > kTolerance ||
        std::abs(rpy.y() - kPitch) > kTolerance ||
        std::abs(rpy.z() - yaw) > kTolerance) {
      std::printf("talos_contract_smoke: FAIL discontinuous Euler branch\n");
      return 1;
    }
  }
  std::printf("talos_contract_smoke: ok\n");
  return 0;
}
