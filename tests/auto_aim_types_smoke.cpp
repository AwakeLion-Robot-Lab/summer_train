#include "l3_estimation/types.hpp"
#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"
#include "l6_telemetry/auto_aim_trace.hpp"
#include "runtime/auto_aim_config.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

int main()
{
  static_assert(std::is_same_v<L4Planning::Plan, L4Planning::AimPlan>);

  L3Estimation::TargetState target;
  target.center = {1.0, 2.0, 3.0};
  target.velocity = {4.0, 5.0, 6.0};
  target.yaw = 0.7;
  target.yaw_rate = 0.8;
  target.radius = 0.22;

  const bool state_order_ok =
    target.center.x() == 1.0 && target.velocity.x() == 4.0 &&
    target.center.y() == 2.0 && target.velocity.y() == 5.0 &&
    target.center.z() == 3.0 && target.velocity.z() == 6.0 &&
    target.yaw == 0.7 && target.yaw_rate == 0.8 &&
    target.radius == 0.22 &&
    target.covariance.rows() == L3Estimation::STATE_DIM &&
    target.covariance.cols() == L3Estimation::STATE_DIM;
  if (!state_order_ok) {
    std::cerr << "Target state order is incorrect\n";
    return 1;
  }

  const auto image_time = L4Planning::TimePoint{std::chrono::milliseconds{100}};
  const auto command_time = image_time + std::chrono::milliseconds{6};
  L4Planning::LatencyConfig latency_config;
  const L4Planning::LatencyCompensator latency_compensator{latency_config};
  const auto latency = latency_compensator.calculate(
    L4Planning::Delay{image_time, command_time, 0.010});
  if (!latency.valid ||
      std::abs(latency.delay.beforeFire() - 0.006) > 1e-12 ||
      std::abs(latency.delay.total() - 0.016) > 1e-12) {
    std::cerr << "Delay aggregation is incorrect\n";
    return 2;
  }
  if (latency_compensator.calculate(
        L4Planning::Delay{command_time, image_time, 0.0}).valid) {
    std::cerr << "Invalid latency input was accepted\n";
    return 2;
  }
  if (latency_compensator.calculate(
        L4Planning::Delay{
          image_time, command_time,
          std::numeric_limits<double>::quiet_NaN()}).valid ||
      latency_compensator.calculate(
        L4Planning::Delay{
          image_time, command_time,
          std::numeric_limits<double>::infinity()}).valid ||
      latency_compensator.calculate(
        L4Planning::Delay{image_time, command_time, -0.001}).valid) {
    std::cerr << "Invalid fire delay was accepted\n";
    return 2;
  }

  runtime::AutoAimConfig config;
  L1Sensor::CameraCalibration calibration;
  if (config.fire.shoot_enable || config.fireReady(calibration)) {
    std::cerr << "Auto aim configuration is not safe by default\n";
    return 3;
  }

  L6Telemetry::AimTrace trace;
  trace.target = target;
  trace.plan.target_id = 3;
  trace.fire.reasons.push_back(L5Control::RejectReason::ShootDisabled);
  if (!trace.target || trace.plan.target_id != 3 ||
      L5Control::toString(trace.fire.reasons.front()) != "shoot_disabled") {
    std::cerr << "AimTrace data contract is incorrect\n";
    return 4;
  }

  std::cout << "Auto aim data types smoke test passed\n";
  return 0;
}
