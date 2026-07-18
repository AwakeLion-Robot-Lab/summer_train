#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"
#include "l6_telemetry/auto_aim_trace.hpp"
#include "runtime/auto_aim_config.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>

int main()
{
  static_assert(std::is_same_v<
    L3Estimation::Armor, L3Estimation::ArmorObservation>);
  static_assert(std::is_same_v<
    L3Estimation::Target, L3Estimation::TargetState>);
  static_assert(std::is_same_v<L4Planning::Plan, L4Planning::AimPlan>);

  L3Estimation::Target target;
  target.position = {1.0, 2.0, 3.0};
  target.velocity = {4.0, 5.0, 6.0};
  target.yaw = 0.7;
  target.v_yaw = 0.8;
  target.radius = 0.22;

  const auto x = target.vector();
  const bool state_order_ok =
    x[0] == 1.0 && x[1] == 4.0 && x[2] == 2.0 && x[3] == 5.0 &&
    x[4] == 3.0 && x[5] == 6.0 && x[6] == 0.7 && x[7] == 0.8 &&
    x[8] == 0.22;
  if (!state_order_ok) {
    std::cerr << "Target state order is incorrect\n";
    return 1;
  }

  L4Planning::Delay delay;
  delay.image_to_plan = 0.001;
  delay.plan_to_send = 0.002;
  delay.send_to_control = 0.003;
  delay.control_to_fire = 0.004;
  delay.fire_to_hit = 0.010;
  if (std::abs(delay.beforeFire() - 0.010) > 1e-12 ||
      std::abs(delay.total() - 0.020) > 1e-12) {
    std::cerr << "Delay aggregation is incorrect\n";
    return 2;
  }

  runtime::AutoAimConfig config;
  L3Estimation::AimCalibration calibration;
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
