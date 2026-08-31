#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"
#include "l6_telemetry/auto_aim_trace.hpp"
#include "runtime/auto_aim_config.hpp"

#include <cmath>
#include <chrono>
#include <iostream>
#include <type_traits>

int main()
{
  static_assert(std::is_same_v<
    L3Estimation::Armor, L3Estimation::ArmorObservation>);

  L4Planning::Delay delay;
  delay.image_to_plan = 0.001;
  delay.plan_to_send = 0.002;
  delay.send_to_control = 0.003;
  delay.control_to_fire = 0.004;
  delay.fire_to_hit = 0.010;
  if (std::abs(delay.beforeFire() - 0.010) > 1e-12 ||
      std::abs(delay.total() - 0.020) > 1e-12) {
    std::cerr << "Delay aggregation is incorrect\n";
    return 1;
  }

  runtime::AutoAimConfig config;
  if (config.fire.shoot_enable) {
    std::cerr << "Auto aim configuration is not safe by default\n";
    return 2;
  }

  L3Estimation::Armor observation;
  observation.name = L3Estimation::ArmorName::Infantry3;
  observation.type = L3Estimation::ArmorType::Small;
  observation.xyz_in_world = {1.0, 0.0, 0.0};
  observation.ypr_in_world = {0.0, 0.0, 0.0};
  observation.ypd_in_world = {0.0, 0.0, 1.0};
  const auto timestamp = std::chrono::steady_clock::now();

  // 确定性构造：旋转中心 (1.2, 0, 0)、半径 0.2、静止。不经观测，因为这里
  // 验的是状态布局契约本身，不是滤波器行为（那在 tests/eskf_target_smoke.cpp）。
  L3Estimation::EskfTarget tracked_target(
    observation.name, 1.2, 0.0, 0.2);

  // 状态的元素顺序是 L3/L4 之间的硬契约：
  // [cx, vcx, cy, vcy, cz, vcz, rot_z, vyaw, r1, r2, h, rot_y, rot_x]。
  // 前九维下标含义不可改——L4 的 planner 按下标读 x[0]/x[2]/x[7]/x[8]。
  // 注意 ekf_x() 对外吐的第 8、9 维是**线性**半径，内部存的是对数。
  const Eigen::VectorXd x0 = tracked_target.ekf_x();
  const bool state_order_ok =
    x0.size() == L3Estimation::EskfTarget::kStateSize &&
    std::abs(x0[0] - 1.2) < 1e-12 && x0[1] == 0.0 &&
    std::abs(x0[2]) < 1e-12 && x0[3] == 0.0 && x0[4] == 0.0 && x0[5] == 0.0 &&
    x0[6] == 0.0 && x0[7] == 0.0 && std::abs(x0[8] - 0.2) < 1e-12;
  if (!state_order_ok) {
    std::cerr << "EskfTarget state order is incorrect\n";
    return 3;
  }

  if (tracked_target.armor_num() != 4) {
    std::cerr << "EskfTarget initial contract is incorrect\n";
    return 4;
  }

  tracked_target.predict(0.01);
  if (tracked_target.armor_xyza_list().size() != 4 ||
      tracked_target.name != observation.name ||
      !tracked_target.ekf_x().allFinite()) {
    std::cerr << "EskfTarget update contract is incorrect\n";
    return 5;
  }

  L6Telemetry::AimTrace trace;
  trace.target = tracked_target;
  trace.track_state = L3Estimation::TrackState::Tracking;
  trace.plan.status = L4Planning::PlanStatus::TrackOnly;
  trace.plan.reason = L4Planning::PlanError::BadBulletSpeed;
  trace.fire.reasons.push_back(L5Control::RejectReason::ShootDisabled);
  if (!trace.target || !trace.plan.valid() || trace.plan.fireAdmissible() ||
      trace.track_state != L3Estimation::TrackState::Tracking ||
      L5Control::toString(trace.fire.reasons.front()) != "shoot_disabled") {
    std::cerr << "AimTrace data contract is incorrect\n";
    return 6;
  }

  std::cout << "Auto aim data types smoke test passed\n";
  return 0;
}
