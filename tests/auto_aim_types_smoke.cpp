// 跨层数据契约的冒烟测试：延迟分段、整车状态的元素顺序、默认配置的安全性。

#include "l3_estimation/tracked_target.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"
#include "runtime/auto_aim_config.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

int main()
{
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

  // 默认配置必须是安全的：没有实车标定就不能解锁开火。
  const runtime::AutoAimConfig config;
  if (config.inference_backend != L2Perception::InferenceBackendKind::OpenVino ||
      config.fire.shoot_enable || config.fire.parametersReady() ||
      config.plan.fireDelayReady()) {
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

  L3Estimation::TargetStateVector initial_state =
    L3Estimation::TargetStateVector::Zero();
  initial_state[L3Estimation::CenterX] = 1.2;
  initial_state[L3Estimation::RadiusA] = 0.2;
  const L3Estimation::TargetCovariance initial_covariance =
    L3Estimation::TargetCovariance::Identity();
  L3Estimation::TrackedTarget tracked_target(
    observation.name, observation.type, 4, timestamp, initial_state,
    initial_covariance);

  // 内部十一维状态的元素顺序是 L3/L4 之间的硬契约：
  // [xc, vx, yc, vy, z, vz, yaw, v_yaw, r1, r2-r1, z2-z1]。
  // 观测在 (1, 0, 0)、板 yaw 为 0、半径 0.2，所以旋转中心是 (1.2, 0, 0)。
  const L3Estimation::TargetStateVector& x0 = tracked_target.state();
  const bool state_order_ok =
    x0.size() == 11 && std::abs(x0[0] - 1.2) < 1e-12 && x0[1] == 0.0 &&
    std::abs(x0[2]) < 1e-12 && x0[3] == 0.0 && x0[4] == 0.0 && x0[5] == 0.0 &&
    x0[6] == 0.0 && x0[7] == 0.0 && std::abs(x0[8] - 0.2) < 1e-12;
  if (!state_order_ok) {
    std::cerr << "TrackedTarget state order is incorrect\n";
    return 3;
  }

  // jumped 是粘滞的：还没关联到 0 号以外的板时保持 false，L4 据此只瞄
  // 当前观测到的那块板。
  if (tracked_target.jumped || tracked_target.armorCount() != 4 ||
      tracked_target.timestamp() != timestamp) {
    std::cerr << "TrackedTarget initial contract is incorrect\n";
    return 4;
  }

  tracked_target.predict(0.01);
  if (tracked_target.armorPoses().size() != 4 ||
      tracked_target.name != observation.name ||
      !tracked_target.state().allFinite() ||
      !tracked_target.covariance().allFinite()) {
    std::cerr << "TrackedTarget update contract is incorrect\n";
    return 5;
  }

  if (L5Control::toString(L5Control::RejectReason::ShootDisabled) != "shoot_disabled") {
    std::cerr << "RejectReason naming is incorrect\n";
    return 6;
  }

  std::cout << "Auto aim data types smoke test passed\n";
  return 0;
}
