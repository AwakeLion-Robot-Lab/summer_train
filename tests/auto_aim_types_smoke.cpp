#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/target_estimator.hpp"
#include "l4_planning/types.hpp"
#include "l4_planning/predictor.hpp"
#include "l5_control/reject_reason.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/auto_aim_trace.hpp"
#include "runtime/auto_aim_config.hpp"
#include "runtime/l4_target_adapter.hpp"

#include <cmath>
#include <chrono>
#include <iostream>
#include <type_traits>

int main()
{
  static_assert(std::is_same_v<
    L3Estimation::Armor, L3Estimation::ArmorObservation>);

  L4Planning::Delay delay;
  delay.camera_timestamp = std::chrono::steady_clock::now();
  delay.command_timestamp = delay.camera_timestamp + std::chrono::milliseconds{10};
  delay.fire_delay = 0.010;
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

  const runtime::AutoAimConfig loaded_config =
    runtime::loadAutoAimConfig("config/auto_aim.yaml");
  if (!loaded_config.auto_layout || loaded_config.debug.force_work_mode != "auto_aim") {
    std::cerr << "Auto aim inference/debug configuration was not loaded\n";
    return 9;
  }
  auto decoder = L2Perception::yolov8_21DecoderConfig();
  runtime::DecoderThresholds thresholds;
  thresholds.confidence_threshold = 0.61F;
  thresholds.minimum_confidence = 1.5F;
  runtime::applyThresholds(thresholds, decoder);
  if (std::abs(decoder.confidence_threshold - 0.61F) > 1e-6F ||
      std::abs(decoder.minimum_confidence - 0.5F) > 1e-6F) {
    std::cerr << "Decoder threshold overrides did not preserve the preset fallback\n";
    return 10;
  }

  L3Estimation::Armor observation;
  observation.name = L3Estimation::ArmorName::Infantry3;
  observation.type = L3Estimation::ArmorType::Small;
  observation.xyz_in_world = {1.0, 0.0, 0.0};
  observation.ypr_in_world = {0.0, 0.0, 0.0};
  observation.ypd_in_world = {0.0, 0.0, 1.0};
  const auto timestamp = std::chrono::steady_clock::now();

  L3Estimation::TrackedTarget tracked_target(
    observation, timestamp, 0.2, 4, Eigen::VectorXd::Ones(L3Estimation::TrackedTarget::kStateSize));

  // 状态的元素顺序是 L3/L4 之间的硬契约：
  // [xc, vx, yc, vy, z, vz, yaw, v_yaw, r1, r2-r1, z2-z1, dz1, dz2]。
  // 前十一维顺序不可改（L4 按下标读），末两维是三板车的板间高度差。
  // 观测在 (1, 0, 0)、板 yaw 为 0、半径 0.2，所以旋转中心是 (1.2, 0, 0)。
  const Eigen::VectorXd x0 = tracked_target.ekf_x();
  const bool state_order_ok =
    x0.size() == L3Estimation::TrackedTarget::kStateSize &&
    std::abs(x0[0] - 1.2) < 1e-12 && x0[1] == 0.0 &&
    std::abs(x0[2]) < 1e-12 && x0[3] == 0.0 && x0[4] == 0.0 && x0[5] == 0.0 &&
    x0[6] == 0.0 && x0[7] == 0.0 && std::abs(x0[8] - 0.2) < 1e-12;
  if (!state_order_ok) {
    std::cerr << "TrackedTarget state order is incorrect\n";
    return 3;
  }

  // jumped 是粘滞的：还没关联到 0 号以外的板时保持 false，L4 据此只瞄
  // 当前观测到的那块板。
  if (tracked_target.jumped || tracked_target.armor_num() != 4 ||
      tracked_target.t() != timestamp) {
    std::cerr << "TrackedTarget initial contract is incorrect\n";
    return 4;
  }

  tracked_target.predict(0.01);
  tracked_target.update(observation);
  if (tracked_target.armor_xyza_list().size() != 4 ||
      tracked_target.name != observation.name ||
      !tracked_target.ekf_x().allFinite() ||
      !tracked_target.ekf().P.allFinite()) {
    std::cerr << "TrackedTarget update contract is incorrect\n";
    return 5;
  }

  L6Telemetry::AimTrace trace;
  trace.target = tracked_target;
  trace.track_state = L3Estimation::TrackState::Tracking;
  trace.plan.valid = true;
  trace.plan.fire_permitted = false;
  trace.fire.reasons.push_back(L5Control::RejectReason::ShootDisabled);
  const auto l4_target = runtime::toL4TargetState(trace.target);
  if (!trace.target || !l4_target ||
      std::abs(l4_target->center.x() - tracked_target.ekf_x()[0]) > 1e-12 ||
      l4_target->robot_id != static_cast<int>(tracked_target.name) ||
      l4_target->timestamp != tracked_target.t() ||
      !trace.plan.valid || trace.plan.fire_permitted ||
      trace.track_state != L3Estimation::TrackState::Tracking ||
      L5Control::toString(trace.fire.reasons.front()) != "shoot_disabled") {
    std::cerr << "AimTrace data contract is incorrect\n";
    return 6;
  }

  // L4 的命中时刻预测应与 L3 EKF 副本的 predict() 一致，且不推进原目标。
  L3Estimation::TrackedTarget observed(
    observation, timestamp, 0.2, 4,
    Eigen::VectorXd::Ones(L3Estimation::TrackedTarget::kStateSize));
  const auto snapshot = runtime::toL4TargetState(observed);
  if (!snapshot || !snapshot->filter_state) {
    std::cerr << "L4 snapshot is missing its EKF copy\n";
    return 7;
  }
  auto expected = observed;
  const auto impact_time = timestamp + std::chrono::milliseconds{50};
  expected.predict(impact_time);
  const auto predicted = L4Planning::Predictor{}.predict({*snapshot, impact_time});
  const auto expected_armor = expected.armor_xyza_list();
  if (!predicted.valid || predicted.armor_candidates.size() != expected_armor.size() ||
      (predicted.predicted_vehicle.covariance -
       expected.ekf().P.topLeftCorner<
         L3Estimation::STATE_DIM, L3Estimation::STATE_DIM>()).norm() > 1e-10 ||
      (predicted.armor_candidates.front().position_world -
       expected_armor.front().head<3>()).norm() > 1e-10 ||
      observed.t() != timestamp ||
      (observed.ekf_x() - snapshot->filter_state->ekf_x()).norm() > 1e-12) {
    std::cerr << "L4 prediction must use an isolated copy of the L3 EKF\n";
    return 8;
  }

  // 只有 Tracking 且规划有效时才画红色命中时刻预测板。
  std::optional<L3Estimation::TrackedTarget> overlay_target = observed;
  L4Planning::AimPlan invalid_plan;
  if (L6Telemetry::trackingRedArmorPose(
        overlay_target, L3Estimation::TrackState::Detecting, invalid_plan) ||
      L6Telemetry::plannedImpactArmorPose(overlay_target, invalid_plan)) {
    std::cerr << "Overlay drew a red/impact armor before tracking or planning\n";
    return 11;
  }
  if (L6Telemetry::trackingRedArmorPose(
        overlay_target, L3Estimation::TrackState::Tracking, invalid_plan)) {
    std::cerr << "Tracking overlay drew a red armor for an invalid plan\n";
    return 12;
  }

  L4Planning::AimPlan valid_plan;
  valid_plan.valid = true;
  valid_plan.armor_id = 1;
  valid_plan.impact_time = impact_time;
  const auto impact_armor =
    L6Telemetry::plannedImpactArmorPose(overlay_target, valid_plan);
  const auto planned_red = L6Telemetry::trackingRedArmorPose(
    overlay_target, L3Estimation::TrackState::Tracking, valid_plan);
  if (!impact_armor || !planned_red ||
      (*impact_armor - *planned_red).norm() > 1e-12) {
    std::cerr << "Valid plan did not select the predicted impact armor\n";
    return 13;
  }

  std::cout << "Auto aim data types smoke test passed\n";
  return 0;
}
