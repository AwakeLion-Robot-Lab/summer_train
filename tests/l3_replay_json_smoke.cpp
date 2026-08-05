#include "l6_telemetry/l3_replay_json.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main()
{
  try {
    L3Estimation::ArmorObservation observation;
    observation.robot_id = 3;
    observation.position_world = {1.0, 2.0, 3.0};
    observation.rpy_raw_world = {0.1, 0.2, 0.3};
    observation.rpy_constrained_world = {0.0, 0.25, 0.35};
    observation.confidence = 0.9F;

    L3Estimation::TargetState target;
    target.robot_id = 3;
    target.center = {1.1, 2.1, 3.1};
    target.velocity = {0.4, 0.5, 0.6};
    target.yaw = 0.35;
    target.yaw_rate = 1.2;
    target.radius = 0.20;
    target.radius_offset = 0.05;

    const auto payload = L6Telemetry::makeL3ReplayJson(
      12, 0.24, {0.01, 0.02, 0.03}, &observation, &target,
      true, 4.0, 0.5, 2.5, 7);
    require(payload["frame_index"] == 12, "frame index is missing");
    require(
      payload["geometry_constraints_enabled"] == 1,
      "geometry constraint state is missing");
    require(
      payload["l3"]["observation"]["valid"] == true
        && payload["l3"]["target"]["valid"] == true,
      "valid L3 samples were not marked valid");
    require(
      std::abs(payload["gimbal"]["rpy_rad"]["yaw"].get<double>() - 0.03) < 1e-12
        && std::abs(payload["l3"]["observation"]["raw_rpy_rad"]["pitch"].get<double>() - 0.2) < 1e-12
        && std::abs(payload["l3"]["observation"]["optimized_rpy_rad"]["yaw"].get<double>() - 0.35) < 1e-12,
      "RPY curve paths contain wrong values");
    require(
      std::abs(payload["performance"]["realtime_lag_ms"].get<double>() - 2.5)
          < 1e-12
        && payload["performance"]["skipped_frames"] == 7,
      "realtime performance paths contain wrong values");
    require(
      std::abs(
        payload["l3"]["target"]["geometry"]["radius_2_m"].get<double>()
        - 0.25) < 1e-12
        && payload["l3"]["target"]["geometry"]
             ["minimum_corner_angle_rad"].get<double>()
             > 50.0 * std::numbers::pi / 180.0,
      "vehicle geometry curve paths contain wrong values");

    const auto empty = L6Telemetry::makeL3ReplayJson(
      13, 0.26, Eigen::Vector3d::Zero(), nullptr, nullptr,
      false, 3.0, 0.4, 0.0, 0);
    require(
      empty["l3"]["observation"]["valid"] == false
        && !empty["l3"]["observation"].contains("raw_rpy_rad")
        && empty["l3"]["target"]["valid"] == false
        && !empty["l3"]["target"].contains("yaw_rad"),
      "missing L3 samples were replaced with fake zero values");
  } catch (const std::exception& error) {
    std::cerr << "L3 replay JSON smoke test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "L3 replay JSON smoke test passed\n";
  return 0;
}
