#include "l6_telemetry/l3_replay_json.hpp"

namespace L6Telemetry {
namespace {

nlohmann::json vectorJson(const Eigen::Vector3d& value)
{
  return {{"x", value.x()}, {"y", value.y()}, {"z", value.z()}};
}

nlohmann::json rpyJson(const Eigen::Vector3d& value)
{
  return {{"roll", value.x()}, {"pitch", value.y()}, {"yaw", value.z()}};
}

}  // namespace

nlohmann::json makeL3ReplayJson(
  int frame_index,
  double time_seconds,
  const Eigen::Vector3d& gimbal_rpy_rad,
  const L3Estimation::ArmorObservation* observation,
  const L3Estimation::TargetState* target,
  bool geometry_constraints_enabled,
  double l2_ms,
  double l3_ms,
  double realtime_lag_ms,
  std::size_t skipped_frames)
{
  nlohmann::json result = {
    {"frame_index", frame_index},
    {"time_s", time_seconds},
    {"gimbal", {{"rpy_rad", rpyJson(gimbal_rpy_rad)}}},
    {"geometry_constraints_enabled", geometry_constraints_enabled ? 1 : 0},
    {"performance", {
      {"l2_ms", l2_ms},
      {"l3_ms", l3_ms},
      {"realtime_lag_ms", realtime_lag_ms},
      {"skipped_frames", skipped_frames},
    }},
    {"l3", {
      {"observation", {{"valid", observation != nullptr}}},
      {"target", {{"valid", target != nullptr}}},
    }},
  };

  if (observation != nullptr) {
    result["l3"]["observation"].update({
      {"robot_id", observation->robot_id},
      {"raw_rpy_rad", rpyJson(observation->rpy_raw_world)},
      {"optimized_rpy_rad", rpyJson(observation->rpy_constrained_world)},
      {"position_m", vectorJson(observation->position_world)},
      {"confidence", observation->confidence},
      {"pnp_error_px", observation->pnp_reprojection_error_px},
      {"optimized_error_px", observation->optimized_reprojection_error_px},
    });
  }

  if (target != nullptr) {
    const double second_radius = target->radius
      + (target->model == L3Estimation::TargetModel::FourArmorVehicle
           ? target->radius_offset
           : 0.0);
    result["l3"]["target"].update({
      {"robot_id", target->robot_id},
      {"center_m", vectorJson(target->center)},
      {"velocity_mps", vectorJson(target->velocity)},
      {"yaw_rad", target->yaw},
      {"yaw_rate_rad_s", target->yaw_rate},
      {"radius_m", target->radius},
      {"geometry", {
        {"radius_1_m", target->radius},
        {"radius_2_m", second_radius},
        {"height_offset_m", target->height_offset},
        {"minimum_corner_angle_rad",
         target->model == L3Estimation::TargetModel::FourArmorVehicle
           ? L3Estimation::fourArmorMinimumCornerAngle(
               target->radius, second_radius)
           : 0.0},
      }},
      {"updated_this_frame", target->updated_this_frame},
    });
  }

  return result;
}

}  // namespace L6Telemetry
