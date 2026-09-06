#include "l3_estimation/target_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace L3Estimation {
namespace {

bool isRotationMatrix(const Eigen::Matrix3d& rotation) noexcept
{
  if (!rotation.allFinite()) {
    return false;
  }
  const Eigen::Matrix3d error =
    rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  return error.norm() < 1e-5
         && std::abs(rotation.determinant() - 1.0) < 1e-5;
}

bool isValidQuaternion(const Eigen::Quaterniond& quaternion) noexcept
{
  return quaternion.coeffs().allFinite()
         && quaternion.norm() > 1e-9;
}

}  // namespace

TargetEstimator::TargetEstimator(
  L1Sensor::CameraCalibration calibration,
  BarrelPoseProvider barrel_pose_provider,
  L3Config config)
  : config_(std::move(config)),
    pnp_solver_(
      calibration,
      config_.armor.dimensions,
      config_.pnp),
    yaw_optimizer_(
      calibration,
      config_.armor.dimensions,
      config_.yaw_search),
    calibration_image_size_(calibration.image_size),
    barrel_pose_provider_(std::move(barrel_pose_provider))
{
  if (!isValidL3Config(config_)) {
    throw std::invalid_argument(
      "TargetEstimator received an invalid L3 config");
  }
  if (!calibration.T_barrel_camera) {
    throw std::invalid_argument(
      "TargetEstimator requires calibrated T_barrel_camera");
  }
  T_barrel_camera_ = *calibration.T_barrel_camera;
  if (!isRotationMatrix(T_barrel_camera_.linear())
      || !T_barrel_camera_.translation().allFinite()) {
    throw std::invalid_argument(
      "TargetEstimator received an invalid T_barrel_camera");
  }
  if (calibration_image_size_.width <= 0
      || calibration_image_size_.height <= 0) {
    throw std::invalid_argument(
      "TargetEstimator requires a positive calibration image size");
  }
  if (!barrel_pose_provider_) {
    throw std::invalid_argument(
      "TargetEstimator requires a barrel pose provider");
  }
}

ArmorSize TargetEstimator::armorSizeFromClass(int class_id) const noexcept
{
  const auto armor_class = L2Perception::armorClassFromId(class_id);
  return armor_class == L2Perception::ArmorClass::Hero
           || armor_class == L2Perception::ArmorClass::BaseLarge
         ? ArmorSize::Large
         : ArmorSize::Small;
}

int TargetEstimator::robotIdFromClass(int class_id) const noexcept
{
  const auto armor_class = L2Perception::armorClassFromId(class_id);
  if (armor_class == L2Perception::ArmorClass::Unknown
      || armor_class == L2Perception::ArmorClass::BaseSmall
      || armor_class == L2Perception::ArmorClass::BaseLarge) {
    return -1;
  }
  return static_cast<int>(armor_class);
}

std::optional<TargetModel> TargetEstimator::targetModelFromClass(
  int class_id) const noexcept
{
  const auto armor_class = L2Perception::armorClassFromId(class_id);
  if (armor_class == L2Perception::ArmorClass::Outpost) {
    return TargetModel::ThreeArmorOutpost;
  }
  if (armor_class == L2Perception::ArmorClass::Unknown
      || armor_class == L2Perception::ArmorClass::BaseSmall
      || armor_class == L2Perception::ArmorClass::BaseLarge) {
    return std::nullopt;
  }
  return TargetModel::FourArmorVehicle;
}

Eigen::Vector3d TargetEstimator::positionInWorld(
  const ArmorPose& pose,
  const Eigen::Quaterniond& R_world_barrel) const noexcept
{
  const Eigen::Vector3d position_camera{
    pose.tvec[0],
    pose.tvec[1],
    pose.tvec[2]};
  const Eigen::Vector3d position_barrel =
    T_barrel_camera_ * position_camera;
  return R_world_barrel * position_barrel;
}

std::optional<ArmorObservation> TargetEstimator::makeObservation(
  const L2Perception::ArmorDetection& armor,
  TimePoint timestamp,
  const Eigen::Quaterniond& R_world_barrel) const
{
  const int robot_id = robotIdFromClass(armor.class_id);
  const auto model = targetModelFromClass(armor.class_id);
  if (robot_id < 0 || !model || !std::isfinite(armor.confidence)) {
    return std::nullopt;
  }

  // 第一版每块检测只求一个 IPPE 位姿。
  const ArmorSize armor_size = armorSizeFromClass(armor.class_id);
  const auto pose = pnp_solver_.solve(armor, armor_size);
  if (!pose) {
    return std::nullopt;
  }

  // 固定 PnP 位置和模型 pitch，以 1°步长遍历世界系 yaw。
  const auto yaw = yaw_optimizer_.optimize(
    armor,
    armor_size,
    *pose,
    R_world_barrel,
    config_.armor.parameters(*model).pitch_rad);
  if (!yaw) {
    return std::nullopt;
  }

  const Eigen::Vector3d position_world =
    positionInWorld(*pose, R_world_barrel);
  if (!position_world.allFinite()) {
    return std::nullopt;
  }

  return ArmorObservation{
    .robot_id = robot_id,
    .armor_class = L2Perception::armorClassFromId(armor.class_id),
    .model = *model,
    .position_world = position_world,
    .yaw_raw_world = yaw->yaw_raw_world,
    .yaw_world = yaw->yaw_optimized_world,
    .confidence = armor.confidence,
    .pnp_reprojection_error_px =
      yaw->pnp_reprojection_error_px,
    .raw_yaw_reprojection_error_px =
      yaw->raw_yaw_reprojection_error_px,
    .optimized_reprojection_error_px =
      yaw->optimized_reprojection_error_px,
    .timestamp = timestamp};
}

std::vector<ArmorObservation> TargetEstimator::buildObservations(
  const std::vector<L2Perception::ArmorDetection>& armors,
  TimePoint timestamp,
  const Eigen::Quaterniond& R_world_barrel) const
{
  std::vector<ArmorObservation> observations;
  observations.reserve(armors.size());
  for (const auto& armor : armors) {
    if (auto observation = makeObservation(
          armor,
          timestamp,
          R_world_barrel)) {
      observations.push_back(std::move(*observation));
    }
  }
  return observations;
}

void TargetEstimator::predictTrackers(TimePoint timestamp)
{
  for (auto& [robot_id, tracker] : trackers_) {
    (void)robot_id;
    tracker.predict(timestamp);
  }
}

void TargetEstimator::updateTrackers(
  const std::vector<ArmorObservation>& observations)
{
  std::unordered_map<int, std::vector<ArmorObservation>>
    observations_by_robot;
  for (const auto& observation : observations) {
    observations_by_robot[observation.robot_id].push_back(observation);
  }

  // 已有 Tracker 即使本帧没有观测也必须完成一次生命周期判断。
  for (auto& [robot_id, tracker] : trackers_) {
    const auto group = observations_by_robot.find(robot_id);
    const std::vector<ArmorObservation> empty;
    const auto& robot_observations =
      group == observations_by_robot.end() ? empty : group->second;
    auto diagnostics = tracker.update(robot_observations);
    last_association_diagnostics_.insert(
      last_association_diagnostics_.end(),
      std::make_move_iterator(diagnostics.begin()),
      std::make_move_iterator(diagnostics.end()));
    if (group != observations_by_robot.end()) {
      observations_by_robot.erase(group);
    }
  }

  // 每个新 robot_id 只建立一个整车假设。
  for (auto& [robot_id, robot_observations] :
       observations_by_robot) {
    if (robot_observations.empty()) {
      continue;
    }
    const TargetModel model = robot_observations.front().model;
    auto [iterator, inserted] = trackers_.try_emplace(
      robot_id,
      robot_id,
      model,
      config_.trackerConfig(model));
    (void)inserted;
    auto diagnostics = iterator->second.update(robot_observations);
    last_association_diagnostics_.insert(
      last_association_diagnostics_.end(),
      std::make_move_iterator(diagnostics.begin()),
      std::make_move_iterator(diagnostics.end()));
  }
}

void TargetEstimator::removeExpiredTrackers(TimePoint timestamp)
{
  std::erase_if(
    trackers_,
    [timestamp](const auto& item) {
      return item.second.expired(timestamp);
    });
}

std::vector<TargetState> TargetEstimator::collectTargets() const
{
  std::vector<TargetState> targets;
  targets.reserve(trackers_.size());
  for (const auto& [robot_id, tracker] : trackers_) {
    (void)robot_id;
    if (tracker.state().timestamp != TimePoint{}) {
      targets.push_back(tracker.state());
    }
  }
  std::sort(
    targets.begin(),
    targets.end(),
    [](const auto& lhs, const auto& rhs) {
      return lhs.robot_id < rhs.robot_id;
    });
  return targets;
}

std::vector<TargetState> TargetEstimator::update(
  const std::vector<L2Perception::ArmorDetection>& armors,
  const FrameContext& frame_context)
{
  if (frame_context.image_size.width <= 0
      || frame_context.image_size.height <= 0) {
    throw std::invalid_argument(
      "TargetEstimator requires a positive frame image size");
  }
  if (config_.require_matching_image_size
      && frame_context.image_size != calibration_image_size_) {
    throw std::invalid_argument(
      "TargetEstimator frame image size does not match calibration");
  }

  last_observations_.clear();
  last_association_diagnostics_.clear();
  predictTrackers(frame_context.timestamp);

  const auto barrel_pose =
    barrel_pose_provider_(frame_context.timestamp);
  if (barrel_pose && isValidQuaternion(*barrel_pose)) {
    const Eigen::Quaterniond R_world_barrel =
      barrel_pose->normalized();
    last_observations_ = buildObservations(
      armors,
      frame_context.timestamp,
      R_world_barrel);
  }
  updateTrackers(last_observations_);
  removeExpiredTrackers(frame_context.timestamp);
  return collectTargets();
}

const std::vector<ArmorObservation>&
TargetEstimator::lastObservations() const noexcept
{
  return last_observations_;
}

const std::vector<AssociationDiagnostic>&
TargetEstimator::lastAssociationDiagnostics() const noexcept
{
  return last_association_diagnostics_;
}

}  // namespace L3Estimation
