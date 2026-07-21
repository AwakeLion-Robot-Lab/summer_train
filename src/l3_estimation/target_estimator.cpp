#include "l3_estimation/target_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>

namespace L3Estimation {
namespace {

double normalizeAngle(double angle) noexcept
{
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  angle = std::remainder(angle, kTwoPi);
  return angle <= -std::numbers::pi ? angle + kTwoPi : angle;
}

bool isRotationMatrix(const Eigen::Matrix3d& rotation) noexcept
{
  if (!rotation.allFinite()) {
    return false;
  }

  const Eigen::Matrix3d orthogonality_error =
    rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  return orthogonality_error.norm() < 1e-5
         && std::abs(rotation.determinant() - 1.0) < 1e-5;
}

bool isValidQuaternion(const Eigen::Quaterniond& quaternion) noexcept
{
  return quaternion.coeffs().allFinite() && quaternion.norm() > 1e-9;
}

Eigen::Matrix3d rotationMatrixFromRvec(const cv::Vec3d& rvec)
{
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  Eigen::Matrix3d rotation;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) = rotation_cv.at<double>(row, col);
    }
  }
  return rotation;
}

}  // namespace

TargetEstimator::TargetEstimator(
  L1Sensor::CameraCalibration calibration,
  Eigen::Matrix3d rotation_camera_to_gimbal,
  Eigen::Vector3d translation_camera_to_gimbal,
  GimbalPoseProvider gimbal_pose_provider)
  : pnp_solver_(std::move(calibration)),
    rotation_camera_to_gimbal_(std::move(rotation_camera_to_gimbal)),
    translation_camera_to_gimbal_(std::move(translation_camera_to_gimbal)),
    gimbal_pose_provider_(std::move(gimbal_pose_provider))
{
  if (!isRotationMatrix(rotation_camera_to_gimbal_)) {
    throw std::invalid_argument(
      "TargetEstimator requires an orthonormal camera-to-gimbal rotation");
  }
  if (!translation_camera_to_gimbal_.allFinite()) {
    throw std::invalid_argument(
      "TargetEstimator requires a finite camera-to-gimbal translation");
  }
  if (!gimbal_pose_provider_) {
    throw std::invalid_argument("TargetEstimator requires a gimbal pose provider");
  }
}

ArmorSize TargetEstimator::armorSizeFromClass(int class_id) const noexcept
{
  const auto armor_class = L2Perception::armorClassFromId(class_id);
  const bool is_large =
    armor_class == L2Perception::ArmorClass::Hero
    || armor_class == L2Perception::ArmorClass::BaseLarge;
  return is_large ? ArmorSize::Large : ArmorSize::Small;
}

int TargetEstimator::robotIdFromClass(int class_id) const noexcept
{
  const auto armor_class = L2Perception::armorClassFromId(class_id);
  if (armor_class == L2Perception::ArmorClass::Unknown) {
    return -1;
  }

  // 大、小基地装甲板属于同一物理目标，必须进入同一个 tracker。
  if (armor_class == L2Perception::ArmorClass::BaseLarge) {
    return static_cast<int>(L2Perception::ArmorClass::BaseSmall);
  }
  return static_cast<int>(armor_class);
}

Eigen::Vector3d TargetEstimator::positionInWorld(
  const ArmorPose& pose,
  const Eigen::Quaterniond& rotation_gimbal_to_world) const noexcept
{
  const Eigen::Vector3d position_camera{
    pose.tvec[0], pose.tvec[1], pose.tvec[2]};
  const Eigen::Vector3d position_gimbal =
    rotation_camera_to_gimbal_ * position_camera
    + translation_camera_to_gimbal_;
  return rotation_gimbal_to_world * position_gimbal;
}

double TargetEstimator::yawInWorld(
  const ArmorPose& pose,
  const Eigen::Quaterniond& rotation_gimbal_to_world) const
{
  const Eigen::Matrix3d rotation_armor_to_camera =
    rotationMatrixFromRvec(pose.rvec);
  const Eigen::Matrix3d rotation_armor_to_world =
    rotation_gimbal_to_world.toRotationMatrix()
    * rotation_camera_to_gimbal_
    * rotation_armor_to_camera;

  // 第一列是装甲板局部 +x 法线在世界系中的方向。
  return normalizeAngle(std::atan2(
    rotation_armor_to_world(1, 0),
    rotation_armor_to_world(0, 0)));
}

std::optional<ArmorObservation> TargetEstimator::makeObservation(
  const L2Perception::ArmorDetection& armor,
  TimePoint timestamp,
  const Eigen::Quaterniond& rotation_gimbal_to_world) const
{
  const int robot_id = robotIdFromClass(armor.class_id);
  if (robot_id < 0 || !std::isfinite(armor.confidence)) {
    return std::nullopt;
  }

  const ArmorSize armor_size = armorSizeFromClass(armor.class_id);
  const auto pose = pnp_solver_.solve(armor, armor_size);
  if (!pose) {
    return std::nullopt;
  }

  const Eigen::Vector3d position_world =
    positionInWorld(*pose, rotation_gimbal_to_world);
  const double yaw_raw_world =
    yawInWorld(*pose, rotation_gimbal_to_world);
  if (!position_world.allFinite() || !std::isfinite(yaw_raw_world)) {
    return std::nullopt;
  }

  return ArmorObservation{
    .robot_id = robot_id,
    .armor_class = L2Perception::armorClassFromId(armor.class_id),
    .position_world = position_world,
    .yaw_raw_world = yaw_raw_world,
    // 第一版先使用 PnP 原始世界 yaw；后续重投影优化只替换这个字段。
    .yaw_world = yaw_raw_world,
    .confidence = armor.confidence,
    // 0 表示当前阶段尚未计算重投影误差。
    .reprojection_error_px = 0.0,
    .timestamp = timestamp};
}

std::vector<ArmorObservation> TargetEstimator::buildObservations(
  const std::vector<L2Perception::ArmorDetection>& armors,
  TimePoint timestamp,
  const Eigen::Quaterniond& rotation_gimbal_to_world) const
{
  std::vector<ArmorObservation> observations;
  observations.reserve(armors.size());

  for (const auto& armor : armors) {
    if (auto observation = makeObservation(
          armor, timestamp, rotation_gimbal_to_world)) {
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
  std::unordered_map<int, std::vector<ArmorObservation>> observations_by_robot;
  for (const auto& observation : observations) {
    observations_by_robot[observation.robot_id].push_back(observation);
  }

  for (auto& [robot_id, robot_observations] : observations_by_robot) {
    auto [tracker, inserted] = trackers_.try_emplace(robot_id, robot_id);
    (void)inserted;
    tracker->second.update(robot_observations);
  }
}

void TargetEstimator::removeExpiredTrackers(TimePoint timestamp)
{
  std::erase_if(trackers_, [timestamp](const auto& item) {
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

  std::sort(targets.begin(), targets.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.robot_id < rhs.robot_id;
  });
  return targets;
}

std::vector<TargetState> TargetEstimator::update(
  const std::vector<L2Perception::ArmorDetection>& armors,
  TimePoint timestamp)
{
  // 预测所有 tracker 到当前帧时刻，保证后续更新使用一致的时间戳。
  predictTrackers(timestamp);


  const auto gimbal_pose = gimbal_pose_provider_(timestamp);
  if (gimbal_pose && isValidQuaternion(*gimbal_pose)) {
    const Eigen::Quaterniond rotation_gimbal_to_world = gimbal_pose->normalized();
    const auto observations =
      buildObservations(armors, timestamp, rotation_gimbal_to_world);
    updateTrackers(observations);
  }

  removeExpiredTrackers(timestamp);
  return collectTargets();
}

}  // namespace L3Estimation

///////////////////////////////
  // buildObservations()
  // 遍历本帧装甲板

  // makeObservation()
  //   单块装甲板 → PnP → 世界观测

  // positionInWorld()
  //   camera → gimbal → world位置转换

  // yawInWorld()
  //   armor → camera → gimbal → world朝向转换

  // predictTrackers()
  //   所有车辆每帧预测一次

  // updateTrackers()
  //   按robot_id分组并更新对应Tracker

  // removeExpiredTrackers()
  //   删除过期车辆

  // collectTargets()
  //   收集并排序TargetState
