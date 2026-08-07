#include "l3_estimation/target_estimator.hpp"
#include "l3_estimation/angle_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace L3Estimation {
namespace {

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
      config_.yaw_optimization),
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
  // 英雄与基地大装甲使用大板宽度，其余用小板。
  const auto armor_class = L2Perception::armorClassFromId(class_id);
  return armor_class == L2Perception::ArmorClass::Hero
           || armor_class == L2Perception::ArmorClass::BaseLarge
         ? ArmorSize::Large
         : ArmorSize::Small;
}

int TargetEstimator::robotIdFromClass(int class_id) const noexcept
{
  // 车辆类别直接作为 robot_id；基地与未知类别返回 -1（不建目标）。
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
  // 前哨站 → 三板模型，其余有效车辆类别 → 四板模型。
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
  const cv::Vec3d& tvec_camera,
  const Eigen::Quaterniond& R_world_barrel) const noexcept
{
  const Eigen::Vector3d position_camera{
    tvec_camera[0],
    tvec_camera[1],
    tvec_camera[2]};
  const Eigen::Vector3d position_barrel =
    T_barrel_camera_ * position_camera;
  return R_world_barrel * position_barrel;
}

std::optional<ArmorObservation> TargetEstimator::makeObservation(
  const L2Perception::ArmorDetection& armor,
  std::size_t source_detection_index,
  TimePoint timestamp,
  const Eigen::Quaterniond& R_world_barrel) const
{
  // 选解流程：
  // 1. PnP 保留最多两个 IPPE 候选；
  // 2. 已确认目标先用预测 face yaw 选候选（角距离须小于半个面间隔），
  //    只对选中候选做 yaw 搜索；
  // 3. 无预测或预测路径失败时，对每个候选做 yaw 搜索并按优化误差选解；
  // 4. 每块检测仍只输出一个观测。
  const int robot_id = robotIdFromClass(armor.class_id);
  const auto model = targetModelFromClass(armor.class_id);
  if (robot_id < 0 || !model || !std::isfinite(armor.confidence)) {
    return std::nullopt;
  }

  const ArmorSize armor_size = armorSizeFromClass(armor.class_id);
  const std::vector<ArmorPose> poses =
    pnp_solver_.solve(armor, armor_size);
  if (poses.empty()) {
    return std::nullopt;
  }

  const double configured_pitch =
    config_.armor.parameters(*model).pitch_rad;

  // 已确认目标：用预测 face yaw 选 IPPE 候选，只对选中候选做 yaw 搜索。
  const auto predicted_faces =
    predictedFaceYaws(robot_id, *model, timestamp);
  if (predicted_faces) {
    double best_distance =
      std::numeric_limits<double>::infinity();
    const ArmorPose* best_pose = selectCandidateByPredictedFace(
      armor,
      armor_size,
      poses,
      R_world_barrel,
      configured_pitch,
      *predicted_faces,
      best_distance);
    const double gate =
      targetModelTraits(*model).face_angle_interval_rad / 2.0;
    if (best_pose != nullptr && best_distance < gate) {
      if (const auto yaw = yaw_optimizer_.optimize(
            armor,
            armor_size,
            *best_pose,
            R_world_barrel,
            configured_pitch)) {
        return buildObservation(
          armor,
          source_detection_index,
          timestamp,
          R_world_barrel,
          robot_id,
          *model,
          *yaw);
      }
    }
    // 预测距离超出门限或选中候选优化失败：回退到误差选解。
  }

  const auto best_yaw = selectCandidateByOptimizedError(
    armor,
    armor_size,
    poses,
    R_world_barrel,
    configured_pitch);
  if (!best_yaw) {
    return std::nullopt;
  }
  return buildObservation(
    armor,
    source_detection_index,
    timestamp,
    R_world_barrel,
    robot_id,
    *model,
    *best_yaw);
}

std::optional<ArmorObservation> TargetEstimator::buildObservation(
  const L2Perception::ArmorDetection& armor,
  std::size_t source_detection_index,
  TimePoint timestamp,
  const Eigen::Quaterniond& R_world_barrel,
  int robot_id,
  TargetModel model,
  const YawOptimizationResult& yaw) const
{
  const Eigen::Vector3d position_world =
    positionInWorld(yaw.tvec_optimized_camera, R_world_barrel);
  if (!position_world.allFinite()) {
    return std::nullopt;
  }
  return ArmorObservation{
    .source_detection_index = source_detection_index,
    .robot_id = robot_id,
    .armor_class = L2Perception::armorClassFromId(armor.class_id),
    .model = model,
    .position_world = position_world,
    .rpy_raw_world = yaw.rpy_raw_world,
    .rpy_constrained_world = {
      0.0,
      config_.armor.parameters(model).pitch_rad,
      yaw.yaw_optimized_world},
    .yaw_raw_world = yaw.yaw_raw_world,
    .yaw_world = yaw.yaw_optimized_world,
    .confidence = armor.confidence,
    .pnp_reprojection_error_px = yaw.pnp_reprojection_error_px,
    .raw_yaw_reprojection_error_px =
      yaw.raw_yaw_reprojection_error_px,
    .optimized_reprojection_error_px =
      yaw.optimized_reprojection_error_px,
    .timestamp = timestamp};
}

const ArmorPose* TargetEstimator::selectCandidateByPredictedFace(
  const L2Perception::ArmorDetection& armor,
  ArmorSize armor_size,
  const std::vector<ArmorPose>& poses,
  const Eigen::Quaterniond& R_world_barrel,
  double configured_pitch,
  const std::vector<double>& predicted_faces,
  double& best_distance_out) const
{
  const ArmorPose* best_pose = nullptr;
  double best_distance = std::numeric_limits<double>::infinity();
  double best_raw_error = std::numeric_limits<double>::infinity();
  for (const ArmorPose& pose : poses) {
    const auto raw_result = yaw_optimizer_.raw(
      armor,
      armor_size,
      pose,
      R_world_barrel,
      configured_pitch);
    if (!raw_result) {
      continue;
    }
    double distance = std::numeric_limits<double>::infinity();
    for (const double face_yaw : predicted_faces) {
      distance = std::min(
        distance,
        std::abs(normalizeAngle(raw_result->yaw_raw_world - face_yaw)));
    }
    const bool better =
      distance < best_distance
      || (distance == best_distance
          && (raw_result->raw_yaw_reprojection_error_px
                < best_raw_error
              || (raw_result->raw_yaw_reprojection_error_px
                    == best_raw_error
                  && pose.ippe_candidate_index
                       < best_pose->ippe_candidate_index)));
    if (better) {
      best_pose = &pose;
      best_distance = distance;
      best_raw_error = raw_result->raw_yaw_reprojection_error_px;
    }
  }
  best_distance_out = best_distance;
  return best_pose;
}

std::optional<YawOptimizationResult>
TargetEstimator::selectCandidateByOptimizedError(
  const L2Perception::ArmorDetection& armor,
  ArmorSize armor_size,
  const std::vector<ArmorPose>& poses,
  const Eigen::Quaterniond& R_world_barrel,
  double configured_pitch) const
{
  std::optional<YawOptimizationResult> best_yaw;
  const ArmorPose* best_pose = nullptr;
  for (const ArmorPose& pose : poses) {
    const auto yaw = yaw_optimizer_.optimize(
      armor,
      armor_size,
      pose,
      R_world_barrel,
      configured_pitch);
    if (!yaw) {
      continue;
    }
    const bool better =
      !best_yaw
      || yaw->optimized_reprojection_error_px
           < best_yaw->optimized_reprojection_error_px
      || (yaw->optimized_reprojection_error_px
            == best_yaw->optimized_reprojection_error_px
          && (yaw->raw_yaw_reprojection_error_px
                < best_yaw->raw_yaw_reprojection_error_px
              || (yaw->raw_yaw_reprojection_error_px
                    == best_yaw->raw_yaw_reprojection_error_px
                  && pose.ippe_candidate_index
                       < best_pose->ippe_candidate_index)));
    if (better) {
      best_yaw = *yaw;
      best_pose = &pose;
    }
  }
  return best_yaw;
}

std::optional<std::vector<double>> TargetEstimator::predictedFaceYaws(
  int robot_id,
  TargetModel model,
  TimePoint timestamp) const
{
  // 仅当开关开启、Tracker 已确认且预测到当前帧时返回各面预测 yaw。
  if (!config_.pnp.enable_predicted_face_yaw_selection) {
    return std::nullopt;
  }
  const auto iterator = trackers_.find(robot_id);
  if (iterator == trackers_.end()) {
    return std::nullopt;
  }
  const auto& state = iterator->second.state();
  const bool confirmed =
    state.tracker_state == TrackerState::Tracking
    || state.tracker_state == TrackerState::TemporaryLost;
  if (!confirmed
      || state.model != model
      || state.timestamp != timestamp
      || !std::isfinite(state.yaw)) {
    return std::nullopt;
  }

  const auto traits = targetModelTraits(model);
  std::vector<double> face_yaws;
  face_yaws.reserve(static_cast<std::size_t>(traits.armor_count));
  for (int face_id = 0; face_id < traits.armor_count; ++face_id) {
    face_yaws.push_back(normalizeAngle(
      state.yaw
      + static_cast<double>(face_id) * traits.face_angle_interval_rad));
  }
  return face_yaws;
}

std::vector<ArmorObservation> TargetEstimator::buildObservations(
  const std::vector<L2Perception::ArmorDetection>& armors,
  TimePoint timestamp,
  const Eigen::Quaterniond& R_world_barrel) const
{
  // 逐检测生成世界系观测，保留与检测框的对应下标。
  std::vector<ArmorObservation> observations;
  observations.reserve(armors.size());
  for (std::size_t index = 0; index < armors.size(); ++index) {
    const auto& armor = armors[index];
    if (auto observation = makeObservation(
          armor,
          index,
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
  // L3 每帧唯一入口：校验图像尺寸 → 预测全部 Tracker →
  // 构建世界系观测 → 更新/新建 Tracker → 清理过期 → 发布目标列表。
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
