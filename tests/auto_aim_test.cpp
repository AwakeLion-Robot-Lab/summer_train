#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/tracker.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr int kYawSearchSamples = 140;
constexpr double kDegreesToRadians = std::numbers::pi / 180.0;
constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi;
constexpr double kMinimumCornerDepth = 1e-6;
constexpr double kCandidateMatchToleranceMeters = 1e-4;
constexpr double kIppeAmbiguityErrorGapPixels = 0.25;
constexpr double kIppeAmbiguityErrorRatio = 1.2;
constexpr double kOverlaySmoothingAlpha = 0.40;
constexpr double kOverlayResetDistancePixels = 120.0;
constexpr int kOverlayHoldFrames = 5;

const std::string kCommandLineKeys =
  "{help h usage ? | false | show command-line help}"
  "{model m | model/armor_model/armor.xml | current newvision armor model}"
  "{calibration c | tests/data/sp_auto_aim/camera_calibration.yaml | calibration YAML}"
  "{device d | CPU | OpenVINO device}"
  "{enemy e | blue | red, blue, or any}"
  "{start-index s | 0 | first frame index}"
  "{end-index n | 0 | last frame index, zero means all}"
  "{show-from-index | -1 | first displayed frame; earlier frames still update Tracker}"
  "{csv | logs/sp_auto_aim_replay.csv | per-frame output path}"
  "{ekf-iterations | 5 | EKF Gauss-Newton relinearizations; 1 disables iteration}"
  "{show | false | show replay window}"
  "{@input-path | tests/data/sp_auto_aim/demo | base path of .avi and .txt}";

struct PoseSample {
  double seconds{0.0};
  Eigen::Quaterniond q_imu_body_to_world{Eigen::Quaterniond::Identity()};
};

struct IppeCandidateDiagnostic {
  int original_index{-1};
  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};
  double yaw_in_world{std::numeric_limits<double>::quiet_NaN()};
  double reprojection_rmse{std::numeric_limits<double>::infinity()};
  double facing{std::numeric_limits<double>::quiet_NaN()};
};

struct YawCostMinimum {
  std::size_t sample_index{0};
  double yaw_in_world{std::numeric_limits<double>::quiet_NaN()};
  double offset_degrees{std::numeric_limits<double>::quiet_NaN()};
  double cost{std::numeric_limits<double>::infinity()};
};

struct PnpCostDiagnostic {
  std::size_t observation_index{0};
  int raw_ippe_solution_count{0};
  std::vector<IppeCandidateDiagnostic> ippe_candidates;
  bool ippe_ambiguous{false};

  double barrel_yaw{std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> curve_yaws;
  std::vector<double> curve_offsets_degrees;
  std::vector<double> curve_costs;
  YawCostMinimum best_curve_minimum;
  std::optional<YawCostMinimum> second_curve_minimum;
  std::size_t local_minimum_count{0};
  bool yaw_search_applied{false};

  double observation_yaw{std::numeric_limits<double>::quiet_NaN()};
  int single_pnp_ippe_index{-1};
  double single_pnp_position_delta{
    std::numeric_limits<double>::infinity()};
  int observation_nearest_ippe{-1};

  bool filter_updated{false};
  int ekf_armor_id{-1};
  double ekf_armor_yaw{std::numeric_limits<double>::quiet_NaN()};
  int ekf_nearest_ippe{-1};
  bool ekf_nearest_ippe_changed{false};
  double ekf_nearest_ippe_angle{
    std::numeric_limits<double>::infinity()};
};

struct ProjectedArmorBox {
  std::size_t armor_id{0};
  std::array<cv::Point2f, 4> corners;
};

struct VehicleOverlayState {
  std::optional<L3Estimation::ArmorName> target_name;
  std::vector<ProjectedArmorBox> boxes;
  int last_update_frame{-1};
};

struct VehicleOverlayResult {
  std::size_t drawn_count{0};
  bool held{false};
  int age_frames{0};
};

struct ReplayStats {
  std::size_t frames{0};
  std::size_t frames_with_raw_detections{0};
  std::size_t frames_with_enemy_detections{0};
  std::size_t raw_detections{0};
  std::size_t enemy_detections{0};
  std::size_t pnp_valid_frames{0};
  std::size_t pnp_valid_observations{0};
  std::size_t target_output_frames{0};
  std::size_t tracking_frames{0};
  std::size_t temp_lost_frames{0};
  std::size_t nonfinite_target_frames{0};
  std::size_t resets_with_valid_pnp{0};
  std::size_t state_transitions{0};
  std::size_t current_tracking_run{0};
  std::size_t longest_tracking_run{0};
  std::size_t cost_diagnostic_frames{0};
  std::size_t ippe_two_solution_frames{0};
  std::size_t ippe_ambiguous_frames{0};
  std::size_t single_pnp_lower_rmse_frames{0};
  std::size_t single_pnp_higher_rmse_frames{0};
  std::size_t single_pnp_unmatched_frames{0};
  std::size_t ekf_nearest_lower_rmse_updates{0};
  std::size_t ekf_nearest_higher_rmse_updates{0};
  std::size_t ekf_nearest_unavailable_updates{0};
  std::size_t ekf_ippe_branch_switches{0};
  double detector_ms_sum{0.0};
  double tracker_ms_sum{0.0};
  double max_position_step{0.0};
  double max_speed{0.0};
};

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] std::string_view stateName(L3Estimation::TrackState state) noexcept
{
  switch (state) {
  case L3Estimation::TrackState::Lost:
    return "lost";
  case L3Estimation::TrackState::Detecting:
    return "detecting";
  case L3Estimation::TrackState::Tracking:
    return "tracking";
  case L3Estimation::TrackState::TempLost:
    return "temp_lost";
  }
  return "unknown";
}

[[nodiscard]] bool readPose(std::istream& input, PoseSample& sample)
{
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (!(input >> sample.seconds >> w >> x >> y >> z)) {
    return false;
  }

  sample.q_imu_body_to_world = {w, x, y, z};
  if (!std::isfinite(sample.seconds) ||
      !sample.q_imu_body_to_world.coeffs().allFinite() ||
      sample.q_imu_body_to_world.squaredNorm() <= 1e-12) {
    throw std::runtime_error("pose text contains a non-finite quaternion");
  }
  sample.q_imu_body_to_world.normalize();
  return true;
}

[[nodiscard]] Eigen::Quaterniond toWorldBarrelPose(const PoseSample& sample)
{
  // 与 SP Solver::set_R_gimbal2world() 保持一致：录像和它配套的 T_barrel_camera
  // 都按 sp_vision 的约定标定，barrel 的 x、y 轴与 IMU 相反。
  //
  // 这里是双边的相似变换，因为 sp 把 world 也跟着 barrel 一起重标记了。
  // 本项目 world 固定为 imu_abs，barrel 单独定义，实机走的是单边复合
  // R_world_imu * R_imu_barrel（见 SerialWorker::toBarrelPose）。
  // 两套约定各自自洽，只差绕 z 轴 180 度：俯仰角和距离一致，方位角和装甲板
  // yaw 整体差 pi，因此回放里的绝对 yaw 数值不能直接和实机日志对比。
  Eigen::Matrix3d R_imu_barrel = Eigen::Matrix3d::Identity();
  R_imu_barrel(0, 0) = -1.0;
  R_imu_barrel(1, 1) = -1.0;
  const Eigen::Matrix3d R_world_barrel =
    R_imu_barrel.transpose() *
    sample.q_imu_body_to_world.toRotationMatrix() *
    R_imu_barrel;
  return Eigen::Quaterniond(R_world_barrel).normalized();
}

[[nodiscard]] L2Perception::ArmorColor parseEnemyColor(
  const std::string& value)
{
  if (value == "red") {
    return L2Perception::ArmorColor::Red;
  }
  if (value == "blue") {
    return L2Perception::ArmorColor::Blue;
  }
  if (value == "any") {
    return L2Perception::ArmorColor::Unknown;
  }
  throw std::invalid_argument("enemy must be red, blue, or any");
}

[[nodiscard]] bool matchesEnemy(
  L2Perception::ArmorColor observed,
  L2Perception::ArmorColor selected) noexcept
{
  return selected == L2Perception::ArmorColor::Unknown || observed == selected;
}

[[nodiscard]] std::filesystem::path withExtension(
  std::filesystem::path path,
  std::string_view extension)
{
  if (path.extension() == extension) {
    return path;
  }
  path.replace_extension(extension);
  return path;
}

[[nodiscard]] cv::Point toPixel(const cv::Point2f& point)
{
  return {
    static_cast<int>(std::lround(point.x)),
    static_cast<int>(std::lround(point.y))};
}

void drawOutlinedText(
  cv::Mat& image,
  const std::string& text,
  cv::Point origin,
  const cv::Scalar& color,
  double scale = 0.62)
{
  cv::putText(
    image,
    text,
    origin,
    cv::FONT_HERSHEY_SIMPLEX,
    scale,
    {0, 0, 0},
    4,
    cv::LINE_AA);
  cv::putText(
    image,
    text,
    origin,
    cv::FONT_HERSHEY_SIMPLEX,
    scale,
    color,
    1,
    cv::LINE_AA);
}

[[nodiscard]] double yawDegrees(const Eigen::Quaterniond& quaternion)
{
  const Eigen::Matrix3d rotation = quaternion.normalized().toRotationMatrix();
  return std::atan2(rotation(1, 0), rotation(0, 0)) *
    180.0 / std::numbers::pi;
}

[[nodiscard]] double angularDistance(double lhs, double rhs)
{
  return std::abs(L6Telemetry::limit_rad(lhs - rhs));
}

[[nodiscard]] std::vector<cv::Point3d> diagnosticArmorPoints(
  L3Estimation::ArmorType type,
  const L3Estimation::ArmorConfig& config)
{
  const double half_width =
    (type == L3Estimation::ArmorType::Big
       ? config.big_width
       : config.small_width) /
    2.0;
  const double half_height = config.height / 2.0;
  return {
    {0.0, half_width, half_height},
    {0.0, -half_width, half_height},
    {0.0, -half_width, -half_height},
    {0.0, half_width, -half_height}};
}

[[nodiscard]] bool validPnpObservation(
  const L3Estimation::Armor& observation) noexcept
{
  return observation.quality.pnp_ok && observation.quality.geometry_ok &&
         observation.quality.reprojection_ok && observation.quality.finite;
}

[[nodiscard]] bool yawSearchApplied(
  const L3Estimation::Armor& observation) noexcept
{
  const bool is_balance =
    observation.type == L3Estimation::ArmorType::Big &&
    (observation.name == L3Estimation::ArmorName::Infantry3 ||
     observation.name == L3Estimation::ArmorName::Infantry4 ||
     observation.name == L3Estimation::ArmorName::Infantry5);
  return !is_balance;
}

[[nodiscard]] Eigen::Matrix3d toEigenRotation(const cv::Mat& rotation)
{
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = rotation.at<double>(row, col);
    }
  }
  return result;
}

[[nodiscard]] std::optional<Eigen::Vector3d> toEigenVector3(
  const cv::Mat& vector)
{
  if (vector.total() != 3) {
    return std::nullopt;
  }
  cv::Mat vector64;
  vector.convertTo(vector64, CV_64F);
  const cv::Mat flattened = vector64.reshape(1, 1);
  return Eigen::Vector3d{
    flattened.at<double>(0, 0),
    flattened.at<double>(0, 1),
    flattened.at<double>(0, 2)};
}

[[nodiscard]] double yawReprojectionCost(
  const L3Estimation::PnpSolver& solver,
  const L3Estimation::Armor& observation,
  double yaw)
{
  const std::vector<cv::Point2f> projected = solver.reproject_armor(
    observation.xyz_in_world,
    yaw,
    observation.type,
    observation.name);
  if (projected.size() != observation.points.size()) {
    return std::numeric_limits<double>::infinity();
  }

  double cost = 0.0;
  for (std::size_t index = 0; index < observation.points.size(); ++index) {
    cost += cv::norm(observation.points[index] - projected[index]);
  }
  return cost;
}

[[nodiscard]] std::optional<std::size_t> selectDiagnosticObservation(
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<L3Estimation::TargetState>& target,
  const std::vector<Eigen::Vector4d>& target_armor_poses,
  const cv::Size& image_size)
{
  std::optional<std::size_t> selected;
  double best_score = std::numeric_limits<double>::infinity();

  if (target && !target_armor_poses.empty()) {
    if (target->armor_id >= 0 &&
        static_cast<std::size_t>(target->armor_id) <
          target_armor_poses.size()) {
      const Eigen::Vector3d selected_armor_position =
        target_armor_poses[static_cast<std::size_t>(target->armor_id)].head<3>();
      for (std::size_t observation_index = 0;
           observation_index < observations.size();
           ++observation_index) {
        const auto& observation = observations[observation_index];
        if (!validPnpObservation(observation) ||
            observation.name != target->name) {
          continue;
        }
        const double score =
          (observation.xyz_in_world - selected_armor_position).squaredNorm();
        if (score < best_score) {
          best_score = score;
          selected = observation_index;
        }
      }
      if (selected) {
        return selected;
      }
    }

    for (std::size_t observation_index = 0;
         observation_index < observations.size();
         ++observation_index) {
      const auto& observation = observations[observation_index];
      if (!validPnpObservation(observation) ||
          observation.name != target->name) {
        continue;
      }
      for (const auto& armor_pose : target_armor_poses) {
        const double score =
          (observation.xyz_in_world - armor_pose.head<3>()).squaredNorm();
        if (score < best_score) {
          best_score = score;
          selected = observation_index;
        }
      }
    }
    if (selected) {
      return selected;
    }
  }

  const cv::Point2f image_center{
    static_cast<float>(image_size.width) * 0.5F,
    static_cast<float>(image_size.height) * 0.5F};
  for (std::size_t observation_index = 0;
       observation_index < observations.size();
       ++observation_index) {
    const auto& observation = observations[observation_index];
    if (!validPnpObservation(observation)) {
      continue;
    }
    cv::Point2f center{};
    for (const auto& point : observation.points) {
      center += point;
    }
    center *= 0.25F;
    const cv::Point2f difference = center - image_center;
    const double score = static_cast<double>(difference.x) * difference.x +
      static_cast<double>(difference.y) * difference.y;
    if (score < best_score) {
      best_score = score;
      selected = observation_index;
    }
  }
  return selected;
}

[[nodiscard]] int nearestIppeCandidateByYaw(
  const std::vector<IppeCandidateDiagnostic>& candidates,
  double yaw,
  double* minimum_angle = nullptr)
{
  int selected = -1;
  double best_angle = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const double angle = angularDistance(yaw, candidates[index].yaw_in_world);
    if (angle < best_angle) {
      best_angle = angle;
      selected = static_cast<int>(index);
    }
  }
  if (minimum_angle != nullptr) {
    *minimum_angle = best_angle;
  }
  return selected;
}

[[nodiscard]] std::optional<PnpCostDiagnostic> buildPnpCostDiagnostic(
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<L3Estimation::TargetState>& target,
  const std::vector<Eigen::Vector4d>& target_armor_poses,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::ArmorConfig& armor_config,
  const Eigen::Quaterniond& q_world_barrel,
  const L3Estimation::PnpSolver& diagnostic_solver)
{
  const auto observation_index = selectDiagnosticObservation(
    observations,
    target,
    target_armor_poses,
    calibration.image_size);
  if (!observation_index || !calibration.T_barrel_camera) {
    return std::nullopt;
  }

  const auto& observation = observations[*observation_index];
  PnpCostDiagnostic diagnostic;
  diagnostic.observation_index = *observation_index;
  diagnostic.observation_yaw = observation.ypr_in_world[0];
  diagnostic.yaw_search_applied = yawSearchApplied(observation);
  diagnostic.filter_updated = target && target->updated;

  const std::vector<cv::Point3d> object_points =
    diagnosticArmorPoints(observation.type, armor_config);
  const std::vector<cv::Point2f> image_points(
    observation.points.begin(), observation.points.end());
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  try {
    diagnostic.raw_ippe_solution_count = cv::solvePnPGeneric(
      object_points,
      image_points,
      calibration.camera_matrix,
      calibration.distortion_coefficients,
      rvecs,
      tvecs,
      false,
      cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception&) {
    diagnostic.raw_ippe_solution_count = 0;
  }

  const Eigen::Matrix3d R_world_barrel =
    q_world_barrel.normalized().toRotationMatrix();
  const Eigen::Matrix3d R_barrel_camera =
    calibration.T_barrel_camera->linear();
  const Eigen::Vector3d t_barrel_camera =
    calibration.T_barrel_camera->translation();
  const std::size_t solution_count = std::min(rvecs.size(), tvecs.size());
  diagnostic.ippe_candidates.reserve(solution_count);
  for (std::size_t solution_index = 0;
       solution_index < solution_count;
       ++solution_index) {
    const auto t_camera = toEigenVector3(tvecs[solution_index]);
    if (!t_camera || !t_camera->allFinite() ||
        t_camera->z() <= kMinimumCornerDepth) {
      continue;
    }

    cv::Mat rotation_cv;
    try {
      cv::Rodrigues(rvecs[solution_index], rotation_cv);
    } catch (const cv::Exception&) {
      continue;
    }
    if (rotation_cv.type() != CV_64FC1 || rotation_cv.rows != 3 ||
        rotation_cv.cols != 3 || !cv::checkRange(rotation_cv)) {
      continue;
    }
    const Eigen::Matrix3d R_camera_armor = toEigenRotation(rotation_cv);

    bool geometry_ok = true;
    for (const auto& point : object_points) {
      const Eigen::Vector3d point_in_camera =
        R_camera_armor * Eigen::Vector3d{point.x, point.y, point.z} +
        *t_camera;
      if (!point_in_camera.allFinite() ||
          point_in_camera.z() <= kMinimumCornerDepth) {
        geometry_ok = false;
        break;
      }
    }
    if (!geometry_ok) {
      continue;
    }

    std::vector<cv::Point2d> projected_points;
    try {
      cv::projectPoints(
        object_points,
        rvecs[solution_index],
        tvecs[solution_index],
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        projected_points);
    } catch (const cv::Exception&) {
      continue;
    }
    if (projected_points.size() != image_points.size()) {
      continue;
    }
    double squared_error_sum = 0.0;
    for (std::size_t point_index = 0;
         point_index < image_points.size();
         ++point_index) {
      const double dx = projected_points[point_index].x -
        image_points[point_index].x;
      const double dy = projected_points[point_index].y -
        image_points[point_index].y;
      squared_error_sum += dx * dx + dy * dy;
    }

    const Eigen::Vector3d xyz_in_barrel =
      R_barrel_camera * *t_camera + t_barrel_camera;
    const Eigen::Vector3d xyz_in_world =
      R_world_barrel * xyz_in_barrel;
    const Eigen::Matrix3d R_world_armor =
      R_world_barrel * R_barrel_camera * R_camera_armor;
    const Eigen::Vector3d ypr_in_world =
      L6Telemetry::eulers(R_world_armor, 2, 1, 0);
    const Eigen::Vector3d front_in_camera = -R_camera_armor.col(0);
    const double facing = front_in_camera.dot(-*t_camera);
    const double rmse = std::sqrt(
      squared_error_sum / static_cast<double>(image_points.size()));
    if (!xyz_in_world.allFinite() || !ypr_in_world.allFinite() ||
        !std::isfinite(facing) || !std::isfinite(rmse)) {
      continue;
    }

    diagnostic.ippe_candidates.push_back(IppeCandidateDiagnostic{
      .original_index = static_cast<int>(solution_index),
      .xyz_in_camera = *t_camera,
      .xyz_in_world = xyz_in_world,
      .yaw_in_world = ypr_in_world[0],
      .reprojection_rmse = rmse,
      .facing = facing});
  }
  std::stable_sort(
    diagnostic.ippe_candidates.begin(),
    diagnostic.ippe_candidates.end(),
    [](const auto& lhs, const auto& rhs) {
      return lhs.reprojection_rmse < rhs.reprojection_rmse;
    });

  if (diagnostic.ippe_candidates.size() >= 2) {
    const double best_rmse =
      diagnostic.ippe_candidates[0].reprojection_rmse;
    const double second_rmse =
      diagnostic.ippe_candidates[1].reprojection_rmse;
    diagnostic.ippe_ambiguous =
      second_rmse <= armor_config.max_reprojection_error &&
      second_rmse <= std::max(
        best_rmse + kIppeAmbiguityErrorGapPixels,
        best_rmse * kIppeAmbiguityErrorRatio);
  }

  for (std::size_t index = 0;
       index < diagnostic.ippe_candidates.size();
       ++index) {
    const double delta =
      (observation.xyz_in_camera -
       diagnostic.ippe_candidates[index].xyz_in_camera)
        .norm();
    if (delta < diagnostic.single_pnp_position_delta) {
      diagnostic.single_pnp_position_delta = delta;
      diagnostic.single_pnp_ippe_index = static_cast<int>(index);
    }
  }
  if (diagnostic.single_pnp_position_delta >
      kCandidateMatchToleranceMeters) {
    diagnostic.single_pnp_ippe_index = -1;
  }
  diagnostic.observation_nearest_ippe = nearestIppeCandidateByYaw(
    diagnostic.ippe_candidates,
    diagnostic.observation_yaw);

  diagnostic.barrel_yaw =
    L6Telemetry::eulers(R_world_barrel, 2, 1, 0)[0];
  diagnostic.curve_yaws.reserve(kYawSearchSamples);
  diagnostic.curve_offsets_degrees.reserve(kYawSearchSamples);
  diagnostic.curve_costs.reserve(kYawSearchSamples);
  for (int sample_index = 0;
       sample_index < kYawSearchSamples;
       ++sample_index) {
    const double offset_degrees =
      static_cast<double>(sample_index) -
      static_cast<double>(kYawSearchSamples) * 0.5;
    const double yaw = L6Telemetry::limit_rad(
      diagnostic.barrel_yaw + offset_degrees * kDegreesToRadians);
    const double cost = yawReprojectionCost(
      diagnostic_solver,
      observation,
      yaw);
    diagnostic.curve_yaws.push_back(yaw);
    diagnostic.curve_offsets_degrees.push_back(offset_degrees);
    diagnostic.curve_costs.push_back(cost);
    if (cost < diagnostic.best_curve_minimum.cost) {
      diagnostic.best_curve_minimum = {
        .sample_index = static_cast<std::size_t>(sample_index),
        .yaw_in_world = yaw,
        .offset_degrees = offset_degrees,
        .cost = cost};
    }
  }

  std::vector<YawCostMinimum> local_minima;
  for (std::size_t index = 0; index < diagnostic.curve_costs.size(); ++index) {
    const double cost = diagnostic.curve_costs[index];
    if (!std::isfinite(cost)) {
      continue;
    }
    const bool no_larger_than_left =
      index == 0 || cost <= diagnostic.curve_costs[index - 1];
    const bool no_larger_than_right =
      index + 1 == diagnostic.curve_costs.size() ||
      cost <= diagnostic.curve_costs[index + 1];
    if (no_larger_than_left && no_larger_than_right) {
      local_minima.push_back({
        .sample_index = index,
        .yaw_in_world = diagnostic.curve_yaws[index],
        .offset_degrees = diagnostic.curve_offsets_degrees[index],
        .cost = cost});
    }
  }
  diagnostic.local_minimum_count = local_minima.size();
  std::stable_sort(
    local_minima.begin(),
    local_minima.end(),
    [](const auto& lhs, const auto& rhs) { return lhs.cost < rhs.cost; });
  for (const auto& minimum : local_minima) {
    const auto sample_distance = static_cast<std::size_t>(std::abs(
      static_cast<long long>(minimum.sample_index) -
      static_cast<long long>(diagnostic.best_curve_minimum.sample_index)));
    if (sample_distance >= 5) {
      diagnostic.second_curve_minimum = minimum;
      break;
    }
  }

  if (target && target->name == observation.name &&
      !target_armor_poses.empty()) {
    if (target->armor_id >= 0 &&
        static_cast<std::size_t>(target->armor_id) <
          target_armor_poses.size()) {
      diagnostic.ekf_armor_id = target->armor_id;
      diagnostic.ekf_armor_yaw =
        target_armor_poses[static_cast<std::size_t>(target->armor_id)].w();
    }
    if (std::isfinite(diagnostic.ekf_armor_yaw)) {
      diagnostic.ekf_nearest_ippe = nearestIppeCandidateByYaw(
        diagnostic.ippe_candidates,
        diagnostic.ekf_armor_yaw,
        &diagnostic.ekf_nearest_ippe_angle);
    }
  }

  return diagnostic;
}

[[nodiscard]] std::string branchName(int branch)
{
  if (branch == 0) {
    return "lower-RMSE #0";
  }
  if (branch == 1) {
    return "higher-RMSE #1";
  }
  return "N/A";
}

[[nodiscard]] cv::Mat drawYawCostPlot(
  const std::optional<PnpCostDiagnostic>& diagnostic,
  int frame_index)
{
  constexpr int kWidth = 1000;
  constexpr int kHeight = 600;
  constexpr int kLeft = 78;
  constexpr int kRight = 28;
  constexpr int kTop = 165;
  constexpr int kBottom = 62;
  const cv::Rect graph{
    kLeft,
    kTop,
    kWidth - kLeft - kRight,
    kHeight - kTop - kBottom};
  cv::Mat plot(kHeight, kWidth, CV_8UC3, cv::Scalar{24, 24, 24});

  drawOutlinedText(
    plot,
    cv::format("frame=%d  newvision yaw-search cost diagnostic", frame_index),
    {14, 28},
    {255, 255, 255},
    0.64);
  if (!diagnostic) {
    drawOutlinedText(
      plot,
      "No valid PnP observation in this frame",
      {14, 62},
      {0, 165, 255},
      0.62);
    return plot;
  }

  const auto& data = *diagnostic;
  const auto candidate_text = [&data](std::size_t index) {
    if (index >= data.ippe_candidates.size()) {
      return std::string{"N/A"};
    }
    const auto& candidate = data.ippe_candidates[index];
    return cv::format(
      "yaw=%.2fdeg rmse=%.3fpx",
      candidate.yaw_in_world * kRadiansToDegrees,
      candidate.reprojection_rmse);
  };
  drawOutlinedText(
    plot,
    "IPPE #0 " + candidate_text(0) + "   #1 " + candidate_text(1),
    {14, 58},
    data.ippe_ambiguous ? cv::Scalar{0, 165, 255}
                        : cv::Scalar{255, 255, 255},
    0.53);
  drawOutlinedText(
    plot,
    cv::format(
      "IPPE ambiguous=%s  solvePnP returned=%s (dt=%.6fm)  search=%s",
      data.ippe_ambiguous ? "YES" : "NO",
      branchName(data.single_pnp_ippe_index).c_str(),
      data.single_pnp_position_delta,
      data.yaw_search_applied ? "ON" : "SKIPPED(balance)"),
    {14, 86},
    data.ippe_ambiguous ? cv::Scalar{0, 165, 255}
                        : cv::Scalar{220, 220, 220},
    0.50);
  drawOutlinedText(
    plot,
    cv::format(
      "search yaw=%.2fdeg cost=%.3f  EKF armor[%d] yaw=%.2fdeg nearest=%s changed=%s updated=%s",
      data.observation_yaw * kRadiansToDegrees,
      data.best_curve_minimum.cost,
      data.ekf_armor_id,
      data.ekf_armor_yaw * kRadiansToDegrees,
      branchName(data.ekf_nearest_ippe).c_str(),
      data.ekf_nearest_ippe_changed ? "YES" : "NO",
      data.filter_updated ? "YES" : "NO"),
    {14, 114},
    {0, 255, 0},
    0.50);
  drawOutlinedText(
    plot,
    cv::format(
      "curve: sum of 4 corner distances [px], fixed pitch, local minima=%zu; EKF-nearest is diagnostic only",
      data.local_minimum_count),
    {14, 142},
    {180, 180, 180},
    0.48);

  std::vector<double> finite_costs;
  finite_costs.reserve(data.curve_costs.size());
  for (const double cost : data.curve_costs) {
    if (std::isfinite(cost)) {
      finite_costs.push_back(cost);
    }
  }
  if (finite_costs.empty() || data.curve_offsets_degrees.size() < 2) {
    drawOutlinedText(
      plot,
      "Cost curve is unavailable",
      {graph.x + 20, graph.y + 40},
      {0, 165, 255});
    return plot;
  }

  const auto [minimum_it, maximum_it] = std::minmax_element(
    finite_costs.begin(), finite_costs.end());
  double minimum_cost = *minimum_it;
  double maximum_cost = *maximum_it;
  if (maximum_cost - minimum_cost < 1e-9) {
    maximum_cost = minimum_cost + 1.0;
  }
  const double padding = 0.05 * (maximum_cost - minimum_cost);
  minimum_cost = std::max(0.0, minimum_cost - padding);
  maximum_cost += padding;

  const double minimum_offset = data.curve_offsets_degrees.front();
  const double maximum_offset = data.curve_offsets_degrees.back();
  const auto x_for_offset = [&](double offset) {
    return graph.x + static_cast<int>(std::lround(
      (offset - minimum_offset) /
      (maximum_offset - minimum_offset) *
      static_cast<double>(graph.width)));
  };
  const auto y_for_cost = [&](double cost) {
    const double clamped = std::clamp(cost, minimum_cost, maximum_cost);
    return graph.y + graph.height - static_cast<int>(std::lround(
      (clamped - minimum_cost) /
      (maximum_cost - minimum_cost) *
      static_cast<double>(graph.height)));
  };

  cv::rectangle(plot, graph, {100, 100, 100}, 1, cv::LINE_AA);
  for (int grid_index = 0; grid_index <= 4; ++grid_index) {
    const double ratio = static_cast<double>(grid_index) / 4.0;
    const int x = graph.x + static_cast<int>(std::lround(ratio * graph.width));
    const int y = graph.y + static_cast<int>(std::lround(ratio * graph.height));
    cv::line(
      plot,
      {x, graph.y},
      {x, graph.y + graph.height},
      {55, 55, 55},
      1,
      cv::LINE_AA);
    cv::line(
      plot,
      {graph.x, y},
      {graph.x + graph.width, y},
      {55, 55, 55},
      1,
      cv::LINE_AA);
    const double offset = minimum_offset +
      ratio * (maximum_offset - minimum_offset);
    const double cost = maximum_cost - ratio * (maximum_cost - minimum_cost);
    cv::putText(
      plot,
      cv::format("%.0f", offset),
      {x - 16, graph.y + graph.height + 24},
      cv::FONT_HERSHEY_SIMPLEX,
      0.45,
      {190, 190, 190},
      1,
      cv::LINE_AA);
    cv::putText(
      plot,
      cv::format("%.1f", cost),
      {8, y + 5},
      cv::FONT_HERSHEY_SIMPLEX,
      0.43,
      {190, 190, 190},
      1,
      cv::LINE_AA);
  }
  cv::putText(
    plot,
    "yaw offset from barrel [deg]",
    {graph.x + graph.width / 2 - 105, kHeight - 12},
    cv::FONT_HERSHEY_SIMPLEX,
    0.50,
    {220, 220, 220},
    1,
    cv::LINE_AA);

  std::optional<cv::Point> previous_point;
  for (std::size_t index = 0; index < data.curve_costs.size(); ++index) {
    if (!std::isfinite(data.curve_costs[index])) {
      previous_point.reset();
      continue;
    }
    const cv::Point point{
      x_for_offset(data.curve_offsets_degrees[index]),
      y_for_cost(data.curve_costs[index])};
    if (previous_point) {
      cv::line(plot, *previous_point, point, {230, 230, 230}, 2, cv::LINE_AA);
    }
    previous_point = point;
  }

  if (std::isfinite(data.best_curve_minimum.cost)) {
    cv::circle(
      plot,
      {x_for_offset(data.best_curve_minimum.offset_degrees),
       y_for_cost(data.best_curve_minimum.cost)},
      6,
      {0, 0, 255},
      cv::FILLED,
      cv::LINE_AA);
  }
  if (data.second_curve_minimum &&
      std::isfinite(data.second_curve_minimum->cost)) {
    cv::circle(
      plot,
      {x_for_offset(data.second_curve_minimum->offset_degrees),
       y_for_cost(data.second_curve_minimum->cost)},
      6,
      {0, 165, 255},
      2,
      cv::LINE_AA);
  }

  const auto draw_yaw_marker = [&](double yaw,
                                   const cv::Scalar& color,
                                   std::string_view label,
                                   int label_row) {
    if (!std::isfinite(yaw)) {
      return;
    }
    const double offset =
      L6Telemetry::limit_rad(yaw - data.barrel_yaw) * kRadiansToDegrees;
    if (offset < minimum_offset || offset > maximum_offset) {
      return;
    }
    const int x = x_for_offset(offset);
    cv::line(
      plot,
      {x, graph.y},
      {x, graph.y + graph.height},
      color,
      1,
      cv::LINE_AA);
    cv::putText(
      plot,
      std::string(label),
      {x + 3, graph.y + 18 + label_row * 17},
      cv::FONT_HERSHEY_SIMPLEX,
      0.43,
      color,
      1,
      cv::LINE_AA);
  };
  if (!data.ippe_candidates.empty()) {
    draw_yaw_marker(
      data.ippe_candidates[0].yaw_in_world,
      {255, 255, 0},
      "IPPE0",
      0);
  }
  if (data.ippe_candidates.size() >= 2) {
    draw_yaw_marker(
      data.ippe_candidates[1].yaw_in_world,
      {255, 0, 255},
      "IPPE1",
      1);
  }
  draw_yaw_marker(data.observation_yaw, {0, 255, 255}, "SEARCH", 2);
  draw_yaw_marker(data.ekf_armor_yaw, {0, 255, 0}, "EKF", 3);

  return plot;
}

[[nodiscard]] std::optional<std::vector<cv::Point2f>> projectWorldPoints(
  const std::vector<Eigen::Vector3d>& points_in_world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  if (!calibration.T_barrel_camera ||
      !q_world_barrel.coeffs().allFinite() ||
      q_world_barrel.squaredNorm() <= 1e-12) {
    return std::nullopt;
  }

  Eigen::Isometry3d T_world_barrel = Eigen::Isometry3d::Identity();
  T_world_barrel.linear() =
    q_world_barrel.normalized().toRotationMatrix();
  const Eigen::Isometry3d T_camera_world =
    (T_world_barrel * *calibration.T_barrel_camera).inverse();

  std::vector<cv::Point3d> points_in_camera;
  points_in_camera.reserve(points_in_world.size());
  for (const auto& point_in_world : points_in_world) {
    const Eigen::Vector3d point_in_camera = T_camera_world * point_in_world;
    if (!point_in_camera.allFinite() || point_in_camera.z() <= 1e-6) {
      return std::nullopt;
    }
    points_in_camera.emplace_back(
      point_in_camera.x(), point_in_camera.y(), point_in_camera.z());
  }

  std::vector<cv::Point2d> projected_points;
  try {
    cv::projectPoints(
      points_in_camera,
      cv::Vec3d::all(0.0),
      cv::Vec3d::all(0.0),
      calibration.camera_matrix,
      calibration.distortion_coefficients,
      projected_points);
  } catch (const cv::Exception&) {
    return std::nullopt;
  }

  if (projected_points.size() != points_in_world.size() ||
      !std::all_of(
        projected_points.begin(),
        projected_points.end(),
        [](const cv::Point2d& point) {
          return std::isfinite(point.x) && std::isfinite(point.y);
        })) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> image_points;
  image_points.reserve(projected_points.size());
  std::transform(
    projected_points.begin(),
    projected_points.end(),
    std::back_inserter(image_points),
    [](const cv::Point2d& point) { return cv::Point2f(point); });
  return image_points;
}

[[nodiscard]] std::array<Eigen::Vector3d, 4> estimatedArmorCorners(
  const Eigen::Vector4d& armor_pose,
  L3Estimation::ArmorType type,
  L3Estimation::ArmorName name,
  const L3Estimation::ArmorConfig& config)
{
  const double width = type == L3Estimation::ArmorType::Big
    ? config.big_width
    : config.small_width;
  const double half_width = width / 2.0;
  const double half_height = config.height / 2.0;
  // 与 SP-Vision Solver::reproject_armor() 一致：普通目标假设装甲板
  // pitch 为 +15 度，前哨站为 -15 度。
  const double pitch = (name == L3Estimation::ArmorName::Outpost ? -15.0 : 15.0) *
    std::numbers::pi / 180.0;
  const Eigen::Matrix3d R_world_armor =
    (Eigen::AngleAxisd(armor_pose.w(), Eigen::Vector3d::UnitZ()) *
     Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()))
      .toRotationMatrix();

  const std::array<Eigen::Vector3d, 4> local_corners{
    Eigen::Vector3d{0.0, half_width, half_height},
    Eigen::Vector3d{0.0, -half_width, half_height},
    Eigen::Vector3d{0.0, -half_width, -half_height},
    Eigen::Vector3d{0.0, half_width, -half_height}};
  std::array<Eigen::Vector3d, 4> world_corners;
  std::transform(
    local_corners.begin(),
    local_corners.end(),
    world_corners.begin(),
    [&armor_pose, &R_world_armor](const Eigen::Vector3d& corner) {
      return armor_pose.head<3>() + R_world_armor * corner;
    });
  return world_corners;
}

[[nodiscard]] L3Estimation::ArmorType estimatedArmorType(
  L3Estimation::ArmorName name) noexcept
{
  return name == L3Estimation::ArmorName::Hero
    ? L3Estimation::ArmorType::Big
    : L3Estimation::ArmorType::Small;
}

[[nodiscard]] std::vector<ProjectedArmorBox> projectFilteredVehicle(
  const L3Estimation::TargetState& target,
  const std::vector<Eigen::Vector4d>& armor_poses,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::ArmorConfig& armor_config,
  const Eigen::Quaterniond& q_world_barrel)
{
  std::vector<ProjectedArmorBox> projected_boxes;
  projected_boxes.reserve(armor_poses.size());
  const auto armor_type = estimatedArmorType(target.name);
  for (std::size_t armor_id = 0; armor_id < armor_poses.size(); ++armor_id) {
    const auto& armor_pose = armor_poses[armor_id];
    if (!armor_pose.allFinite()) {
      continue;
    }
    const auto world_corners = estimatedArmorCorners(
      armor_pose, armor_type, target.name, armor_config);
    const std::vector<Eigen::Vector3d> world_corner_points(
      world_corners.begin(), world_corners.end());
    const auto image_corners = projectWorldPoints(
      world_corner_points,
      calibration,
      q_world_barrel);
    if (!image_corners || image_corners->size() != 4) {
      continue;
    }

    ProjectedArmorBox box{.armor_id = armor_id};
    std::copy(
      image_corners->begin(),
      image_corners->end(),
      box.corners.begin());
    projected_boxes.push_back(box);
  }
  return projected_boxes;
}

[[nodiscard]] VehicleOverlayResult drawFilteredVehicle(
  cv::Mat& image,
  const std::optional<L3Estimation::TargetState>& target,
  const std::vector<Eigen::Vector4d>& armor_poses,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::ArmorConfig& armor_config,
  const Eigen::Quaterniond& q_world_barrel,
  int frame_index,
  VehicleOverlayState& state)
{
  bool received_new_projection = false;
  if (target) {
    if (state.target_name && *state.target_name != target->name) {
      state = {};
    }

    auto projected_boxes = projectFilteredVehicle(
      *target,
      armor_poses,
      calibration,
      armor_config,
      q_world_barrel);
    if (!projected_boxes.empty()) {
      bool smooth =
        state.target_name && *state.target_name == target->name &&
        state.boxes.size() == projected_boxes.size() &&
        state.last_update_frame + 1 == frame_index;
      if (smooth) {
        double maximum_corner_step = 0.0;
        for (std::size_t armor_id = 0;
             armor_id < projected_boxes.size();
             ++armor_id) {
          if (projected_boxes[armor_id].armor_id !=
              state.boxes[armor_id].armor_id) {
            smooth = false;
            break;
          }
          for (std::size_t corner_id = 0; corner_id < 4; ++corner_id) {
            maximum_corner_step = std::max(
              maximum_corner_step,
              static_cast<double>(cv::norm(
                projected_boxes[armor_id].corners[corner_id] -
                state.boxes[armor_id].corners[corner_id])));
          }
        }
        // Tracker 重置或关联跳变时不要把两套相距很远的框拖出残影。
        smooth = smooth &&
          maximum_corner_step <= kOverlayResetDistancePixels;
      }

      if (smooth) {
        for (std::size_t armor_id = 0;
             armor_id < projected_boxes.size();
             ++armor_id) {
          for (std::size_t corner_id = 0; corner_id < 4; ++corner_id) {
            projected_boxes[armor_id].corners[corner_id] =
              state.boxes[armor_id].corners[corner_id] *
                static_cast<float>(1.0 - kOverlaySmoothingAlpha) +
              projected_boxes[armor_id].corners[corner_id] *
                static_cast<float>(kOverlaySmoothingAlpha);
          }
        }
      }

      state.target_name = target->name;
      state.boxes = std::move(projected_boxes);
      state.last_update_frame = frame_index;
      received_new_projection = true;
    }
  }

  VehicleOverlayResult result;
  result.age_frames = state.last_update_frame < 0
    ? kOverlayHoldFrames + 1
    : frame_index - state.last_update_frame;
  if (state.boxes.empty() || result.age_frames > kOverlayHoldFrames) {
    if (result.age_frames > kOverlayHoldFrames) {
      state = {};
    }
    return result;
  }

  result.held = !received_new_projection;
  const cv::Scalar color = result.held
    ? cv::Scalar{0, 150, 0}
    : cv::Scalar{0, 255, 0};
  const int thickness = result.held ? 2 : 3;
  for (const auto& box : state.boxes) {
    // 与 SP-Vision auto_aim_test.cpp 一致，当前滤波整车装甲板统一为绿色。
    for (std::size_t index = 0; index < box.corners.size(); ++index) {
      cv::line(
        image,
        toPixel(box.corners[index]),
        toPixel(box.corners[(index + 1) % box.corners.size()]),
        color,
        thickness,
        cv::LINE_AA);
    }
    cv::putText(
      image,
      std::to_string(box.armor_id),
      toPixel(box.corners[0]) + cv::Point{5, -5},
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      color,
      thickness,
      cv::LINE_AA);
    ++result.drawn_count;
  }
  return result;
}

void drawReplay(
  cv::Mat& image,
  const std::vector<L2Perception::Armor>& detections,
  const L3Estimation::Tracker& tracker,
  const std::optional<L3Estimation::TargetState>& target,
  const std::vector<Eigen::Vector4d>& target_armor_poses,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::ArmorConfig& armor_config,
  const Eigen::Quaterniond& q_world_barrel,
  const PnpCostDiagnostic* cost_diagnostic,
  int frame_index,
  VehicleOverlayState& vehicle_overlay)
{
  for (const auto& detection : detections) {
    const cv::Scalar color = detection.color == L2Perception::ArmorColor::Red
      ? cv::Scalar{0, 0, 255}
      : detection.color == L2Perception::ArmorColor::Blue
        ? cv::Scalar{255, 0, 0}
        : cv::Scalar{0, 255, 255};
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
      cv::line(
        image,
        toPixel(detection.corners[index]),
        toPixel(detection.corners[(index + 1) % detection.corners.size()]),
        color,
        2,
        cv::LINE_AA);
    }
  }

  for (const auto& observation : tracker.observations()) {
    const cv::Scalar color =
      observation.quality.pnp_ok && observation.quality.geometry_ok &&
          observation.quality.reprojection_ok && observation.quality.finite
        ? cv::Scalar{255, 255, 0}
        : cv::Scalar{0, 165, 255};
    for (const auto& point : observation.points) {
      cv::circle(image, toPixel(point), 4, color, cv::FILLED, cv::LINE_AA);
    }
  }

  const VehicleOverlayResult overlay_result = drawFilteredVehicle(
    image,
    target,
    target_armor_poses,
    calibration,
    armor_config,
    q_world_barrel,
    frame_index,
    vehicle_overlay);

  std::size_t pnp_valid_count = 0;
  for (const auto& observation : tracker.observations()) {
    if (observation.quality.pnp_ok && observation.quality.geometry_ok &&
        observation.quality.reprojection_ok && observation.quality.finite) {
      ++pnp_valid_count;
    }
  }

  std::string model_status;
  if (!target) {
    model_status = overlay_result.drawn_count > 0
      ? "HOLD(" + std::to_string(overlay_result.drawn_count) +
          ",age=" + std::to_string(overlay_result.age_frames) + ")"
      : "N/A";
  } else if (overlay_result.held && overlay_result.drawn_count > 0) {
    model_status = "HOLD(" + std::to_string(overlay_result.drawn_count) +
      ",age=" + std::to_string(overlay_result.age_frames) + ")";
  } else if (overlay_result.drawn_count > 0) {
    model_status = "SMOOTH(" + std::to_string(overlay_result.drawn_count) + "/" +
      std::to_string(target_armor_poses.size()) + ")";
  } else if (!target_armor_poses.empty()) {
    model_status = "PROJECTION_FAILED(0/" +
      std::to_string(target_armor_poses.size()) + ")";
  } else {
    model_status = "EMPTY";
  }

  drawOutlinedText(
    image,
    cv::format(
      "frame=%d state=%s det=%zu pnp=%zu waitKey=30ms",
      frame_index,
      std::string(stateName(tracker.state())).c_str(),
      detections.size(),
      pnp_valid_count),
    {12, 32},
    {255, 255, 255});
  drawOutlinedText(
    image,
    cv::format("gimbal yaw=%.2f deg", yawDegrees(q_world_barrel)),
    {12, 62},
    {255, 255, 255});

  if (!tracker.observations().empty()) {
    const auto& observation = tracker.observations().front();
    drawOutlinedText(
      image,
      cv::format(
        "PnP xyz=(%.2f,%.2f,%.2f)m yaw=%.2fdeg rmse=%.2fpx",
        observation.xyz_in_world.x(),
        observation.xyz_in_world.y(),
        observation.xyz_in_world.z(),
        observation.ypr_in_world[0] * 180.0 / std::numbers::pi,
        observation.reprojection_error),
      {12, 92},
      {255, 255, 0});
  } else {
    drawOutlinedText(image, "PnP observation=N/A", {12, 92}, {0, 165, 255});
  }

  if (target) {
    drawOutlinedText(
      image,
      cv::format(
        "EKF center=(%.2f,%.2f,%.2f)m velocity=(%.2f,%.2f,%.2f)m/s",
        target->position.x(),
        target->position.y(),
        target->position.z(),
        target->velocity.x(),
        target->velocity.y(),
        target->velocity.z()),
      {12, 122},
      {0, 255, 0});
    drawOutlinedText(
      image,
      cv::format(
        "EKF yaw=%.2fdeg v_yaw=%.2frad/s r1=%.3fm last_id=%d NIS=%.2f model=%s",
        target->yaw * 180.0 / std::numbers::pi,
        target->v_yaw,
        target->radius,
        target->armor_id,
        target->nis,
        model_status.c_str()),
      {12, 152},
      {0, 255, 0});
  } else {
    drawOutlinedText(
      image,
      "EKF target=N/A model=N/A",
      {12, 122},
      {0, 165, 255});
  }

  drawOutlinedText(
    image,
    cost_diagnostic
      ? cv::format(
          "IPPE ambiguous=%s solvePnP=%s EKF-nearest=%s changed=%s curve-minima=%zu",
          cost_diagnostic->ippe_ambiguous ? "YES" : "NO",
          branchName(cost_diagnostic->single_pnp_ippe_index).c_str(),
          branchName(cost_diagnostic->ekf_nearest_ippe).c_str(),
          cost_diagnostic->ekf_nearest_ippe_changed ? "YES" : "NO",
          cost_diagnostic->local_minimum_count)
      : std::string{"IPPE/cost diagnostic=N/A"},
    {12, 182},
    cost_diagnostic && cost_diagnostic->ippe_ambiguous
      ? cv::Scalar{0, 165, 255}
      : cv::Scalar{255, 255, 255},
    0.54);
  drawOutlinedText(
    image,
    "red/blue: detector  cyan: PnP  green: smoothed EKF  dark green: held <=5f",
    {12, 208},
    {255, 255, 255},
    0.54);
  drawOutlinedText(
    image,
    "EKF-nearest is diagnostic: current filter receives only searched yaw",
    {12, 234},
    {255, 255, 255},
    0.54);
  drawOutlinedText(
    image,
    "space: pause/resume  n: single-step while paused  q/esc: quit",
    {12, 260},
    {255, 255, 255},
    0.54);
}

void writeCsvHeader(std::ostream& output)
{
  output
    << "frame,time_s,raw_detections,enemy_detections,pnp_valid,model_armors,"
       "state,updated,class_id,armor_id,x,vx,y,vy,z,vz,yaw,"
       "v_yaw,radius,nis,P_trace,detector_ms,tracker_ms,diag_observation,"
       "ippe_solutions,ippe_ambiguous,ippe0_yaw,ippe0_rmse,ippe1_yaw,"
       "ippe1_rmse,single_pnp_ippe,single_pnp_position_delta,searched_yaw,"
       "searched_yaw_nearest_ippe,curve_best_yaw,curve_best_cost,"
       "curve_local_minima,curve_second_yaw,curve_second_cost,filter_updated,"
       "ekf_armor_id,ekf_armor_yaw,ekf_nearest_ippe,"
       "ekf_nearest_ippe_changed,ekf_nearest_ippe_angle\n";
}

void writeCsvRow(
  std::ostream& output,
  int frame_index,
  const PoseSample& pose,
  std::size_t raw_detection_count,
  std::size_t enemy_detection_count,
  std::size_t pnp_valid_count,
  std::size_t model_armor_count,
  L3Estimation::TrackState state,
  const std::optional<L3Estimation::TargetState>& target,
  const std::optional<PnpCostDiagnostic>& diagnostic,
  double detector_ms,
  double tracker_ms)
{
  output << frame_index << ',' << pose.seconds << ',' << raw_detection_count
         << ',' << enemy_detection_count << ',' << pnp_valid_count << ','
         << model_armor_count << ',' << stateName(state) << ',';
  if (target) {
    output << target->updated << ',' << static_cast<int>(target->name) << ','
           << target->armor_id << ',' << target->position.x() << ','
           << target->velocity.x() << ',' << target->position.y() << ','
           << target->velocity.y() << ',' << target->position.z() << ','
           << target->velocity.z() << ',' << target->yaw << ','
           << target->v_yaw << ',' << target->radius << ',' << target->nis
           << ',' << target->P.trace();
  } else {
    // updated 到 P_trace 共 14 个空字段；state 后的逗号已经由上面写出。
    for (int field = 1; field < 14; ++field) {
      output << ',';
    }
  }
  output << ',' << detector_ms << ',' << tracker_ms;
  constexpr int kDiagnosticFieldCount = 22;
  if (!diagnostic) {
    for (int field = 0; field < kDiagnosticFieldCount; ++field) {
      output << ',';
    }
    output << '\n';
    return;
  }

  const auto& data = *diagnostic;
  output << ',' << data.observation_index
         << ',' << data.raw_ippe_solution_count
         << ',' << data.ippe_ambiguous;
  output << ',';
  if (!data.ippe_candidates.empty()) {
    output << data.ippe_candidates[0].yaw_in_world;
  }
  output << ',';
  if (!data.ippe_candidates.empty()) {
    output << data.ippe_candidates[0].reprojection_rmse;
  }
  output << ',';
  if (data.ippe_candidates.size() >= 2) {
    output << data.ippe_candidates[1].yaw_in_world;
  }
  output << ',';
  if (data.ippe_candidates.size() >= 2) {
    output << data.ippe_candidates[1].reprojection_rmse;
  }
  output << ',' << data.single_pnp_ippe_index
         << ',' << data.single_pnp_position_delta
         << ',' << data.observation_yaw
         << ',' << data.observation_nearest_ippe
         << ',' << data.best_curve_minimum.yaw_in_world
         << ',' << data.best_curve_minimum.cost
         << ',' << data.local_minimum_count;
  output << ',';
  if (data.second_curve_minimum) {
    output << data.second_curve_minimum->yaw_in_world;
  }
  output << ',';
  if (data.second_curve_minimum) {
    output << data.second_curve_minimum->cost;
  }
  output << ',' << data.filter_updated
         << ',' << data.ekf_armor_id;
  output << ',';
  if (std::isfinite(data.ekf_armor_yaw)) {
    output << data.ekf_armor_yaw;
  }
  output << ',' << data.ekf_nearest_ippe
         << ',' << data.ekf_nearest_ippe_changed;
  output << ',';
  if (std::isfinite(data.ekf_nearest_ippe_angle)) {
    output << data.ekf_nearest_ippe_angle;
  }
  output << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
    if (cli.get<bool>("help")) {
      cli.printMessage();
      return 0;
    }
    if (!cli.check()) {
      cli.printErrors();
      return 1;
    }

    const std::filesystem::path input_base{cli.get<std::string>(0)};
    const std::filesystem::path video_path =
      withExtension(input_base, ".avi");
    const std::filesystem::path pose_path =
      withExtension(input_base, ".txt");
    const std::filesystem::path model_path{cli.get<std::string>("model")};
    const std::filesystem::path calibration_path{
      cli.get<std::string>("calibration")};
    const std::filesystem::path csv_path{cli.get<std::string>("csv")};
    const std::string device = cli.get<std::string>("device");
    const auto enemy_color = parseEnemyColor(cli.get<std::string>("enemy"));
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const int ekf_iterations = cli.get<int>("ekf-iterations");
    require(ekf_iterations >= 1, "--ekf-iterations must be at least 1");
    const int requested_show_from_index = cli.get<int>("show-from-index");
    const int show_from_index = requested_show_from_index < 0
      ? start_index
      : requested_show_from_index;
    const bool show = cli.get<bool>("show");
    require(start_index >= 0, "start-index must not be negative");
    require(end_index == 0 || end_index >= start_index,
            "end-index must be zero or no smaller than start-index");
    require(
      show_from_index >= start_index,
      "show-from-index must not be smaller than start-index");

    L6Telemetry::initLogger();

    const YAML::Node calibration_yaml = YAML::LoadFile(calibration_path.string());
    const auto calibration = L1Sensor::loadCameraCalibration(
      calibration_yaml["calibration"], calibration_path.string());

    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = model_path;
    model_config.device = device;
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);
    require(backend->ready(), "current OpenVINO armor backend is not ready");
    L2Perception::ArmorDetector detector(std::move(backend));
    require(detector.ready(), "current ArmorDetector is not ready");

    const L3Estimation::ArmorConfig armor_config;
    // 迭代次数走命令行，方便同一段回放对照单次线性化和 Gauss-Newton 迭代。
    L3Estimation::TrackerConfig tracker_config;
    tracker_config.ekf_max_iterations = ekf_iterations;
    L3Estimation::Tracker tracker(calibration, armor_config, tracker_config);
    require(tracker.ready(), "current Tracker rejected replay calibration");
    std::cout << "ekf_max_iterations = " << tracker_config.ekf_max_iterations
              << '\n';
    L3Estimation::PnpSolver diagnostic_solver(calibration, armor_config);
    require(
      diagnostic_solver.ready(),
      "diagnostic PnP solver rejected replay calibration");

    cv::VideoCapture video(video_path.string());
    require(video.isOpened(), "failed to open replay video: " + video_path.string());
    std::ifstream pose_input(pose_path);
    require(pose_input.is_open(), "failed to open pose text: " + pose_path.string());

    if (!csv_path.parent_path().empty()) {
      std::filesystem::create_directories(csv_path.parent_path());
    }
    std::ofstream csv(csv_path);
    require(csv.is_open(), "failed to open replay CSV: " + csv_path.string());
    csv << std::setprecision(12);
    writeCsvHeader(csv);

    PoseSample skipped_pose;
    for (int index = 0; index < start_index; ++index) {
      require(video.grab(), "video ended before start-index");
      require(readPose(pose_input, skipped_pose),
              "pose text ended before start-index");
    }

    cv::Mat frame;
    PoseSample pose;
    require(video.read(frame) && !frame.empty(), "replay video contains no selected frame");
    require(readPose(pose_input, pose), "pose text contains no selected row");
    require(calibration.matchesImageSize(frame.size()),
            "replay frame size does not match calibration");

    // 只预热当前识别链路，不向 Tracker 重复发送第一帧。
    (void)detector.detect(frame);

    if (show) {
      cv::namedWindow("newvision auto_aim replay", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("newvision yaw cost", cv::WINDOW_AUTOSIZE);
    }

    ReplayStats stats;
    std::optional<Eigen::Vector3d> previous_position;
    L3Estimation::TrackState previous_state = tracker.state();
    std::optional<int> previous_ekf_ippe_branch;
    std::optional<L3Estimation::ArmorName> previous_ekf_target_name;
    int previous_ekf_branch_frame = -1;
    const auto replay_epoch = std::chrono::steady_clock::now();
    bool keep_running = true;
    bool paused = false;
    VehicleOverlayState vehicle_overlay;
    int frame_index = start_index;

    while (keep_running) {
      if (end_index > 0 && frame_index > end_index) {
        break;
      }

      const auto detector_begin = std::chrono::steady_clock::now();
      auto detections = detector.detect(frame);
      const auto detector_end = std::chrono::steady_clock::now();
      const std::size_t raw_detection_count = detections.size();
      std::erase_if(detections, [enemy_color](const auto& detection) {
        return !matchesEnemy(detection.color, enemy_color);
      });

      const auto replay_offset = std::chrono::duration_cast<
        std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(pose.seconds));
      const auto timestamp = replay_epoch + replay_offset;
      const std::optional<Eigen::Quaterniond> q_world_barrel{
        toWorldBarrelPose(pose)};
      diagnostic_solver.set_R_world_barrel(q_world_barrel);

      const auto tracker_begin = std::chrono::steady_clock::now();
      const auto target = tracker.track(detections, q_world_barrel, timestamp);
      const auto target_armor_poses = tracker.targetArmorPoses();
      const auto tracker_end = std::chrono::steady_clock::now();
      auto cost_diagnostic = buildPnpCostDiagnostic(
        tracker.observations(),
        target,
        target_armor_poses,
        calibration,
        armor_config,
        *q_world_barrel,
        diagnostic_solver);
      const double detector_ms = std::chrono::duration<double, std::milli>(
        detector_end - detector_begin).count();
      const double tracker_ms = std::chrono::duration<double, std::milli>(
        tracker_end - tracker_begin).count();

      std::size_t pnp_valid_count = 0;
      for (const auto& observation : tracker.observations()) {
        if (observation.quality.pnp_ok && observation.quality.geometry_ok &&
            observation.quality.reprojection_ok && observation.quality.finite) {
          ++pnp_valid_count;
        }
      }

      ++stats.frames;
      stats.raw_detections += raw_detection_count;
      stats.enemy_detections += detections.size();
      stats.pnp_valid_observations += pnp_valid_count;
      stats.detector_ms_sum += detector_ms;
      stats.tracker_ms_sum += tracker_ms;
      if (cost_diagnostic) {
        ++stats.cost_diagnostic_frames;
        if (cost_diagnostic->ippe_candidates.size() >= 2) {
          ++stats.ippe_two_solution_frames;
        }
        if (cost_diagnostic->ippe_ambiguous) {
          ++stats.ippe_ambiguous_frames;
        }
        if (cost_diagnostic->single_pnp_ippe_index == 0) {
          ++stats.single_pnp_lower_rmse_frames;
        } else if (cost_diagnostic->single_pnp_ippe_index == 1) {
          ++stats.single_pnp_higher_rmse_frames;
        } else {
          ++stats.single_pnp_unmatched_frames;
        }

        if (cost_diagnostic->filter_updated) {
          if (cost_diagnostic->ekf_nearest_ippe == 0) {
            ++stats.ekf_nearest_lower_rmse_updates;
          } else if (cost_diagnostic->ekf_nearest_ippe == 1) {
            ++stats.ekf_nearest_higher_rmse_updates;
          } else {
            ++stats.ekf_nearest_unavailable_updates;
          }

          const int current_branch = cost_diagnostic->ekf_nearest_ippe;
          if (target && (current_branch == 0 || current_branch == 1)) {
            if (previous_ekf_ippe_branch && previous_ekf_target_name &&
                *previous_ekf_target_name == target->name &&
                previous_ekf_branch_frame + 1 == frame_index &&
                *previous_ekf_ippe_branch != current_branch) {
              ++stats.ekf_ippe_branch_switches;
              cost_diagnostic->ekf_nearest_ippe_changed = true;
              std::cout
                << "replay frame " << frame_index
                << ": EKF-nearest IPPE branch changed "
                << *previous_ekf_ippe_branch << " -> " << current_branch
                << " (diagnostic only)\n";
            }
            previous_ekf_ippe_branch = current_branch;
            previous_ekf_target_name = target->name;
            previous_ekf_branch_frame = frame_index;
          }
        }
      }
      if (!target || tracker.state() == L3Estimation::TrackState::Lost) {
        previous_ekf_ippe_branch.reset();
        previous_ekf_target_name.reset();
        previous_ekf_branch_frame = -1;
      }
      if (raw_detection_count > 0) {
        ++stats.frames_with_raw_detections;
      }
      if (!detections.empty()) {
        ++stats.frames_with_enemy_detections;
      }
      if (pnp_valid_count > 0) {
        ++stats.pnp_valid_frames;
      }
      if (tracker.state() != previous_state) {
        ++stats.state_transitions;
        previous_state = tracker.state();
      }
      if (tracker.state() == L3Estimation::TrackState::Tracking) {
        ++stats.tracking_frames;
        ++stats.current_tracking_run;
        stats.longest_tracking_run = std::max(
          stats.longest_tracking_run,
          stats.current_tracking_run);
      } else if (tracker.state() == L3Estimation::TrackState::TempLost) {
        ++stats.temp_lost_frames;
        stats.current_tracking_run = 0;
      } else {
        stats.current_tracking_run = 0;
      }
      // 有几何有效的 PnP，却在本次更新后没有任何目标状态，说明当前
      // 关联/EKF 路径主动丢弃了已有目标；结合日志可定位半径发散重置。
      if (pnp_valid_count > 0 && !target &&
          tracker.state() == L3Estimation::TrackState::Lost) {
        ++stats.resets_with_valid_pnp;
      }

      if (target) {
        ++stats.target_output_frames;
        const bool finite = target->vector().allFinite() && target->P.allFinite() &&
          std::isfinite(target->nis);
        if (!finite) {
          ++stats.nonfinite_target_frames;
        }
        stats.max_speed = std::max(stats.max_speed, target->velocity.norm());
        if (previous_position) {
          stats.max_position_step = std::max(
            stats.max_position_step,
            (target->position - *previous_position).norm());
        }
        previous_position = target->position;
      } else {
        previous_position.reset();
      }

      writeCsvRow(
        csv,
        frame_index,
        pose,
        raw_detection_count,
        detections.size(),
        pnp_valid_count,
        target_armor_poses.size(),
        tracker.state(),
        target,
        cost_diagnostic,
        detector_ms,
        tracker_ms);

      if (show && frame_index >= show_from_index) {
        cv::Mat drawing = frame.clone();
        drawReplay(
          drawing,
          detections,
          tracker,
          target,
          target_armor_poses,
          calibration,
          armor_config,
          *q_world_barrel,
          cost_diagnostic ? &*cost_diagnostic : nullptr,
          frame_index,
          vehicle_overlay);
        cv::resize(drawing, drawing, {}, 0.5, 0.5, cv::INTER_AREA);
        cv::Mat cost_plot = drawYawCostPlot(cost_diagnostic, frame_index);
        cv::imshow("newvision auto_aim replay", drawing);
        cv::imshow("newvision yaw cost", cost_plot);
        // 与 SP-Vision auto_aim_test 保持一致：每帧处理完后固定等待 30 ms。
        while (keep_running) {
          const int key = cv::waitKey(paused ? 0 : 30);
          if (key == 27 || key == 'q' || key == 'Q') {
            keep_running = false;
            break;
          }
          if (key == ' ') {
            paused = !paused;
            if (paused) {
              continue;
            }
            break;
          }
          if (paused) {
            if (key == 'n' || key == 'N') {
              break;
            }
            continue;
          }
          break;
        }
      }

      if (stats.frames % 50 == 0) {
        std::cout << "replay frame " << frame_index
                  << ": detections=" << detections.size()
                  << " pnp=" << pnp_valid_count
                  << " state=" << stateName(tracker.state()) << '\n';
      }

      ++frame_index;
      if (!keep_running || !video.read(frame) || frame.empty()) {
        break;
      }
      if (!readPose(pose_input, pose)) {
        std::cout << "pose text ended; replay stopped at the last paired frame\n";
        break;
      }
    }

    if (show) {
      cv::destroyWindow("newvision auto_aim replay");
      cv::destroyWindow("newvision yaw cost");
    }
    csv.flush();
    require(csv.good(), "failed while writing replay CSV");
    require(stats.frames > 0, "no replay frame was processed");

    const double average_detector_ms =
      stats.detector_ms_sum / static_cast<double>(stats.frames);
    const double average_tracker_ms =
      stats.tracker_ms_sum / static_cast<double>(stats.frames);
    std::cout
      << "\nnewvision auto_aim replay summary\n"
      << "frames: " << stats.frames << '\n'
      << "raw detection frames: " << stats.frames_with_raw_detections << '\n'
      << "enemy detection frames: " << stats.frames_with_enemy_detections << '\n'
      << "raw detections: " << stats.raw_detections << '\n'
      << "enemy detections: " << stats.enemy_detections << '\n'
      << "PnP-valid frames: " << stats.pnp_valid_frames << '\n'
      << "PnP-valid observations: " << stats.pnp_valid_observations << '\n'
      << "target output frames: " << stats.target_output_frames << '\n'
      << "Tracking frames: " << stats.tracking_frames << '\n'
      << "longest continuous Tracking run: " << stats.longest_tracking_run
      << " frames\n"
      << "TempLost frames: " << stats.temp_lost_frames << '\n'
      << "state transitions: " << stats.state_transitions << '\n'
      << "resets with valid PnP: " << stats.resets_with_valid_pnp << '\n'
      << "cost diagnostic frames: " << stats.cost_diagnostic_frames << '\n'
      << "IPPE two-solution frames: " << stats.ippe_two_solution_frames << '\n'
      << "IPPE ambiguous frames: " << stats.ippe_ambiguous_frames << '\n'
      << "single solvePnP matched lower-RMSE IPPE: "
      << stats.single_pnp_lower_rmse_frames << '\n'
      << "single solvePnP matched higher-RMSE IPPE: "
      << stats.single_pnp_higher_rmse_frames << '\n'
      << "single solvePnP candidate unmatched: "
      << stats.single_pnp_unmatched_frames << '\n'
      << "updated EKF nearest lower-RMSE IPPE: "
      << stats.ekf_nearest_lower_rmse_updates << '\n'
      << "updated EKF nearest higher-RMSE IPPE: "
      << stats.ekf_nearest_higher_rmse_updates << '\n'
      << "updated EKF IPPE comparison unavailable: "
      << stats.ekf_nearest_unavailable_updates << '\n'
      << "consecutive EKF-nearest IPPE branch switches: "
      << stats.ekf_ippe_branch_switches << '\n'
      << "non-finite target frames: " << stats.nonfinite_target_frames << '\n'
      << "max target position step: " << stats.max_position_step << " m\n"
      << "max target speed: " << stats.max_speed << " m/s\n"
      << "average detector time: " << average_detector_ms << " ms\n"
      << "average tracker time: " << average_tracker_ms << " ms\n"
      << "CSV: " << csv_path.string() << '\n';

    L6Telemetry::flushLogger();
    if (stats.pnp_valid_observations == 0) {
      std::cerr << "VERDICT: recognition/PnP produced no usable observation; "
                   "the filter cannot be evaluated on this replay.\n";
      return 2;
    }
    if (stats.tracking_frames == 0) {
      std::cerr << "VERDICT: observations exist, but the filter never reached Tracking.\n";
      return 3;
    }
    if (stats.nonfinite_target_frames != 0) {
      std::cerr << "VERDICT: filter produced non-finite state or covariance.\n";
      return 4;
    }
    if (stats.resets_with_valid_pnp != 0) {
      std::cerr << "VERDICT: filter/association reset despite valid PnP; "
                   "it is not stable enough for planning yet.\n";
      return 5;
    }

    std::cout << "VERDICT: filter completed the replay with finite Tracking output.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "newvision auto_aim replay failed: " << error.what() << '\n';
    return 1;
  }
}
