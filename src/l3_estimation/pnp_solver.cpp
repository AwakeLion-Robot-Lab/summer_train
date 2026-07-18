#include "l3_estimation/pnp_solver.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace L3Estimation {
namespace {

constexpr double kMinimumCornerDepth = 1e-6;

struct PnpCandidate {
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  Eigen::Matrix3d R_camera_armor{Eigen::Matrix3d::Identity()};
  double reprojection_error{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] constexpr std::optional<ArmorType>
armorTypeFromClassId(int class_id) noexcept
{
  const auto armor_class = L2Perception::armorClassFromId(class_id);

  if (armor_class == L2Perception::ArmorClass::Unknown) {
    return std::nullopt;
  }

  return armor_class == L2Perception::ArmorClass::Hero ? ArmorType::Big
                                                       : ArmorType::Small;
}

[[nodiscard]] std::vector<cv::Point3d> armorPoints(
  ArmorType type,
  const ArmorConfig& config)
{
  const double half_width =
    (type == ArmorType::Big ? config.big_width : config.small_width) / 2.0;
  const double half_height = config.height / 2.0;

  // 与 Armor::points 的左上、右上、右下、左下顺序一致。
  return {
    {0.0, half_width, half_height},
    {0.0, -half_width, half_height},
    {0.0, -half_width, -half_height},
    {0.0, half_width, -half_height}};
}

[[nodiscard]] bool validCalibration(
  const L1Sensor::CameraCalibration& calibration)
{
  const std::size_t distortion_count =
    calibration.distortion_coefficients.total();
  const bool distortion_count_ok =
    distortion_count == 4 || distortion_count == 5 ||
    distortion_count == 8 || distortion_count == 12 ||
    distortion_count == 14;

  if (calibration.image_size.width <= 0 ||
      calibration.image_size.height <= 0 ||
      calibration.camera_matrix.type() != CV_64FC1 ||
      calibration.camera_matrix.rows != 3 ||
      calibration.camera_matrix.cols != 3 ||
      calibration.distortion_coefficients.type() != CV_64FC1 ||
      calibration.distortion_coefficients.empty() ||
      !distortion_count_ok ||
      !cv::checkRange(calibration.camera_matrix) ||
      !cv::checkRange(calibration.distortion_coefficients) ||
      calibration.camera_matrix.at<double>(0, 0) <= 0.0 ||
      calibration.camera_matrix.at<double>(1, 1) <= 0.0 ||
      !calibration.T_barrel_camera) {
    return false;
  }

  const auto& transform = *calibration.T_barrel_camera;
  const Eigen::Matrix3d rotation = transform.linear();
  constexpr double kRotationTolerance = 1e-3;
  return transform.matrix().allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <=
           kRotationTolerance &&
         std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

[[nodiscard]] bool validConfig(const ArmorConfig& config)
{
  return std::isfinite(config.small_width) && config.small_width > 0.0 &&
         std::isfinite(config.big_width) && config.big_width > 0.0 &&
         std::isfinite(config.height) && config.height > 0.0 &&
         std::isfinite(config.corner_noise) && config.corner_noise > 0.0 &&
         std::isfinite(config.max_reprojection_error) &&
         config.max_reprojection_error > 0.0 &&
         std::isfinite(config.ambiguity_error_gap) &&
         config.ambiguity_error_gap >= 0.0 &&
         std::isfinite(config.ambiguity_error_ratio) &&
         config.ambiguity_error_ratio >= 1.0 &&
         std::isfinite(config.min_area) && config.min_area >= 0.0;
}

void resetPnpOutput(Armor& armor)
{
  armor.name = ArmorName::Unknown;
  armor.type = ArmorType::Small;
  armor.xyz_in_camera.setZero();
  armor.xyz_in_world.setZero();
  armor.rpy_in_camera.setZero();
  armor.rpy_in_world.setZero();
  armor.ypd_in_world.setZero();
  armor.reprojection_error = std::numeric_limits<double>::infinity();
  armor.second_reprojection_error = std::numeric_limits<double>::infinity();
  armor.facing = 0.0;
  armor.mode = ObservationMode::Single;
  armor.R = ArmorCovariance::Identity();
  armor.quality = {};
}

[[nodiscard]] bool finiteImagePoints(
  const std::array<cv::Point2f, 4>& points)
{
  return std::all_of(points.begin(), points.end(), [](const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
  });
}

[[nodiscard]] std::optional<cv::Vec3d> toVec3d(const cv::Mat& value)
{
  if (value.total() != 3 || !cv::checkRange(value)) {
    return std::nullopt;
  }

  cv::Mat converted;
  value.reshape(1, 3).convertTo(converted, CV_64FC1);
  const cv::Vec3d vector{
    converted.at<double>(0, 0),
    converted.at<double>(1, 0),
    converted.at<double>(2, 0)};
  if (!std::isfinite(vector[0]) || !std::isfinite(vector[1]) ||
      !std::isfinite(vector[2])) {
    return std::nullopt;
  }
  return vector;
}

[[nodiscard]] Eigen::Matrix3d toEigen(const cv::Matx33d& rotation)
{
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = rotation(row, col);
    }
  }
  return result;
}

[[nodiscard]] std::optional<PnpCandidate> evaluateCandidate(
  const cv::Mat& rvec_mat,
  const cv::Mat& tvec_mat,
  const std::vector<cv::Point3d>& object_points,
  const std::vector<cv::Point2f>& image_points,
  const L1Sensor::CameraCalibration& calibration)
{
  const auto rvec = toVec3d(rvec_mat);
  const auto tvec = toVec3d(tvec_mat);
  if (!rvec || !tvec || (*tvec)[2] <= 0.0) {
    return std::nullopt;
  }

  cv::Matx33d rotation;
  cv::Rodrigues(*rvec, rotation);
  if (!cv::checkRange(cv::Mat(rotation))) {
    return std::nullopt;
  }

  for (const auto& point : object_points) {
    const cv::Vec3d point_in_camera =
      rotation * cv::Vec3d{point.x, point.y, point.z} + *tvec;
    if (!std::isfinite(point_in_camera[0]) ||
        !std::isfinite(point_in_camera[1]) ||
        !std::isfinite(point_in_camera[2]) ||
        point_in_camera[2] <= kMinimumCornerDepth) {
      return std::nullopt;
    }
  }

  // 装甲正面是局部 -X；正面方向与“板中心到相机”方向应同向。
  const cv::Vec3d front_in_camera{
    -rotation(0, 0), -rotation(1, 0), -rotation(2, 0)};
  const cv::Vec3d center_to_camera{-(*tvec)[0], -(*tvec)[1], -(*tvec)[2]};
  const double facing = front_in_camera.dot(center_to_camera);
  if (!std::isfinite(facing) || facing <= 0.0) {
    return std::nullopt;
  }

  std::vector<cv::Point2d> reprojected_points;
  cv::projectPoints(
    object_points,
    *rvec,
    *tvec,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    reprojected_points);
  if (reprojected_points.size() != image_points.size()) {
    return std::nullopt;
  }

  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    const double dx = reprojected_points[index].x - image_points[index].x;
    const double dy = reprojected_points[index].y - image_points[index].y;
    if (!std::isfinite(dx) || !std::isfinite(dy)) {
      return std::nullopt;
    }
    squared_error_sum += dx * dx + dy * dy;
  }

  const double reprojection_error =
    std::sqrt(squared_error_sum / static_cast<double>(image_points.size()));
  if (!std::isfinite(reprojection_error)) {
    return std::nullopt;
  }

  return PnpCandidate{
    *rvec, *tvec, toEigen(rotation), reprojection_error};
}

}  // namespace

PnpSolver::PnpSolver(
  const L1Sensor::CameraCalibration& calibration,
  ArmorConfig config)
  : calibration_(calibration), config_(config),
    ready_(validCalibration(calibration_) && validConfig(config_))
{
}

bool PnpSolver::ready() const noexcept
{
  return ready_;
}

void PnpSolver::set_R_world_barrel(
  const std::optional<Eigen::Quaterniond>& barrel_pose)
{
  world_barrel_ready_ = false;
  if (!barrel_pose || !barrel_pose->coeffs().allFinite() ||
      barrel_pose->squaredNorm() <= 1e-12) {
    return;
  }

  R_world_barrel_ = barrel_pose->normalized().toRotationMatrix();
  world_barrel_ready_ = R_world_barrel_.allFinite();
}

void PnpSolver::single_pnp(Armor& armor) const
{
  // Armor 可能被跨帧复用；任何提前返回都只能留下明确的无效输出。
  resetPnpOutput(armor);

  const auto armor_type = armorTypeFromClassId(armor.class_id);
  if (!ready_ || !world_barrel_ready_ || !armor_type ||
      !finiteImagePoints(armor.points)) {
    return;
  }

  const auto object_points = armorPoints(*armor_type, config_);
  const std::vector<cv::Point2f> image_points(
    armor.points.begin(), armor.points.end());
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;

  int solution_count = 0;
  try {
    solution_count = cv::solvePnPGeneric(
      object_points,
      image_points,
      calibration_.camera_matrix,
      calibration_.distortion_coefficients,
      rvecs,
      tvecs,
      false,
      cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception&) {
    return;
  }

  if (solution_count <= 0 || rvecs.empty() || tvecs.empty()) {
    return;
  }
  armor.quality.pnp_ok = true;

  std::vector<PnpCandidate> candidates;
  const std::size_t candidate_count = std::min(rvecs.size(), tvecs.size());
  candidates.reserve(candidate_count);
  try {
    for (std::size_t index = 0; index < candidate_count; ++index) {
      auto candidate = evaluateCandidate(
        rvecs[index],
        tvecs[index],
        object_points,
        image_points,
        calibration_);
      if (candidate) {
        candidates.push_back(*candidate);
      }
    }
  } catch (const cv::Exception&) {
    candidates.clear();
  }

  if (candidates.empty()) {
    return;
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const PnpCandidate& lhs, const PnpCandidate& rhs) {
      return lhs.reprojection_error < rhs.reprojection_error;
    });

  armor.quality.geometry_ok = true;
  const auto& best = candidates.front();
  const double second_error = candidates.size() > 1
    ? candidates[1].reprojection_error
    : std::numeric_limits<double>::infinity();

  const Eigen::Vector3d xyz_in_camera{
    best.tvec[0], best.tvec[1], best.tvec[2]};
  const Eigen::Vector3d xyz_in_barrel =
    *calibration_.T_barrel_camera * xyz_in_camera;
  const Eigen::Vector3d xyz_in_world = R_world_barrel_ * xyz_in_barrel;

  // solvePnP 给出 armor -> camera；静态外参再转换到 barrel 和 world。
  const Eigen::Matrix3d R_barrel_armor =
    calibration_.T_barrel_camera->linear() * best.R_camera_armor;
  const Eigen::Matrix3d R_world_armor =
    R_world_barrel_ * R_barrel_armor;
  const Eigen::Vector3d rpy_in_camera =
    L6Telemetry::rotationToRpy(best.R_camera_armor);
  const Eigen::Vector3d rpy_in_world =
    L6Telemetry::rotationToRpy(R_world_armor);
  const Eigen::Vector3d ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);

  const bool finite =
    xyz_in_camera.allFinite() && xyz_in_barrel.allFinite() &&
    xyz_in_world.allFinite() && best.R_camera_armor.allFinite() &&
    R_world_armor.allFinite() && rpy_in_camera.allFinite() &&
    rpy_in_world.allFinite() && ypd_in_world.allFinite() &&
    std::isfinite(best.reprojection_error);
  if (!finite) {
    armor.quality.finite = false;
    return;
  }

  armor.name = L2Perception::armorClassFromId(armor.class_id);
  armor.type = *armor_type;
  armor.xyz_in_camera = xyz_in_camera;
  armor.xyz_in_world = xyz_in_world;
  armor.rpy_in_camera = rpy_in_camera;
  armor.rpy_in_world = rpy_in_world;
  armor.ypd_in_world = ypd_in_world;
  armor.reprojection_error = best.reprojection_error;
  armor.second_reprojection_error = second_error;
  armor.quality.finite = true;
  armor.quality.reprojection_ok =
    std::isfinite(best.reprojection_error) &&
    best.reprojection_error <= config_.max_reprojection_error;

  const double ambiguity_limit = std::max(
    best.reprojection_error + config_.ambiguity_error_gap,
    best.reprojection_error * config_.ambiguity_error_ratio);
  armor.quality.yaw_ambiguous =
    std::isfinite(second_error) &&
    second_error <= config_.max_reprojection_error &&
    second_error <= ambiguity_limit;

  // 协方差尚未实现，故 covariance_ok 保持 false，禁止进入 EKF 正式更新。
}

}  // namespace L3Estimation
