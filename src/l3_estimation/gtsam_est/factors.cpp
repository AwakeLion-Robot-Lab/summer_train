#include "l3_estimation/gtsam_est/factors.hpp"

#ifdef NEWVISION_USE_GTSAM

#include "l6_telemetry/math.hpp"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/PinholeCamera.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace L3Estimation::GtsamEst {
namespace {

[[nodiscard]] double armorYawOffset(int armor_index, int armor_count)
{
  return static_cast<double>(armor_index) * 2.0 * std::numbers::pi /
         static_cast<double>(armor_count);
}

[[nodiscard]] double logistic(double raw, double minimum, double maximum)
{
  const double unit = raw > 0.0 ? 1.0 / (1.0 + std::exp(-raw))
                                : std::exp(raw) / (1.0 + std::exp(raw));
  return minimum + unit * (maximum - minimum);
}

// derivative of logistic(raw, minimum, maximum), expressed using its output.
[[nodiscard]] double logisticDerivative(
  double output,
  double minimum,
  double maximum)
{
  return (output - minimum) * (maximum - output) / (maximum - minimum);
}

[[nodiscard]] Eigen::Isometry3d toEigen(const gtsam::Pose3& pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = pose.rotation().matrix();
  result.translation() = pose.translation();
  return result;
}

[[nodiscard]] gtsam::Point3 armorPoint(
  ArmorType type,
  const ArmorConfig& config,
  int point_index)
{
  const double width = type == ArmorType::Big ? config.big_width : config.small_width;
  const double half_width = width * 0.5;
  const double half_height = config.height * 0.5;
  const std::array<gtsam::Point3, 4> points{
    gtsam::Point3{0.0, half_width, half_height},
    gtsam::Point3{0.0, -half_width, half_height},
    gtsam::Point3{0.0, -half_width, -half_height},
    gtsam::Point3{0.0, half_width, -half_height}};
  return points.at(static_cast<std::size_t>(point_index));
}

}  // namespace

TranslationFactor::TranslationFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key x_pre,
  gtsam::Key v_pre,
  gtsam::Key x_cur,
  double dt)
: Base(model, x_pre, v_pre, x_cur), dt_(dt)
{
}

gtsam::Vector TranslationFactor::evaluateError(
  const gtsam::Point3& x_pre,
  const gtsam::Vector3& v_pre,
  const gtsam::Point3& x_cur,
  gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2,
  gtsam::OptionalMatrixType H3) const
{
  const gtsam::Vector3 error = x_cur - (x_pre + v_pre * dt_);
  if (H1) {
    *H1 = -gtsam::Matrix3::Identity();
  }
  if (H2) {
    *H2 = -dt_ * gtsam::Matrix3::Identity();
  }
  if (H3) {
    *H3 = gtsam::Matrix3::Identity();
  }
  return error;
}

YawFactor::YawFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key r_pre,
  gtsam::Key w_pre,
  gtsam::Key r_cur,
  double dt)
: Base(model, r_pre, w_pre, r_cur), dt_(dt)
{
}

gtsam::Vector YawFactor::evaluateError(
  const gtsam::Rot2& r_pre,
  const double& w_pre,
  const gtsam::Rot2& r_cur,
  gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2,
  gtsam::OptionalMatrixType H3) const
{
  const gtsam::Vector1 error =
    (r_pre * gtsam::Rot2::fromAngle(w_pre * dt_)).localCoordinates(r_cur);
  if (H1) {
    *H1 = gtsam::Matrix::Constant(1, 1, -1.0);
  }
  if (H2) {
    *H2 = gtsam::Matrix::Constant(1, 1, -dt_);
  }
  if (H3) {
    *H3 = gtsam::Matrix::Identity(1, 1);
  }
  return error;
}

VelocityFactor::VelocityFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key v_pre,
  gtsam::Key v_cur)
: Base(model, v_pre, v_cur)
{
}

gtsam::Vector VelocityFactor::evaluateError(
  const gtsam::Vector3& v_pre,
  const gtsam::Vector3& v_cur,
  gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2) const
{
  const gtsam::Vector3 error = v_cur - v_pre;
  if (H1) {
    *H1 = -gtsam::Matrix3::Identity();
  }
  if (H2) {
    *H2 = gtsam::Matrix3::Identity();
  }
  return error;
}

VyawFactor::VyawFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key w_pre,
  gtsam::Key w_cur)
: Base(model, w_pre, w_cur)
{
}

gtsam::Vector VyawFactor::evaluateError(
  const double& w_pre,
  const double& w_cur,
  gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2) const
{
  const gtsam::Vector1 error{w_cur - w_pre};
  if (H1) {
    *H1 = -gtsam::Matrix1::Identity();
  }
  if (H2) {
    *H2 = gtsam::Matrix1::Identity();
  }
  return error;
}

ArmorRadiusCenterZFactor::ArmorRadiusCenterZFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key armor_pose_key,
  gtsam::Key radius_key,
  gtsam::Key center_yaw_key,
  gtsam::Key center_point_key,
  const Eigen::Isometry3d& T_world_camera,
  int armor_index,
  double radius_min,
  double radius_max,
  int armor_count)
: Base(model, armor_pose_key, radius_key, center_yaw_key, center_point_key),
  T_world_camera_(T_world_camera),
  armor_index_(armor_index),
  radius_min_(radius_min),
  radius_max_(radius_max),
  armor_count_(armor_count)
{
}

gtsam::Vector ArmorRadiusCenterZFactor::evaluateError(
  const gtsam::Pose3& armor_pose_camera,
  const double& radius,
  const gtsam::Rot2& center_yaw,
  const gtsam::Point3& center_point,
  gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2,
  gtsam::OptionalMatrixType H3,
  gtsam::OptionalMatrixType H4) const
{
  const Eigen::Isometry3d armor_pose_world =
    T_world_camera_ * toEigen(armor_pose_camera);
  const double armor_yaw = L6Telemetry::rotationToYpr(armor_pose_world.linear()).x();
  const Eigen::Vector3d armor_position = armor_pose_world.translation();
  const double physical_radius = logistic(radius, radius_min_, radius_max_);
  const double nx = std::cos(armor_yaw);
  const double ny = std::sin(armor_yaw);
  const double tx = -ny;
  const double ty = nx;
  const double dx = center_point.x() - armor_position.x();
  const double dy = center_point.y() - armor_position.y();
  const double tangential_error = tx * dx + ty * dy;
  const double radial_error = nx * dx + ny * dy - physical_radius;
  const double z_error = center_point.z() - armor_position.z();
  const gtsam::Rot2 predicted_yaw = gtsam::Rot2::fromAngle(
    center_yaw.theta() + armorYawOffset(armor_index_, armor_count_));
  const double yaw_error =
    gtsam::Rot2::fromAngle(armor_yaw).localCoordinates(predicted_yaw).x();

  if (H1) {
    Eigen::Matrix<double, 4, 3> position_jacobian;
    position_jacobian <<
      -tx, -ty, 0.0,
      -nx, -ny, 0.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, 0.0;
    const double radial_projection = nx * dx + ny * dy;
    Eigen::Matrix<double, 4, 1> error_yaw_jacobian;
    error_yaw_jacobian << -radial_projection, tangential_error, 0.0, -1.0;
    const Eigen::Vector3d ypr = L6Telemetry::rotationToYpr(armor_pose_world.linear());
    const double roll = ypr.z();
    const double pitch = ypr.y();
    const double safe_cos_pitch = std::copysign(
      std::max(std::abs(std::cos(pitch)), 1e-9), std::cos(pitch));
    Eigen::Matrix<double, 1, 3> yaw_rotation_jacobian;
    yaw_rotation_jacobian <<
      0.0, std::sin(roll) / safe_cos_pitch, std::cos(roll) / safe_cos_pitch;

    Eigen::Matrix<double, 4, 6> jacobian;
    jacobian.leftCols<3>() = error_yaw_jacobian * yaw_rotation_jacobian;
    jacobian.rightCols<3>() =
      position_jacobian * armor_pose_world.rotation();
    *H1 = jacobian;
  }
  if (H2) {
    *H2 = (gtsam::Matrix(4, 1) << 0.0,
      -logisticDerivative(physical_radius, radius_min_, radius_max_),
      0.0, 0.0).finished();
  }
  if (H3) {
    *H3 = (gtsam::Matrix(4, 1) << 0.0, 0.0, 0.0, 1.0).finished();
  }
  if (H4) {
    *H4 = (gtsam::Matrix(4, 3) <<
      tx, ty, 0.0,
      nx, ny, 0.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 0.0).finished();
  }
  return gtsam::Vector4{
    tangential_error, radial_error, z_error, yaw_error};
}

ArmorRadiusDZFactor::ArmorRadiusDZFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key armor_pose_key,
  gtsam::Key radius_key,
  gtsam::Key dz_key,
  gtsam::Key center_yaw_key,
  gtsam::Key center_point_key,
  const Eigen::Isometry3d& T_world_camera,
  int armor_index,
  double radius_min,
  double radius_max,
  int armor_count)
: Base(model, armor_pose_key, radius_key, dz_key, center_yaw_key, center_point_key),
  T_world_camera_(T_world_camera),
  armor_index_(armor_index),
  radius_min_(radius_min),
  radius_max_(radius_max),
  armor_count_(armor_count)
{
}

gtsam::Vector ArmorRadiusDZFactor::evaluateError(
  const gtsam::Pose3& armor_pose_camera,
  const double& radius,
  const double& dz,
  const gtsam::Rot2& center_yaw,
  const gtsam::Point3& center_point,
  gtsam::OptionalMatrixType H1,
  gtsam::OptionalMatrixType H2,
  gtsam::OptionalMatrixType H3,
  gtsam::OptionalMatrixType H4,
  gtsam::OptionalMatrixType H5) const
{
  const Eigen::Isometry3d armor_pose_world =
    T_world_camera_ * toEigen(armor_pose_camera);
  const double armor_yaw = L6Telemetry::rotationToYpr(armor_pose_world.linear()).x();
  const Eigen::Vector3d armor_position = armor_pose_world.translation();
  const double physical_radius = logistic(radius, radius_min_, radius_max_);
  const double nx = std::cos(armor_yaw);
  const double ny = std::sin(armor_yaw);
  const double tx = -ny;
  const double ty = nx;
  const double dx = center_point.x() - armor_position.x();
  const double dy = center_point.y() - armor_position.y();
  const double tangential_error = tx * dx + ty * dy;
  const double radial_error = nx * dx + ny * dy - physical_radius;
  const double z_error = center_point.z() + dz - armor_position.z();
  const gtsam::Rot2 predicted_yaw = gtsam::Rot2::fromAngle(
    center_yaw.theta() + armorYawOffset(armor_index_, armor_count_));
  const double yaw_error =
    gtsam::Rot2::fromAngle(armor_yaw).localCoordinates(predicted_yaw).x();

  if (H1) {
    Eigen::Matrix<double, 4, 3> position_jacobian;
    position_jacobian <<
      -tx, -ty, 0.0,
      -nx, -ny, 0.0,
      0.0, 0.0, -1.0,
      0.0, 0.0, 0.0;
    const double radial_projection = nx * dx + ny * dy;
    Eigen::Matrix<double, 4, 1> error_yaw_jacobian;
    error_yaw_jacobian << -radial_projection, tangential_error, 0.0, -1.0;
    const Eigen::Vector3d ypr = L6Telemetry::rotationToYpr(armor_pose_world.linear());
    const double roll = ypr.z();
    const double pitch = ypr.y();
    const double safe_cos_pitch = std::copysign(
      std::max(std::abs(std::cos(pitch)), 1e-9), std::cos(pitch));
    Eigen::Matrix<double, 1, 3> yaw_rotation_jacobian;
    yaw_rotation_jacobian <<
      0.0, std::sin(roll) / safe_cos_pitch, std::cos(roll) / safe_cos_pitch;
    Eigen::Matrix<double, 4, 6> jacobian;
    jacobian.leftCols<3>() = error_yaw_jacobian * yaw_rotation_jacobian;
    jacobian.rightCols<3>() =
      position_jacobian * armor_pose_world.rotation();
    *H1 = jacobian;
  }
  if (H2) {
    *H2 = (gtsam::Matrix(4, 1) << 0.0,
      -logisticDerivative(physical_radius, radius_min_, radius_max_),
      0.0, 0.0).finished();
  }
  if (H3) {
    *H3 = (gtsam::Matrix(4, 1) << 0.0, 0.0, 1.0, 0.0).finished();
  }
  if (H4) {
    *H4 = (gtsam::Matrix(4, 1) << 0.0, 0.0, 0.0, 1.0).finished();
  }
  if (H5) {
    *H5 = (gtsam::Matrix(4, 3) <<
      tx, ty, 0.0,
      nx, ny, 0.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 0.0).finished();
  }
  return gtsam::Vector4{
    tangential_error, radial_error, z_error, yaw_error};
}

ArmorReprojFactor::ArmorReprojFactor(
  const gtsam::SharedNoiseModel& model,
  gtsam::Key armor_pose_key,
  const cv::Mat& camera_matrix,
  const cv::Mat& distortion_coefficients,
  ArmorType type,
  const ArmorConfig& armor_config,
  int point_index,
  Eigen::Vector2d px_point)
: Base(model, armor_pose_key),
  px_point_(std::move(px_point)),
  armor_point_(armorPoint(type, armor_config, point_index))
{
  const cv::Mat distortion = distortion_coefficients.reshape(1, 1);
  const auto coefficient = [&distortion](int index) {
    return index < distortion.cols ? distortion.at<double>(0, index) : 0.0;
  };
  calib_ = gtsam::Cal3DS2(
    camera_matrix.at<double>(0, 0),
    camera_matrix.at<double>(1, 1),
    camera_matrix.at<double>(0, 1),
    camera_matrix.at<double>(0, 2),
    camera_matrix.at<double>(1, 2),
    coefficient(0), coefficient(1), coefficient(2), coefficient(3));
}

gtsam::Vector ArmorReprojFactor::evaluateError(
  const gtsam::Pose3& armor_pose_camera,
  gtsam::OptionalMatrixType H) const
{
  gtsam::Matrix36 transform_jacobian;
  const gtsam::Point3 point_camera = armor_pose_camera.transformFrom(
    armor_point_, H ? &transform_jacobian : nullptr, nullptr);
  gtsam::Matrix23 project_jacobian;
  const gtsam::Point2 normalized =
    gtsam::PinholeCamera<gtsam::Cal3DS2>::Project(
      point_camera, H ? &project_jacobian : nullptr);
  gtsam::Matrix22 calibration_jacobian;
  const gtsam::Point2 pixel = calib_.uncalibrate(
    normalized, {}, H ? &calibration_jacobian : nullptr);
  if (H) {
    *H = calibration_jacobian * project_jacobian * transform_jacobian;
  }
  return pixel - px_point_;
}

}  // namespace L3Estimation::GtsamEst

#endif  // NEWVISION_USE_GTSAM
