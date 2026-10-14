#include "rune/f2-building/builder.hpp"

#include <opencv2/calib3d.hpp>

#include <array>
#include <cmath>

namespace Rune {

std::optional<Pose> BuildingStage::run(const Observation& observation,
                                       const Frame& frame) const {
  if (!frame.T_barrel_camera || frame.camera_matrix.empty() ||
      frame.camera_matrix.rows != 3 || frame.camera_matrix.cols != 3 ||
      !std::isfinite(geometry_.radius_m) ||
      !std::isfinite(geometry_.armor_half_width_m) ||
      !std::isfinite(geometry_.armor_half_height_m) ||
      !std::isfinite(geometry_.max_reprojection_error_px) ||
      geometry_.radius_m <= 0 || geometry_.armor_half_width_m <= 0 ||
      geometry_.armor_half_height_m <= 0 ||
      geometry_.max_reprojection_error_px <= 0 ||
      observation.armor.empty() || observation.arm.empty() ||
      observation.center_r.empty()) return std::nullopt;

  const double radius = geometry_.radius_m;
  const double half_width = geometry_.armor_half_width_m;
  const double half_height = geometry_.armor_half_height_m;
  const std::vector<cv::Point3f> model{
    {0, static_cast<float>(radius + half_height), 0},
    {-static_cast<float>(half_width), static_cast<float>(radius), 0},
    {static_cast<float>(half_width), static_cast<float>(radius), 0},
    {0, static_cast<float>(radius - half_height), 0},
    {0, 0, 0}};
  const auto& k = observation.keypoints;
  const std::vector<cv::Point2f> image{k.top, k.left, k.right, k.bottom, k.center_r};
  cv::Mat rotation_vector, translation_vector, rotation;
  try {
    if (!cv::solvePnP(model, image, frame.camera_matrix, frame.distortion,
                      rotation_vector, translation_vector, false,
                      cv::SOLVEPNP_ITERATIVE)) return std::nullopt;
    cv::Rodrigues(rotation_vector, rotation);
  } catch (const cv::Exception&) { return std::nullopt; }

  std::vector<cv::Point2f> projected;
  try {
    cv::projectPoints(model, rotation_vector, translation_vector,
                      frame.camera_matrix, frame.distortion, projected);
  } catch (const cv::Exception&) { return std::nullopt; }
  double squared_error = 0.0;
  for (std::size_t i = 0; i < image.size(); ++i) {
    const cv::Point2f delta = image[i] - projected[i];
    squared_error += delta.dot(delta);
  }
  if (!std::isfinite(squared_error) ||
      std::sqrt(squared_error / image.size()) >
        geometry_.max_reprojection_error_px) return std::nullopt;

  cv::Mat r64, t64;
  rotation.convertTo(r64, CV_64F);
  translation_vector.convertTo(t64, CV_64F);
  if (!cv::checkRange(r64) || !cv::checkRange(t64)) return std::nullopt;
  Eigen::Matrix3d R;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col) R(row, col) = r64.at<double>(row, col);
  const Eigen::Vector3d t(t64.at<double>(0), t64.at<double>(1), t64.at<double>(2));
  if (t.z() <= 0 || !frame.T_barrel_camera->matrix().allFinite())
    return std::nullopt;

  Pose pose;
  const auto& transform = *frame.T_barrel_camera;
  pose.center_barrel = transform * t;
  pose.target_barrel = transform * (t + R * Eigen::Vector3d(0, radius, 0));
  pose.normal_barrel = (transform.linear() * R.col(2)).normalized();
  if (pose.normal_barrel.x() < 0) pose.normal_barrel = -pose.normal_barrel;
  const Eigen::Vector3d up_projection = Eigen::Vector3d::UnitZ() -
    pose.normal_barrel * pose.normal_barrel.z();
  if (up_projection.norm() < 1e-3 || pose.target_barrel.x() <= 0 ||
      !pose.center_barrel.allFinite() || !pose.target_barrel.allFinite())
    return std::nullopt;
  const Eigen::Vector3d up = up_projection.normalized();
  const Eigen::Vector3d right = pose.normal_barrel.cross(up);
  const Eigen::Vector3d radial = (pose.target_barrel - pose.center_barrel).normalized();
  pose.phase_rad = std::atan2(radial.dot(right), radial.dot(up));
  pose.activated = k.class_id == 1;
  pose.captured_at = frame.captured_at;
  return pose;
}

}  // namespace Rune
