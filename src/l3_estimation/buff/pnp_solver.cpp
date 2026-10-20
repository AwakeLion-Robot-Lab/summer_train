#include "l3_estimation/buff/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace L3Estimation::Buff {
namespace {

Eigen::Matrix3d rosToOpenCv()
{
  Eigen::Matrix3d matrix;
  matrix << 0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
            1.0, 0.0, 0.0;
  return matrix;
}

std::array<cv::Point3f, 5> objectPoints()
{
  // 源 RunePagePoints 顺序：Icon, T, L, B, R。
  const std::array<Eigen::Vector3d, 5> ros_points{
    PagePoints::icon(), PagePoints::top(), PagePoints::left(),
    PagePoints::bottom(), PagePoints::right()};
  const Eigen::Matrix3d matrix = rosToOpenCv();
  std::array<cv::Point3f, 5> points{};
  for (std::size_t i = 0; i < ros_points.size(); ++i) {
    const Eigen::Vector3d point = matrix * ros_points[i];
    points[i] = cv::Point3f{
      static_cast<float>(point.x()), static_cast<float>(point.y()),
      static_cast<float>(point.z())};
  }
  return points;
}

}  // namespace

PnpSolver::PnpSolver(const L1Sensor::CameraCalibration& calibration)
: calibration_(calibration)
{
}

std::optional<PnpResult> PnpSolver::solve(
  const L2Perception::Buff::Icon& icon,
  const L2Perception::Buff::Bullseye& bullseye) const
{
  try {
    if (calibration_.camera_matrix.empty() ||
        calibration_.distortion_coefficients.empty()) return std::nullopt;
    if (calibration_.camera_matrix.rows != 3 ||
        calibration_.camera_matrix.cols != 3) return std::nullopt;
    if (calibration_.camera_matrix.at<double>(0, 0) <= 0.0 ||
        calibration_.camera_matrix.at<double>(1, 1) <= 0.0) return std::nullopt;

    const cv::Point2f icon_point = icon.center;
    const cv::Point2f center = bullseye.center;
    const auto& corners = bullseye.corners;
    std::size_t index_b = 0;
    std::size_t index_t = 0;
    double min_distance = std::numeric_limits<double>::max();
    double max_distance = -std::numeric_limits<double>::max();
    for (std::size_t index = 0; index < corners.size(); ++index) {
      const double distance = std::hypot(
        static_cast<double>(corners[index].x - icon_point.x),
        static_cast<double>(corners[index].y - icon_point.y));
      if (distance < min_distance) {
        index_b = index;
        min_distance = distance;
      }
      if (distance > max_distance) {
        index_t = index;
        max_distance = distance;
      }
    }
    if (index_b == index_t) return std::nullopt;

    std::array<std::size_t, 2> remaining{};
    std::size_t remaining_count = 0;
    for (std::size_t index = 0; index < corners.size(); ++index) {
      if (index == index_b || index == index_t) continue;
      if (remaining_count >= remaining.size()) return std::nullopt;
      remaining[remaining_count++] = index;
    }
    if (remaining_count != remaining.size()) return std::nullopt;

    const cv::Point2f bottom = corners[index_b];
    const cv::Point2f top = corners[index_t];
    const cv::Point2f z_axis_raw = top - bottom;
    const double z_axis_length = std::hypot(z_axis_raw.x, z_axis_raw.y);
    if (z_axis_length <= 1e-6) return std::nullopt;
    const cv::Point2d z_axis{
      z_axis_raw.x / z_axis_length, z_axis_raw.y / z_axis_length};
    const cv::Point2d y_axis{z_axis.y, -z_axis.x};
    const auto first = corners[remaining[0]];
    const auto second = corners[remaining[1]];
    const double first_projection =
      (first.x - center.x) * y_axis.x + (first.y - center.y) * y_axis.y;
    const double second_projection =
      (second.x - center.x) * y_axis.x + (second.y - center.y) * y_axis.y;
    if (std::abs(first_projection - second_projection) <= 1e-6) return std::nullopt;
    const cv::Point2f left = first_projection > second_projection ? first : second;
    const cv::Point2f right = first_projection > second_projection ? second : first;

    const auto points = objectPoints();
    const std::vector<cv::Point3f> object{points.begin(), points.end()};
    const std::vector<cv::Point2f> image{icon_point, top, left, bottom, right};
    cv::Vec3d rotation_vector;
    cv::Vec3d translation_vector;
    if (!cv::solvePnP(object, image, calibration_.camera_matrix,
                      calibration_.distortion_coefficients, rotation_vector,
                      translation_vector, false, cv::SOLVEPNP_EPNP) ||
        translation_vector[2] <= 0.0) return std::nullopt;
    if (!cv::solvePnP(object, image, calibration_.camera_matrix,
                      calibration_.distortion_coefficients, rotation_vector,
                      translation_vector, true, cv::SOLVEPNP_ITERATIVE) ||
        translation_vector[2] <= 0.0) return std::nullopt;

    cv::Mat rotation_cv;
    cv::Rodrigues(rotation_vector, rotation_cv);
    Eigen::Matrix3d rotation_ocv;
    Eigen::Vector3d translation_ocv;
    cv::cv2eigen(rotation_cv, rotation_ocv);
    cv::cv2eigen(translation_vector, translation_ocv);

    const Eigen::Matrix3d cv_to_ros = rosToOpenCv().transpose();
    PnpResult result;
    result.translation_camera = cv_to_ros * translation_ocv;
    result.orientation_camera = Eigen::Quaterniond{
      cv_to_ros * rotation_ocv * rosToOpenCv()}.normalized();
    return result;
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
}

}  // namespace L3Estimation::Buff
