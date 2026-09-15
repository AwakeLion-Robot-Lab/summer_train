#include "l6_telemetry/aim_overlay.hpp"

#include "l5_control/reject_reason.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace L6Telemetry {
namespace {

const char* trackStateName(L3Estimation::TrackState state) noexcept
{
  switch (state) {
  case L3Estimation::TrackState::Lost:      return "lost";
  case L3Estimation::TrackState::Detecting: return "detecting";
  case L3Estimation::TrackState::Tracking:  return "tracking";
  case L3Estimation::TrackState::TempLost:  return "temp-lost";
  }
  return "?";
}

}  // namespace

void drawAimOverlay(
  cv::Mat& image, const AimOverlayInput& input,
  const L3Estimation::PnpSolver& solver,
  const L1Sensor::CameraCalibration& calibration)
{
  // 蓝色：L2 网络给出的原始角点，还没经过任何 PnP。
  for (const auto& detection : input.detections) {
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
      cv::line(
        image, toPixel(detection.corners[index]),
        toPixel(detection.corners[(index + 1) % detection.corners.size()]),
        {255, 128, 0}, 1, cv::LINE_AA);
    }
  }

  if (input.target) {
    const auto type = L3Estimation::armorTypeOf(input.target->name)
                        .value_or(L3Estimation::ArmorType::Small);
    // 绿色：EKF 展开的全部物理装甲板，直接压在图像上，贴不贴板可以目视判断。
    drawVehicle(
      image, input.target->armor_xyza_list(), type, input.target->name, solver,
      {0, 255, 0}, 2);
    // 红色：Plan 选中的命中时刻实体板，它领先绿框是延迟补偿的正常结果。
    if (input.plan.valid() && input.plan.fire.has_value()) {
      drawVehicle(
        image, {input.plan.fire->armor_pose}, type, input.target->name, solver,
        {0, 0, 255}, 2);
    }
  }

  // 一行状态：跟踪状态、规划是否可开火、以及不开火的第一个原因。
  std::string status = trackStateName(input.track_state);
  status += input.plan.fireAdmissible() ? " | plan:fire-ready" : " | plan:track-only";
  if (!input.fire.reasons.empty()) {
    status += " | " + L5Control::toString(input.fire.reasons.front());
    if (input.fire.reasons.size() > 1) {
      status += " +" + std::to_string(input.fire.reasons.size() - 1);
    }
  }
  drawOutlinedText(
    image, status, {12, 28},
    input.fire.shoot ? cv::Scalar{0, 0, 255} : cv::Scalar{0, 255, 255}, 0.7);
}

cv::Point toPixel(const cv::Point2f& point)
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
  double scale)
{
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 4,
    cv::LINE_AA);
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

// 把一个世界系点投到图像上。整车的旋转中心不是装甲板，用不了
// reproject_armor，所以这里单独走一次 world -> camera -> pixel。
std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& point_in_world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  if (!calibration.T_barrel_camera || !point_in_world.allFinite()) {
    return std::nullopt;
  }
  Eigen::Isometry3d T_world_barrel = Eigen::Isometry3d::Identity();
  T_world_barrel.linear() = q_world_barrel.toRotationMatrix();
  const Eigen::Vector3d point_in_camera =
    (T_world_barrel * *calibration.T_barrel_camera).inverse() * point_in_world;
  // 相机后方的点投影出来是镜像的假点，直接丢掉。
  if (!point_in_camera.allFinite() || point_in_camera.z() <= 1e-6) {
    return std::nullopt;
  }

  std::vector<cv::Point2d> projected;
  try {
    cv::projectPoints(
      std::vector<cv::Point3d>{
        {point_in_camera.x(), point_in_camera.y(), point_in_camera.z()}},
      cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), calibration.camera_matrix,
      calibration.distortion_coefficients, projected);
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
  if (projected.size() != 1 || !std::isfinite(projected[0].x) ||
      !std::isfinite(projected[0].y)) {
    return std::nullopt;
  }
  return cv::Point2f(projected[0]);
}

// 把一组整车装甲板位姿画成闭合四边形。
void drawVehicle(
  cv::Mat& image,
  const std::vector<Eigen::Vector4d>& armor_poses,
  L3Estimation::ArmorType type,
  L3Estimation::ArmorName name,
  const L3Estimation::PnpSolver& solver,
  const cv::Scalar& color,
  int thickness,
  cv::Point image_offset)
{
  for (const Eigen::Vector4d& xyza : armor_poses) {
    const auto image_points =
      solver.reproject_armor(xyza.head<3>(), xyza[3], type, name);
    for (std::size_t index = 0; index < image_points.size(); ++index) {
      cv::line(
        image, toPixel(image_points[index]) + image_offset,
        toPixel(image_points[(index + 1) % image_points.size()]) + image_offset,
        color,
        thickness, cv::LINE_AA);
    }
  }
}

}  // namespace L6Telemetry
