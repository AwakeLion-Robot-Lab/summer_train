#include "l6_telemetry/aim_overlay.hpp"

#include "l5_control/reject_reason.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace L6Telemetry {
namespace {

[[nodiscard]] const char* trackStateName(L3Estimation::TrackState state) noexcept
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

[[nodiscard]] bool isFilterInputArmor(const L3Estimation::Armor& armor)
{
  // 与 Tracker::observationUsable 保持一致，避免把被滤掉的坏解画出来。
  return armor.name != L3Estimation::ArmorName::Unknown &&
    armor.xyz_in_world.allFinite() &&
    std::isfinite(armor.ypr_in_world[0]);
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
[[nodiscard]] std::optional<cv::Point2f> projectWorldPoint(
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

// 将当前帧实际送入目标滤波器的单板 PnP 位姿重投影为红框。
void drawFilterInputArmors(
  cv::Mat& image,
  const std::vector<L3Estimation::Armor>& observations,
  const L3Estimation::PnpSolver& solver,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  for (const auto& armor : observations) {
    if (!isFilterInputArmor(armor)) {
      continue;
    }

    const auto image_points = solver.reproject_armor(
      armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);
    if (image_points.size() != armor.points.size()) {
      continue;
    }
    for (std::size_t index = 0; index < image_points.size(); ++index) {
      cv::line(
        image, toPixel(image_points[index]),
        toPixel(image_points[(index + 1) % image_points.size()]),
        {0, 255, 0}, 2, cv::LINE_AA);
    }

    // armorPoints 使用局部 x=0 的 y-z 平面，因此局部 +x 是装甲板法向；
    // 它在世界系中的方向正是 yaw 所表示的朝向。
    const double pitch = armor.name == L3Estimation::ArmorName::Outpost
      ? -15.0 * std::numbers::pi / 180.0
      : 15.0 * std::numbers::pi / 180.0;
    const double yaw = armor.ypr_in_world[0];
    const Eigen::Vector3d normal_in_world{
      std::cos(yaw) * std::cos(pitch),
      std::sin(yaw) * std::cos(pitch),
      -std::sin(pitch)};
    constexpr double kArrowLengthMeters = 0.16;
    const auto arrow_start = projectWorldPoint(
      armor.xyz_in_world, calibration, q_world_barrel);
    const auto arrow_end = projectWorldPoint(
      armor.xyz_in_world + kArrowLengthMeters * normal_in_world,
      calibration, q_world_barrel);
    if (arrow_start && arrow_end && image_points.size() == 4) {
      // 透视投影不保持垂直关系。为了让图像上的箭头直观看起来垂直于
      // 装甲板，使用投影后长边的二维垂线；三维法向投影只用于决定正负方向。
      const cv::Point2f long_edge =
        (image_points[1] - image_points[0]) +
        (image_points[2] - image_points[3]);
      const double long_edge_length =
        std::hypot(long_edge.x, long_edge.y);
      if (long_edge_length > 1e-3) {
        cv::Point2f perpendicular{
          static_cast<float>(-long_edge.y / long_edge_length),
          static_cast<float>(long_edge.x / long_edge_length)};
        const cv::Point2f physical_direction =
          *arrow_end - *arrow_start;
        if (perpendicular.x * physical_direction.x +
              perpendicular.y * physical_direction.y < 0.0F) {
          perpendicular *= -1.0F;
        }

        const double short_edge_length =
          0.5 * (cv::norm(image_points[3] - image_points[0]) +
                 cv::norm(image_points[2] - image_points[1]));
        const double arrow_length =
          std::clamp(0.8 * short_edge_length, 12.0, 64.0);
        const cv::Point2f arrow_tip =
          *arrow_start + perpendicular * static_cast<float>(arrow_length);
        cv::arrowedLine(
          image, toPixel(*arrow_start), toPixel(arrow_tip),
          {0, 255, 0}, 2, cv::LINE_AA, 0, 0.25);
      }
    }
  }
}

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

  if (input.q_world_barrel) {
    // 绿色：当前帧真正送进滤波器的单板位姿及其朝向。
    drawFilterInputArmors(
      image, input.observations, solver, calibration, *input.q_world_barrel);
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

}  // namespace L6Telemetry
