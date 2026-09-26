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

const char* trackingPhaseName(L4Planning::ArmorTrackingPhase phase) noexcept
{
  switch (phase) {
  case L4Planning::ArmorTrackingPhase::Unlocked:    return "unlocked";
  case L4Planning::ArmorTrackingPhase::Tracking:    return "tracking";
  case L4Planning::ArmorTrackingPhase::Switching:   return "switching";
  case L4Planning::ArmorTrackingPhase::Stabilizing: return "stabilizing";
  }
  return "?";
}

std::string rejectReasons(const L5Control::FireDecision& fire)
{
  std::string result;
  for (const auto reason : fire.reasons) {
    if (!result.empty()) {
      result += ',';
    }
    result += L5Control::toString(reason);
  }
  return result;
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

  if (input.q_world_barrel) {
    // 绿色：当前帧真正送进滤波器的单板位姿及其朝向。
    drawFilterInputArmors(
      image, input.observations, solver, calibration, *input.q_world_barrel);
  }

  if (input.target) {
    const auto type = L3Estimation::armorTypeOf(input.target->name)
                        .value_or(L3Estimation::ArmorType::Small);
    const auto current_armors = input.target->armor_xyza_list();
    // 绿色：EKF 展开的全部物理装甲板，直接压在图像上，贴不贴板可以目视判断。
    drawVehicle(
      image, current_armors, type, input.target->name, solver,
      {0, 255, 0}, 2);
    const auto tracked_armor = trackingRedArmorPose(
      input.target, input.track_state, input.plan);
    if (tracked_armor) {
      // fire_feasible 时先画同一命中预测板的粗紫框，再在上面
      // 画细红框，让紫色作为红框外沿始终可见。使用 fire_feasible
      // 而不是 shoot，使总开火开关关闭时仍能可视化判定结果。
      if (input.fire.fire_feasible) {
        drawVehicle(
          image, {*tracked_armor}, type, input.target->name, solver,
          {255, 0, 255}, 4);
      }

      // 红色：只有 Tracker 处于 Tracking 且 Plan 有效时，
      // 才画 Plan 选中的命中时刻预测板。
      drawVehicle(
        image, {*tracked_armor}, type, input.target->name, solver,
        {0, 0, 255}, 2);
    }
  }

  // 第一行展开 L3/L4 状态和全部开火拒绝原因，避免“+N”
  // 把真正的第二个门控隐藏掉。
  std::string status = trackStateName(input.track_state);
  status += " | l4:";
  status += trackingPhaseName(input.plan.tracked_phase);
  status += input.plan.fire_permitted ? "/fire-ready" : "/track-only";
  const std::string reasons = rejectReasons(input.fire);
  if (!reasons.empty()) {
    status += " | " + reasons;
  }
  drawOutlinedText(
    image, status, {12, 28},
    input.fire.shoot ? cv::Scalar{0, 0, 255} : cv::Scalar{0, 255, 255}, 0.7);

  // 第二行直接显示 outside_hit_window 合并前的两个条件，
  // 以及实际跟随误差/本帧容差（单位 degree）。
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  const std::string details = cv::format(
    "lock=%d window=%d | err yaw=%.2f/%.2f pitch=%.2f/%.2f deg",
    input.plan.tracked_ready ? 1 : 0,
    input.plan.within_firing_window ? 1 : 0,
    input.fire.yaw_error * kRadToDeg,
    input.fire.tolerance.yaw * kRadToDeg,
    input.fire.pitch_error * kRadToDeg,
    input.fire.tolerance.pitch * kRadToDeg);
  drawOutlinedText(image, details, {12, 54}, {0, 255, 255}, 0.6);
}

std::optional<Eigen::Vector4d> plannedImpactArmorPose(
  const std::optional<L3Estimation::TrackedTarget>& target,
  const L4Planning::AimPlan& plan)
{
  if (!target || !plan.valid || plan.armor_id < 0 ||
      plan.impact_time < target->t()) {
    return std::nullopt;
  }

  L3Estimation::TrackedTarget predicted = *target;
  predicted.predict(plan.impact_time);
  const auto predicted_armors = predicted.armor_xyza_list();
  const auto selected = static_cast<std::size_t>(plan.armor_id);
  if (selected >= predicted_armors.size() ||
      !predicted_armors[selected].allFinite()) {
    return std::nullopt;
  }
  return predicted_armors[selected];
}

std::optional<Eigen::Vector4d> trackingRedArmorPose(
  const std::optional<L3Estimation::TrackedTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::AimPlan& plan)
{
  if (!target || track_state != L3Estimation::TrackState::Tracking) {
    return std::nullopt;
  }
  return plannedImpactArmorPose(target, plan);
}

cv::Point toPixel(const cv::Point2f& point)
{
  return {
    static_cast<int>(std::lround(point.x)),
    static_cast<int>(std::lround(point.y))};
}

void drawImageCenter(cv::Mat& image)
{
  if (image.empty()) {
    return;
  }
  cv::circle(
    image, {image.cols / 2, image.rows / 2}, 3, {0, 0, 255}, cv::FILLED,
    cv::LINE_AA);
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

bool isFilterInputArmor(const L3Estimation::Armor& armor)
{
  // 与 Tracker::observationUsable 保持一致，避免把被滤掉的坏解画出来。
  return armor.name != L3Estimation::ArmorName::Unknown &&
    armor.xyz_in_world.allFinite() &&
    std::isfinite(armor.ypr_in_world[0]);
}

// 将当前帧实际送入目标滤波器的单板 PnP 位姿重投影为绿框。
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
    const double pitch = L3Estimation::armorPitchOf(armor.name);
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

}  // namespace L6Telemetry
