// Daedalus 仿真器（Talos 共享内存）闭环自瞄工具。
//
// 链路：TalosCamera.read → ArmorDetector.detect → TargetEstimator.update
//       → 直接瞄准（L4 Planner 当前为 stub，setpoint 在本工具内计算）
//       → TalosSerial.updateCommand 写回仿真器云台。
// 同时读取 GroundTruth 叠加到画面并输出检测精度指标。
//
// 注意：
//  - 仿真器需要先启动（cargo run），并按 F5 开启自瞄订阅后命令才会被消费；
//  - 单消费者：一次只允许运行一个读取 Talos 共享内存的进程。

#include "l1_sensor/camera/talos_camera.hpp"
#include "l1_sensor/serial/talos_serial.hpp"
#include "l1_sensor/talos/talos_config.hpp"
#include "l1_sensor/talos/talos_reader.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/config.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l3_estimation/types.hpp"
#include "l5_control/serial_command.hpp"
#include "l6_telemetry/logger.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kRadToDeg = 180.0 / std::numbers::pi;

const char* kCommandLineKeys =
  "{help h usage ? | | 显示命令行帮助}"
  "{camera-config c | config/talos_camera_config.yaml | 相机与 Talos 配置 YAML}"
  "{l3-config | config/l3_config.yaml | L3 参数 YAML}"
  "{model-path m | model/armor_model/armor.xml | OpenVINO 装甲模型}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方阵营：red 或 blue}"
  "{shoot | 0 | 是否允许开火：1 或 0}"
  "{no-gui | | 关闭 OpenCV 界面}"
  "{timeout-ms | 50 | 单帧读取超时毫秒}"
  "{shm-dir | /tmp | Talos 共享内存目录}"
  "{gt-gate-m | 0.6 | 真值匹配距离门限（米）}"
  "{prediction-ms | 100 | 绘制多少毫秒后的整车预测，0 仅关闭未来层}"
  "{show-armor-text | | 显示装甲类别、置信度、XYZ、RPY}";

L1Sensor::EnemyColor parseEnemy(const std::string& value)
{
  if (value == "blue" || value == "Blue") {
    return L1Sensor::EnemyColor::Blue;
  }
  return L1Sensor::EnemyColor::Red;
}

L2Perception::ArmorDetector makeArmorDetector(
  const std::filesystem::path& model_path, const std::string& device)
{
  try {
    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = model_path;
    model_config.device = device;
    // armor.xml 的宿主输入来自 OpenCV BGR 图像，送入模型前必须转换为 RGB。
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);
    L2Perception::ArmorDetector detector{std::move(backend)};
    if (!detector.ready()) {
      throw std::runtime_error("armor detector is not ready");
    }
    L6Telemetry::logInfo(
      "armor model loaded", model_path.string(), device);
    return detector;
  } catch (const std::exception& error) {
    L6Telemetry::logError(
      "armor model unavailable", model_path.string(), error.what());
    return {};
  }
}

bool isEnemyArmor(
  L2Perception::ArmorColor observed,
  L1Sensor::EnemyColor expected) noexcept
{
  switch (expected) {
  case L1Sensor::EnemyColor::Red:
    return observed == L2Perception::ArmorColor::Red;
  case L1Sensor::EnemyColor::Blue:
    return observed == L2Perception::ArmorColor::Blue;
  case L1Sensor::EnemyColor::Unknown:
    return false;
  }
  return false;
}

void drawText(
  cv::Mat& image,
  const std::string& text,
  cv::Point origin,
  const cv::Scalar& color,
  double scale = 0.48)
{
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
    scale, {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
    scale, color, 1, cv::LINE_AA);
}

std::string armorClassName(int class_id)
{
  static const std::vector<std::string> names{
    "guard", "hero", "engineer", "infantry3", "infantry4",
    "infantry5", "outpost", "base-small", "base-large"};
  return class_id >= 0 && static_cast<std::size_t>(class_id) < names.size()
           ? names[static_cast<std::size_t>(class_id)]
           : "unknown";
}

std::string trackerStateName(L3Estimation::TrackerState state)
{
  switch (state) {
  case L3Estimation::TrackerState::Lost:
    return "lost";
  case L3Estimation::TrackerState::Detecting:
    return "detecting";
  case L3Estimation::TrackerState::Tracking:
    return "tracking";
  case L3Estimation::TrackerState::TemporaryLost:
    return "temporary-lost";
  }
  return "unknown";
}

// 绝对世界系 → 相机系刚体变换（用于 GroundTruth 投影）：
// [R_world_barrelᵀ | -R_world_barrelᵀ * p_gimbal]。
Eigen::Isometry3d cameraFromWorld(
  const Eigen::Quaterniond& R_world_barrel,
  const Eigen::Vector3d& p_gimbal)
{
  const Eigen::Matrix3d rotation =
    R_world_barrel.normalized().toRotationMatrix().transpose();
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = rotation;
  transform.translation() = -rotation * p_gimbal;
  return transform;
}

// 纯旋转的世界系 → 相机系变换（L3 世界系 = 云台系，相机为原点，
// 与 l3_video_replay 的 T_barrel_world.translation() = 0 语义一致）。
Eigen::Isometry3d rotationOnlyCameraFromWorld(
  const Eigen::Quaterniond& R_world_barrel)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() =
    R_world_barrel.normalized().toRotationMatrix().transpose();
  return transform;
}

std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& point_world,
  const Eigen::Isometry3d& T_camera_world,
  const L1Sensor::CameraCalibration& calibration)
{
  const Eigen::Vector3d point_camera = T_camera_world * point_world;
  if (!point_camera.allFinite() || point_camera.z() <= 0.01) {
    return std::nullopt;
  }
  std::vector<cv::Point3d> points{{
    point_camera.x(), point_camera.y(), point_camera.z()}};
  std::vector<cv::Point2d> pixels;
  cv::projectPoints(
    points,
    cv::Vec3d{0.0, 0.0, 0.0},
    cv::Vec3d{0.0, 0.0, 0.0},
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    pixels);
  if (pixels.size() != 1
      || !std::isfinite(pixels[0].x)
      || !std::isfinite(pixels[0].y)
      || std::abs(pixels[0].x) > 1e6
      || std::abs(pixels[0].y) > 1e6) {
    return std::nullopt;
  }
  return cv::Point2f{
    static_cast<float>(pixels[0].x),
    static_cast<float>(pixels[0].y)};
}

void drawGroundTruth(
  cv::Mat& image,
  const L1Sensor::talos::GroundTruthBatch& ground_truth,
  const Eigen::Isometry3d& T_camera_world,
  const L1Sensor::CameraCalibration& calibration,
  L1Sensor::EnemyColor enemy)
{
  const std::uint8_t team = enemy == L1Sensor::EnemyColor::Red ? 0 : 1;
  for (std::uint32_t i = 0; i < ground_truth.target_count; ++i) {
    const auto& target = ground_truth.targets[i];
    if (target.team != team) {
      continue;
    }
    const Eigen::Vector3d point_world(
      target.position[0], target.position[1], target.position[2]);
    const auto pixel =
      projectWorldPoint(point_world, T_camera_world, calibration);
    if (!pixel) {
      continue;
    }
    cv::drawMarker(
      image, *pixel, cv::Scalar(255, 255, 0), cv::MARKER_CROSS, 20, 2);
    const std::string label =
      "GT" + std::to_string(static_cast<int>(target.armor_label))
      + (target.team == 0 ? "R" : "B");
    drawText(
      image, label, cv::Point(pixel->x + 10, pixel->y - 10),
      {255, 255, 0}, 0.45);
  }
}

cv::Scalar detectionColor(L2Perception::ArmorColor color)
{
  if (color == L2Perception::ArmorColor::Red) {
    return {0, 0, 255};
  }
  if (color == L2Perception::ArmorColor::Blue) {
    return {255, 0, 0};
  }
  return {0, 255, 255};
}

// 瞄准换算集中在 computeAimSetpoint，实测若方向/符号不对只改这里。
struct AimSetpoint {
  double yaw{0.0};      // rad，相对底盘
  double pitch{0.0};    // rad，相对水平面（正为向上）
  double distance{0.0}; // m
  bool valid{false};
};

AimSetpoint computeAimSetpoint(
  const L3Estimation::TargetState& target,
  const L1Sensor::talos::ChassisObservation& chassis)
{
  // L3 世界系 = 云台系（相机为原点）：target.center 就是相机到目标的位移。
  const Eigen::Vector3d p_rel = target.center;
  const double xy = std::hypot(p_rel.x(), p_rel.y());
  AimSetpoint out;
  if (!p_rel.allFinite() || xy < 1e-3) {
    return out;
  }
  // 仿真器世界系 Z-up、yaw=0 指向 +X（与 ros_yaw 约定一致）；
  // local_yaw 是相对底盘的绝对角，因此减去底盘世界 yaw。
  const double bearing = std::atan2(p_rel.y(), p_rel.x());
  const double elevation = std::atan2(p_rel.z(), xy);
  out.yaw = bearing - static_cast<double>(chassis.rpy_rad[2]);
  out.pitch = elevation;
  out.distance = p_rel.norm();
  out.valid = true;
  return out;
}

std::optional<L3Estimation::TargetState> selectTarget(
  const std::vector<L3Estimation::TargetState>& targets)
{
  const L3Estimation::TargetState* best = nullptr;
  // L3 世界系 = 云台系：center 的模长就是相机到目标的距离。
  const auto closer = [](
                        const L3Estimation::TargetState* current,
                        const L3Estimation::TargetState& candidate) {
    return current == nullptr
           || candidate.center.norm() < current->center.norm();
  };
  for (const auto& target : targets) {
    if (target.tracker_state == L3Estimation::TrackerState::Tracking
        && target.updated_this_frame && closer(best, target)) {
      best = &target;
    }
  }
  if (best == nullptr) {
    for (const auto& target : targets) {
      if (target.tracker_state == L3Estimation::TrackerState::Detecting
          && target.updated_this_frame && closer(best, target)) {
        best = &target;
      }
    }
  }
  return best
           ? std::optional<L3Estimation::TargetState>(*best)
           : std::nullopt;
}

struct GtMetrics {
  std::uint64_t frames{0};
  std::uint64_t observations{0};
  std::uint64_t matched{0};
  std::uint64_t gt_total{0};
  double distance_error_sum{0.0};
};

void accumulateMetrics(
  GtMetrics& metrics,
  const std::vector<L3Estimation::ArmorObservation>& observations,
  const L1Sensor::talos::GroundTruthBatch& ground_truth,
  L1Sensor::EnemyColor enemy,
  double gate_m)
{
  const std::uint8_t team = enemy == L1Sensor::EnemyColor::Red ? 0 : 1;
  std::vector<Eigen::Vector3d> gt_positions;
  gt_positions.reserve(ground_truth.target_count);
  for (std::uint32_t i = 0; i < ground_truth.target_count; ++i) {
    const auto& target = ground_truth.targets[i];
    if (target.team == team) {
      gt_positions.emplace_back(
        target.position[0], target.position[1], target.position[2]);
    }
  }
  metrics.gt_total += gt_positions.size();
  ++metrics.frames;

  for (const auto& observation : observations) {
    if (!observation.position_world.allFinite()) {
      continue;
    }
    ++metrics.observations;
    double best_distance = gate_m;
    bool matched = false;
    for (const auto& gt : gt_positions) {
      const double distance = (observation.position_world - gt).norm();
      if (distance < best_distance) {
        best_distance = distance;
        matched = true;
      }
    }
    if (matched) {
      ++metrics.matched;
      metrics.distance_error_sum += best_distance;
    }
  }
}

void reportMetrics(const GtMetrics& metrics)
{
  const double precision =
    metrics.observations > 0
      ? static_cast<double>(metrics.matched) / metrics.observations
      : 0.0;
  const double recall =
    metrics.gt_total > 0
      ? static_cast<double>(metrics.matched) / metrics.gt_total
      : 0.0;
  const double mean_error =
    metrics.matched > 0
      ? metrics.distance_error_sum / metrics.matched
      : 0.0;
  std::cout << std::fixed << std::setprecision(3)
            << "[talos] frames=" << metrics.frames
            << " gt=" << metrics.gt_total
            << " obs=" << metrics.observations
            << " precision=" << precision
            << " recall=" << recall
            << " mean_dist_err_m=" << mean_error << '\n';
  L6Telemetry::logInfo(
    "talos GT metrics",
    "frames", metrics.frames,
    "gt", metrics.gt_total,
    "obs", metrics.observations,
    "precision", precision,
    "recall", recall,
    "mean_dist_err_m", mean_error);
}

// 整车 EKF 模型叠加：中心、各装甲面轮廓与 P0.. 编号、中心连线。
void drawVehicleModel(
  cv::Mat& image,
  const L3Estimation::TargetState& target,
  const L3Estimation::L3Config& config,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Isometry3d& T_camera_world,
  double prediction_seconds,
  const cv::Scalar& color,
  int thickness,
  bool show_armor_text)
{
  const Eigen::Vector3d center =
    target.center + target.velocity * prediction_seconds;
  const double yaw = target.yaw + target.yaw_rate * prediction_seconds;
  if (!center.allFinite() || !std::isfinite(yaw)) {
    return;
  }

  const auto traits = L3Estimation::targetModelTraits(target.model);
  const auto& model_parameters = config.armor.parameters(target.model);
  const double armor_width =
    target.robot_id == static_cast<int>(L2Perception::ArmorClass::Hero)
      ? config.armor.dimensions.large_width
      : config.armor.dimensions.small_width;
  const double half_width = armor_width / 2.0;
  const double half_height = config.armor.dimensions.height / 2.0;
  const std::array<Eigen::Vector3d, 4> local_corners{{
    {0.0, half_width, half_height},
    {0.0, -half_width, half_height},
    {0.0, -half_width, -half_height},
    {0.0, half_width, -half_height}}};

  const auto center_pixel =
    projectWorldPoint(center, T_camera_world, calibration);
  std::vector<std::optional<cv::Point2f>> armor_centers;
  armor_centers.reserve(static_cast<std::size_t>(traits.armor_count));

  for (int face_id = 0; face_id < traits.armor_count; ++face_id) {
    const double angle = yaw
      + static_cast<double>(face_id) * traits.face_angle_interval_rad;
    const bool second_group =
      traits.uses_alternating_radius_and_height && face_id % 2 != 0;
    const double radius = target.radius
      + (second_group ? target.radius_offset : 0.0);
    const Eigen::Vector3d armor_center{
      center.x() - radius * std::cos(angle),
      center.y() - radius * std::sin(angle),
      center.z() + (second_group ? target.height_offset : 0.0)};
    armor_centers.push_back(
      projectWorldPoint(armor_center, T_camera_world, calibration));

    const Eigen::Matrix3d R_world_armor =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix()
      * Eigen::AngleAxisd(
          model_parameters.pitch_rad,
          Eigen::Vector3d::UnitY()).toRotationMatrix();
    std::array<std::optional<cv::Point2f>, 4> corners;
    bool all_visible = true;
    for (std::size_t corner = 0; corner < corners.size(); ++corner) {
      corners[corner] = projectWorldPoint(
        armor_center + R_world_armor * local_corners[corner],
        T_camera_world,
        calibration);
      all_visible = all_visible && corners[corner].has_value();
    }
    if (all_visible) {
      for (std::size_t corner = 0; corner < corners.size(); ++corner) {
        cv::line(
          image,
          *corners[corner],
          *corners[(corner + 1) % corners.size()],
          color,
          thickness,
          cv::LINE_AA);
      }
    }
    if (show_armor_text && armor_centers.back()) {
      drawText(
        image,
        "P" + std::to_string(face_id),
        cv::Point{
          cvRound(armor_centers.back()->x) + 4,
          cvRound(armor_centers.back()->y) - 4},
        color,
        0.42);
    }
  }

  for (int face_id = 0; face_id < traits.armor_count; ++face_id) {
    const auto& begin = armor_centers[static_cast<std::size_t>(face_id)];
    const auto& end = armor_centers[
      static_cast<std::size_t>((face_id + 1) % traits.armor_count)];
    if (begin && end) {
      cv::line(image, *begin, *end, color, thickness, cv::LINE_AA);
    }
    if (center_pixel && begin) {
      cv::line(image, *center_pixel, *begin, color, 1, cv::LINE_AA);
    }
  }

  if (center_pixel) {
    cv::circle(image, *center_pixel, 4, color, cv::FILLED, cv::LINE_AA);
    std::ostringstream label;
    label << (prediction_seconds > 0.0 ? "+" : "EKF #")
          << (prediction_seconds > 0.0
                ? cvRound(prediction_seconds * 1000.0)
                : target.robot_id)
          << (prediction_seconds > 0.0 ? "ms" : "")
          << (target.updated_this_frame ? " U" : " P");
    drawText(
      image,
      label.str(),
      cv::Point{
        cvRound(center_pixel->x) + 7,
        cvRound(center_pixel->y) - 7},
      color,
      0.45);
  }
}

// 单帧完整叠加：整车模型、检测框与文字、状态行、目标列表、GT 指标。
void drawFrame(
  cv::Mat& image,
  std::uint64_t frame_seq,
  double l2_ms,
  double l3_ms,
  const std::vector<L2Perception::Armor>& detections,
  const std::vector<L3Estimation::ArmorObservation>& observations,
  const std::vector<L3Estimation::TargetState>& targets,
  const std::optional<int>& selected_robot,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::L3Config& l3_parameters,
  const Eigen::Isometry3d& T_camera_world,
  double prediction_seconds,
  const AimSetpoint& aim,
  bool fire,
  const GtMetrics& metrics,
  bool show_armor_text,
  bool paused)
{
  for (const auto& target : targets) {
    if (prediction_seconds > 0.0) {
      drawVehicleModel(
        image, target, l3_parameters, calibration, T_camera_world,
        prediction_seconds, {255, 0, 255}, 1, false);
    }
    drawVehicleModel(
      image, target, l3_parameters, calibration, T_camera_world,
      0.0,
      target.updated_this_frame
        ? cv::Scalar{0, 255, 0}
        : cv::Scalar{0, 165, 255},
      2,
      show_armor_text);
  }

  for (std::size_t index = 0; index < detections.size(); ++index) {
    const auto& detection = detections[index];
    const cv::Scalar color = detectionColor(detection.color);
    for (std::size_t corner = 0; corner < detection.corners.size(); ++corner) {
      cv::line(
        image,
        detection.corners[corner],
        detection.corners[(corner + 1) % detection.corners.size()],
        color, 2, cv::LINE_AA);
    }

    if (show_armor_text) {
      const cv::Point label_origin{
        cvRound(detection.corners[0].x),
        std::max(20, cvRound(detection.corners[0].y) - 8)};
      std::ostringstream label;
      label << '#' << index << ' ' << armorClassName(detection.class_id)
            << " conf=" << std::fixed << std::setprecision(2)
            << detection.confidence;
      drawText(image, label.str(), label_origin, color);

      const auto observation = std::find_if(
        observations.begin(), observations.end(),
        [index](const auto& value) {
          return value.source_detection_index == index;
        });
      if (observation != observations.end()) {
        std::ostringstream position;
        position << "xyz=" << std::fixed << std::setprecision(2)
                 << observation->position_world.transpose();
        drawText(
          image, position.str(), label_origin + cv::Point{0, 20}, {0, 255, 0});

        const auto rpyText = [](const char* name, const Eigen::Vector3d& rpy) {
          std::ostringstream text;
          text << name << " rpy=" << std::fixed << std::setprecision(1)
               << (rpy * kRadToDeg).transpose() << "deg";
          return text.str();
        };
        drawText(
          image,
          rpyText("raw", observation->rpy_raw_world),
          label_origin + cv::Point{0, 40}, {0, 255, 0});
        drawText(
          image,
          rpyText("opt", observation->rpy_constrained_world),
          label_origin + cv::Point{0, 60}, {0, 255, 0});
      }
    }
  }

  std::ostringstream status;
  status << "seq=" << frame_seq
         << " L2=" << std::fixed << std::setprecision(1) << l2_ms << "ms"
         << " L3=" << l3_ms << "ms"
         << " det=" << detections.size()
         << " obs=" << observations.size()
         << " target=" << targets.size()
         << " aim=" << std::setprecision(1)
         << aim.yaw * kRadToDeg << '/'
         << aim.pitch * kRadToDeg << "deg"
         << " dist=" << std::setprecision(1) << aim.distance
         << " fire=" << (fire ? 1 : 0);
  if (selected_robot) {
    status << " sel=" << *selected_robot;
  }
  if (paused) {
    status << " [PAUSED]";
  }
  drawText(image, status.str(), {12, 28}, {255, 255, 255}, 0.6);

  int text_y = 55;
  for (const auto& target : targets) {
    const double second_radius = target.radius
      + (target.model == L3Estimation::TargetModel::FourArmorVehicle
           ? target.radius_offset
           : 0.0);
    std::ostringstream line;
    line << "robot=" << target.robot_id << ' '
         << trackerStateName(target.tracker_state)
         << " c=" << std::fixed << std::setprecision(2)
         << target.center.transpose()
         << " v=" << target.velocity.transpose()
         << " yaw=" << std::setprecision(1)
         << target.yaw * kRadToDeg << "deg"
         << " w=" << std::setprecision(2) << target.yaw_rate
         << " r=" << target.radius << '/' << second_radius;
    if (target.model == L3Estimation::TargetModel::FourArmorVehicle) {
      line << " angle=" << std::setprecision(1)
           << L3Estimation::fourArmorMinimumCornerAngle(
                target.radius, second_radius) * kRadToDeg
           << "deg";
    }
    drawText(
      image, line.str(), {12, text_y},
      selected_robot && target.robot_id == *selected_robot
        ? cv::Scalar{0, 255, 0}
        : cv::Scalar{220, 220, 220});
    text_y += 22;
  }

  const double precision =
    metrics.observations > 0
      ? static_cast<double>(metrics.matched) / metrics.observations
      : 0.0;
  const double recall =
    metrics.gt_total > 0
      ? static_cast<double>(metrics.matched) / metrics.gt_total
      : 0.0;
  std::ostringstream gt_line;
  gt_line << "GT frames=" << metrics.frames
          << " gt=" << metrics.gt_total
          << " obs=" << metrics.observations
          << " precision=" << std::fixed << std::setprecision(2)
          << precision
          << " recall=" << recall;
  drawText(image, gt_line.str(), {12, image.rows - 60}, {255, 255, 0});
  drawText(
    image,
    "Vehicle: green=updated orange=predict-only magenta=future",
    {12, image.rows - 38}, {255, 255, 255});
  drawText(
    image,
    "Space: pause/resume  N/Right: step  L: armor text  Q/Esc: quit",
    {12, image.rows - 16}, {255, 255, 255});
}

int runTalosAutoAim(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, kCommandLineKeys);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  const std::string camera_config = parser.get<std::string>("camera-config");
  const std::string l3_config_path = parser.get<std::string>("l3-config");
  const std::string model_path = parser.get<std::string>("model-path");
  const std::string device = parser.get<std::string>("device");
  const std::string enemy = parser.get<std::string>("enemy");
  const bool shoot = parser.get<int>("shoot") != 0;
  const bool gui_enabled = !parser.has("no-gui");
  const int timeout_ms = parser.get<int>("timeout-ms");
  const std::string shm_dir = parser.get<std::string>("shm-dir");
  const double gt_gate_m = parser.get<double>("gt-gate-m");
  const int prediction_ms = parser.get<int>("prediction-ms");
  if (!parser.check() || timeout_ms <= 0 || gt_gate_m <= 0.0
      || prediction_ms < 0) {
    parser.printErrors();
    return 2;
  }

  L1Sensor::talos::TalosSimConfig sim_config =
    L1Sensor::talos::loadTalosSimConfig(camera_config);
  sim_config.enemy_color = parseEnemy(enemy);
  sim_config.shoot_enable = shoot;
  sim_config.timeout = std::chrono::milliseconds{timeout_ms};
  sim_config.shm_dir = shm_dir;
  sim_config.gt_match_gate_m = gt_gate_m;

  auto reader =
    std::make_shared<L1Sensor::talos::TalosReader>(sim_config.shm_dir);
  if (!reader->open()) {
    std::cerr << "talos shared memory not found in " << shm_dir
              << "; is the Daedalus simulator running?\n";
    return 2;
  }
  L1Sensor::TalosCamera camera{reader, sim_config};
  L1Sensor::TalosSerial serial{reader, sim_config};

  const L1Sensor::CameraCalibration& calibration = *camera.calibration();
  const L3Estimation::L3Config l3_config =
    L3Estimation::loadL3Config(l3_config_path);
  L2Perception::ArmorDetector detector =
    makeArmorDetector(model_path, device);
  L3Estimation::TargetEstimator estimator{
    calibration,
    [&serial](L3Estimation::TimePoint timestamp) {
      return serial.gimbalPoseAt(timestamp);
    },
    l3_config};

  const auto camera_info = reader->cameraInfo();
  std::cout << "talos auto aim started: "
            << camera_info.width << "x" << camera_info.height
            << " fx=" << camera_info.fx
            << " fy=" << camera_info.fy
            << " cx=" << camera_info.cx
            << " cy=" << camera_info.cy
            << " enemy=" << enemy
            << " shoot=" << (sim_config.shoot_enable ? 1 : 0) << '\n';
  L6Telemetry::logInfo(
    "talos auto aim started",
    "enemy", enemy,
    "shoot", sim_config.shoot_enable,
    "detector_ready", detector.ready());

  if (gui_enabled) {
    cv::namedWindow("talos auto aim", cv::WINDOW_NORMAL);
    cv::resizeWindow("talos auto aim", 960, 720);
  }

  GtMetrics metrics;
  auto last_report = Clock::now();
  bool running = true;
  bool enemy_warned = false;
  bool paused = false;
  bool step_once = false;
  bool show_armor_text = parser.has("show-armor-text");
  const double prediction_seconds =
    static_cast<double>(prediction_ms) / 1000.0;
  std::uint64_t frame_seq = 0;
  while (running) {
    if (gui_enabled && paused) {
      const int key = cv::waitKeyEx(30);
      if (key == 'q' || key == 'Q' || key == 27) {
        running = false;
      } else if (key == 'l' || key == 'L') {
        show_armor_text = !show_armor_text;
      } else if (key == ' ' || key == 'n' || key == 'N' || key == 65363) {
        paused = false;
        step_once = (key == 'n' || key == 'N' || key == 65363);
      }
      if (running && paused) {
        continue;
      }
    }

    cv::Mat frame;
    Clock::time_point timestamp;
    if (!camera.read(frame, timestamp, sim_config.timeout)) {
      if (gui_enabled) {
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
          running = false;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
      }
      continue;
    }
    ++frame_seq;

    const auto l2_begin = Clock::now();
    std::vector<L2Perception::Armor> detections =
      detector.ready() ? detector.detect(frame)
                       : std::vector<L2Perception::Armor>{};
    const auto l2_end = Clock::now();
    std::erase_if(detections, [&sim_config](const auto& armor) {
      return !isEnemyArmor(armor.color, sim_config.enemy_color);
    });

    const auto l3_begin = Clock::now();
    const std::vector<L3Estimation::TargetState> targets =
      estimator.update(
        detections, L3Estimation::FrameContext{timestamp, frame.size()});
    const auto l3_end = Clock::now();
    const std::vector<L3Estimation::ArmorObservation>& observations =
      estimator.lastObservations();
    const double l2_ms =
      std::chrono::duration<double, std::milli>(l2_end - l2_begin).count();
    const double l3_ms =
      std::chrono::duration<double, std::milli>(l3_end - l3_begin).count();

    const auto odom = reader->pose(L1Sensor::talos::PoseIndex::Odom);
    const auto pose_gimbal = serial.gimbalPoseAt(timestamp);
    Eigen::Vector3d p_gimbal = Eigen::Vector3d::Zero();
    if (odom) {
      p_gimbal = Eigen::Vector3d(
        odom->position[0], odom->position[1], odom->position[2]);
    }
    const auto selected = selectTarget(targets);

    const AimSetpoint aim =
      selected
        ? computeAimSetpoint(
            *selected, reader->chassisObservation())
        : AimSetpoint{};
    const bool tracking =
      selected
      && selected->tracker_state == L3Estimation::TrackerState::Tracking
      && selected->updated_this_frame;
    const bool in_range = aim.valid && aim.distance > 0.3 && aim.distance < 25.0;
    const bool fire = sim_config.shoot_enable && tracking && in_range;
    serial.updateCommand(
      L5Control::SerialCommand{aim.yaw, aim.pitch, fire}, aim.distance);

    const L1Sensor::talos::GroundTruthBatch ground_truth =
      reader->groundTruth();
    if (!enemy_warned) {
      const std::uint8_t team =
        sim_config.enemy_color == L1Sensor::EnemyColor::Red ? 0 : 1;
      bool found = false;
      for (std::uint32_t i = 0; i < ground_truth.target_count; ++i) {
        if (ground_truth.targets[i].team == team) {
          found = true;
          break;
        }
      }
      if (!found) {
        std::cout << "[talos] warning: no ground truth target of team "
                  << enemy << " in scene; check --enemy\n";
      }
      enemy_warned = true;
    }
    accumulateMetrics(
      metrics, observations, ground_truth, sim_config.enemy_color,
      sim_config.gt_match_gate_m);

    if (gui_enabled) {
      // L3 世界系 = 云台系：整车叠加用纯旋转变换（与 replay 一致）；
      // GroundTruth 是绝对世界坐标，投影需带相机位置平移。
      const Eigen::Quaterniond R_barrel =
        pose_gimbal.value_or(Eigen::Quaterniond::Identity());
      const Eigen::Isometry3d T_camera_world_model =
        calibration.T_barrel_camera->inverse()
        * rotationOnlyCameraFromWorld(R_barrel);
      const Eigen::Isometry3d T_camera_world_gt =
        calibration.T_barrel_camera->inverse()
        * cameraFromWorld(R_barrel, p_gimbal);
      cv::Mat drawing = frame.clone();
      drawFrame(
        drawing, frame_seq, l2_ms, l3_ms,
        detections, observations, targets,
        selected ? std::optional<int>{selected->robot_id} : std::nullopt,
        calibration, l3_config, T_camera_world_model,
        prediction_seconds, aim, fire, metrics,
        show_armor_text, paused);
      drawGroundTruth(
        drawing, ground_truth, T_camera_world_gt,
        calibration, sim_config.enemy_color);
      cv::imshow("talos auto aim", drawing);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        running = false;
      } else if (key == 'l' || key == 'L') {
        show_armor_text = !show_armor_text;
      } else if (key == ' ') {
        paused = true;
      }
    }

    if (step_once) {
      paused = true;
      step_once = false;
    }

    if (Clock::now() - last_report
        >= std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{sim_config.report_interval_s})) {
      reportMetrics(metrics);
      metrics = {};
      last_report = Clock::now();
    }
  }

  if (gui_enabled) {
    cv::destroyWindow("talos auto aim");
  }
  serial.stop();
  camera.stop();
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  L6Telemetry::initLogger();
  try {
    const int result = runTalosAutoAim(argc, argv);
    L6Telemetry::flushLogger();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "talos auto aim failed: " << error.what() << '\n';
    L6Telemetry::logError("talos auto aim failed", error.what());
    L6Telemetry::flushLogger();
    return 1;
  }
}
