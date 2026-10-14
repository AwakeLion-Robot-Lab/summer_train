#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <chrono>
#include <functional>
#include <optional>
#include <vector>

namespace Rune {

using TimePoint = std::chrono::steady_clock::time_point;

enum class Mode { Small, Big };
enum class Color { Red, Blue };
enum class Status {
  Ready,
  NoDetection,
  InvalidInput,
  NoContour,
  PoseFailed,
  MotionNotReady,
  BallisticFailed,
  AimNotReady,
  AlreadyFired
};

// All image points use full-resolution pixel coordinates. Order is top, left,
// right, bottom, R. class_id: 0 inactive, 1 activated.
struct Keypoints {
  cv::Point2f top{}, left{}, right{}, bottom{}, center_r{};
  int class_id{-1};
  float confidence{0.0F};
};

struct Frame {
  cv::Mat bgr;
  TimePoint captured_at{};
  Mode mode{Mode::Small};
  Color color{Color::Red};
  std::vector<Keypoints> detections;
  cv::Mat camera_matrix;
  cv::Mat distortion;
  // Camera optical frame -> barrel frame. Barrel axes: x forward, y left, z up.
  std::optional<Eigen::Isometry3d> T_barrel_camera;
};

struct Region {
  Keypoints keypoints;
  cv::Rect bounds;
  cv::Mat bgr;  // owned ROI; source image can be released after extraction
};

struct Observation {
  Keypoints keypoints;
  std::vector<cv::Point> armor;
  std::vector<cv::Point> arm;
  std::vector<cv::Point> center_r;
};

struct Geometry {
  // Measured from the actual rune. Defaults are deliberately invalid.
  double radius_m{0.0};
  double armor_half_width_m{0.0};
  double armor_half_height_m{0.0};
  double max_reprojection_error_px{8.0};
};

struct Pose {
  Eigen::Vector3d center_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d target_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal_barrel{Eigen::Vector3d::Zero()};
  double phase_rad{0.0};
  bool activated{false};
  TimePoint captured_at{};
};

struct Motion {
  Pose pose;
  double angular_speed_rad_s{0.0};
  bool ready{false};
  // Optional immutable prediction snapshot: phase change at seconds after
  // capture. Required for a fitted big-rune motion model.
  std::function<std::optional<double>(double)> phase_delta_rad;
};

struct ControlInput {
  TimePoint command_at{};
  double bullet_speed_m_s{0.0};
  double fire_delay_s{0.0};
  double actual_yaw_rad{0.0};
  double actual_pitch_rad{0.0};
};

struct Command {
  Status status{Status::NoDetection};
  Mode mode{Mode::Small};
  double yaw_rad{0.0};
  double pitch_rad{0.0};
  double flight_time_s{0.0};
  bool found{false};
  bool fire{false};  // one command-cycle pulse
};

}  // namespace Rune
