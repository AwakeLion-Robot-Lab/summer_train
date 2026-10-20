#pragma once

#include "l2_perception/buff/types.hpp"

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <cstddef>
#include <vector>

namespace L3Estimation::Buff {

using TimePoint = std::chrono::steady_clock::time_point;

constexpr double kGlobalRadius = 0.7;
constexpr double kBullseyeRadius = 0.15;
constexpr double kIconProminentDistance = 0.1;

struct PagePoints {
  static Eigen::Vector3d icon() { return {-kIconProminentDistance, 0.0, 0.0}; }
  static Eigen::Vector3d top() { return {0.0, 0.0, kGlobalRadius + kBullseyeRadius}; }
  static Eigen::Vector3d left() { return {0.0, kBullseyeRadius, kGlobalRadius}; }
  static Eigen::Vector3d bottom() { return {0.0, 0.0, kGlobalRadius - kBullseyeRadius}; }
  static Eigen::Vector3d right() { return {0.0, -kBullseyeRadius, kGlobalRadius}; }
};

// 状态下标必须与源 RuneModel 一致，所有 Jacobian 和预测公式均依赖此顺序。
struct State {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  TimePoint start_timestamp{};
  // 最近一次滤波状态对应的图像时刻；与拟合预热用的 start_timestamp 分开。
  TimePoint timestamp{};
  double rotation_speed{0.0};
  double rotation_angle{0.0};
  double face_yaw{0.0};
  std::array<bool, 5> inactive{};

  bool use_prediction_speed{false};
  double prediction_cost{0.0};
  double sine_C{0.0};
  double sine_v{0.0};
  double sine_a{0.0};
  double sine_omega{0.0};
  double sine_phase{0.0};
  double sine_t{0.0};
  bool sine_valid{false};
  std::size_t update_count{0};

  void transition(double seconds) noexcept;
  [[nodiscard]] Eigen::Vector3d direction() const noexcept { return {x, y, z}; }
  [[nodiscard]] double speed() const noexcept { return rotation_speed; }
  [[nodiscard]] std::vector<Eigen::Vector3d> aimpoints() const;
};

struct Config {
  double noise_x{1e-5};
  double noise_y{1e-5};
  double noise_z{1e-5};
  double noise_rotation_angle{1e-3};
  double noise_rotation_speed{1.0};
  double noise_face_yaw{1e-5};
  double noise_observation{20.0};
  double gate_threshold{13.816};
  double init_seed_mean_error{10.0};
  double init_seed_max_error{20.0};
  double init_center_gate{30.0};
  double init_pitch_bound_deg{20.0};
  double diverge_face_angle_deg{45.0};
  double timeout_seconds{1.5};
  double inactive_timeout_seconds{0.1};
};

struct Addition {
  struct Feature {
    int feature_id{-1};
    cv::Point2f point{};
  };
  std::vector<Feature> tracked;
  std::vector<Feature> predicted;
};

}  // namespace L3Estimation::Buff
