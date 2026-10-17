#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "armor/l3_planning/types.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
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
  AlreadyFired,
  WarmingUp,
  CenterHold,
  Idle,
  ShotCooldown,
  ShootDisabled,
  InvalidFeedback,
  CommandJump,
  PlanStale
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
  // Exposure-time barrel -> gravity-aligned world orientation (armor's
  // SerialWorker::gimbalPoseAt(captured_at)). Rotation alone enables the prior.
  std::optional<Eigen::Quaterniond> q_world_barrel;
  // A full pose is REQUIRED for accumulation across moving-camera frames.
  // Translation is metres; change world_frame_id when localization resets.
  std::optional<Eigen::Isometry3d> T_world_barrel;
  std::uint64_t world_frame_id{0};
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

struct BladeModel {
  std::vector<cv::Point3d> armor_boundary_m;
  std::vector<cv::Point3d> arm_boundary_m;
  std::vector<cv::Point3d> center_r_boundary_m;
  // Optional measured PCA envelope corners: outer-left, inner-left,
  // inner-right, outer-right. Separate physical points from NN anchors.
  std::vector<cv::Point3d> pca_box_anchors_m;
};

struct Geometry {
  // Measured from the actual rune. Defaults are deliberately invalid.
  double radius_m{0.0};
  double armor_half_width_m{0.0};
  double armor_half_height_m{0.0};
  double max_reprojection_error_px{8.0};
  // Optional measured boundary samples in metres in the SAME frame as the
  // five anchors: R=(0,0,0), target=(0,radius,0), +x toward right.
  // Samples need not form a single connected polygon. Empty disables that term.
  std::vector<cv::Point3d> armor_boundary_m;
  std::vector<cv::Point3d> arm_boundary_m;
  std::vector<cv::Point3d> center_r_boundary_m;
  int refinement_iterations{15}; // 0 disables contour refinement
  int max_samples_per_contour{96};
  int distance_field_padding_px{24};
  int max_distance_field_pixels{1024 * 1024}; // per semantic contour
  double contour_weight{2.0};
  double huber_delta_px{3.0};
  double max_chamfer_error_px{5.0};
  std::vector<cv::Point3d> pca_box_anchors_m;
  std::optional<BladeModel> small_activated_model;
  std::optional<BladeModel> big_activated_model;
  bool use_pca_anchors{true};
  double max_sector_error_rad{0.30};
  double max_center_disagreement_px{15.0};
  double normal_vertical_weight_px{20.0}; // 0 disables gravity prior
  double plane_window_duration_s{0.5};
  int plane_window_max_frames{30};
  int plane_window_min_frames{3};
  double plane_window_max_gap_s{0.2};
  double plane_window_max_center_jump_m{0.3};
  double plane_window_max_normal_angle_rad{0.35};
  double plane_window_max_rms_m{0.03};
  double plane_window_max_correction_m{0.05};
};

struct BladeMatch {
  std::size_t observation_index{0};
  int sector{0}; // 0..4, reference blade is 0; successive +72 degree rotations
  bool activated{false};
  bool pca_anchors{false};
  Eigen::Vector3d target_barrel{Eigen::Vector3d::Zero()};
};

// State coordinates use the full world pose when supplied, otherwise barrel.
// x = [center x/y/z, normal yaw/elevation, unwrapped phase, angular speed].
struct FilterState {
  Eigen::Matrix<double,7,1> x{Eigen::Matrix<double,7,1>::Zero()};
  Eigen::Matrix<double,7,7> covariance{Eigen::Matrix<double,7,7>::Identity()};
  bool world_frame{false};
  int target_sector{0};
};

struct FeatureMatch {
  std::size_t observation_index{0};
  int feature_id{0}; // 0: shared R; 1..5: persistent blade identities
  double mahalanobis_squared{0};
};

struct Pose {
  Eigen::Vector3d center_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d target_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal_barrel{Eigen::Vector3d::Zero()};
  double phase_rad{0.0};
  bool activated{false};
  TimePoint captured_at{};
  double reprojection_error_px{0.0}; // five-anchor RMS
  double chamfer_error_px{0.0}; // semantic contour RMS; 0 if unavailable
  bool contour_refined{false};
  // Full model -> barrel transform; also maps any measured boundary sample.
  Eigen::Isometry3d T_barrel_model{Eigen::Isometry3d::Identity()};
  std::vector<BladeMatch> blades;
  int plane_window_frames{0};
  bool plane_smoothed{false};
  bool attitude_constrained{false};
  double normal_vertical_component{0.0};
  bool target_switched{false};
  bool history_reset{false};
  // Gravity expressed in barrel coordinates, used to give phase a fixed up.
  Eigen::Vector3d up_barrel{Eigen::Vector3d::UnitZ()};
  std::optional<FilterState> filter_state;
  std::vector<FeatureMatch> feature_matches;
  // Exposure-time orientation. Planning rotates barrel-relative points into
  // gravity-aligned absolute axes before using the shared ballistic solver.
  std::optional<Eigen::Quaterniond> q_world_barrel;
};

enum class MotionModel { Unready, Linear, Sine, External };

struct Motion {
  Pose pose;
  double angular_speed_rad_s{0.0};
  bool ready{false};
  // Optional immutable prediction snapshot: phase change at seconds after
  // capture. Required for a fitted big-rune motion model.
  std::function<std::optional<double>(double)> phase_delta_rad;
  MotionModel model{MotionModel::Unready};
  double fit_cost_rad2{0};
  double linear_cost_rad2{0};
  double sine_cost_rad2{0};
  double sine_frequency_rad_s{0};
  // Optional signed-time snapshot for the past half of a centered MPC window.
  // The ordinary phase_delta_rad API continues to reject negative horizons.
  std::function<std::optional<double>(double)> phase_delta_signed_rad;
};

struct ControlInput {
  TimePoint command_at{};
  double bullet_speed_m_s{0.0};
  double fire_delay_s{0.0};
  // Absolute gravity-aligned angles; pitch is positive upwards.
  // Unknown feedback must not look like an aligned zero-angle measurement.
  double actual_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double actual_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
  // Optional feedback timestamp; zero means the caller supplies synchronous feedback.
  TimePoint feedback_at{};
  bool shoot_requested{true};
};

struct Command {
  Status status{Status::NoDetection};
  Mode mode{Mode::Small};
  double yaw_rad{0.0};
  double pitch_rad{0.0};
  double flight_time_s{0.0};
  bool found{false};
  bool fire{false};  // one command-cycle pulse
  bool fire_feasible{false}; // independent of the master shoot switch
  bool single_shoot{true};
  bool center_hold{false};
  bool using_mpc{false};
  int target_sector{-1};
  double yaw_rate_rad_s{0.0};
  double pitch_rate_rad_s{0.0};
  double yaw_acceleration_rad_s2{0.0};
  double pitch_acceleration_rad_s2{0.0};
  std::vector<L4Planning::AimSample> samples;
};

}  // namespace Rune
