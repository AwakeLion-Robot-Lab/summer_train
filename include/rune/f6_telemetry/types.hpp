#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "armor/l3_planning/types.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace Rune {

using TimePoint = std::chrono::steady_clock::time_point;

enum class Mode { Small, Big };
enum class Color { Red, Blue };
enum class TargetRingStatus { Unchecked, Verified, Rejected };
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

struct InactiveArmOutline {
  std::array<cv::Point2f,4> corners;
  std::array<std::optional<std::array<cv::Point2f,2>>,2> measured_edges{};
  // 左、右长边的最后一次原图观测时刻。两端为显示用的发光范围。
  std::array<TimePoint,2> side_observed_at{};
};

struct DetectedContours {
  std::vector<cv::Point> armor, arm, center_r;
  Color color{Color::Red};
  std::optional<cv::RotatedRect> target_ellipse;
  std::array<std::optional<cv::Point2f>, 8> gap_points;
  std::vector<cv::Point2f> fan_points;
  std::vector<int> fan_point_ids;
  std::vector<bool> fan_point_valid;
  // 显示预测按语义编号保存，独立于当前帧的观测有效性。
  std::array<std::optional<cv::Point2f>, 6> fan_display_predictions{};
  std::array<std::optional<cv::Point2f>, 8> gap_display_predictions{};
  std::optional<InactiveArmOutline> inactive_arm_outline;
  [[nodiscard]] int fanPointId(std::size_t i) const {
    return fan_point_ids.size() == fan_points.size() ? fan_point_ids[i] : int(i);
  }
  [[nodiscard]] bool fanPointValid(std::size_t i) const {
    return fan_point_valid.size() != fan_points.size() || fan_point_valid[i];
  }
  [[nodiscard]] std::optional<cv::Point2f> fanDisplayPrediction(std::size_t i) const {
    const auto id = fanPointId(i);
    if (id < 0 || std::size_t(id) >= fan_display_predictions.size()) return {};
    return fan_display_predictions[id];
  }
};

// All image points use full-resolution pixel coordinates. Order is top, left,
// right, bottom, R. class_id: 0 inactive, 1 activated.
struct Keypoints {
  cv::Point2f top{}, left{}, right{}, bottom{}, center_r{};
  int class_id{-1};
  float confidence{0.0F};
  // 关键点顺序为 top、left、right、bottom、R。位姿与滤波只使用有效点。
  std::array<float, 5> point_confidences{1, 1, 1, 1, 1};
  std::array<bool, 5> point_valid{true, true, true, true, true};
  // 靶圈由轮廓检测确认，图像防抖可撤销不一致的观测。
  TargetRingStatus target_ring_status{TargetRingStatus::Unchecked};
  std::shared_ptr<const DetectedContours> image_contours;
  // 像素观测方差的倍率，分别用于 R 标中心和靶圈中心。
  double center_r_variance_scale{1.0};
  double target_variance_scale{1.0};

  [[nodiscard]] std::optional<cv::Point2f> armorCenter() const {
    if (point_valid[0] && point_valid[1] && point_valid[2] && point_valid[3])
      return (top + left + right + bottom) * 0.25F;
    if (point_valid[0] && point_valid[3]) return (top + bottom) * 0.5F;
    if (point_valid[1] && point_valid[2]) return (left + right) * 0.5F;
    return std::nullopt;
  }
};

struct DisplayBladeOutline {
  cv::RotatedRect target;
  std::array<std::optional<cv::Point2f>,6> arm_points{};
  int class_id{-1};
};

struct ReobservedTargetRing {
  cv::RotatedRect ellipse;
  double arc_coverage;
  double fit_rms_px;
  bool from_shared_motion;
  bool from_history{false};
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
  std::vector<DisplayBladeOutline> display_outline_predictions;
  std::vector<ReobservedTargetRing> reobserved_target_rings;
};

struct Region {
  Keypoints keypoints;
  cv::Rect bounds;
  cv::Mat bgr;  // 只读候选区域，持有源图像存储；外部裸指针输入使用独立副本。
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
  // inner-right, outer-right. 这些是独立于五点检测锚点的物理位置。
  std::vector<cv::Point3d> pca_box_anchors_m;
};

// 与 DetectedContours 的编号对应，坐标使用以 R 标为原点的模型系，单位为米。
struct NumberedAnchors {
  std::array<std::optional<cv::Point3d>,8> gaps;
  std::array<std::optional<cv::Point3d>,6> fan;
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
  // 依次为未激活叶片、小符已激活叶片、大符已激活叶片。
  std::array<NumberedAnchors,3> numbered_anchors{};
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
  double reprojection_error_px{0.0}; // reliable-anchor RMS
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
  std::size_t numbered_anchor_count{0};
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
