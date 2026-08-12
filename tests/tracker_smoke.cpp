#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/filter_est/tracker.hpp"
#include "l3_estimation/gtsam_est/tracker.hpp"
#include "l3_estimation/tracker.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kSmallWidth = 0.135;
constexpr double kArmorHeight = 0.056;

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

[[nodiscard]] std::vector<cv::Point3d> armorPoints()
{
  const double half_width = kSmallWidth / 2.0;
  const double half_height = kArmorHeight / 2.0;
  return {
    {0.0, half_width, half_height},
    {0.0, -half_width, half_height},
    {0.0, -half_width, -half_height},
    {0.0, half_width, -half_height}};
}

[[nodiscard]] L2Perception::Armor makeDetection(
  const L1Sensor::CameraCalibration& calibration,
  L2Perception::ArmorClass armor_class = L2Perception::ArmorClass::Infantry3,
  cv::Vec3d translation = cv::Vec3d{0.05, -0.03, 3.0})
{
  const cv::Matx33d optical_alignment{
    0.0, -1.0, 0.0,
    0.0, 0.0, -1.0,
    1.0, 0.0, 0.0};
  const Eigen::Matrix3d rotation =
    L6Telemetry::toEigen(optical_alignment) *
    L6Telemetry::yprToRotation({0.16, -0.12, 0.08});
  cv::Vec3d rvec;
  cv::Rodrigues(L6Telemetry::toCv(rotation), rvec);

  std::vector<cv::Point2d> projected;
  cv::projectPoints(
    armorPoints(),
    rvec,
    translation,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    projected);

  L2Perception::Armor detection;
  detection.class_id = static_cast<int>(armor_class);
  detection.color = L2Perception::ArmorColor::Blue;
  detection.confidence = 0.95F;
  for (std::size_t index = 0; index < detection.corners.size(); ++index) {
    detection.corners[index] = cv::Point2f(projected[index]);
    detection.center += detection.corners[index];
  }
  detection.center = detection.center * 0.25F;
  return detection;
}

}  // namespace

int main()
{
  const auto camera_config =
    YAML::LoadFile("tests/data/camera_calibration_inline.yaml");
  const auto calibration = L1Sensor::loadCameraCalibration(
    camera_config["calibration"], "Tracker smoke config");
  const auto detection = makeDetection(calibration);
  const std::vector<L2Perception::Armor> detections{detection};

  L3Estimation::TrackerConfig tracker_config;
  tracker_config.min_detect_count = 2;
  tracker_config.max_temp_lost_count = 1;
  tracker_config.outpost_max_temp_lost_count = 2;
  L3Estimation::FilterEst::Tracker tracker(calibration, {}, tracker_config);
  expect(tracker.ready(), "Tracker rejected valid calibration and config");

  const auto t0 = std::chrono::steady_clock::now();
  const auto converted = L3Estimation::toArmorObservation(detection, t0);
  expect(
    converted.class_id == detection.class_id &&
      converted.points == detection.corners &&
      converted.center == detection.center &&
      std::abs(converted.confidence - detection.confidence) < 1e-6 &&
      converted.area > 20.0 && converted.timestamp == t0,
    "explicit L2-to-L3 conversion lost detection fields");

  const std::optional<Eigen::Quaterniond> world_barrel{
    Eigen::Quaterniond::Identity()};
  const auto first = tracker.track(detections, world_barrel, t0);
  if (!first && !tracker.observations().empty()) {
    const auto& observation = tracker.observations().front();
    std::cerr << "Tracker first-observation diagnostics: area="
              << observation.area << " rmse=" << observation.reprojection_error
              << " committed="
              << (observation.name != L3Estimation::ArmorName::Unknown) << '\n';
  }
  expect(first.has_value(), "first valid observation did not initialize target");
  expect(
    tracker.state() == L3Estimation::TrackState::Detecting && first,
    "first observation did not enter Detecting state");
  const auto first_armor_poses = tracker.targetArmorPoses();
  expect(
    first_armor_poses.size() == 4 &&
      std::all_of(
        first_armor_poses.begin(),
        first_armor_poses.end(),
        [](const Eigen::Vector4d& pose) { return pose.allFinite(); }),
    "Tracker did not expose the current finite EKF armor model");
  expect(
    tracker.observations().size() == 1 &&
      tracker.observations().front().name != L3Estimation::ArmorName::Unknown,
    "Tracker did not retain a committed PnP observation");

  const auto second = tracker.track(
    detections, world_barrel, t0 + std::chrono::milliseconds(10));
  expect(second.has_value(), "second observation lost initialized target");
  expect(
    tracker.state() == L3Estimation::TrackState::Tracking && second &&
      second->state().allFinite() && second->covariance().allFinite(),
    "second observation did not enter a finite Tracking state");

  const auto missing_pose = tracker.track(
    detections, std::nullopt, t0 + std::chrono::milliseconds(20));
  expect(
    missing_pose.has_value() &&
      tracker.state() == L3Estimation::TrackState::TempLost,
    "missing image-time barrel pose was not treated as a temporary loss");
  expect(
    tracker.observations().size() == 1 &&
      tracker.observations().front().name == L3Estimation::ArmorName::Unknown,
    "missing pose retained a previous PnP result");

  const auto recovered = tracker.track(
    detections, world_barrel, t0 + std::chrono::milliseconds(30));
  expect(
    recovered.has_value() &&
      tracker.state() == L3Estimation::TrackState::Tracking,
    "Tracker did not recover from a temporary loss");

  const auto short_loss = tracker.track(
    {}, world_barrel, t0 + std::chrono::milliseconds(60));
  expect(
    short_loss.has_value() &&
      tracker.state() == L3Estimation::TrackState::TempLost,
    "empty detection frame did not enter TempLost");

  const auto expired = tracker.track(
    {}, world_barrel, t0 + std::chrono::milliseconds(90));
  expect(
    !expired.has_value() && tracker.state() == L3Estimation::TrackState::Lost,
    "temporary-loss timeout did not clear the target");
  expect(
    tracker.targetArmorPoses().empty(),
    "Lost Tracker retained a stale EKF armor model");

  // 100 ms 的大 dt 保护只在已有跟踪目标时生效。Lost 状态可以在长间隔后
  // 正常初始化；随后 Detecting 状态遇到大 dt 必须重置而不能累计检测次数。
  const auto initialized_after_gap = tracker.track(
    detections, world_barrel, t0 + std::chrono::milliseconds(200));
  expect(
    initialized_after_gap.has_value() &&
      tracker.state() == L3Estimation::TrackState::Detecting,
    "Lost tracker could not initialize after a long idle interval");
  const auto reset_after_large_dt = tracker.track(
    detections, world_barrel, t0 + std::chrono::milliseconds(310));
  expect(
    reset_after_large_dt.has_value() &&
      tracker.state() == L3Estimation::TrackState::Detecting,
    "the 100 ms frame-interval guard did not restart Detecting");

  // Tracker 只负责中心排序和同车辆关联；多目标优先级由更上层的决策
  // 写入，不在这个入口硬编码。
  const auto engineer = makeDetection(
    calibration,
    L2Perception::ArmorClass::Engineer,
    cv::Vec3d{0.0, 0.0, 3.0});
  const auto infantry3 = makeDetection(
    calibration,
    L2Perception::ArmorClass::Infantry3,
    cv::Vec3d{0.0, -0.35, 3.0});
  const std::vector<L2Perception::Armor> mixed_detections{engineer, infantry3};

  // 已稳定跟踪工程后，同帧偶然出现 3 号步兵不得重新初始化 EKF；否则单帧误分类
  // 会让新目标随后 TempLost，零速度初值便会造成 cmd_yaw 平台。按优先级
  // 本管线没有全向感知输入，因此不存在强制切目标。
  L3Estimation::FilterEst::Tracker switch_tracker(calibration, {}, tracker_config);
  const std::vector<L2Perception::Armor> engineer_only{engineer};
  const auto engineer_first = switch_tracker.track(
    engineer_only, world_barrel, t0);
  const auto engineer_tracking = switch_tracker.track(
    engineer_only, world_barrel, t0 + std::chrono::milliseconds(10));
  expect(
    engineer_first && engineer_tracking &&
      engineer_tracking->name == L3Estimation::ArmorName::Engineer &&
      switch_tracker.state() == L3Estimation::TrackState::Tracking,
    "low-priority Engineer did not reach Tracking before switch test");

  const auto ignored_higher_priority = switch_tracker.track(
    mixed_detections, world_barrel, t0 + std::chrono::milliseconds(20));
  expect(
    ignored_higher_priority &&
      ignored_higher_priority->name == L3Estimation::ArmorName::Engineer &&
      switch_tracker.state() == L3Estimation::TrackState::Tracking,
    "Tracker switched on a transient higher-priority class");

  const std::vector<L2Perception::Armor> infantry3_only{infantry3};
  const auto unmatched_higher_priority = switch_tracker.track(
    infantry3_only, world_barrel, t0 + std::chrono::milliseconds(30));
  expect(
    unmatched_higher_priority &&
      unmatched_higher_priority->name == L3Estimation::ArmorName::Engineer &&
      switch_tracker.state() == L3Estimation::TrackState::TempLost,
    "Tracker switched class instead of entering TempLost");

  const auto engineer_recovered = switch_tracker.track(
    engineer_only, world_barrel, t0 + std::chrono::milliseconds(40));
  expect(
    engineer_recovered &&
      engineer_recovered->name == L3Estimation::ArmorName::Engineer &&
      switch_tracker.state() == L3Estimation::TrackState::Tracking,
    "Tracker did not recover after transient misclassification");

  L3Estimation::TrackerConfig invalid_tracker_config = tracker_config;
  invalid_tracker_config.min_detect_count = 0;
  L3Estimation::FilterEst::Tracker invalid_tracker(
    calibration, {}, invalid_tracker_config);
  expect(!invalid_tracker.ready(), "Tracker accepted invalid state-machine config");

#ifdef NEWVISION_USE_GTSAM
  // 同一组像素检测走完整的 GTSAM Tracker 接缝。因子图冷启动按三帧联合优化，
  // 这里验证 PnP -> ISAM2 Target -> 公共快照和四态状态机能闭环。
  L3Estimation::GtsamEst::Config graph_config;
  graph_config.first_update_batch_size = 3;
  L3Estimation::GtsamEst::Tracker graph_tracker(
    calibration, {}, {}, graph_config);
  expect(graph_tracker.ready(), "GTSAM Tracker rejected valid calibration and config");
  std::optional<L3Estimation::TrackedTarget> graph_target;
  for (int frame = 0; frame < 5; ++frame) {
    graph_target = graph_tracker.track(
      detections, world_barrel,
      t0 + std::chrono::milliseconds(400 + frame * 10));
  }
  expect(
    graph_target.has_value() && graph_target->valid() &&
      graph_tracker.state() == L3Estimation::TrackState::Tracking &&
      graph_tracker.targetArmorPoses().size() == 4,
    "GTSAM Tracker did not produce a finite Tracking snapshot");
#endif

  if (failure_count != 0) {
    std::cerr << failure_count << " Tracker smoke assertion(s) failed\n";
    return 1;
  }

  std::cout << "Tracker smoke test passed\n";
  return 0;
}
