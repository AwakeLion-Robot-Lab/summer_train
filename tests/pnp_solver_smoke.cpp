#include "l3_estimation/pnp_solver.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {1280, 1024};
  calibration.camera_matrix = (cv::Mat_<double>(3, 3)
    << 1200.0, 0.0, 640.0,
       0.0, 1200.0, 512.0,
       0.0, 0.0, 1.0);
  calibration.distortion_coefficients =
    cv::Mat::zeros(1, 5, CV_64FC1);
  return calibration;
}

L2Perception::ArmorDetection makeDetection(
  const L1Sensor::CameraCalibration& calibration,
  const cv::Vec3d& rvec,
  const cv::Vec3d& tvec)
{
  const std::array<cv::Point3f, 4> object_points{{
    {0.0F,  0.0675F,  0.0275F},
    {0.0F, -0.0675F,  0.0275F},
    {0.0F, -0.0675F, -0.0275F},
    {0.0F,  0.0675F, -0.0275F}}};
  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    object_points,
    rvec,
    tvec,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    projected);

  L2Perception::ArmorDetection detection;
  std::copy(
    projected.begin(),
    projected.end(),
    detection.corners.begin());
  detection.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  detection.confidence = 1.0F;
  return detection;
}

L2Perception::ArmorDetection makeDetectionFromPose(
  const L1Sensor::CameraCalibration& calibration,
  double yaw_world_rad,
  double pitch_rad,
  const cv::Vec3d& tvec)
{
  // 与 l3_baseline_smoke 一致：R_world_armor = Rz(yaw) * Ry(pitch)，
  // T_barrel_camera 为 camera→barrel 的固定旋转。
  const cv::Matx33d rz{
    std::cos(yaw_world_rad), -std::sin(yaw_world_rad), 0.0,
    std::sin(yaw_world_rad),  std::cos(yaw_world_rad), 0.0,
    0.0, 0.0, 1.0};
  const cv::Matx33d ry{
    std::cos(pitch_rad), 0.0, std::sin(pitch_rad),
    0.0, 1.0, 0.0,
    -std::sin(pitch_rad), 0.0, std::cos(pitch_rad)};
  const cv::Matx33d T_barrel_camera{
    0.0, 0.0, 1.0,
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0};
  const cv::Matx33d R_camera_armor =
    T_barrel_camera.t() * (rz * ry);
  cv::Vec3d rvec;
  cv::Rodrigues(R_camera_armor, rvec);
  return makeDetection(calibration, rvec, tvec);
}

}  // namespace

int main()
{
  try {
    const auto calibration = makeCalibration();
    L3Estimation::PnpSolver solver{calibration};
    const cv::Matx33d rotation_armor_to_camera{
       0.0, -1.0,  0.0,
       0.0,  0.0, -1.0,
       1.0,  0.0,  0.0};
    cv::Vec3d rvec;
    cv::Rodrigues(rotation_armor_to_camera, rvec);
    const cv::Vec3d expected_tvec{0.08, -0.03, 3.2};
    const auto detection = makeDetection(
      calibration,
      rvec,
      expected_tvec);

    // 近正视目标返回两个都通过几何检查的 IPPE 候选。
    const auto poses = solver.solve(
      detection,
      L3Estimation::ArmorSize::Small);
    require(
      poses.size() == 2,
      "near-frontal IPPE did not return two candidates");
    require(
      poses[0].ippe_candidate_index == 0
        && poses[1].ippe_candidate_index == 1,
      "IPPE candidate indices are not 0 and 1");
    require(
      cv::norm(poses[0].tvec - expected_tvec) < 1e-3,
      "first IPPE candidate translation is inaccurate");
    require(
      cv::norm(poses[1].tvec - expected_tvec) > 1e-3
        || cv::norm(poses[0].rvec - poses[1].rvec) > 1e-6,
      "IPPE candidates are not distinct");
    require(
      std::isfinite(poses[0].reprojection_error_px)
        && poses[0].reprojection_error_px < 1e-3,
      "first IPPE candidate reprojection RMSE is invalid");
    require(
      std::isfinite(poses[1].reprojection_error_px),
      "second IPPE candidate reprojection RMSE is invalid");

    // 收紧 RMSE 门限时，镜像候选被拒绝，只剩一个有效候选。
    L3Estimation::PnpSolverConfig tight_config;
    tight_config.maximum_reprojection_error_px = 0.1;
    L3Estimation::PnpSolver tight_solver{
      calibration,
      {},
      tight_config};
    constexpr double kDegree = 3.141592653589793 / 180.0;
    const auto oblique = makeDetectionFromPose(
      calibration,
      15.0 * kDegree,
      15.0 * kDegree,
      {0.05, -0.02, 3.0});
    const auto tight_poses = tight_solver.solve(
      oblique,
      L3Estimation::ArmorSize::Small);
    require(
      tight_poses.size() == 1
        && tight_poses.front().ippe_candidate_index == 0,
      "tight RMSE gate did not leave a single candidate");

    // 关闭双候选时恢复旧单候选路径。
    L3Estimation::PnpSolverConfig single_config;
    single_config.enable_ippe_dual_candidates = false;
    L3Estimation::PnpSolver single_solver{
      calibration,
      {},
      single_config};
    const auto single_poses = single_solver.solve(
      detection,
      L3Estimation::ArmorSize::Small);
    require(
      single_poses.size() == 1
        && single_poses.front().ippe_candidate_index == 0,
      "disabled dual candidates did not return one pose");
    require(
      cv::norm(single_poses.front().tvec - expected_tvec) < 1e-3,
      "single-candidate path translation is inaccurate");

    auto degenerate = detection;
    degenerate.corners.fill({100.0F, 100.0F});
    require(
      solver.solve(degenerate, L3Estimation::ArmorSize::Small).empty(),
      "degenerate quadrilateral was accepted");

    auto non_finite = detection;
    non_finite.corners[0].x =
      std::numeric_limits<float>::quiet_NaN();
    require(
      solver.solve(non_finite, L3Estimation::ArmorSize::Small).empty(),
      "non-finite corner was accepted");
  } catch (const std::exception& error) {
    std::cerr << "PnpSolver smoke test failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "PnpSolver smoke test passed\n";
  return 0;
}
