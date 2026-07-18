#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kSmallWidth = 0.135;
constexpr double kBigWidth = 0.225;
constexpr double kArmorHeight = 0.055;

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

[[nodiscard]] std::vector<cv::Point3d> armorPoints(double width)
{
  const double half_width = width / 2.0;
  const double half_height = kArmorHeight / 2.0;
  return {
    {0.0, half_width, half_height},
    {0.0, -half_width, half_height},
    {0.0, -half_width, -half_height},
    {0.0, half_width, -half_height}};
}

[[nodiscard]] cv::Matx33d toCv(const Eigen::Matrix3d& rotation)
{
  cv::Matx33d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = rotation(row, col);
    }
  }
  return result;
}

[[nodiscard]] Eigen::Matrix3d toEigen(const cv::Matx33d& rotation)
{
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = rotation(row, col);
    }
  }
  return result;
}

[[nodiscard]] cv::Vec3d rotationVector(const cv::Matx33d& rotation)
{
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  return rvec;
}

[[nodiscard]] std::array<cv::Point2f, 4> projectArmor(
  const L1Sensor::CameraCalibration& calibration,
  double width,
  const cv::Matx33d& rotation,
  const cv::Vec3d& translation)
{
  std::vector<cv::Point2d> projected;
  cv::projectPoints(
    armorPoints(width),
    rotationVector(rotation),
    translation,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    projected);

  return {
    cv::Point2f(projected[0]),
    cv::Point2f(projected[1]),
    cv::Point2f(projected[2]),
    cv::Point2f(projected[3])};
}

[[nodiscard]] double manualReprojectionRmse(
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::Armor& armor)
{
  const double width = armor.type == L3Estimation::ArmorType::Big
    ? kBigWidth
    : kSmallWidth;
  const cv::Matx33d rotation =
    toCv(L6Telemetry::rpyToRotation(armor.rpy_in_camera));
  const cv::Vec3d translation{
    armor.xyz_in_camera.x(),
    armor.xyz_in_camera.y(),
    armor.xyz_in_camera.z()};

  std::vector<cv::Point2d> projected;
  cv::projectPoints(
    armorPoints(width),
    rotationVector(rotation),
    translation,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    projected);

  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < armor.points.size(); ++index) {
    const double dx = projected[index].x - armor.points[index].x;
    const double dy = projected[index].y - armor.points[index].y;
    squared_error_sum += dx * dx + dy * dy;
  }
  return std::sqrt(squared_error_sum / armor.points.size());
}

[[nodiscard]] bool outputsCleared(const L3Estimation::Armor& armor)
{
  return armor.xyz_in_camera.isZero(0.0) &&
         armor.xyz_in_world.isZero(0.0) &&
         armor.rpy_in_camera.isZero(0.0) &&
         armor.rpy_in_world.isZero(0.0) &&
         armor.ypd_in_world.isZero(0.0) &&
         armor.name == L3Estimation::ArmorName::Unknown &&
         armor.type == L3Estimation::ArmorType::Small &&
         std::isinf(armor.reprojection_error) &&
         std::isinf(armor.second_reprojection_error) &&
         armor.facing == 0.0 &&
         armor.mode == L3Estimation::ObservationMode::Single &&
         armor.R.isIdentity(0.0);
}

[[nodiscard]] bool allQualityFlagsClear(const L3Estimation::Armor& armor)
{
  return !armor.quality.pnp_ok && !armor.quality.covariance_ok &&
         !armor.quality.reprojection_ok && !armor.quality.geometry_ok &&
         !armor.quality.finite && !armor.quality.yaw_ambiguous &&
         !armor.quality.valid();
}

[[nodiscard]] L1Sensor::CameraCalibration cloneCalibration(
  const L1Sensor::CameraCalibration& calibration)
{
  auto clone = calibration;
  clone.camera_matrix = calibration.camera_matrix.clone();
  clone.distortion_coefficients =
    calibration.distortion_coefficients.clone();
  return clone;
}

}  // namespace

int main()
{
  const auto camera_config =
    YAML::LoadFile("tests/data/camera_calibration_inline.yaml");
  const auto calibration = L1Sensor::loadCameraCalibration(
    camera_config["calibration"], "PnP smoke config");

  L3Estimation::PnpSolver solver(calibration);
  expect(solver.ready(), "PnpSolver rejected valid calibration");

  auto intrinsic_only_node = YAML::Clone(camera_config["calibration"]);
  intrinsic_only_node.remove("T_barrel_camera");
  const auto intrinsic_only = L1Sensor::loadCameraCalibration(
    intrinsic_only_node, "PnP intrinsic-only smoke config");
  L3Estimation::PnpSolver solver_without_extrinsics(intrinsic_only);
  expect(
    !solver_without_extrinsics.ready(),
    "PnpSolver accepted missing barrel extrinsics");

  L3Estimation::ArmorConfig invalid_config;
  invalid_config.small_width = 0.0;
  L3Estimation::PnpSolver solver_with_invalid_config(
    calibration, invalid_config);
  expect(
    !solver_with_invalid_config.ready(),
    "PnpSolver accepted invalid armor dimensions");

  L3Estimation::ArmorConfig invalid_ambiguity_config;
  invalid_ambiguity_config.ambiguity_error_ratio = 0.9;
  L3Estimation::PnpSolver solver_with_invalid_ambiguity_config(
    calibration, invalid_ambiguity_config);
  expect(
    !solver_with_invalid_ambiguity_config.ready(),
    "PnpSolver accepted an ambiguity ratio below one");

  // 基准姿态：局部 +X 大致指向相机 +Z，因此局部 -X 正面朝向相机。
  const cv::Matx33d optical_alignment{
    0.0, -1.0, 0.0,
    0.0, 0.0, -1.0,
    1.0, 0.0, 0.0};
  const Eigen::Matrix3d R_camera_armor =
    toEigen(optical_alignment) *
    L6Telemetry::rpyToRotation({0.08, -0.12, 0.16});
  const cv::Matx33d R_camera_armor_cv = toCv(R_camera_armor);
  const cv::Vec3d tvec{0.05, -0.03, 3.0};

  const Eigen::Vector3d world_barrel_rpy{0.07, -0.09, 0.24};
  const Eigen::Matrix3d R_world_barrel =
    L6Telemetry::rpyToRotation(world_barrel_rpy);
  const Eigen::Quaterniond q_world_barrel(R_world_barrel);
  solver.set_R_world_barrel(
    std::optional<Eigen::Quaterniond>{q_world_barrel});

  L3Estimation::Armor armor;
  armor.class_id = static_cast<int>(L2Perception::ArmorClass::Infantry3);
  armor.points = projectArmor(
    calibration, kSmallWidth, R_camera_armor_cv, tvec);
  solver.single_pnp(armor);

  const Eigen::Vector3d expected_camera{tvec[0], tvec[1], tvec[2]};
  const Eigen::Vector3d expected_barrel =
    *calibration.T_barrel_camera * expected_camera;
  const Eigen::Vector3d expected_world =
    R_world_barrel * expected_barrel;
  const Eigen::Matrix3d expected_world_rotation =
    R_world_barrel * calibration.T_barrel_camera->linear() *
    R_camera_armor;
  const double camera_rotation_error =
    (L6Telemetry::rpyToRotation(armor.rpy_in_camera) - R_camera_armor)
      .norm();
  const double world_rotation_error =
    (L6Telemetry::rpyToRotation(armor.rpy_in_world) -
     expected_world_rotation)
      .norm();
  const Eigen::Vector3d expected_ypd = L6Telemetry::xyz2ypd(expected_world);

  expect(armor.quality.pnp_ok, "valid synthetic armor did not solve PnP");
  expect(armor.quality.geometry_ok, "valid synthetic armor failed geometry checks");
  expect(armor.quality.finite, "valid synthetic armor produced non-finite output");
  expect(
    armor.quality.reprojection_ok,
    "near-zero synthetic reprojection failed its gate");
  expect(
    !armor.quality.covariance_ok && !armor.quality.valid(),
    "PnP-only observation was incorrectly enabled for EKF use");
  expect(
    armor.type == L3Estimation::ArmorType::Small,
    "infantry armor did not use the small model");
  expect(
    (armor.xyz_in_camera - expected_camera).norm() < 1e-3,
    "camera-frame translation is incorrect");
  expect(
    (armor.xyz_in_world - expected_world).norm() < 1e-3,
    "camera/barrel/world translation chain is incorrect");
  expect(camera_rotation_error < 1e-3, "camera-frame RPY is incorrect");
  expect(world_rotation_error < 1e-3, "world-frame RPY is incorrect");
  expect(
    (armor.ypd_in_world - expected_ypd).norm() < 1e-3,
    "world-frame yaw/pitch/distance is incorrect");
  expect(
    armor.reprojection_error < 1e-3,
    "noise-free synthetic armor does not have near-zero pixel RMSE");

  // Hero 独占大装甲尺寸；包括 BaseLarge 在内的其余合法 class_id 都用小装甲。
  for (int class_id = static_cast<int>(L2Perception::ArmorClass::Guard);
       class_id <= static_cast<int>(L2Perception::ArmorClass::BaseLarge);
       ++class_id) {
    const bool is_hero =
      class_id == static_cast<int>(L2Perception::ArmorClass::Hero);
    const double width = is_hero ? kBigWidth : kSmallWidth;
    L3Estimation::Armor sized_armor;
    sized_armor.class_id = class_id;
    sized_armor.points = projectArmor(
      calibration, width, R_camera_armor_cv, tvec);
    solver.single_pnp(sized_armor);

    expect(
      sized_armor.quality.pnp_ok && sized_armor.quality.geometry_ok &&
        sized_armor.quality.finite,
      "a legal armor class did not produce a usable PnP result");
    expect(
      (sized_armor.xyz_in_camera - expected_camera).norm() < 1e-3,
      "an armor class used the wrong physical width");
    expect(
      sized_armor.type == (is_hero ? L3Estimation::ArmorType::Big
                                   : L3Estimation::ArmorType::Small),
      "armor type classification disagrees with its class_id");
  }

  // 非对称微扰保证不能被一个理想矩形姿态完全解释。
  L3Estimation::Armor noisy_armor;
  noisy_armor.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry4);
  noisy_armor.points = projectArmor(
    calibration, kSmallWidth, R_camera_armor_cv, tvec);
  const std::array<cv::Point2f, 4> noise{
    cv::Point2f{0.45F, -0.20F},
    cv::Point2f{-0.30F, 0.35F},
    cv::Point2f{0.25F, 0.15F},
    cv::Point2f{-0.40F, -0.25F}};
  for (std::size_t index = 0; index < noisy_armor.points.size(); ++index) {
    noisy_armor.points[index] += noise[index];
  }
  solver.single_pnp(noisy_armor);

  const double manual_rmse = manualReprojectionRmse(calibration, noisy_armor);
  expect(
    noisy_armor.quality.pnp_ok && noisy_armor.quality.geometry_ok &&
      noisy_armor.quality.finite,
    "noisy armor did not retain a valid PnP pose");
  expect(
    noisy_armor.reprojection_error > 1e-3,
    "corner noise unexpectedly produced zero reprojection error");
  expect(
    std::abs(noisy_armor.reprojection_error - manual_rmse) < 1e-8,
    "reported reprojection error is not four-corner pixel RMSE");

  L3Estimation::ArmorConfig strict_config;
  strict_config.max_reprojection_error = 0.05;
  L3Estimation::PnpSolver strict_solver(calibration, strict_config);
  strict_solver.set_R_world_barrel(
    std::optional<Eigen::Quaterniond>{q_world_barrel});
  L3Estimation::Armor rejected_by_reprojection = noisy_armor;
  strict_solver.single_pnp(rejected_by_reprojection);
  expect(
    rejected_by_reprojection.quality.pnp_ok &&
      rejected_by_reprojection.quality.geometry_ok &&
      rejected_by_reprojection.quality.finite,
    "reprojection gate incorrectly discarded the PnP pose itself");
  expect(
    rejected_by_reprojection.reprojection_error >
      strict_config.max_reprojection_error &&
      !rejected_by_reprojection.quality.reprojection_ok,
    "above-threshold corner perturbation passed reprojection gating");
  expect(
    !rejected_by_reprojection.quality.valid(),
    "rejected reprojection was marked valid for EKF use");

  // 近正视平面会产生两个重投影几乎等价的 IPPE 姿态。
  L3Estimation::Armor ambiguous_armor;
  ambiguous_armor.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry5);
  ambiguous_armor.points = projectArmor(
    calibration, kSmallWidth, optical_alignment, tvec);
  solver.single_pnp(ambiguous_armor);
  expect(
    ambiguous_armor.quality.pnp_ok && ambiguous_armor.quality.geometry_ok &&
      ambiguous_armor.quality.reprojection_ok,
    "near-frontal planar observation failed PnP");
  expect(
    std::isfinite(ambiguous_armor.second_reprojection_error),
    "IPPE did not expose a second geometrically valid candidate");
  expect(
    ambiguous_armor.quality.yaw_ambiguous,
    "near-equal IPPE candidates were not marked yaw-ambiguous");
  expect(
    ambiguous_armor.second_reprojection_error <=
      std::max(
        ambiguous_armor.reprojection_error + 0.25,
        ambiguous_armor.reprojection_error * 1.2),
    "yaw ambiguity flag disagrees with the configured error rule");

  // 失败路径都复用一个已有结果，验证不会泄漏上一帧状态。
  L3Estimation::Armor nan_armor = ambiguous_armor;
  nan_armor.points[1].x = std::numeric_limits<float>::quiet_NaN();
  solver.single_pnp(nan_armor);
  expect(outputsCleared(nan_armor), "NaN input retained previous PnP output");
  expect(allQualityFlagsClear(nan_armor), "NaN input retained quality flags");

  solver.set_R_world_barrel(std::nullopt);
  L3Estimation::Armor missing_pose_armor = ambiguous_armor;
  solver.single_pnp(missing_pose_armor);
  expect(
    outputsCleared(missing_pose_armor),
    "missing image-time barrel pose retained previous PnP output");
  expect(
    allQualityFlagsClear(missing_pose_armor),
    "missing image-time barrel pose retained quality flags");
  solver.set_R_world_barrel(
    std::optional<Eigen::Quaterniond>{q_world_barrel});

  // 相同角点由背面朝向相机的姿态生成，PnP 可求解但必须被正面约定拒绝。
  const cv::Matx33d back_facing_rotation{
    0.0, 1.0, 0.0,
    0.0, 0.0, -1.0,
    -1.0, 0.0, 0.0};
  L3Estimation::Armor wrong_geometry_armor = ambiguous_armor;
  wrong_geometry_armor.points = projectArmor(
    calibration, kSmallWidth, back_facing_rotation, tvec);
  solver.single_pnp(wrong_geometry_armor);
  expect(
    wrong_geometry_armor.quality.pnp_ok,
    "back-facing test geometry did not reach candidate validation");
  expect(
    !wrong_geometry_armor.quality.geometry_ok &&
      !wrong_geometry_armor.quality.finite &&
      !wrong_geometry_armor.quality.reprojection_ok,
    "back-facing armor passed geometric validation");
  expect(
    outputsCleared(wrong_geometry_armor),
    "back-facing armor retained previous PnP output");

  auto invalid_calibration = cloneCalibration(calibration);
  invalid_calibration.camera_matrix.at<double>(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  L3Estimation::PnpSolver invalid_calibration_solver(invalid_calibration);
  expect(
    !invalid_calibration_solver.ready(),
    "PnpSolver accepted non-finite camera calibration");
  invalid_calibration_solver.set_R_world_barrel(
    std::optional<Eigen::Quaterniond>{q_world_barrel});
  L3Estimation::Armor invalid_calibration_armor = ambiguous_armor;
  invalid_calibration_solver.single_pnp(invalid_calibration_armor);
  expect(
    outputsCleared(invalid_calibration_armor),
    "invalid calibration retained previous PnP output");
  expect(
    allQualityFlagsClear(invalid_calibration_armor),
    "invalid calibration retained quality flags");

  L3Estimation::Armor invalid_class_armor = ambiguous_armor;
  invalid_class_armor.class_id = -1;
  solver.single_pnp(invalid_class_armor);
  expect(
    outputsCleared(invalid_class_armor) &&
      allQualityFlagsClear(invalid_class_armor),
    "invalid class_id retained previous PnP state");

  if (failure_count != 0) {
    std::cerr << failure_count << " PnpSolver smoke assertion(s) failed\n";
    return 1;
  }

  std::cout << "PnpSolver smoke test passed\n";
  return 0;
}
