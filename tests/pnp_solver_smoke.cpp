#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <string_view>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kSmallWidth = 0.135;
constexpr double kBigWidth = 0.230;
constexpr double kArmorHeight = 0.056;

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

// 复刻 PnpSolver 内部的 armor -> world 旋转：安装倾角固定 15 度，只有 yaw 自由。
// 这里必须独立写一遍而不是调求解器，否则两边一起写错就测不出来。
[[nodiscard]] Eigen::Matrix3d armorRotationInWorldReference(double yaw)
{
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double pitch = 15.0 * std::numbers::pi / 180.0;
  const double sin_pitch = std::sin(pitch);
  const double cos_pitch = std::cos(pitch);
  return Eigen::Matrix3d{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch},
    {-sin_pitch, 0.0, cos_pitch}};
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
    L6Telemetry::toCv(L6Telemetry::yprToRotation(armor.ypr_in_camera));
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
         armor.ypr_in_camera.isZero(0.0) &&
         armor.ypr_in_world.isZero(0.0) &&
         armor.ypd_in_world.isZero(0.0) &&
         armor.name == L3Estimation::ArmorName::Unknown &&
         armor.type == L3Estimation::ArmorType::Small &&
         std::isinf(armor.reprojection_error);
}

[[nodiscard]] bool allQualityFlagsClear(const L3Estimation::Armor& armor)
{
  return !armor.quality.pnp_ok && !armor.quality.reprojection_ok &&
         !armor.quality.geometry_ok && !armor.quality.finite &&
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

  // 基准姿态：局部 +X 大致指向相机 +Z，因此局部 -X 正面朝向相机。
  const cv::Matx33d optical_alignment{
    0.0, -1.0, 0.0,
    0.0, 0.0, -1.0,
    1.0, 0.0, 0.0};
  const Eigen::Matrix3d R_camera_armor =
    L6Telemetry::toEigen(optical_alignment) *
    L6Telemetry::yprToRotation({0.16, -0.12, 0.08});
  const cv::Matx33d R_camera_armor_cv = L6Telemetry::toCv(R_camera_armor);
  const cv::Vec3d tvec{0.05, -0.03, 3.0};

  const Eigen::Vector3d world_barrel_ypr{0.24, -0.09, 0.07};
  const Eigen::Matrix3d R_world_barrel =
    L6Telemetry::yprToRotation(world_barrel_ypr);
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
    (L6Telemetry::yprToRotation(armor.ypr_in_camera) - R_camera_armor)
      .norm();
  const Eigen::Vector3d expected_world_ypr =
    L6Telemetry::eulers(expected_world_rotation, 2, 1, 0);
  const double world_pitch_roll_error =
    (armor.ypr_in_world.tail<2>() - expected_world_ypr.tail<2>()).norm();
  const Eigen::Vector3d expected_ypd = L6Telemetry::xyz2ypd(expected_world);

  expect(armor.quality.pnp_ok, "valid synthetic armor did not solve PnP");
  expect(armor.quality.geometry_ok, "valid synthetic armor failed geometry checks");
  expect(armor.quality.finite, "valid synthetic armor produced non-finite output");
  expect(
    armor.quality.reprojection_ok,
    "near-zero synthetic reprojection failed its gate");
  expect(
    armor.quality.valid(),
    "valid PnP observation did not pass the combined quality check");
  expect(
    armor.type == L3Estimation::ArmorType::Small,
    "infantry armor did not use the small model");
  expect(
    (armor.xyz_in_camera - expected_camera).norm() < 1e-3,
    "camera-frame translation is incorrect");
  expect(
    (armor.xyz_in_world - expected_world).norm() < 1e-3,
    "camera/barrel/world translation chain is incorrect");
  expect(camera_rotation_error < 1e-3, "camera-frame YPR is incorrect");
  expect(
    world_pitch_roll_error < 1e-3,
    "world-frame pitch/roll changed during yaw optimization");
  expect(
    (armor.ypd_in_world - expected_ypd).norm() < 1e-3,
    "world-frame yaw/pitch/distance is incorrect");
  expect(
    armor.reprojection_error < 1e-3,
    "noise-free synthetic armor does not have near-zero pixel RMSE");
  // yaw_sigma 来自高斯牛顿收敛点的曲率。上界取得很宽松：这里只确认它确实
  // 被算了出来且量级合理，真正的标定留给实机数据。为零说明曲率被当成了
  // 无穷大，那种情况下 EKF 会无条件相信这个 yaw，比不给还危险。
  expect(
    std::isfinite(armor.yaw_sigma) && armor.yaw_sigma > 0.0 &&
      armor.yaw_sigma < 1.0,
    "yaw sigma from Gauss-Newton curvature is missing or implausible");

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

  // 当前与 SP-Vision 一致，solvePnP + IPPE 使用单解。
  L3Estimation::Armor frontal_armor;
  frontal_armor.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry5);
  frontal_armor.points = projectArmor(
    calibration, kSmallWidth, optical_alignment, tvec);
  solver.single_pnp(frontal_armor);
  expect(
    frontal_armor.quality.pnp_ok && frontal_armor.quality.geometry_ok &&
      frontal_armor.quality.reprojection_ok,
    "near-frontal planar observation failed PnP");
  // 失败路径都复用一个已有结果，验证不会泄漏上一帧状态。
  L3Estimation::Armor nan_armor = frontal_armor;
  nan_armor.points[1].x = std::numeric_limits<float>::quiet_NaN();
  solver.single_pnp(nan_armor);
  expect(outputsCleared(nan_armor), "NaN input retained previous PnP output");
  expect(allQualityFlagsClear(nan_armor), "NaN input retained quality flags");

  solver.set_R_world_barrel(std::nullopt);
  L3Estimation::Armor missing_pose_armor = frontal_armor;
  solver.single_pnp(missing_pose_armor);
  expect(
    outputsCleared(missing_pose_armor),
    "missing image-time barrel pose retained previous PnP output");
  expect(
    allQualityFlagsClear(missing_pose_armor),
    "missing image-time barrel pose retained quality flags");
  solver.set_R_world_barrel(
    std::optional<Eigen::Quaterniond>{q_world_barrel});

  // 与 SP-Vision 一致，不使用正面方向作为拒绝门限。
  const cv::Matx33d back_facing_rotation{
    0.0, 1.0, 0.0,
    0.0, 0.0, -1.0,
    -1.0, 0.0, 0.0};
  L3Estimation::Armor wrong_geometry_armor = frontal_armor;
  wrong_geometry_armor.points = projectArmor(
    calibration, kSmallWidth, back_facing_rotation, tvec);
  solver.single_pnp(wrong_geometry_armor);
  expect(
    wrong_geometry_armor.quality.pnp_ok,
    "back-facing test geometry did not reach pose validation");
  expect(
    wrong_geometry_armor.quality.geometry_ok &&
      wrong_geometry_armor.quality.finite &&
      wrong_geometry_armor.quality.reprojection_ok,
    "SP-compatible solvePnP rejected a finite back-facing pose");

  auto invalid_calibration = cloneCalibration(calibration);
  invalid_calibration.camera_matrix.at<double>(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  L3Estimation::PnpSolver invalid_calibration_solver(invalid_calibration);
  expect(
    !invalid_calibration_solver.ready(),
    "PnpSolver accepted non-finite camera calibration");
  invalid_calibration_solver.set_R_world_barrel(
    std::optional<Eigen::Quaterniond>{q_world_barrel});
  L3Estimation::Armor invalid_calibration_armor = frontal_armor;
  invalid_calibration_solver.single_pnp(invalid_calibration_armor);
  expect(
    outputsCleared(invalid_calibration_armor),
    "invalid calibration retained previous PnP output");
  expect(
    allQualityFlagsClear(invalid_calibration_armor),
    "invalid calibration retained quality flags");

  L3Estimation::Armor invalid_class_armor = frontal_armor;
  invalid_class_armor.class_id = -1;
  solver.single_pnp(invalid_class_armor);
  expect(
    outputsCleared(invalid_class_armor) &&
      allQualityFlagsClear(invalid_class_armor),
    "invalid class_id retained previous PnP state");

  // reproject_armor 内部为了 yaw 搜索的速度自己实现了 Brown-Conrady 畸变，
  // 而不是每次都调 cv::projectPoints。这一段是那份复刻的唯一防线：只要两者
  // 出现分歧，yaw 搜索最小化的就不是真正的重投影误差，而这种错误在整车
  // 结果上表现为缓慢的偏差，不会有任何显式报错。
  {
    double worst_pixel_error = 0.0;
    for (int yaw_degrees = -180; yaw_degrees < 180; yaw_degrees += 7) {
      const double yaw =
        static_cast<double>(yaw_degrees) * std::numbers::pi / 180.0;
      const auto fast = solver.reproject_armor(
        expected_world, yaw, L3Estimation::ArmorType::Small,
        L3Estimation::ArmorName::Infantry3);
      if (fast.empty()) {
        continue;
      }

      // 用求解器给出的位姿反推 armor -> camera，再走一次 OpenCV 参考实现。
      const Eigen::Matrix3d R_armor_world = armorRotationInWorldReference(yaw);
      const Eigen::Matrix3d R_camera_barrel =
        calibration.T_barrel_camera->linear();
      const Eigen::Vector3d t_camera_barrel =
        calibration.T_barrel_camera->translation();
      const Eigen::Matrix3d R_armor_camera = R_camera_barrel.transpose() *
        R_world_barrel.transpose() * R_armor_world;
      const Eigen::Vector3d t_armor_camera = R_camera_barrel.transpose() *
        (R_world_barrel.transpose() * expected_world - t_camera_barrel);

      std::vector<cv::Point2d> reference;
      cv::projectPoints(
        armorPoints(kSmallWidth),
        rotationVector(L6Telemetry::toCv(R_armor_camera)),
        cv::Vec3d{t_armor_camera.x(), t_armor_camera.y(), t_armor_camera.z()},
        calibration.camera_matrix, calibration.distortion_coefficients,
        reference);
      expect(
        reference.size() == fast.size(),
        "fast reprojection returned a different number of corners");
      for (std::size_t index = 0; index < reference.size(); ++index) {
        // reproject_armor 对外返回 Point2f，700 像素量级上 float 本身就有约
        // 6e-5 的量化台阶。把参考值同样降到 float 再比，剩下的差异才是两套
        // 投影数学的差异；否则测的是浮点格式而不是公式。
        worst_pixel_error = std::max(
          worst_pixel_error,
          std::hypot(
            static_cast<double>(static_cast<float>(reference[index].x)) -
              static_cast<double>(fast[index].x),
            static_cast<double>(static_cast<float>(reference[index].y)) -
              static_cast<double>(fast[index].y)));
      }
    }
    expect(
      worst_pixel_error < 1e-4,
      "fast reprojection disagrees with cv::projectPoints");
  }

  if (failure_count != 0) {
    std::cerr << failure_count << " PnpSolver smoke assertion(s) failed\n";
    return 1;
  }

  std::cout << "PnpSolver smoke test passed\n";
  return 0;
}
