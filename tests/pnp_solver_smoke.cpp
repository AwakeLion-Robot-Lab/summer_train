#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
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

std::vector<cv::Point3f> armorPoints(double width)
{
  const float half_width = static_cast<float>(width / 2.0);
  const float half_height = static_cast<float>(kArmorHeight / 2.0);
  return {
    {0.0F, half_width, half_height},
    {0.0F, -half_width, half_height},
    {0.0F, -half_width, -half_height},
    {0.0F, half_width, -half_height}};
}

// 复刻 PnpSolver 内部的 armor -> world 旋转：安装倾角固定 15 度，只有 yaw 自由。
// 这里必须独立写一遍而不是调求解器，否则两边一起写错就测不出来。
Eigen::Matrix3d armorRotationInWorldReference(double yaw)
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

cv::Vec3d rotationVector(const cv::Matx33d& rotation)
{
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  return rvec;
}

std::array<cv::Point2f, 4> projectArmor(
  const L1Sensor::CameraCalibration& calibration,
  double width,
  const cv::Matx33d& rotation,
  const cv::Vec3d& translation)
{
  std::vector<cv::Point2f> projected;
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

double manualReprojectionRmse(
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

  std::vector<cv::Point2f> projected;
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

bool outputsCleared(const L3Estimation::Armor& armor)
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

// PnP 是否成功提交了位姿。name 只在提交的那一步才被赋值，任何失败路径上都
// 保持 Unknown——这是 Tracker 唯一的观测门限，ArmorQuality 已经不存在了。
bool poseCommitted(const L3Estimation::Armor& armor)
{
  return armor.name != L3Estimation::ArmorName::Unknown;
}

// PnpSolver 内部的角度归一化。必须照抄而不是用 limit_rad：后者走 std::remainder，
// 与这里的循环相差一个 ulp，做逐位对照时会变成噪声。
double spLimitRadReference(double angle)
{
  while (angle > std::numbers::pi) angle -= 2.0 * std::numbers::pi;
  while (angle <= -std::numbers::pi) angle += 2.0 * std::numbers::pi;
  return angle;
}

// 只用公开的 reproject_armor 独立复刻整个 yaw 搜索：140 点整步枚举加三点抛物线
// 细化。求解器内部走的是绕开 cv::Mat 与 Rodrigues 的快路，两条路必须给出同一个
// 答案——这是那条快路唯一能从外部验证的方式。
double referenceYawSearch(
  const L3Estimation::PnpSolver& solver,
  const L3Estimation::Armor& armor,
  double barrel_yaw)
{
  constexpr double kDegree = std::numbers::pi / 180.0;
  constexpr int kSteps = 140;
  const double yaw0 = spLimitRadReference(barrel_yaw - 70.0 * kDegree);

  std::array<double, kSteps> costs{};
  double best_cost = std::numeric_limits<double>::infinity();
  int best_index = -1;
  double best_yaw = armor.ypr_in_world[0];
  for (int index = 0; index < kSteps; ++index) {
    const double yaw = spLimitRadReference(yaw0 + index * kDegree);
    const std::vector<cv::Point2f> projected =
      solver.reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
    if (projected.size() != armor.points.size()) {
      costs[index] = std::numeric_limits<double>::infinity();
      continue;
    }
    double cost = 0.0;
    for (std::size_t point = 0; point < projected.size(); ++point) {
      cost += cv::norm(armor.points[point] - projected[point]);
    }
    costs[index] = cost;
    if (cost < best_cost) {
      best_cost = cost;
      best_yaw = yaw;
      best_index = index;
    }
  }
  if (best_index <= 0 || best_index + 1 >= kSteps) {
    return best_yaw;
  }

  const double left = costs[best_index - 1];
  const double center = costs[best_index];
  const double right = costs[best_index + 1];
  if (!std::isfinite(left) || !std::isfinite(center) || !std::isfinite(right)) {
    return best_yaw;
  }
  const double curvature = left - 2.0 * center + right;
  if (std::abs(curvature) < 1e-9) {
    return best_yaw;
  }
  const double offset = 0.5 * (left - right) / curvature;
  if (std::abs(offset) > 0.5) {
    return best_yaw;
  }
  return spLimitRadReference(best_yaw + offset * kDegree);
}

// 在指定的世界系 yaw 和世界系位置上合成一块无噪声的板：按参考旋转造出
// armor -> camera 位姿，再投影出四个角点。真值 yaw 已知，因此可以直接量
// 搜索输出离真值差多少——这是亚度细化唯一能被验证的方式。
L3Estimation::Armor synthesizeArmorAtWorldYaw(
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Matrix3d& R_world_barrel,
  const Eigen::Vector3d& xyz_in_world,
  double yaw)
{
  const Eigen::Matrix3d R_camera_barrel = calibration.T_barrel_camera->linear();
  const Eigen::Vector3d t_camera_barrel =
    calibration.T_barrel_camera->translation();
  const Eigen::Matrix3d R_armor_world = armorRotationInWorldReference(yaw);
  const Eigen::Matrix3d R_armor_camera = R_camera_barrel.transpose() *
    R_world_barrel.transpose() * R_armor_world;
  const Eigen::Vector3d t_armor_camera = R_camera_barrel.transpose() *
    (R_world_barrel.transpose() * xyz_in_world - t_camera_barrel);

  L3Estimation::Armor armor;
  armor.class_id = static_cast<int>(L2Perception::ArmorClass::Infantry3);
  armor.points = projectArmor(
    calibration, kSmallWidth, L6Telemetry::toCv(R_armor_camera),
    cv::Vec3d{
      t_armor_camera.x(), t_armor_camera.y(), t_armor_camera.z()});
  return armor;
}

L1Sensor::CameraCalibration cloneCalibration(
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

  expect(poseCommitted(armor), "valid synthetic armor did not solve PnP");
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
  // sp_vision 的 1 度离散搜索不估计 yaw 标准差；兼容字段保持无穷。
  expect(
    std::isinf(armor.yaw_sigma),
    "SP-compatible discrete yaw search unexpectedly estimated yaw sigma");

  // 回归用例：人为生成一个在枪管背后 180° 的零误差平面解。整周搜索
  // 会把这个背面极小值选中；SP 的行为是无论整周哪里代价更小，输出都必须
  // 留在枪管 yaw 的 [-70°, +69°] 枚举窗口内。
  {
    const double barrel_yaw =
      L6Telemetry::eulers(R_world_barrel, 2, 1, 0)[0];
    const double back_yaw =
      L6Telemetry::limit_rad(barrel_yaw + std::numbers::pi);
    const Eigen::Matrix3d R_armor_world =
      armorRotationInWorldReference(back_yaw);
    const Eigen::Matrix3d R_camera_barrel =
      calibration.T_barrel_camera->linear();
    const Eigen::Vector3d t_camera_barrel =
      calibration.T_barrel_camera->translation();
    const Eigen::Matrix3d R_armor_camera = R_camera_barrel.transpose() *
      R_world_barrel.transpose() * R_armor_world;
    const Eigen::Vector3d t_armor_camera = R_camera_barrel.transpose() *
      (R_world_barrel.transpose() * expected_world - t_camera_barrel);

    L3Estimation::Armor back_minimum;
    back_minimum.class_id =
      static_cast<int>(L2Perception::ArmorClass::Infantry3);
    back_minimum.points = projectArmor(
      calibration, kSmallWidth, L6Telemetry::toCv(R_armor_camera),
      cv::Vec3d{
        t_armor_camera.x(), t_armor_camera.y(), t_armor_camera.z()});
    solver.single_pnp(back_minimum);

    const double optimized_offset = L6Telemetry::limit_rad(
      back_minimum.ypr_in_world[0] - barrel_yaw);
    expect(poseCommitted(back_minimum), "back-minimum PnP pose was not committed");
    expect(
      optimized_offset >= -70.0 * std::numbers::pi / 180.0 - 1e-12 &&
        optimized_offset <= 69.0 * std::numbers::pi / 180.0 + 1e-12,
      "yaw optimizer escaped SP's barrel-centered search window");
  }

  // 亚度细化。1 度整步只能给出栅格上的点，栅格锚在 barrel_yaw - 70 度，所以
  // 相对枪管 yaw 的栅格点恰好是整数度——把真值故意放在两个整数度之间，那个
  // 小数就是整步必然吃下的量化误差，细化要做的就是把它收回来。
  //
  // 这里不复用上面那份合成标定：它的 T_barrel_camera 是绕 z 的 90 度，相机
  // 光轴落在 barrel +z 上，而搜索窗口锚在 barrel yaw，两者对不上，板子摆在
  // 光轴正前方时 yaw 根本不在窗口里。换成 CLAUDE.md 记录的那份真实轴置换
  // （光学系 z 前/x 右/y 下 -> barrel x 前/y 左/z 上），几何才自洽。
  {
    auto physical = cloneCalibration(calibration);
    Eigen::Isometry3d T_barrel_camera = Eigen::Isometry3d::Identity();
    T_barrel_camera.linear() = (Eigen::Matrix3d{} <<
      0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0,
      0.0, -1.0, 0.0).finished();
    T_barrel_camera.translation() = Eigen::Vector3d{0.02, 0.0, 0.05};
    physical.T_barrel_camera = T_barrel_camera;

    L3Estimation::PnpSolver physical_solver(physical);
    expect(
      physical_solver.ready(),
      "PnpSolver rejected the physically consistent calibration");
    physical_solver.set_R_world_barrel(
      std::optional<Eigen::Quaterniond>{q_world_barrel});

    constexpr double kDegree = std::numbers::pi / 180.0;
    const double barrel_yaw =
      L6Telemetry::eulers(R_world_barrel, 2, 1, 0)[0];
    // 板心摆在枪口正前方 3 米，四个角点都在相机前方。
    const Eigen::Vector3d plate_world =
      R_world_barrel * Eigen::Vector3d{3.0, 0.0, 0.0};

    // 覆盖靠近格心与靠近格边、两个方向。刻意不取正负 0.5：那是相邻两格代价
    // 相等的简并点，胜者由浮点比较的先后决定，本来就没有确定答案。
    for (double fraction : {0.15, 0.37, -0.28, 0.45}) {
      const double true_yaw = L6Telemetry::limit_rad(
        barrel_yaw + (12.0 + fraction) * kDegree);
      const double grid_yaw =
        L6Telemetry::limit_rad(barrel_yaw + 12.0 * kDegree);

      L3Estimation::Armor refined = synthesizeArmorAtWorldYaw(
        physical, R_world_barrel, plate_world, true_yaw);
      physical_solver.single_pnp(refined);
      expect(
        poseCommitted(refined),
        "sub-degree synthetic armor did not produce a PnP pose");

      const double error =
        std::abs(L6Telemetry::limit_rad(refined.ypr_in_world[0] - true_yaw));
      const double quantized = std::abs(fraction) * kDegree;
      const double moved =
        std::abs(L6Telemetry::limit_rad(refined.ypr_in_world[0] - grid_yaw));

      // 一、必须比整步的量化误差更接近真值，否则细化没有意义。
      expect(
        error < quantized,
        "parabolic refinement did not beat the 1-degree grid");
      // 二、绝不许离开赢下来的那一格。这条保证细化动不了 argmin，
      //     引入不了整步枚举本来没有的失效模式。
      expect(
        moved <= 0.5 * kDegree + 1e-12,
        "parabolic refinement left the winning grid cell");
      // 三、无噪声合成板在真解处代价恰好为零，四个角点残差同时归零，底部
      //     是折线而不是抛物线，细化只能收回一部分。0.15 度是这种最不利
      //     形状下的上界；真实录像里残差不会同时归零，底部光滑得多，
      //     sp demo 的 526 块板上实测中位 0.025 度、p95 0.197 度。
      expect(
        error < 0.15 * kDegree,
        "parabolic refinement is worse than the piecewise-linear bound");

      // 四、快路交叉核对。求解器内部不再走 cv::projectPoints，这里用公开的
      //     reproject_armor 把整个搜索独立算一遍，两者必须落在同一个答案上。
      //     两条路唯一的系统性差异是投影点收窄成 float 的那一步（约 1e-4 px
      //     的代价差），传到角度上远小于 1e-3 度。
      const double reference =
        referenceYawSearch(physical_solver, refined, barrel_yaw);
      expect(
        std::abs(L6Telemetry::limit_rad(
          refined.ypr_in_world[0] - reference)) < 1e-3 * kDegree,
        "fast yaw-cost path disagrees with the cv::projectPoints reference");
    }
  }

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
      poseCommitted(sized_armor),
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
    poseCommitted(noisy_armor), "noisy armor did not retain a valid PnP pose");
  expect(
    noisy_armor.reprojection_error > 1e-3,
    "corner noise unexpectedly produced zero reprojection error");
  expect(
    std::abs(noisy_armor.reprojection_error - manual_rmse) < 1e-8,
    "reported reprojection error is not four-corner pixel RMSE");

  // 重投影误差只是诊断量，不再是门限：与 sp_vision 一致，无论 RMSE 多大，
  // 位姿照样提交，Tracker 照样把它喂进 EKF。
  L3Estimation::Armor large_error_armor = noisy_armor;
  for (auto& point : large_error_armor.points) {
    point.x += 6.0F;
  }
  solver.single_pnp(large_error_armor);
  expect(
    poseCommitted(large_error_armor),
    "a large reprojection error must no longer reject the observation");

  // 当前与 SP-Vision 一致，solvePnP + IPPE 使用单解。
  L3Estimation::Armor frontal_armor;
  frontal_armor.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry5);
  frontal_armor.points = projectArmor(
    calibration, kSmallWidth, optical_alignment, tvec);
  solver.single_pnp(frontal_armor);
  expect(
    poseCommitted(frontal_armor), "near-frontal planar observation failed PnP");
  // 失败路径都复用一个已有结果，验证不会泄漏上一帧状态。
  L3Estimation::Armor nan_armor = frontal_armor;
  nan_armor.points[1].x = std::numeric_limits<float>::quiet_NaN();
  solver.single_pnp(nan_armor);
  expect(outputsCleared(nan_armor), "NaN input retained previous PnP output");

  solver.set_R_world_barrel(std::nullopt);
  L3Estimation::Armor missing_pose_armor = frontal_armor;
  solver.single_pnp(missing_pose_armor);
  expect(
    outputsCleared(missing_pose_armor),
    "missing image-time barrel pose retained previous PnP output");
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
    poseCommitted(wrong_geometry_armor),
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

  L3Estimation::Armor invalid_class_armor = frontal_armor;
  invalid_class_armor.class_id = -1;
  solver.single_pnp(invalid_class_armor);
  expect(
    outputsCleared(invalid_class_armor),
    "invalid class_id retained previous PnP state");

  // reproject_armor 必须和 SP 一样走 Point3f/Point2f 的 cv::projectPoints；
  // 逐个 yaw 与独立参考调用核对，防止以后又换回另一条手写投影路径。
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

      std::vector<cv::Point2f> reference;
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
        worst_pixel_error = std::max(
          worst_pixel_error,
          std::hypot(
            static_cast<double>(reference[index].x - fast[index].x),
            static_cast<double>(reference[index].y - fast[index].y)));
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
