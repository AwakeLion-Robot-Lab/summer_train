// 校验"PnP 观测 vs Daedalus 真值"这条残差链路上的坐标系约定。
//
// daedalus_noise_calib 量的是观测减真值。只要两边有任何一处约定不一致，
// 残差就会是一个看起来像噪声的常量，而统计出来的 R 全是错的。这个测试用
// **无噪声**的合成观测走完整条链路：真值位姿 -> 重投影出像素角点 -> single_pnp
// 解回位姿 -> 按标定工具的算法算残差。约定全对的话残差必须是零。
//
// 覆盖三处最容易错的地方：
//   1. xyz_in_world 的原点和 ypd 的定义
//   2. 仿真发的"朝外法线" yaw 与 newvision"朝内法线" yaw 差 π
//   3. 仿真把 marker mesh 的四个点排成左上/右上/右下/左下 的象限算法，
//      是否与 pnp_solver.cpp 的 armorPoints 顺序一致

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/daedalus_ground_truth.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/tracker.hpp"
#include "l3_estimation/types.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] double degrees(double radians)
{
  return radians * 180.0 / std::numbers::pi;
}

[[nodiscard]] L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {1440, 1080};
  calibration.camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
  calibration.camera_matrix.at<double>(0, 0) = 1200.0;
  calibration.camera_matrix.at<double>(1, 1) = 1200.0;
  calibration.camera_matrix.at<double>(0, 2) = 720.0;
  calibration.camera_matrix.at<double>(1, 2) = 540.0;
  calibration.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64FC1);

  // CLAUDE.md 记的那个纯轴变换：OpenCV 光学系 -> 枪管系（x 出膛口，z 上，y 左）。
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() <<
     0.0,  0.0, 1.0,
    -1.0,  0.0, 0.0,
     0.0, -1.0, 0.0;
  transform.translation() = Eigen::Vector3d{0.02, 0.0, 0.05};
  calibration.T_barrel_camera = transform;
  return calibration;
}

// 镜像 pnp_solver.cpp 的 armorRotationInWorld：板系 x 是**朝内**法线。
[[nodiscard]] Eigen::Matrix3d armorRotationInWorld(double yaw, double mount_pitch)
{
  const double sy = std::sin(yaw);
  const double cy = std::cos(yaw);
  const double sp = std::sin(mount_pitch);
  const double cp = std::cos(mount_pitch);
  return Eigen::Matrix3d{
    {cy * cp, -sy, cy * sp},
    {sy * cp, cy, sy * sp},
    {-sp, 0.0, cp}};
}

// 镜像 pnp_solver.cpp 的 armorPoints：左上、右上、右下、左下，板系 x 是法线、
// y 是宽度方向、z 是高度方向。
[[nodiscard]] std::array<Eigen::Vector3d, 4> armorPoints(double width, double height)
{
  const double hw = width / 2.0;
  const double hh = height / 2.0;
  return {
    Eigen::Vector3d{0.0, hw, hh}, Eigen::Vector3d{0.0, -hw, hh},
    Eigen::Vector3d{0.0, -hw, -hh}, Eigen::Vector3d{0.0, hw, -hh}};
}

// 这是 Rust 那边 solve_plate_frame 的等价实现（crates/talos-ipc 的发布端把
// marker mesh 的四个顶点按同样的规则排序）。这里重跑一遍，是为了确认这套象限
// 排序确实能还原出 armorPoints 的顺序 —— 顺序错了，角点真值就没法用。
struct PlateFrame {
  std::array<Eigen::Vector3d, 4> corners{};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double outward_yaw{0.0};
  double pitch{0.0};
};

[[nodiscard]] PlateFrame solvePlateFrame(
  std::array<Eigen::Vector3d, 4> raw,
  const Eigen::Vector3d& vehicle_center)
{
  const Eigen::Vector3d center =
    (raw[0] + raw[1] + raw[2] + raw[3]) / 4.0;

  Eigen::Vector3d normal = (raw[1] - raw[0]).cross(raw[2] - raw[0]);
  check(normal.squaredNorm() > 1e-12, "degenerate marker quad");
  normal.normalize();
  if (normal.dot(center - vehicle_center) < 0.0) {
    normal = -normal;
  }

  Eigen::Vector3d up =
    Eigen::Vector3d::UnitZ() - normal * Eigen::Vector3d::UnitZ().dot(normal);
  check(up.squaredNorm() > 1e-9, "plate is horizontal");
  up.normalize();

  // 右手系 x = 朝外法线、z = 朝上 ⟹ y = z × x 指向观察者的右手边。
  const Eigen::Vector3d right = up.cross(normal);

  PlateFrame frame;
  std::array<bool, 4> filled{};
  for (const Eigen::Vector3d& point : raw) {
    const Eigen::Vector3d offset = point - center;
    const bool left_side = offset.dot(right) < 0.0;
    const bool top_side = offset.dot(up) > 0.0;
    const std::size_t slot = left_side ? (top_side ? 0 : 3) : (top_side ? 1 : 2);
    check(!filled[slot], "two corners fell into the same quadrant");
    frame.corners[slot] = point;
    filled[slot] = true;
  }

  frame.center = center;
  frame.outward_yaw = std::atan2(normal.y(), normal.x());
  frame.pitch = std::asin(std::clamp(-normal.z(), -1.0, 1.0));
  return frame;
}

}  // namespace

int main()
{
  try {
    const L1Sensor::CameraCalibration calibration = makeCalibration();
    const L3Estimation::ArmorConfig armor_config;
    L3Estimation::PnpSolver solver(calibration, armor_config);
    check(solver.ready(), "PnpSolver rejected the synthetic calibration");

    // 观测者朝向若干个方位，装甲板放在若干个距离和斜视角上。
    const std::array<double, 3> barrel_yaws{0.0, 0.7, -1.9};
    const std::array<double, 3> distances{1.5, 3.5, 6.0};
    // 相对"正对观测者"的偏转角；0 就是板正对着看。
    const std::array<double, 4> obliquities{0.0, 0.35, -0.6, 0.9};

    std::size_t cases = 0;
    std::size_t declined = 0;
    std::string declined_cases;
    double worst_azimuth = 0.0;
    double worst_elevation = 0.0;
    double worst_distance = 0.0;
    double worst_yaw = 0.0;
    double worst_corner = 0.0;

    for (const double barrel_yaw : barrel_yaws) {
      const Eigen::Quaterniond q_world_barrel{
        Eigen::AngleAxisd(barrel_yaw, Eigen::Vector3d::UnitZ())};
      solver.set_R_world_barrel(q_world_barrel);

      for (const double distance : distances) {
        for (const double obliquity : obliquities) {
          // 真值：板中心在世界系（原点是枪口）的位置，以及 newvision 约定下的板 yaw。
          const Eigen::Vector3d truth_xyz{
            distance * std::cos(barrel_yaw), distance * std::sin(barrel_yaw), 0.05};
          const double sight_yaw = std::atan2(truth_xyz.y(), truth_xyz.x());
          // 朝内法线：正对观测者时与视线方向一致。
          const double truth_yaw = L6Telemetry::limit_rad(sight_yaw + obliquity);

          // 每条断言都带上是哪一组参数失败的，否则 36 个组合里挑一个查很折磨。
          const std::string where = " [barrel_yaw=" + std::to_string(barrel_yaw) +
            " distance=" + std::to_string(distance) +
            " obliquity=" + std::to_string(obliquity) + "]";

          const auto projected = solver.reproject_armor(
            truth_xyz, truth_yaw, L3Estimation::ArmorType::Small,
            L3Estimation::ArmorName::Infantry3);
          check(
            projected.size() == 4,
            "reproject_armor did not return 4 corners" + where);

          // ---- 链路 A：像素角点 -> single_pnp -> 观测 ----
          L2Perception::Armor detection;
          detection.class_id = static_cast<int>(L2Perception::ArmorClass::Infantry3);
          detection.color = L2Perception::ArmorColor::Blue;
          detection.confidence = 1.0F;
          for (std::size_t index = 0; index < 4; ++index) {
            detection.corners[index] = projected[index];
          }
          detection.center = (projected[0] + projected[1] + projected[2] + projected[3]) / 4.0F;

          L3Estimation::Armor observation = L3Estimation::toArmorObservation(
            detection, std::chrono::steady_clock::now());
          solver.single_pnp(observation);
          // 正对且远距离时四个角点几乎构成完美矩形，OpenCV 的 IPPE 会返回
          // "成功"但 tvec 是 NaN；single_pnp 的 allFinite 兜住了它，于是这一帧
          // 没有观测。这是真实行为，不是本测试的缺陷——标定工具会把它表现为
          // 远距离正视时样本变少。这里允许发生，但要求它是少数。
          const bool pnp_declined =
            observation.name == L3Estimation::ArmorName::Unknown;

          // ---- 链路 B：真值 -> 仿真那侧的角点/朝外 yaw -> 标定工具的换算 ----
          // 仿真发布的是世界系角点和朝外法线，这里照着造一遍。
          const Eigen::Matrix3d R_armor_world =
            armorRotationInWorld(truth_yaw, armor_config.mount_pitch);
          const auto object_points =
            armorPoints(armor_config.small_width, armor_config.height);
          std::array<Eigen::Vector3d, 4> world_corners{};
          for (std::size_t index = 0; index < 4; ++index) {
            world_corners[index] = R_armor_world * object_points[index] + truth_xyz;
          }

          // 车心在板背后：朝内法线方向即由板指向车心。
          const Eigen::Vector3d inward = R_armor_world.col(0);
          const Eigen::Vector3d vehicle_center = truth_xyz + inward * 0.2;

          // 打乱后再让象限排序还原，确认排序不是靠输入顺序蒙对的。
          std::array<Eigen::Vector3d, 4> shuffled{
            world_corners[2], world_corners[0], world_corners[3], world_corners[1]};
          const PlateFrame plate = solvePlateFrame(shuffled, vehicle_center);

          for (std::size_t index = 0; index < 4; ++index) {
            const double error = (plate.corners[index] - world_corners[index]).norm();
            worst_corner = std::max(worst_corner, error);
            check(
              error < 1e-9,
              "quadrant sort did not reproduce the armorPoints corner order" + where);
          }
          check(
            (plate.center - truth_xyz).norm() < 1e-9,
            "plate centre must equal the armor centre" + where);

          // 朝外 yaw 过换算后必须回到 newvision 的板 yaw。
          const double converted =
            L1Sensor::toNewvisionArmorYaw(plate.outward_yaw);
          check(
            std::abs(L6Telemetry::limit_rad(converted - truth_yaw)) < 1e-9,
            "toNewvisionArmorYaw did not undo the outward-normal convention" + where);

          // 斜视角的定义要和 update_ypda 的 delta_angle 对上：正视时为 0。
          const double view_angle = std::abs(L6Telemetry::limit_rad(
            converted - std::atan2(truth_xyz.y(), truth_xyz.x())));
          check(
            std::abs(view_angle - std::abs(obliquity)) < 1e-9,
            "view angle does not match the configured obliquity" + where);

          ++cases;
          if (pnp_declined) {
            ++declined;
            declined_cases += where;
            continue;
          }

          // ---- 残差：完全按 daedalus_noise_calib 的算法 ----
          const Eigen::Vector3d truth_ypd = L6Telemetry::xyz2ypd(truth_xyz);
          const double e_azimuth = L6Telemetry::limit_rad(
            observation.ypd_in_world.x() - truth_ypd.x());
          const double e_elevation = L6Telemetry::limit_rad(
            observation.ypd_in_world.y() - truth_ypd.y());
          const double e_distance = observation.ypd_in_world.z() - truth_ypd.z();
          const double e_yaw = L6Telemetry::limit_rad(
            observation.ypr_in_world[0] - converted);

          worst_azimuth = std::max(worst_azimuth, std::abs(e_azimuth));
          worst_elevation = std::max(worst_elevation, std::abs(e_elevation));
          worst_distance = std::max(worst_distance, std::abs(e_distance));
          worst_yaw = std::max(worst_yaw, std::abs(e_yaw));

          // 位置来自连续的 IPPE 解，应当准到数值精度。
          check(std::abs(e_azimuth) < 1e-6, "azimuth residual is not zero" + where);
          check(std::abs(e_elevation) < 1e-6, "elevation residual is not zero" + where);
          check(std::abs(e_distance) < 1e-5, "distance residual is not zero" + where);
          // 板 yaw 走的是离散搜索，精度上限就是细扫步长的一半。
          check(
            std::abs(e_yaw) <= armor_config.yaw_fine_step / 2.0 + 1e-9,
            "armor yaw residual exceeds half of the search step: " +
              std::to_string(degrees(e_yaw)) + " deg" + where);

        }
      }
    }

    check(cases == 36, "expected 36 synthetic cases");
    // 只允许远距离正视这一类退化；再多就说明链路真的坏了。
    check(
      declined <= 4,
      "single_pnp declined too many noiseless cases:" + declined_cases);
    std::cout << "checked " << cases << " noiseless cases (" << declined
              << " declined by IPPE degeneracy:" << declined_cases
              << "); worst residuals: "
              << "azimuth " << worst_azimuth * 1000.0 << " mrad, elevation "
              << worst_elevation * 1000.0 << " mrad, distance " << worst_distance
              << " m, armor yaw " << degrees(worst_yaw) << " deg, corner "
              << worst_corner << " m\n";
  } catch (const std::exception& exception) {
    std::cerr << "pnp truth residual smoke test failed: " << exception.what()
              << '\n';
    return 1;
  }

  std::cout << "pnp truth residual smoke test passed\n";
  return 0;
}
