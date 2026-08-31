// 误差状态整车目标的初始化、关联与闭环检查。
//
// 用合成真值驱动：由一个已知的十三维整车状态投影出装甲板角点当作检测，喂进
// EskfTarget，看它能否初始化、把观测关联到正确的板号、并收敛回真值。不经过
// 相机、网络和 PnP，所以失败一定出在观测模型、关联或 ⊞/⊟ 上。
//
// 注意整车模型对 yaw 有 2π/N 的对称性：滤波器的"0 号板"未必是真值的 0 号板，
// 所以收敛断言写在**装甲板集合**上，关联断言写在**相对编号**上。

#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <numbers>
#include <optional>
#include <string_view>
#include <vector>

namespace VM = L3Estimation::VehicleModel;

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

void expectNear(double actual, double expected, double tolerance, std::string_view message)
{
  if (!(std::abs(actual - expected) <= tolerance)) {
    std::cerr << "FAIL: " << message << "  actual=" << actual << " expected=" << expected
              << " diff=" << std::abs(actual - expected) << '\n';
    ++failure_count;
  }
}

constexpr auto kName = L3Estimation::ArmorName::Infantry3;
constexpr int kArmorNum = 4;
constexpr double kDt = 0.005;
using State = Eigen::Matrix<double, VM::kStateSize, 1>;

L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = cv::Size(1440, 1080);
  calibration.camera_matrix =
    (cv::Mat_<double>(3, 3) << 1210.0, 0.0, 721.5, 0.0, 1208.0, 539.5, 0.0, 0.0, 1.0);
  calibration.distortion_coefficients =
    (cv::Mat_<double>(1, 5) << -0.12, 0.03, 0.0004, -0.0002, 0.0);
  return calibration;
}

Eigen::Isometry3d makeCameraPose()
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = Eigen::Matrix3d{{0.0, 0.0, 1.0}, {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}};
  pose.translation() = Eigen::Vector3d{0.02, -0.01, 0.06};
  return pose;
}

State makeTruth()
{
  State x = State::Zero();
  x[VM::idx::CX] = 3.2;
  x[VM::idx::CY] = 0.40;
  x[VM::idx::CZ] = 0.08;
  x[VM::idx::VCX] = 0.35;
  x[VM::idx::VCY] = -0.15;
  x[VM::idx::ROT_Z] = 0.25;
  x[VM::idx::VYAW] = 1.8;
  x[VM::idx::LOG_R1] = std::log(0.265);
  x[VM::idx::LOG_R2] = std::log(0.285);
  x[VM::idx::HEIGHT] = 0.04;
  return x;
}

bool facesCamera(const State & x, int id, const Eigen::Isometry3d & camera)
{
  const auto pose_in_world = VM::armorPose<double>(x.data(), id, kArmorNum, kName);
  const Eigen::Isometry3d pose_in_camera = camera.inverse() * pose_in_world;
  return (-pose_in_camera.linear().col(0)).dot(-pose_in_camera.translation()) > 0.0;
}

// 由真值状态合成一块板的检测：四角来自投影，位姿字段模拟 PnP 的输出。
L3Estimation::Armor synthesizeDetection(
  const State & x, int id, const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera, const L3Estimation::ArmorConfig & armor_config,
  L3Estimation::TimePoint timestamp)
{
  L3Estimation::Armor armor;
  armor.name = kName;
  armor.type = L3Estimation::ArmorType::Small;
  armor.timestamp = timestamp;

  L3Estimation::UvlContext ctx;
  ctx.armor_num = kArmorNum;
  ctx.id = id;
  ctx.name = kName;
  ctx.armor_config = armor_config;
  ctx.camera_in_world = camera;
  ctx.camera_matrix = calibration.camera_matrix;
  ctx.distortion_coefficients = calibration.distortion_coefficients;

  ctx.is_left = true;
  const auto left = L3Estimation::UvlMeasure{ctx}.projectedPoints(x);
  ctx.is_left = false;
  const auto right = L3Estimation::UvlMeasure{ctx}.projectedPoints(x);

  // 左上、右上、右下、左下
  armor.points = {left.first, right.first, right.second, left.second};
  armor.center = (armor.points[0] + armor.points[1] + armor.points[2] + armor.points[3]) / 4.0F;

  const auto pose = VM::armorPose<double>(x.data(), id, kArmorNum, kName);
  armor.xyz_in_world = pose.translation();
  armor.ypr_in_world = L6Telemetry::rotationToYpr(pose.linear());
  return armor;
}

}  // namespace

int main()
{
  const L1Sensor::CameraCalibration calibration = makeCalibration();
  const Eigen::Isometry3d camera = makeCameraPose();
  const State truth = makeTruth();

  L3Estimation::EskfTargetConfig config;
  config.iteration_num = 5;

  const auto start = L3Estimation::TimePoint{} + std::chrono::seconds(1);

  // --- 1. 由一块板反推整车 ------------------------------------------
  {
    L3Estimation::EskfTarget target;
    const auto detection =
      synthesizeDetection(truth, 0, calibration, camera, config.armor, start);
    target.reset(detection, config, start, calibration, camera);

    expect(target.initialized(), "reset 后应当已初始化");
    expect(!target.jumped, "reset 后 jumped 必须为 false");
    expect(target.armor_num() == kArmorNum, "板数推断错误");

    // 假设的是 0 号板且半径取先验 0.26（真值 0.265），所以车心应当很接近。
    const Eigen::VectorXd state = target.ekf_x();
    const Eigen::Vector3d center(
      state[VM::idx::CX], state[VM::idx::CY], state[VM::idx::CZ]);
    const Eigen::Vector3d truth_center(
      truth[VM::idx::CX], truth[VM::idx::CY], truth[VM::idx::CZ]);
    expect((center - truth_center).norm() < 0.02, "由 0 号板反推的车心偏差过大");

    // ekf_x 必须吐线性半径而不是对数 —— L4 按 abs(x[8]) <= 2.0 判物理半径。
    expectNear(
      state[VM::idx::LOG_R1], config.initial_radius, 1e-12,
      "ekf_x 的第 8 维应当是线性半径");
    expect(target.rawState()[VM::idx::LOG_R1] < 0.0, "内部应当仍存对数半径");

    // Tracker 下发的是不带 filter_ 的 snapshot；L4 必须能直接消费这份 IESKF
    // 目标并在副本上做命中时刻外推。
    L4Planning::Planner planner;
    L1Sensor::RobotState robot_state;
    robot_state.bullet_speed = 23.0;
    const auto plan = planner.plan(
      std::optional<L3Estimation::EskfTarget>{target.snapshot()}, robot_state,
      start, false);
    expect(plan.valid(), "L4 Planner 未接受 IESKF 目标快照");

    // 认错板号只是标签的循环平移：从 2 号板初始化，几何仍自洽，只是整车 yaw
    // 差 π。装甲板集合应当能对上。
    L3Estimation::EskfTarget from_other;
    const auto other_detection =
      synthesizeDetection(truth, 2, calibration, camera, config.armor, start);
    from_other.reset(other_detection, config, start, calibration, camera);
    const auto poses = from_other.armor_xyza_list();
    expect(poses.size() == static_cast<std::size_t>(kArmorNum), "板位姿列表长度错");
    double best = std::numeric_limits<double>::max();
    for (const auto & pose : poses) {
      best = std::min(best, (pose.head<3>() - other_detection.xyz_in_world).norm());
    }
    expect(best < 0.02, "从 2 号板初始化后，板集合里应当有一块落在观测处");
  }

  // --- 2. 关联把观测分到正确的板号 -----------------------------------
  {
    L3Estimation::EskfTarget target;
    const auto detection =
      synthesizeDetection(truth, 0, calibration, camera, config.armor, start);
    target.reset(detection, config, start, calibration, camera);

    // 同一时刻、同一状态下合成所有可见板的检测。
    std::vector<L3Estimation::Armor> detections;
    std::vector<int> truth_ids;
    for (int id = 0; id < kArmorNum; ++id) {
      if (!facesCamera(truth, id, camera)) {
        continue;
      }
      detections.push_back(
        synthesizeDetection(truth, id, calibration, camera, config.armor, start));
      truth_ids.push_back(id);
    }
    expect(detections.size() >= 2, "合成场景里应当至少有两块板可见");

    const auto matched = target.matchArmor(detections, start, calibration, camera);
    expect(matched.size() == detections.size(), "关联数量与可见板数不符");

    // 初始化时假设看到的是 0 号板，所以关联结果应当与真值编号一致。
    for (std::size_t k = 0; k < matched.size(); ++k) {
      bool found = false;
      for (std::size_t m = 0; m < detections.size(); ++m) {
        if (cv::norm(matched[k].second.points[0] - detections[m].points[0]) < 1e-6) {
          expect(matched[k].first == truth_ids[m], "关联把观测分到了错误的板号");
          found = true;
          break;
        }
      }
      expect(found, "关联结果里出现了不存在的观测");
    }
  }

  // --- 3. 独立灯条与单板深度差组合观测 -------------------------------
  {
    L3Estimation::EskfTarget target;
    const auto detection =
      synthesizeDetection(truth, 0, calibration, camera, config.armor, start);
    target.reset(detection, config, start, calibration, camera);

    const auto matched = target.matchArmor(
      std::vector<L3Estimation::Armor>{detection}, start, calibration, camera);
    expect(matched.size() == 1, "单板场景应当关联到一块完整板");

    const int id = matched.empty() ? 0 : matched.front().first;
    const auto predicted_light = target.predictLight(
      id, true, target.rawState(), calibration, camera);
    L2Perception::Light light;
    light.top = predicted_light.first;
    light.bottom = predicted_light.second;
    light.center = (light.top + light.bottom) * 0.5F;
    light.length = cv::norm(light.top - light.bottom);
    light.width = light.length * 0.1;
    light.color = L2Perception::ArmorColor::Blue;

    const auto matched_lights = target.matchLight(
      std::vector<L2Perception::Light>{light}, matched, start, calibration, camera);
    expect(matched_lights.size() == 1, "独立灯条没有关联到预测物理灯条");
    expect(
      target.matchLight(
        std::vector<L2Perception::Light>{light}, {}, start, calibration, camera)
        .empty(),
      "没有完整板关联时不应启用独立灯条关联");

    L3Estimation::UvlContext depth_context;
    depth_context.armor_num = target.armor_num();
    depth_context.id = id;
    depth_context.name = target.name;
    depth_context.armor_config = config.armor;
    depth_context.camera_in_world = camera;
    depth_context.camera_matrix = calibration.camera_matrix;
    depth_context.distortion_coefficients = calibration.distortion_coefficients;
    double depth_difference_data[1]{};
    L3Estimation::DepthDiffMeasure{depth_context}(
      target.rawState().data(), depth_difference_data);

    const int observation_blocks = target.update(
      matched, matched_lights, depth_difference_data[0], start, calibration, camera);
    expect(
      observation_blocks == 4,
      "单完整板 + 独立灯条 + 深度差应产生四个观测块");
    expect(
      target.lastNisDof() == 13,
      "两条板灯 UVL、独立灯条 UVL 和一维深度差应合计 13 维");
  }

  // --- 4. 闭环收敛 ---------------------------------------------------
  {
    L3Estimation::EskfTarget target;
    const auto detection =
      synthesizeDetection(truth, 0, calibration, camera, config.armor, start);
    target.reset(detection, config, start, calibration, camera);

    State truth_now = truth;
    const VM::Motion truth_motion{.dt = kDt, .name = kName};
    auto now = start;

    for (int step = 0; step < 600; ++step) {
      State next;
      truth_motion(truth_now.data(), next.data());
      truth_now = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));

      std::vector<L3Estimation::Armor> detections;
      for (int id = 0; id < kArmorNum; ++id) {
        if (!facesCamera(truth_now, id, camera)) {
          continue;
        }
        detections.push_back(
          synthesizeDetection(truth_now, id, calibration, camera, config.armor, now));
      }

      target.predictEkf(now);
      const auto matched = target.matchArmor(detections, now, calibration, camera);
      target.update(matched, now, calibration, camera);
    }

    expect(target.jumped, "跑完 600 帧后应当已经见过 0 号以外的板");
    expect(!target.diverged(), "滤波器发散了");
    expect(target.converged(), "目标未标记为收敛");

    const Eigen::VectorXd estimate = target.rawState();
    double max_armor_error = 0.0;
    for (int id = 0; id < kArmorNum; ++id) {
      const auto truth_pose =
        VM::armorPose<double>(truth_now.data(), id, kArmorNum, kName);
      double best = std::numeric_limits<double>::max();
      for (int other = 0; other < kArmorNum; ++other) {
        const auto estimate_pose =
          VM::armorPose<double>(estimate.data(), other, kArmorNum, kName);
        best =
          std::min(best, (truth_pose.translation() - estimate_pose.translation()).norm());
      }
      max_armor_error = std::max(max_armor_error, best);
    }
    expect(max_armor_error < 0.02, "闭环后装甲板位置未收敛");
    if (max_armor_error >= 0.02) {
      std::cerr << "  max_armor_error = " << max_armor_error << " m\n";
    }

    expectNear(
      estimate[VM::idx::CX], truth_now[VM::idx::CX], 0.02, "闭环后车心 x 未收敛");
    expectNear(
      std::abs(estimate[VM::idx::VYAW]), std::abs(truth[VM::idx::VYAW]), 0.35,
      "闭环后角速度未收敛");
    // 半径本来就靠观测拉回来：初值给的是 0.26，真值 0.265 / 0.285。
    expect(
      std::exp(estimate[VM::idx::LOG_R1]) > 0.2 &&
        std::exp(estimate[VM::idx::LOG_R1]) < 0.35,
      "半径被推到了物理范围之外");
  }

  // --- 5. snapshot 不携带滤波器 --------------------------------------
  {
    L3Estimation::EskfTarget target;
    const auto detection =
      synthesizeDetection(truth, 0, calibration, camera, config.armor, start);
    target.reset(detection, config, start, calibration, camera);

    // reset 之后速度与角速度都是零，外推不改变任何东西，测不出隔离性。先跑
    // 几十帧让状态动起来。
    State truth_now = truth;
    const VM::Motion truth_motion{.dt = kDt, .name = kName};
    auto now = start;
    for (int step = 0; step < 80; ++step) {
      State next;
      truth_motion(truth_now.data(), next.data());
      truth_now = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));

      std::vector<L3Estimation::Armor> detections;
      for (int id = 0; id < kArmorNum; ++id) {
        if (facesCamera(truth_now, id, camera)) {
          detections.push_back(
            synthesizeDetection(truth_now, id, calibration, camera, config.armor, now));
        }
      }
      target.predictEkf(now);
      target.update(target.matchArmor(detections, now, calibration, camera), now,
                    calibration, camera);
    }
    expect(
      std::abs(target.rawState()[VM::idx::VYAW]) > 0.5,
      "跑了 80 帧后角速度仍接近零，后面的隔离性测试没有意义");

    const State before = target.rawState();
    const auto before_time = target.t();

    L3Estimation::EskfTarget copy = target.snapshot();
    copy.predict(0.5);  // 在副本上大幅外推

    expect(
      (copy.rawState() - target.rawState()).norm() > 1e-6, "副本外推后应当与原目标不同");
    expect(target.t() == before_time, "在副本上外推不应改变原目标的时间戳");
    expect(
      (target.rawState() - before).norm() < 1e-15, "原目标的状态被副本操作污染了");
    expect(copy.t() > before_time, "副本的时间戳应当已经推进");
  }

  if (failure_count != 0) {
    std::cerr << "eskf target smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "eskf target smoke test passed\n";
  return 0;
}
