// 调试叠加层的行为冒烟测试。它挡的是一个静默失败：叠加层什么都不画
// （求解器没就绪、重投影返回空、目标为 nullopt），画面看着"没识别到"，
// 而真正的检测和滤波其实是好的。不依赖相机、串口和推理后端。

#include "l6_telemetry/aim_overlay.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace {

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "aim overlay smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

// 画上去的像素数。原图是纯黑，任何非零像素都来自叠加层。
int inkedPixels(const cv::Mat & image)
{
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  return cv::countNonZero(gray);
}

}  // namespace

int main()
{
  try {
    const auto camera_config =
      YAML::LoadFile("tests/data/camera_calibration_inline.yaml");
    const auto calibration = L1Sensor::loadCameraCalibration(
      camera_config["calibration"], "aim overlay smoke config");

    const L3Estimation::ArmorConfig armor_config;
    L3Estimation::PnpSolver solver(calibration, armor_config);
    require(solver.ready(), "PnpSolver rejected the test calibration");

    // 枪管与世界系对齐，正前方 3 m 处放一辆四板车。
    const Eigen::Quaterniond q_world_barrel = Eigen::Quaterniond::Identity();
    solver.set_R_world_barrel(q_world_barrel);
    const std::optional<Eigen::Quaterniond> pose = q_world_barrel;

    L3Estimation::EskfTarget target(
      L3Estimation::ArmorName::Infantry3, 3.0, 0.0, 0.2);
    const std::optional<L3Estimation::EskfTarget> tracked = target;

    L4Planning::Plan plan;
    plan.status = L4Planning::PlanStatus::FireReady;
    plan.reason = L4Planning::PlanError::None;
    plan.fire = L4Planning::FireReference{0, target.armor_xyza_list().front()};

    L5Control::FireDecision decision;
    decision.reasons.push_back(L5Control::RejectReason::ShootDisabled);

    const std::vector<L2Perception::Armor> no_detections;

    cv::Mat image(
      calibration.image_size.height, calibration.image_size.width, CV_8UC3,
      cv::Scalar::all(0));
    const int before = inkedPixels(image);
    require(before == 0, "the test canvas must start black");

    L6Telemetry::drawAimOverlay(
      image,
      {.detections = no_detections,
       .target = tracked,
       .track_state = L3Estimation::TrackState::Tracking,
       .plan = plan,
       .fire = decision,
       .q_world_barrel = pose},
      solver, calibration);

    const int after = inkedPixels(image);
    require(after > before, "the overlay drew nothing at all");
    // 一辆四板车 + 命中板 + 一行状态文字，远不止几十个像素。
    require(after > 500, "the overlay drew far less than a vehicle outline");

    // 没有目标时只该画状态文字，不该画整车。
    cv::Mat blank(
      calibration.image_size.height, calibration.image_size.width, CV_8UC3,
      cv::Scalar::all(0));
    const std::optional<L3Estimation::EskfTarget> no_target;
    L4Planning::Plan rejected;
    L6Telemetry::drawAimOverlay(
      blank,
      {.detections = no_detections,
       .target = no_target,
       .track_state = L3Estimation::TrackState::Lost,
       .plan = rejected,
       .fire = decision,
       .q_world_barrel = pose},
      solver, calibration);
    const int lost_ink = inkedPixels(blank);
    require(lost_ink > 0, "the status line must be drawn even without a target");
    require(lost_ink < after, "a lost frame must draw less than a tracked one");

    std::cout << "aim overlay smoke passed (tracked " << after << " px, lost "
              << lost_ink << " px)\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "aim overlay smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
