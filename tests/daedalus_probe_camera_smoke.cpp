#include "../examples/daedalus_probe_camera.hpp"
#include "l3_estimation/armor/tracker.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message)
{
  if (!ok) throw std::runtime_error(message);
}

L2Perception::Armor detection(
  const L1Sensor::CameraCalibration& calibration, const Eigen::Quaterniond& pose,
  const Eigen::Vector3d& point, double yaw)
{
  L3Estimation::PnpSolver projector(calibration);
  projector.set_R_world_barrel(pose);
  const auto pixels = projector.reproject_armor(
    point, yaw, L3Estimation::ArmorType::Small, L3Estimation::ArmorName::Infantry3);
  require(pixels.size() == 4, "projection failed");
  L2Perception::Armor armor;
  armor.class_id = static_cast<int>(L2Perception::ArmorClass::Infantry3);
  armor.confidence = 0.99F;
  for (int i = 0; i < 4; ++i) {
    armor.corners[i] = pixels[i];
    armor.center += pixels[i] * 0.25F;
  }
  return armor;
}
}  // namespace

int main()
{
  try {
    L1Sensor::DaedalusFrame frame;
    frame.image_bgr = cv::Mat(1080, 1440, CV_8UC3);
    frame.camera_info = {1, 1303.675, 1303.675, 720, 540, {}, 1440, 1080};
    frame.poses[2].position = {0.05F, 0.0F, 0.09F};
    frame.poses[3].position = {-15.0F, 0.0F, 10.0F};
    require(!DaedalusProbe::cameraNearBarrel(frame), "startup overview must be rejected");
    const auto stale = DaedalusProbe::makeCalibration(frame);
    L3Estimation::Tracker tracker(stale);
    frame.poses[3].position = {0.137F, 0.002F, 0.152F};
    require(DaedalusProbe::cameraNearBarrel(frame), "mounted camera must be accepted");
    const Eigen::Quaterniond pose(Eigen::AngleAxisd(-1.81, Eigen::Vector3d::UnitZ()));
    frame.poses[0].quaternion = {float(pose.w()), float(pose.x()), float(pose.y()), float(pose.z())};
    const Eigen::Vector3d point = pose * Eigen::Vector3d(4.0, 0.25, 0.03);
    const auto now = L3Estimation::TimePoint{};
    for (int i = 0; i < 12; ++i) {
      // 同一目标，相机安装位置和焦距发生变化；更新后的 PnP 必须保持世界点一致。
      frame.poses[3].position[1] += 0.005F;
      frame.camera_info.fx += 2.0;
      const auto calibration = DaedalusProbe::makeCalibration(frame);
      require(tracker.setCalibration(calibration), "current calibration rejected");
      auto armor = detection(calibration, pose, point, -1.6);
      const auto target = tracker.track({armor}, pose, now + std::chrono::milliseconds(10 * i));
      require(target.has_value(), "camera refresh lost the target");
      require(tracker.resetCount() == 0, "valid camera updates must preserve tracking");
      require((tracker.observations().front().xyz_in_world - point).norm() < 1e-4,
        "stale startup extrinsics corrupted the world position");
      const auto reconstructed = DaedalusProbe::barrelPose(frame);
      require(reconstructed.has_value() &&
        (reconstructed->toRotationMatrix() - pose.toRotationMatrix()).norm() < 1e-6,
        "simulator yaw quaternion convention changed");
    }
    auto invalid = DaedalusProbe::makeCalibration(frame);
    invalid.T_barrel_camera.reset();
    require(!tracker.setCalibration(invalid) && !tracker.ready(),
      "invalid calibration must not reuse old extrinsics");
    require(tracker.state() == L3Estimation::TrackState::Lost,
      "invalid calibration must clear target history");
    std::cout << "Daedalus camera smoke passed: startup view, per-frame calibration, stable target\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
