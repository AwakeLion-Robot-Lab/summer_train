#include "l5_telemetry/aim_overlay.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>

namespace VM = L3Estimation::VehicleModel;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

L1Sensor::CameraCalibration calibration() {
  L1Sensor::CameraCalibration c;
  c.image_size = {1440, 1080};
  c.camera_matrix = (cv::Mat_<double>(3, 3) << 1210, 0, 721.5, 0, 1208, 539.5, 0, 0, 1);
  c.distortion_coefficients = (cv::Mat_<double>(1, 5) << -0.12, 0.03, 0.0004, -0.0002, 0);
  Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
  extrinsic.linear() = Eigen::Matrix3d{{0, 0, 1}, {-1, 0, 0}, {0, -1, 0}};
  extrinsic.translation() = Eigen::Vector3d(0.02, -0.01, 0.06);
  c.T_barrel_camera = extrinsic;
  return c;
}

// An independent OpenCV projection of full physical plate poses. This verifies
// winding, dimensions, distortion, exposure pose and preservation of roll/pitch.
cv::Mat reference(const L3Estimation::EskfTarget& target,
                  const L1Sensor::CameraCalibration& c,
                  const Eigen::Quaterniond& q, cv::Point offset) {
  cv::Mat result(c.image_size, CV_8UC3, cv::Scalar::all(0));
  const auto camera = L3Estimation::cameraInWorld(c, q);
  const auto& geometry = target.config().armor;
  const bool big = L3Estimation::armorTypeOf(target.name) == L3Estimation::ArmorType::Big;
  const float w = (big ? geometry.big_width : geometry.small_width) / 2;
  const float h = geometry.height / 2;
  const std::vector<cv::Point3f> object{{0,w,h}, {0,-w,h}, {0,-w,-h}, {0,w,-h}};
  for (int id = 0; id < target.armor_num(); ++id) {
    const Eigen::Isometry3d pose = camera.inverse() *
      VM::armorPose<double>(target.rawState().data(), id, target.armor_num(), target.name);
    cv::Mat rotation;
    cv::eigen2cv(Eigen::Matrix3d(pose.linear()), rotation);
    cv::Vec3d rvec;
    cv::Rodrigues(rotation, rvec);
    std::vector<cv::Point2f> corners;
    cv::projectPoints(object, rvec,
      cv::Vec3d(pose.translation().x(), pose.translation().y(), pose.translation().z()),
      c.camera_matrix, c.distortion_coefficients, corners);
    for (int i = 0; i < 4; ++i) {
      cv::line(result, L6Telemetry::toPixel(corners[i]) + offset,
        L6Telemetry::toPixel(corners[(i + 1) % 4]) + offset, {0,255,0}, 2, cv::LINE_AA);
    }
  }
  return result;
}
} // namespace

int main() {
  try {
    const auto c = calibration();
    for (const auto name : {L3Estimation::ArmorName::Infantry3,
                           L3Estimation::ArmorName::Hero,
                           L3Estimation::ArmorName::Outpost}) {
      for (const bool tilted : {false, true}) {
        L3Estimation::EskfTarget target(name, 3.0, 0.0, 0.27, 0.3, 0.04);
        // Test fixture only: the synthetic target has no live filter; initialize
        // its full attitude without adding a mutable production state API.
        auto& state = const_cast<L3Estimation::EskfTarget::State&>(target.rawState());
        state[VM::idx::ROT_X] = tilted ? 0.24 : 0;
        state[VM::idx::ROT_Y] = tilted ? -0.17 : 0;
        const auto before = target.rawState();
        const auto before_time = target.t();
        for (const bool moving_camera : {false, true}) {
          const Eigen::Quaterniond q = moving_camera
            ? Eigen::Quaterniond(Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ()) *
                                 Eigen::AngleAxisd(-0.06, Eigen::Vector3d::UnitY()))
            : Eigen::Quaterniond::Identity();
          const cv::Point offset{11, -17};
          const auto expected = reference(target, c, q, offset);
          cv::Mat actual(c.image_size, CV_8UC3, cv::Scalar::all(0));
          L6Telemetry::drawTrackedVehicle(actual, target, c, q, offset);
          require(cv::norm(actual, expected, cv::NORM_INF) == 0,
                  "green corners differ from full-pose OpenCV projection");
          require(cv::countNonZero(actual.reshape(1)) > 0, "fixture drew no green plate");

          L3Estimation::PnpSolver solver(c, target.config().armor);
          solver.set_R_world_barrel(q);
          if (tilted && name != L3Estimation::ArmorName::Outpost) {
            cv::Mat legacy(c.image_size, CV_8UC3, cv::Scalar::all(0));
            L6Telemetry::drawVehicle(legacy, target.armor_xyza_list(),
              *L3Estimation::armorTypeOf(name), name, solver, {0,255,0}, 2, offset);
            require(cv::norm(legacy, expected, cv::NORM_INF) > 0,
                    "tilted fixture does not distinguish old yaw-only rendering");
          }

          std::optional<L3Estimation::EskfTarget> snapshot = target.snapshot();
          std::optional<Eigen::Quaterniond> pose = q;
          L4Planning::AimPlan plan;
          L5Control::FireDecision fire;
          const std::vector<L2Perception::Armor> detections;
          const std::vector<L3Estimation::Armor> observations;
          cv::Mat overlay(c.image_size, CV_8UC3, cv::Scalar::all(0));
          L6Telemetry::drawAimOverlay(overlay,
            {detections, observations, snapshot, L3Estimation::TrackState::Tracking, plan, fire, pose},
            solver, c);
          const auto expected_overlay = reference(target, c, q, {});
          // Status text occupies the top strip; plate pixels are below it.
          require(cv::norm(overlay.rowRange(80, overlay.rows),
                           expected_overlay.rowRange(80, overlay.rows), cv::NORM_INF) == 0,
                  "production overlay still uses simplified pose");
          pose.reset(); // Solver deliberately retains its previous pose.
          overlay.setTo(cv::Scalar::all(0));
          L6Telemetry::drawAimOverlay(overlay,
            {detections, observations, snapshot, L3Estimation::TrackState::Tracking, plan, fire, pose},
            solver, c);
          require(cv::countNonZero(overlay.rowRange(80, overlay.rows).reshape(1)) == 0,
                  "missing exposure pose reused stale green geometry");
        }
        require((target.rawState() - before).norm() == 0 && target.t() == before_time,
                "drawing mutated or advanced the filter state");
      }
    }
    cv::Mat image(c.image_size, CV_8UC3, cv::Scalar::all(0));
    L3Estimation::EskfTarget behind(L3Estimation::ArmorName::Infantry3, -3, 0, 0.27);
    L6Telemetry::drawTrackedVehicle(image, behind, c, Eigen::Quaterniond::Identity());
    require(cv::countNonZero(image.reshape(1)) == 0, "behind-camera plates were drawn");
    L3Estimation::EskfTarget target(L3Estimation::ArmorName::Infantry3, 3, 0, 0.27);
    auto missing = c;
    missing.T_barrel_camera.reset();
    L6Telemetry::drawTrackedVehicle(image, target, missing, Eigen::Quaterniond::Identity());
    require(cv::countNonZero(image.reshape(1)) == 0, "missing extrinsics drew green plates");
    std::cout << "aim overlay full-pose projection smoke passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
