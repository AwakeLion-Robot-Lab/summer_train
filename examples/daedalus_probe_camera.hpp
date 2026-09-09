#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/simulator/daedalus_client.hpp"
#include <Eigen/Geometry>
#include <optional>
#include <stdexcept>

namespace DaedalusProbe {

inline bool cameraNearBarrel(const L1Sensor::DaedalusFrame& frame)
{
  // 当前 IPC 没有发布自由相机的朝向。只支持 Robot 视角；用与枪口的距离
  // 拦住启动远景和常规第三人称视角，不能把它们当成枪管上的光学相机。
  const auto& camera = frame.pose(L1Sensor::DaedalusPoseKind::Camera).position;
  const auto& muzzle = frame.pose(L1Sensor::DaedalusPoseKind::Muzzle).position;
  const Eigen::Vector3d offset{
    camera[0] - muzzle[0], camera[1] - muzzle[1], camera[2] - muzzle[2]};
  return offset.allFinite() && offset.norm() < 1.0;
}

inline L1Sensor::CameraCalibration makeCalibration(
  const L1Sensor::DaedalusFrame& frame)
{
  const auto& camera = frame.camera_info;
  if (!camera.valid() ||
      camera.width != static_cast<std::uint32_t>(frame.image_bgr.cols) ||
      camera.height != static_cast<std::uint32_t>(frame.image_bgr.rows)) {
    throw std::runtime_error("Daedalus camera metadata is missing or inconsistent");
  }

  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {
    static_cast<int>(camera.width), static_cast<int>(camera.height)};
  calibration.camera_matrix = (
    cv::Mat_<double>(3, 3) <<
      camera.fx, 0.0, camera.cx,
      0.0, camera.fy, camera.cy,
      0.0, 0.0, 1.0);
  calibration.distortion_coefficients = cv::Mat(1, 5, CV_64FC1);
  for (std::size_t index = 0; index < camera.distortion.size(); ++index) {
    calibration.distortion_coefficients.at<double>(0, static_cast<int>(index)) =
      camera.distortion[index];
  }

  // Daedalus publishes the Camera slot translation in barrel coordinates.
  // Optical axes are z-forward/x-right/y-down, while newvision's barrel axes
  // are x-forward/y-left/z-up, hence p_barrel = [z, -x, -y].
  Eigen::Isometry3d T_barrel_camera = Eigen::Isometry3d::Identity();
  T_barrel_camera.linear() <<
     0.0,  0.0,  1.0,
    -1.0,  0.0,  0.0,
     0.0, -1.0,  0.0;
  const auto& camera_pose = frame.pose(L1Sensor::DaedalusPoseKind::Camera);
  T_barrel_camera.translation() = Eigen::Vector3d{
    camera_pose.position[0], camera_pose.position[1], camera_pose.position[2]};
  if (!T_barrel_camera.matrix().allFinite()) {
    throw std::runtime_error("Daedalus camera extrinsics contain a non-finite value");
  }
  calibration.T_barrel_camera = T_barrel_camera;
  return calibration;
}

inline std::optional<Eigen::Quaterniond> barrelPose(
  const L1Sensor::DaedalusFrame& frame)
{
  const auto& source = frame.pose(L1Sensor::DaedalusPoseKind::Gimbal).quaternion;
  Eigen::Quaterniond pose{source[0], source[1], source[2], source[3]};
  if (!pose.coeffs().allFinite() || pose.squaredNorm() <= 1e-12) {
    return std::nullopt;
  }
  pose.normalize();
  return pose;
}


}  // namespace DaedalusProbe
