#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <Eigen/Geometry>
#include <optional>

namespace L1Sensor::Buff {

struct CameraPose {
  Eigen::Quaterniond q_world_camera{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d t_world_camera{Eigen::Vector3d::Zero()};
};

// SerialWorker 提供 world<-barrel。Buff PnP 会先把 OpenCV 光学坐标转换成
// x前/y左/z上的 camera_ros 坐标，因此这里要把标定的 optical->barrel 外参
// 与 camera_ros->optical 固定换轴组合一次，不能把光学换轴重复施加到 PnP 结果。
[[nodiscard]] std::optional<CameraPose> composeCameraPose(
  const Eigen::Quaterniond& q_world_barrel,
  const CameraCalibration& calibration) noexcept;

}  // namespace L1Sensor::Buff
