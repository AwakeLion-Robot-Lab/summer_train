#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <optional>

namespace L3Estimation {

class PnpSolver {
public:
  explicit PnpSolver(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig config = {});

  [[nodiscard]] bool ready() const noexcept;

  // 姿态表示枪管坐标系到世界坐标系的旋转，必须对应图像曝光时刻。
  void set_R_world_barrel(
    const std::optional<Eigen::Quaterniond>& barrel_pose);

  // 原地补充 Armor 的相机系和世界系 PnP 结果。
  void single_pnp(Armor& armor) const;

private:
  L1Sensor::CameraCalibration calibration_;
  ArmorConfig config_;
  Eigen::Matrix3d R_world_barrel_{Eigen::Matrix3d::Identity()};
  bool world_barrel_ready_{false};
  bool ready_{false};
};

}  // namespace L3Estimation
