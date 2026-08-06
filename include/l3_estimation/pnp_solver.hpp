#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <optional>
#include <vector>

namespace L3Estimation {

// 将二维装甲板角点恢复为相机位姿，并转换到枪管系和世界系。
// 求解器同时提供基于重投影误差的世界系 yaw 优化。
class PnpSolver {
public:
  // 标定或配置无效时对象仍可构造，但 ready() 返回 false。
  explicit PnpSolver(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig config = {});

  // 返回相机标定、静态外参和装甲板配置是否可用于求解。
  [[nodiscard]] bool ready() const noexcept;

  // 姿态表示枪管坐标系到世界坐标系的旋转，必须对应图像曝光时刻。
  void set_R_world_barrel(
    const std::optional<Eigen::Quaterniond>& barrel_pose);

  // 原地补充 Armor 的相机系和世界系 PnP 结果；失败时质量标志保持无效。
  void single_pnp(Armor& armor) const;

  // 校验并替换相机标定，同时缓存 camera -> barrel 外参。
  [[nodiscard]] bool setCalibration(
    const L1Sensor::CameraCalibration& calibration);

  // 将给定世界系装甲板重投影到图像；前置条件不满足时返回空数组。
  [[nodiscard]] std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d& xyz_in_world,
    double yaw,
    ArmorType type,
    ArmorName name) const;

private:
  // 在枪管朝向附近搜索使角点重投影误差最小的世界系 yaw。
  void optimize_yaw(Armor& armor) const;

  // 返回四角点重投影距离之和；无法重投影时返回正无穷。
  [[nodiscard]] double armor_reprojection_error(
    const Armor& armor,
    double yaw) const;

  // 静态 camera -> barrel 外参，以及逐帧更新的 barrel -> world 旋转。
  L1Sensor::CameraCalibration calibration_;
  Eigen::Matrix3d R_camera2barrel_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_camera2barrel_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_barrel2world_{Eigen::Matrix3d::Identity()};
  ArmorConfig config_;
  bool world_barrel_ready_{false};
  bool ready_{false};
};

}  // namespace L3Estimation
