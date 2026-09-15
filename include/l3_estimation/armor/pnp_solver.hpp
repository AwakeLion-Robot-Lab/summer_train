#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/types.hpp"

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
  bool ready() const noexcept;

  // 姿态表示枪管坐标系到世界坐标系的旋转，必须对应图像曝光时刻。
  void set_R_world_barrel(
    const std::optional<Eigen::Quaterniond>& barrel_pose);

  // 原地补充 Armor 的相机系和世界系 PnP 结果；失败时质量标志保持无效。
  void single_pnp(Armor& armor) const;

  // Awakening 单完整板约束专用：IPPE 求全部候选解，选择板正面朝向相机的
  // 一支，只返回左右灯条中心在相机 z 轴上的深度差；前哨沿用其固定俯仰
  // 与黄金分割 yaw 修正分支。
  std::optional<double> armor_lights_depth_difference(
    const Armor& armor) const;

  // 校验并替换相机标定，同时缓存 camera -> barrel 外参。
  [[nodiscard]] bool setCalibration(
    const L1Sensor::CameraCalibration& calibration);

  // 将给定世界系装甲板重投影到图像；前置条件不满足时返回空数组。
  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d& xyz_in_world,
    double yaw,
    ArmorType type,
    ArmorName name) const;

private:
  // 复刻 sp_vision：以枪管 yaw 为中心，在左右各 70 度内按 1 度步长枚举，
  // 用四角点重投影距离之和选择装甲板世界系 yaw。
  void optimize_yaw(Armor& armor) const;

  // 单板在给定世界系 yaw 下的四角点重投影代价，与 optimize_yaw 共用同一支：
  // 重投影不可用时返回无穷，使调用方的比较自然跳过该采样点。
  double yaw_cost(const Armor& armor, double yaw) const;

  // 静态 camera -> barrel 外参，以及逐帧更新的 barrel -> world 旋转。
  L1Sensor::CameraCalibration calibration_;
  Eigen::Matrix3d R_camera2barrel_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_camera2barrel_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_barrel2world_{Eigen::Matrix3d::Identity()};
  ArmorConfig config_;
  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> big_armor_points_;
  bool world_barrel_ready_{false};
  bool ready_{false};
};

}  // namespace L3Estimation
