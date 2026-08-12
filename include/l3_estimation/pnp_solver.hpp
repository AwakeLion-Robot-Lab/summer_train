#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <optional>
#include <vector>

namespace L3Estimation {

// 单板 PnP：把二维角点恢复成相机系位姿，再经静态外参和曝光时刻的云台姿态
// 转换到枪管系与世界系，最后用重投影搜索修正世界系 yaw。
class PnpSolver {
public:
  // 标定无效时对象仍可构造，但 ready() 为 false，single_pnp 不会写出位姿。
  explicit PnpSolver(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig config = {});

  [[nodiscard]] bool ready() const noexcept;

  // 每帧调用：姿态是枪管系到世界系的旋转，必须对应图像曝光时刻。
  void set_R_world_barrel(
    const std::optional<Eigen::Quaterniond>& barrel_pose);

  // 原地补充 Armor 的位姿字段。失败时 name 保持 Unknown，Tracker 据此丢弃。
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

  // 把装甲板按给定世界系 yaw 重投影后的搜索代价：角点像素距离的 Huber 加上
  // 相邻边夹角的 Huber 形状项。无法重投影时返回正无穷。
  //
  // 公开是有意的：离线回放要画的代价曲线必须和 optimize_yaw 真正在极小化的
  // 函数是同一个，否则曲线上的最低点不是求解器找的那个点，对账就没有意义。
  [[nodiscard]] double armor_reprojection_error(const Armor& armor, double yaw) const;

private:
  // 以枪管 yaw 为中心，粗扫定位代价盆地后在盆地内细扫，只改动世界系 yaw。
  void optimize_yaw(Armor& armor) const;

  // 给定世界系 yaw 和车型固定倾角的 armor -> world 旋转。
  [[nodiscard]] Eigen::Matrix3d armorRotationInWorld(
    double yaw, ArmorName name) const;

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
