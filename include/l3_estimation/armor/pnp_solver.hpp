#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/types.hpp"

#include <optional>
#include <vector>

namespace L3Estimation {

// 把装甲板的四个图像角点解成位姿：IPPE 求 armor -> camera，再用静态外参和
// 曝光时刻的枪管姿态转到枪管系、世界系，最后用重投影误差搜一遍世界系 yaw。
class PnpSolver {
public:
  // 标定或几何配置无效时对象照样构造得出来，只是 ready() 为 false，
  // single_pnp 会直接返回无效结果。
  explicit PnpSolver(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig config = {});

  // 相机标定、静态外参、装甲板几何是否都通过了校验。
  bool ready() const noexcept;

  // 逐帧设置 barrel -> world 旋转，必须取图像曝光时刻的枪管姿态。传入空值或
  // 退化四元数时清掉有效标志，不会沿用上一帧的姿态。
  void set_R_world_barrel(
    const std::optional<Eigen::Quaterniond>& barrel_pose);

  // 原地填充 armor 的位姿字段：IPPE 解算 → 检查板中心在相机前方且四个角点
  // 都可见 → 算重投影 RMSE（仅作诊断）→ 转到枪管系和世界系 → 搜 yaw。
  // 任何一步不过关都只清除派生结果，把类别置为 Unknown，不写半成品。
  void single_pnp(Armor& armor) const;

  // 左右灯条中心在相机 z 轴上的深度差，作为端点观测之外的一维补充观测。用 solvePnPGeneric
  // 取出 IPPE 的全部候选解，按重投影误差排序后选第一个正面朝向相机的；
  // 前哨额外把俯仰固定为 -15°，在 IPPE 的 yaw 左右各 70° 内用黄金分割重搜。
  std::optional<double> lights_depth_diff(
    const Armor& armor) const;

  // 校验并替换相机标定：检查内参、畸变长度、静态外参的正交性，通过后拆出
  // camera -> barrel 的 R 和 t 缓存起来。失败返回 false 并保持 ready() 为 false。
  [[nodiscard]] bool setCalibration(
    const L1Sensor::CameraCalibration& calibration);

  // 把世界系里位姿已知的一块板重投影成四个像素角点，顺序同样是 TL/TR/BR/BL。
  // 标定或枪管姿态缺失时返回空数组。
  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d& xyz_in_world,
    double yaw,
    ArmorType type,
    ArmorName name) const;

private:
  // 以枪管 yaw 为中心、左右各 70° 按 1° 步长枚举，取 yaw_cost 最小的一个写回
  // armor。3/4/5 号的大板跳过这一步，保留 IPPE 原始 yaw。
  void optimize_yaw(Armor& armor) const;

  // 给定世界系 yaw 时四个角点的重投影距离之和。重投影不可用时返回无穷大，
  // 这样调用方的比较会自然跳过该采样点。
  double yaw_cost(const Armor& armor, double yaw) const;

  // 静态 camera -> barrel 外参，加上逐帧更新的 barrel -> world 旋转。
  // 两套物点在构造时按 TL/TR/BR/BL 生成一次，yaw 搜索反复用，不再分配。
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
