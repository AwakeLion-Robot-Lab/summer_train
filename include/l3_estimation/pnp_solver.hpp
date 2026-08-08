#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <array>
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
  // 整周粗扫锁定重投影代价的全局极小谷，再在谷内做高斯牛顿细化，
  // 只改动世界系 yaw。同时给出该 yaw 的标准差 Armor::yaw_sigma。
  void optimize_yaw(Armor& armor) const;

  // 不分配堆内存的重投影核心。reproject_armor 和 yaw 搜索共用它，
  // 保证代价曲线和求解器看到的是同一个函数，不会悄悄分叉。
  [[nodiscard]] bool project_armor_points(
    const Eigen::Vector3d& xyz_in_world,
    double yaw,
    ArmorType type,
    ArmorName name,
    std::array<cv::Point2d, 4>& image_points) const;

  // 四角点的八维像素残差 [dx0, dy0, ..., dx3, dy3]，观测减重投影。
  // 高斯牛顿要的是残差向量而不是标量代价，因此单独提供。
  [[nodiscard]] bool yaw_residual(
    const Armor& armor,
    double yaw,
    std::array<double, 8>& residual) const;

  // 上面那个残差的平方和；无法重投影时返回正无穷。
  [[nodiscard]] double yaw_squared_cost(const Armor& armor, double yaw) const;

  // 针孔 + Brown-Conrady 内参的展开缓存，避免逐次从 cv::Mat 取元素。
  // 只覆盖零斜切且畸变系数为 4 或 5 个的常规标定；其余情况 usable 为 false，
  // 重投影回退到 cv::projectPoints，数值以后者为准。
  struct PinholeIntrinsics {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    double k1{0.0};
    double k2{0.0};
    double p1{0.0};
    double p2{0.0};
    double k3{0.0};
    bool usable{false};
  };

  // 静态 camera -> barrel 外参，以及逐帧更新的 barrel -> world 旋转。
  L1Sensor::CameraCalibration calibration_;
  PinholeIntrinsics intrinsics_{};
  Eigen::Matrix3d R_camera2barrel_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_camera2barrel_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_barrel2world_{Eigen::Matrix3d::Identity()};
  ArmorConfig config_;
  bool world_barrel_ready_{false};
  bool ready_{false};
};

}  // namespace L3Estimation
