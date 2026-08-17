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

  // 对本帧全部单板结果做第二遍双板联合 yaw 优化，移植 rm.cv.fans 的双板拟合。
  // 入参必须是同一帧、同一次 set_R_world_barrel 之后由 single_pnp 填好的观测：
  // 位置沿用各自的单板 PnP，联合优化只重写 ypr_in_world[0]。
  //
  // 配对失败或联合解不可信时，两块板的单板 yaw 原样保留——不制造比单板更差的
  // 观测是这个入口唯一的硬要求。
  void refine_double_armor(std::vector<Armor>& armors) const;

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
  // 复刻 sp_vision：以枪管 yaw 为中心，在左右各 70 度内按 1 度步长枚举，
  // 用四角点重投影距离之和选择装甲板世界系 yaw。
  void optimize_yaw(Armor& armor) const;

  // 单板在给定世界系 yaw 下的四角点重投影代价，与 optimize_yaw 共用同一支：
  // 重投影不可用时返回无穷，使调用方的比较自然跳过该采样点。
  [[nodiscard]] double yaw_cost(const Armor& armor, double yaw) const;

  // 双板联合搜索。窗口、步长与单板完全一致，只把代价换成两块板之和，右板 yaw
  // 恒为左板 + 2π/n；成功时改写两块板的 ypr_in_world[0] 并返回 true。
  [[nodiscard]] bool optimize_yaw_pair(Armor& left, Armor& right) const;

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
