#pragma once

#include <Eigen/Core>

namespace L4Planning {

// 单次弹道求解输入。坐标单位为 m，速度为 m/s，重力为 m/s^2。
struct BallisticRequest {
  // 目标相对枪口的位置；当前求解器要求 z 轴与重力反方向一致。
  Eigen::Vector3d target_position_barrel{Eigen::Vector3d::Zero()};
  double bullet_speed{0.0};  // 弹丸出膛速度
  double gravity{9.80665};   // 重力加速度绝对值
  // false：真空弹道；true：线性空气阻力 a_drag=-k*v。
  bool enable_air_resistance{false};
  // 线性阻力系数 k，单位 s^-1；启用空气阻力时必须大于 0。
  double linear_drag_coefficient{0.0};
};

// 一次弹道求解结果；valid=false 时其余数值不可用于控制。
struct BallisticSolution {
  double pitch{0.0};       // 优先低弹道，失败时回退到高弹道，rad
  double yaw{0.0};         // 水平方位角，rad
  double fly_time{0.0};    // 从出膛到命中的时间，s
  bool valid{false};       // 输入合法、目标可达且数值求解成功
  bool used_air_resistance{false}; // 本次有效解是否启用了线性阻力
};

// 真空/线性阻力弹道求解器。两种模型均优先低弹道，失败时尝试高弹道。
class BallisticSolver {
public:
  // 根据请求选择弹道模型，同时计算 yaw、pitch 和 fly_time。
  [[nodiscard]] BallisticSolution solve(const BallisticRequest& request) const;
};

}  // namespace L4Planning
