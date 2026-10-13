#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>

namespace L4Planning {

/// 单调时钟时刻
using TimePoint = std::chrono::steady_clock::time_point;

/**
 * @brief 规划失败的原因
 *
 * None 表示规划成功；其余都表示本帧没有可下发的瞄准角，L5 保持上一条命令并关火。
 */
enum class PlanError : std::uint8_t {
  None,             ///< 规划成功
  NoTarget,         ///< 没有目标
  OutOfWindow,      ///< 命中时刻没有板在可击打窗口内（小陀螺的正常间歇）。
  BallisticFailed   ///< 弹道无解或解出的角度、飞行时间非有限
};

/**
 * @brief 一帧观测从曝光到命中的延迟链，所有字段单位均为秒
 *
 * 五段不能合成一个标量：image_to_plan、plan_to_send 每帧实测，send_to_control
 * 只能实车标定，control_to_fire 按转速档取，fire_to_hit 是弹道飞行时间。
 */
struct Delay {
  double image_to_plan{0.0};      ///< 曝光中点 -> 开始规划
  double plan_to_send{0.0};       ///< 规划完成 -> 串口发送
  double send_to_control{0.0};    ///< 串口发送 -> 电控执行
  double control_to_fire{0.0};    ///< 电控执行 -> 弹丸离膛
  double fire_to_hit{0.0};        ///< 弹丸离膛 -> 命中目标

  /**
   * @brief 曝光到弹丸离膛的总延迟
   * @return 前四段之和，s
   */
  double beforeFire() const noexcept
  {
    return image_to_plan + plan_to_send + send_to_control + control_to_fire;
  }
};

/**
 * @brief 命中点及其对应的世界系枪管角命令
 */
struct AimReference {
  /// 命中点，世界系，meter
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};

  /// 枪管 yaw 命令，世界系绝对角，rad
  double yaw{0.0};

  /// 枪管 pitch 命令，世界系绝对角，rad
  double pitch{0.0};
};

/**
 * @brief L5 判定的目标板
 *
 * L5 判定始终落在一块实体装甲板上；armor_pose = [x, y, z, normal_yaw]。
 */
struct FireReference {
  /// 整车展开后的物理板编号，与 armor_xyza_list() 的下标一致。
  int armor_id{-1};

  /// 板在命中时刻的世界系位姿 [x, y, z, normal_yaw]
  Eigen::Vector4d armor_pose{Eigen::Vector4d::Zero()};

  /**
   * @brief 板心位置
   * @return [x, y, z]，世界系，meter
   */
  Eigen::Vector3d point() const noexcept
  {
    return armor_pose.head<3>();
  }

  /**
   * @brief 板面相对视线的偏角
   * @return 视线方向减板面法向，归一化到 [-pi, pi]，rad
   */
  double facingAngle() const noexcept
  {
    // 视线方向减板面法向并归一化到 [-pi, pi]。
    const double line_of_sight = std::atan2(armor_pose.y(), armor_pose.x());
    return std::remainder(
      line_of_sight - armor_pose.w(), 2.0 * std::numbers::pi);
  }
};

/**
 * @brief 一次规划的时间信息
 */
struct PlanTiming {
  /// 最终瞄准点所对应的目标时刻，当前即预计命中时刻。
  TimePoint prediction_time{};

  /// 弹道飞行时间，s
  double fly_time{0.0};

  /// 延迟链
  Delay delay;
};

/**
 * @brief 一帧的规划结果
 */
struct Plan {
  /// 失败原因，None 表示成功
  PlanError error{PlanError::NoTarget};

  /// 瞄准点与角度命令
  AimReference aim;

  /// 目标板。规划成功时必有；失败时为空。
  std::optional<FireReference> fire;

  /// 时间信息
  PlanTiming timing;

  /**
   * @brief 规划是否成功
   * @return error 为 None 时返回 true
   */
  bool valid() const noexcept
  {
    return error == PlanError::None;
  }
};

/**
 * @brief 规划器参数，角度单位 rad、时间单位 s
 */
struct PlanConfig {
  /// 飞行时间与目标位置相互依赖，迭代到相邻两次飞行时间之差小于该阈值。这是
  /// 迭代次数上限。
  int max_iterations{10};

  /// 飞行时间收敛阈值
  std::chrono::microseconds fly_time_tolerance{1000};

  /// 整车有符号 yaw 角速度高于 decision_speed 时的发射延迟（control_to_fire），s
  double high_speed_delay_time{0.030};

  /// 其余情况下的发射延迟，s
  double low_speed_delay_time{0.015};

  /// 选择发射延迟档位的角速度分界，rad/s；只有正向超过它才用高速档
  double decision_speed{8.0};

  /// 叠加到 yaw 命令上的固定偏置，rad
  double yaw_offset{0.0};

  /// 叠加到弹道仰角上的固定偏置（取反成世界系 pitch 之前），rad
  double pitch_offset{0.0};

  /// 电控没发弹速、或发来的低于 min_valid_bullet_speed 时按它解弹道，照常开火。m/s
  double default_bullet_speed{23.0};

  /// 可信弹速下限，m/s
  double min_valid_bullet_speed{14.0};

  /// 串口发出到电控执行的耗时，只能在实车上标定，没标按 0。其余四段：
  /// image_to_plan 和 plan_to_send 由 runtime 实测，control_to_fire 用上面的
  /// 高低速档，fire_to_hit 是弹道飞行时间。
  double send_to_control{0.0};

  /// 选板：候选板进入可击打区域的角度。
  double coming_angle{60.0 / 57.3};

  /// 选板：结合旋转方向排除即将离开的板的角度。
  double leaving_angle{20.0 / 57.3};

  /// 前哨站的进入角。前哨站转速固定且板面更窄，进入角放宽、离开角收紧，与普通车分开配。
  double outpost_coming_angle{70.0 / 57.3};

  /// 前哨站的离开角
  double outpost_leaving_angle{30.0 / 57.3};
};

}  // namespace L4Planning
