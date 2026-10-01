#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

// None 表示规划成功；其余都表示本帧没有可下发的瞄准角，L5 保持上一条命令并关火。
enum class PlanError : std::uint8_t {
  None,
  NoTarget,
  // 命中时刻没有板在可击打窗口内（小陀螺的正常间歇）。
  OutOfWindow,
  BallisticFailed
};

// 一帧观测从曝光到命中的延迟链，所有字段单位均为秒。
struct Delay {
  double image_to_plan{0.0};      // 曝光中点 -> 开始规划
  double plan_to_send{0.0};       // 规划完成 -> 串口发送
  double send_to_control{0.0};    // 串口发送 -> 电控执行
  double control_to_fire{0.0};    // 电控执行 -> 弹丸离膛
  double fire_to_hit{0.0};        // 弹丸离膛 -> 命中目标

  double beforeFire() const noexcept
  {
    return image_to_plan + plan_to_send + send_to_control + control_to_fire;
  }
};

// 命中点及其对应的世界系枪管角命令。
struct AimReference {
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double pitch{0.0};
};

// L5 判定始终落在一块实体装甲板上；armor_pose = [x, y, z, normal_yaw]。
struct FireReference {
  // 整车展开后的物理板编号，与 armor_xyza_list() 的下标一致。
  int armor_id{-1};
  Eigen::Vector4d armor_pose{Eigen::Vector4d::Zero()};

  Eigen::Vector3d point() const noexcept
  {
    return armor_pose.head<3>();
  }

  double facingAngle() const noexcept
  {
    // 视线方向减板面法向并归一化到 [-pi, pi]。
    const double line_of_sight = std::atan2(armor_pose.y(), armor_pose.x());
    return std::remainder(
      line_of_sight - armor_pose.w(), 2.0 * std::numbers::pi);
  }
};

struct PlanTiming {
  // 最终瞄准点所对应的目标时刻，当前即预计命中时刻。
  TimePoint prediction_time{};
  double fly_time{0.0};  // s
  Delay delay;
};

struct Plan {
  PlanError error{PlanError::NoTarget};
  AimReference aim;
  // 规划成功时必有；失败时为空。
  std::optional<FireReference> fire;
  PlanTiming timing;

  bool valid() const noexcept
  {
    return error == PlanError::None;
  }
};

struct PlanConfig {
  // 飞行时间与目标位置相互依赖，迭代到相邻两次飞行时间之差小于该阈值。
  int max_iterations{10};
  std::chrono::microseconds fly_time_tolerance{1000};

  // 根据整车 yaw 角速度选择的发射延迟，单位秒和 rad/s。
  double high_speed_delay_time{0.030};
  double low_speed_delay_time{0.015};
  double decision_speed{8.0};
  double yaw_offset{0.0};
  double pitch_offset{0.0};

  // 电控没发弹速、或发来的低于 min_valid_bullet_speed 时按它解弹道，照常开火。
  double default_bullet_speed{23.0};
  double min_valid_bullet_speed{14.0};

  // 串口发出到电控执行的耗时，只能在实车上标定，没标按 0。其余四段：
  // image_to_plan 和 plan_to_send 由 runtime 实测，control_to_fire 用上面的
  // 高低速档，fire_to_hit 是弹道飞行时间。
  double send_to_control{0.0};

  // 选板：候选板进入可击打区域的角度，以及结合旋转方向排除即将离开的板的角度。
  double coming_angle{60.0 / 57.3};
  double leaving_angle{20.0 / 57.3};
  // 前哨站转速固定且板面更窄，进入角放宽、离开角收紧，与普通车分开配。
  double outpost_coming_angle{70.0 / 57.3};
  double outpost_leaving_angle{30.0 / 57.3};
};

}  // namespace L4Planning
