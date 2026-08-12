#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

// Plan 只带一个 error。valid == false 时它是拒绝原因；valid == true 时它是降级
// 原因——瞄准角可用但不该开火。完整的拒绝原因列表由 L5 的 RejectReason 负责。
enum class PlanError : std::uint8_t {
  None,
  NoTarget,
  NoArmor,
  BadBulletSpeed,
  BallisticFailed,
  // 有瞄准角，但命中时刻没有装甲板落在可击发窗口内。高速旋转时这是常态
  // 间歇，不是错误——云台照常跟随，只是不开火。
  OutOfWindow
};

// 延迟分段，单位秒。拆开保存，禁止在 runtime 里只维护一个含义不清的总延迟：
// 每一段的来源和标定方式都不同，合成一个数之后就没法回答"慢在哪一段"。
struct Delay {
  double image_to_plan{0.0};   // 曝光 -> 规划完成，可实测
  double plan_to_send{0.0};    // 规划 -> 指令下发
  double send_to_control{0.0}; // 下发 -> 云台开始执行，需实车标定
  double control_to_fire{0.0}; // 执行 -> 弹丸出膛，需实车标定
  double fire_to_hit{0.0};     // 弹丸飞行时间，由弹道解算填入

  [[nodiscard]] double beforeFire() const noexcept
  {
    return image_to_plan + plan_to_send + send_to_control + control_to_fire;
  }

  [[nodiscard]] double total() const noexcept { return beforeFire() + fire_to_hit; }
};

struct Ballistic {
  double pitch{0.0};
  double fly_time{0.0};
};

// L4 的完整输出。
//
// valid 和 fire_admissible 是两件事，必须分开：
//   valid            云台该不该跟随这个角度。有目标就该跟，哪怕当前打不中。
//   fire_admissible  这一帧允不允许考虑开火。
// 合成一个 bool 的后果是：高速小陀螺的正常击发间隙里，云台会因为"不能打"而
// 停止跟随，等窗口回来时已经指偏了。
struct Plan {
  int target_id{-1};
  int armor_id{-1};

  TimePoint plan_time{};
  TimePoint fire_time{};
  TimePoint hit_time{};

  Eigen::Vector3d aim_point{Eigen::Vector3d::Zero()};  // 枪管系，单位 m

  double yaw{0.0};
  double pitch{0.0};

  double fly_time{0.0};
  Delay delay;
  bool ballistic_valid{false};

  // 火控视角的实体装甲板：命中时刻最接近正对枪口的那块，及其法线夹角和位置。
  // L5 用 fire_armor_point 把装甲板的物理尺寸换算成该距离上的角度容差。
  int fire_armor_id{-1};
  double fire_delta_angle{0.0};
  Eigen::Vector3d fire_armor_point{Eigen::Vector3d::Zero()};
  bool fire_admissible{false};

  PlanError error{PlanError::NoTarget};
  bool valid{false};
};

// 选板窗口。角度一律用弧度存储，YAML 侧写度数。
struct SelectorConfig {
  // 普通车辆：|板法线与视线的夹角| 超过它就不考虑这块板。
  double front_window{60.0 * std::numbers::pi / 180.0};

  // 前哨站：转速固定且只有 3 块板，用"转入侧优先"的窗口。coming 是窗口半宽，
  // leaving 是转出侧的截止线——板越过这条线后，等子弹飞到时已经背对枪口了。
  double outpost_coming_angle{70.0 * std::numbers::pi / 180.0};
  double outpost_leaving_angle{30.0 * std::numbers::pi / 180.0};
};

struct BallisticConfig {
  double gravity{9.7833};

  // **二次**空气阻力系数，单位 1/m。0 表示真空模型，两者共用同一个闭式反解。
  //
  // k = ρ·C_d·A/(2m)。17mm 弹丸（3.2 g，C_d≈0.45）理论值约 0.019。注意别把
  // 别的项目里单位为 1/s 的线性阻力系数填到这里，数值接近但物理含义不同。
  double drag_coefficient{0.0};

  // 超过该发射角认为目标在射程外。
  double max_pitch{std::numbers::pi / 2.5};
};

struct PlanConfig {
  // 飞行时间迭代：目标在子弹飞行期间还会移动，所以要反复"预测到命中时刻 ->
  // 重新解弹道"直到飞行时间收敛。
  int max_iterations{10};
  std::chrono::microseconds fly_time_tolerance{1000};

  // 从指令下发到弹丸出膛的固定延迟，按目标转速二选一。
  double high_speed_delay_time{0.030};
  double low_speed_delay_time{0.015};
  double decision_speed{8.0};

  // 系统性瞄准偏置，单位 radian。
  double yaw_offset{0.0};
  double pitch_offset{0.0};

  // 弹速小于 min_valid_bullet_speed 时认为下位机还没回传有效值，改用
  // fallback 值继续解算，但这一帧不允许开火。
  double fallback_bullet_speed{23.0};
  double min_valid_bullet_speed{14.0};

  [[nodiscard]] bool bulletSpeedValid(double speed) const noexcept
  {
    return std::isfinite(speed) && speed >= min_valid_bullet_speed;
  }

  BallisticConfig ballistic;
  SelectorConfig selector;

  // 实车标定前保持空值；空值表示不能解锁开火。
  std::optional<double> send_to_control;  // second
  std::optional<double> control_to_fire;  // second

  [[nodiscard]] bool fireDelayReady() const noexcept
  {
    return send_to_control.has_value() && control_to_fire.has_value();
  }
};

}  // namespace L4Planning
