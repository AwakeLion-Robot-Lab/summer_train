#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <numbers>
#include <optional>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

enum class PlanType : std::uint8_t {
  Setpoint,
  QuinticSwitch,
  TinyMpc
};

// Plan 只带一个 error。valid == false 时它是拒绝原因，valid == true 时它是
// 降级原因——瞄准角可用但不该开火。两者不会同时出现，因为拒绝的 Plan 本来
// 就不允许开火。多个降级原因并存时按 BadBulletSpeed > OutOfWindow 取一个，
// 完整的拒绝原因列表由 L5 的 FireDecision::RejectReason 负责。
enum class PlanError : std::uint8_t {
  None,
  NoTarget,
  NotTracking,
  NoArmor,
  NoCalibration,
  BadBulletSpeed,
  BallisticFailed,
  NotConverged,
  OutOfRange,
  Switching,
  // 有瞄准角，但命中时刻没有装甲板落在可击发窗口内。高速旋转时这是常态
  // 间歇，不是错误——云台照常跟随，只是不开火。
  OutOfWindow
};

// 单位统一为秒。拆开保存，禁止在 runtime 中只维护一个含义不清的
// 总延迟。
struct Delay {
  double image_to_plan{0.0};
  double plan_to_send{0.0};
  double send_to_control{0.0};
  double control_to_fire{0.0};
  double fire_to_hit{0.0};

  [[nodiscard]] double beforeFire() const noexcept
  {
    return image_to_plan + plan_to_send + send_to_control + control_to_fire;
  }

  [[nodiscard]] double total() const noexcept
  {
    return beforeFire() + fire_to_hit;
  }
};

struct Ballistic {
  double yaw{0.0};
  double pitch{0.0};
  double fly_time{0.0};
  bool valid{false};
};

// 瞄准档位。定义和迟滞逻辑在 aim_phase.hpp，放在这里是为了让 Plan 能带上
// 它而不必反向包含。
enum class AimPhase : std::uint8_t;

// L4 的完整输出。第一版 Setpoint 规划器将速度和加速度保持为 0。
//
// valid 和 fire_admissible 是两件事，必须分开——这一点抄自 talos 的
// ControlIntent（TrackCommand / ShotCommand / HoldCommand 三态 variant）：
//   valid            云台该不该跟随这个角度。有目标就该跟，哪怕当前打不中。
//   fire_admissible  这一帧允不允许考虑开火。
// 合成一个 bool 的后果是：高速小陀螺的正常击发间隙里，云台会因为"不能打"
// 而停止跟随，等窗口回来时已经指偏了。
struct Plan {
  int target_id{-1};
  int armor_id{-1};

  TimePoint plan_time{};
  TimePoint fire_time{};
  TimePoint hit_time{};

  Eigen::Vector3d aim_point{Eigen::Vector3d::Zero()};  // muzzle frame, meter

  double yaw{0.0};
  double pitch{0.0};
  double yaw_vel{0.0};
  double pitch_vel{0.0};
  double yaw_acc{0.0};
  double pitch_acc{0.0};

  double fly_time{0.0};
  Delay delay;
  bool ballistic_valid{false};

  // 当前瞄准档位，以及本帧是否真的瞄在实体装甲板上。WholeCarCenter 档瞄的
  // 是旋转圆上的代理点而不是某块板，此时 aim_on_armor 为 false，火控必须改
  // 用 fire_armor_id 那块实体板来判断。
  AimPhase aim_phase{};
  bool aim_on_armor{true};

  // 火控视角的实体装甲板：命中时刻最接近正对枪口的那块，及其法线夹角。
  // 与 armor_id 的区别只在 WholeCarCenter 档才显现。
  int fire_armor_id{-1};
  double fire_delta_angle{0.0};
  bool fire_admissible{false};

  PlanType type{PlanType::Setpoint};
  PlanError error{PlanError::NoTarget};
  bool valid{false};
};

// 选板策略参数。角度一律用弧度存储，YAML 侧再做度数换算。
struct SelectorConfig {
  // 选板前置窗口：法线夹角超过该值的板背对枪口，不作为瞄准候选。窗口内
  // 一块板都没有时不放弃跟随，退化成取夹角最小的那块并标记降级。
  double front_window{60.0 * std::numbers::pi / 180.0};

  // 锁定迟滞：已锁定的板需要比竞争者差过这个角度才允许换板。两块板都接近
  // 窗口边缘时会逐帧互换，命令抖动到云台根本跟不上。
  double switch_hysteresis{5.0 * std::numbers::pi / 180.0};

  // 击发窗口。sp_vision 和 Climber_Vision 拿它当选板判据，这里只拿它当**火控**
  // 判据：coming/leaving 回答的是"子弹飞到时这块板还正对枪口吗"，那是开火
  // 问题不是指向问题。当选板判据用会让云台在正常击发间隙里直接停止跟随。
  //
  // coming 是窗口半宽，leaving 是转出侧的截止线——板越过这条线后，等子弹
  // 飞到已经背对枪口了。
  double coming_angle{40.0 * std::numbers::pi / 180.0};
  double leaving_angle{15.0 * std::numbers::pi / 180.0};

  // 前哨站转速固定且只有 3 块板，窗口比普通车辆宽。
  double outpost_coming_angle{70.0 * std::numbers::pi / 180.0};
  double outpost_leaving_angle{30.0 * std::numbers::pi / 180.0};
};

// 瞄准档位的切换阈值，单位 radian/second。上下行阈值不同构成施密特触发，
// 数值沿用 talos AimerConfig 的标定值。
struct AimPhaseConfig {
  double single_to_whole_up{1.5};
  double single_to_whole_down{1.0};
  double whole_to_center_up{16.5};
  double whole_to_center_down{15.0};

  // 条件连续成立多少帧才换档。相机 200 fps 时 30 帧约 150 ms。
  int transfer_count{30};
};

// 弹道求解参数。
struct BallisticConfig {
  // 上海地区重力加速度，与参考实现保持一致，便于对拍。
  double gravity{9.7833};

  // 线性空气阻力系数，单位 1/m。0 表示真空模型，走闭式解不迭代。
  // 实测标定值约 0.01~0.03。
  double drag_coefficient{0.0};

  // 反解发射角的迭代上限和高度收敛门限 (m)。真空模型用不到。
  int max_iterations{20};
  double height_tolerance{5e-3};

  // 超过该发射角认为目标在射程外。
  double max_pitch{std::numbers::pi / 2.5};
};

struct PlanConfig {
  int max_iterations{10};
  std::chrono::microseconds fly_time_tolerance{100};
  double switch_dead_zone{5.0};  // degree

  // 弹速缺失或明显异常时使用的兜底初速，单位 m/s。裁判系统上电初期会
  // 回传 0，此时用兜底值仍可解算，但 L5 会因 BadBulletSpeed 拒绝开火。
  //
  // 弹速门限只此一处。BallisticSolver 不再自带业务门限，它只判"这个数学
  // 问题有没有解"——两处各设一道且数值不一致的话，落在夹缝里的弹速会既
  // 不触发兜底、又被求解器拒绝，最后报成 BallisticFailed 而不是
  // BadBulletSpeed，把真正的原因藏掉。
  double fallback_bullet_speed{23.0};
  double min_valid_bullet_speed{21.0};

  BallisticConfig ballistic;
  AimPhaseConfig aim_phase;
  SelectorConfig selector;

  // 云台角加速度上限，单位 radian/second²。定点规划器不使用；五次多项式
  // 靠它决定过渡段时长（逐步增大直到峰值加速度落在限内），MPC 用作硬约
  // 束。实车标定前保持空值，轨迹类规划器必须据此退化成定点输出。
  std::optional<double> max_yaw_acceleration;
  std::optional<double> max_pitch_acceleration;

  [[nodiscard]] bool trajectoryLimitsReady() const noexcept
  {
    return max_yaw_acceleration.has_value() && max_pitch_acceleration.has_value();
  }

  // 实车标定前保持空值；空值表示不能解锁开火。
  std::optional<double> send_to_control;  // second
  std::optional<double> control_to_fire;  // second

  [[nodiscard]] bool fireDelayReady() const noexcept
  {
    return send_to_control.has_value() && control_to_fire.has_value();
  }
};

// 兼容当前代码中已经使用的名称。
using AimPlan = Plan;
using DelayBreakdown = Delay;
using BallisticResult = Ballistic;
using PlannerType = PlanType;
using PlanRejectReason = PlanError;

}  // namespace L4Planning
