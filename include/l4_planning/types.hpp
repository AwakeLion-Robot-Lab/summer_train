#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cmath>
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

  // 火控视角的实体装甲板：命中时刻最接近正对枪口的那块，及其法线夹角和位置。
  // 与 armor_id 的区别只在 WholeCarCenter 档才显现。talos 用两条独立的
  // reference trajectory 表达同一件事（fire_aim_phase 把 WholeCarCenter 降级成
  // WholeCarArmor），这里用同一个 Plan 上的两组字段，省掉一次完整重算。
  //
  // fire_armor_point 是命中时刻这块板在枪管系下的位置，L5 用它把装甲板的物理
  // 尺寸换算成该距离上的角度容差。
  int fire_armor_id{-1};
  double fire_delta_angle{0.0};
  Eigen::Vector3d fire_armor_point{Eigen::Vector3d::Zero()};
  bool fire_admissible{false};

  PlanType type{PlanType::Setpoint};
  PlanError error{PlanError::NoTarget};
  bool valid{false};
};

// 选板策略参数。角度一律用弧度存储，YAML 侧再做度数换算。
struct SelectorConfig {
  // sp_vision 普通车选板窗口。窗口为空时直接返回无效，不做 fallback。
  // SP 源码用 57.3 做 degree -> radian，这里保留同一数值口径。
  double front_window{60.0 / 57.3};

  // 击发窗口。sp_vision 和 Climber_Vision 拿它当选板判据，这里只拿它当**火控**
  // 判据：coming/leaving 回答的是"子弹飞到时这块板还正对枪口吗"，那是开火
  // 问题不是指向问题。当选板判据用会让云台在正常击发间隙里直接停止跟随。
  //
  // coming 是窗口半宽，leaving 是转出侧的截止线——板越过这条线后，等子弹
  // 飞到已经背对枪口了。
  double coming_angle{60.0 / 57.3};
  double leaving_angle{20.0 / 57.3};

  // 前哨站转速固定且只有 3 块板，窗口比普通车辆宽。
  double outpost_coming_angle{70.0 * std::numbers::pi / 180.0};
  double outpost_leaving_angle{30.0 * std::numbers::pi / 180.0};
};

// 瞄准档位的切换阈值，单位 radian/second。上下行阈值不同构成施密特触发。
//
// single_to_whole 取 awakening 与 talos 共有的 1.5 / 1.0。
//
// whole_to_center 原先取的是 awakening `config/leg.yaml` 的 pair_center_up/down
// = 16.5 / 15.0。那是**头文件默认值级别**的数字：talos 的 `config.hpp` 里也写着
// 同样的 16.5 / 15.0，但它实际部署的 `vision_base.toml` 覆盖成了 9.0 / 6.0。
// 已经上过场的那份配置更可信，这里跟它。
//
// 16.5 rad/s 是 945 度/秒，实战里的小陀螺很少到这个量级，这一档几乎永远进不去
// ——records/3m_run_mid 的目标转 3 rad/s，全程 0 帧进入 WholeCarCenter。
//
// 注意 9.0 仍然高于常见的 2~4 rad/s 小陀螺，所以那类目标依然逐板瞄准，瞄准点在
// 每次换板时会跨过相邻两板的间距（3 米处约 180 像素）。这是逐板瞄准的固有代价，
// 不是缺陷：转速 3 rad/s 时装甲板有约 89% 的时间落在 ±coming_angle 窗口内，换成
// 中心档等于把这段可击发时间让出去。要不要为了指令连续性而下调到 3 以下，是一个
// 需要云台阶跃响应数据才能回答的取舍，所有参考实现都没有这么低。
//
// 顺带记录：awakening 的 leg.yaml 里 whole_pair_up = 17.5 > pair_center_up = 16.5，
// talos toml 里 whole_pair_up = 8.5 < whole_pair_down = 9.5，两处的上下行阈值都是
// 拧着的。本项目把 Pair 并进了 WholeCarArmor，只留一组边界，不复制那个问题。
struct AimPhaseConfig {
  double single_to_whole_up{1.5};
  double single_to_whole_down{1.0};
  double whole_to_center_up{9.0};
  double whole_to_center_down{6.0};

  // 条件连续成立多少帧才换档。相机 200 fps 时 30 帧约 150 ms。
  int transfer_count{30};
};

// 弹道求解参数。
struct BallisticConfig {
  // 上海地区重力加速度，与参考实现保持一致，便于对拍。
  double gravity{9.7833};

  // **二次**空气阻力系数，单位 1/m（见 ballistic_model.hpp 的推导）。
  // 0 表示真空模型。两种模型都有闭式反解，不迭代。
  //
  // k = ρ·C_d·A/(2m)。17mm 弹丸（3.2 g，C_d≈0.45）理论值约 0.019，
  // jlu_vision_26 标定出 0.01903，与理论吻合；FYT2024 / awakening 用的是
  // 0.092，高一个量级，来源不明。实车标定前保持 0（真空）。
  //
  // **不要拿 Climber_Vision 的 air_resistance_k（0.0229）填这里**：那是两轴
  // 线性阻力的系数，单位 1/s，和这里的 1/m 不是一回事。两者数值碰巧接近，
  // 但按 1/s 解出来的阻力比物理值小约 20 倍（6 m 上速度只掉 0.6%，而二次
  // 阻力掉 10.8%），填错了等于把阻力关掉还以为开着。
  double drag_coefficient{0.0};

  // 高度补偿迭代的上限和收敛门限 (m)。**只有没有闭式反解的模型才会用到**，
  // 两个内置模型都用不上；保留是为了将来加 RK4 全阻力模型时不必改求解器。
  int max_iterations{20};
  double height_tolerance{5e-3};

  // 超过该发射角认为目标在射程外。
  double max_pitch{std::numbers::pi / 2.5};
};

struct PlanConfig {
  int max_iterations{10};
  // SP Aimer 的迭代收敛门限固定为 1 ms。
  std::chrono::microseconds fly_time_tolerance{1000};

  // 逐项对应 SP configs/newvision_record.yaml 的 Aimer 参数。Aimer 使用有符号
  // v_yaw 与 decision_speed 比较，并把选中的延迟加在曝光到击发之前。
  double high_speed_delay_time{0.030};
  double low_speed_delay_time{0.015};
  double decision_speed{8.0};
  double yaw_offset{0.0};
  double pitch_offset{0.0};

  // SP Aimer 只在弹速小于 14 m/s 时回退到 23 m/s，没有上限门槛。
  double fallback_bullet_speed{23.0};
  double min_valid_bullet_speed{14.0};

  [[nodiscard]] bool bulletSpeedValid(double speed) const noexcept
  {
    // SP 只在 bullet_speed < 14 时换成 23 m/s；没有上限门槛。
    return std::isfinite(speed) && speed >= min_valid_bullet_speed;
  }

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
