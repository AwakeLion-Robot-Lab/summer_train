#pragma once

#include <chrono>
#include <functional>
#include <optional>

namespace L4Planning {

// 轨迹上一点的位置、速度、加速度。角度单位 rad，时间单位 s。
struct AxisState {
  double position{0.0};
  double velocity{0.0};
  double acceleration{0.0};
};

// 射击轨迹在某一时刻的 yaw/pitch 状态。
struct AimState {
  AxisState yaw;
  AxisState pitch;
};

// 采样射击轨迹：t 是相对本帧的秒数，非负表示未来。
//
// 用回调而不是预采样表，是因为搜索过渡时长时要在任意 t 处取值，而表只能
// 插值——位置插一次、速度加速度还要再差分一次，糊掉的恰好是加速度约束的
// 输入本身。回调只在提交过渡段的那一帧被调用十几次，不在每帧的热路径上。
//
// 只有切板后那条轨迹需要采样器：过渡起点恒为"本帧"，是一个定点，由调用方
// 直接给出。二分会反复改变时长，如果起点也用采样器表示，同一个 t = 0 会被
// 重复求值十几次，而每次求值在真实链路上是一次整车外推。
using TrajectorySampler = std::function<AimState(double t)>;

// [0, duration] 上的一条五次多项式。
struct Quintic {
  double coefficient[6]{};
  double duration{0.0};

  // 两端各给位置、速度、加速度共六个边界条件，唯一确定六个系数。
  // 传入的必须是同一支连续角度：直接把两个绝对角喂进来，目标扫过 ±pi 时
  // 会解出绕整整一圈的过渡段。归一化由 fitBlend() 负责。
  static Quintic fit(
    const AxisState& start, const AxisState& end, double duration);

  double position(double tau) const;
  double velocity(double tau) const;
  double acceleration(double tau) const;

  // [0, duration] 上 |加速度| 的精确最大值。a(tau) 是三次多项式，a'(tau)
  // 是二次的，求根后与两个端点比较即可。不用采样求：采样只会低估峰值，
  // 而低估的方向恰好是"以为满足约束、其实超了"。
  double peakAbsAcceleration() const;
  // 检查 a(t)=0 的所有内部根及端点；不靠离散采样估计速度峰值。
  double peakAbsVelocity() const;
};

struct BlendLimits {
  // 云台能产生的最大角加速度 = 电机最大扭矩 / 云台惯量，rad/s^2。
  // 必须向机械/电控要这两个数，不能拿 IMU 微分去估——两次微分放大噪声，
  // 而且量到的是闭环实际加速度，不是能力上限。
  double max_yaw_acceleration{50.0};
  double max_pitch_acceleration{100.0};

  // 过渡时长的夹紧区间，s。下限避免解出短到没有意义的过渡段，上限保证
  // 过渡不会长到吃掉整个跟随段——重合度是被过渡时长直接扣掉的。
  double min_duration{0.020};
  double max_duration{0.200};

  // 二分次数。8 次把 [20, 200] ms 分到 0.7 ms，远细于图像帧周期。
  int search_iterations{8};

  // 迟到容差，s。**不参与时长计算**，只用来判定 Output::late。
  //
  // 过渡终点总会比切板时刻晚一点：切板时刻按前视网格离散、帧周期还在抖，
  // 触发条件不可能正好卡在等号上。晚一帧属于正常量化误差，晚很多才说明
  // 目标是突然出现的、根本没来得及提前减速。按一个图像帧周期取值。
  double commit_margin{0.020};
};

struct BlendSolution {
  bool valid{false};
  double duration{0.0};
  Quintic yaw;
  Quintic pitch;
  double peak_yaw_acceleration{0.0};
  double peak_pitch_acceleration{0.0};
  // 到 max_duration 仍然超加速度限。过渡段照发，但要让遥测看得见：这时
  // 重合度上不去是云台能力的物理限制，不是参数没调好。
  bool acceleration_limited{false};
  // 拟合采用的实际边界（已展开角度），供独立检查六个边界残差。
  AimState start;
  AimState end;
  bool feasible() const noexcept { return valid && !acceleration_limited; }
};

// 在给定过渡时长下拟合过渡段。start 是切板前轨迹在本帧（t = 0）的状态，
// 终点取切板后轨迹在 duration 之后的状态。
//
// 起点固定在 t = 0 而不是"切板时刻减去时长"，有两个好处：过渡段起点必然与
// 本帧正在下发的角三阶连续，不需要额外对接；而且永远不会去采样过去的时刻。
// "提前减速"体现在**提交时机**上——切板还有 duration 秒时才提交，过渡正好
// 在切板时刻结束。
BlendSolution fitBlend(
  const AimState& start,
  const TrajectorySampler& after,
  double duration,
  const BlendLimits& limits);

// 二分出满足加速度限制的最小过渡时长。找最小而不是随便找一个可行值：过渡
// 越短，偏离射击轨迹的时间越短，重合度越高。
//
// 二分依赖 peak(T) 随 T 下降；即便在某些构型下不严格单调，返回的解也始终
// 带着自己实算的 acceleration_limited，不会谎报可行。
BlendSolution solveBlend(
  const AimState& start,
  const TrajectorySampler& after,
  const BlendLimits& limits);

// 小陀螺切板会让射击轨迹出现阶跃，不连续点处的速度和加速度无定义，云台强行
// 跟随必然超调或滞后。这个类在切板前插入一段五次多项式过渡，使规划后轨迹的
// 加速度不超过云台能力上限。跟随段完全不参与——过渡窗口之外输出逐位等于
// 射击轨迹原值。
//
// 它只吃标量轨迹采样，不引用任何目标/规划类型，所以两条估计器分支都能直接
// 用，也能脱离相机和串口离线验证。
class AimSmoother {
public:
  using TimePoint = std::chrono::steady_clock::time_point;

  struct Forecast {
    double switch_time{0.0};  // 相对本帧的切板时刻，s
    AimState before;           // 切板前那块板的射击轨迹在本帧的状态
    TrajectorySampler after;   // 切板后那块板的射击轨迹
  };

  struct Output {
    double yaw{0.0};
    double pitch{0.0};
    bool blending{false};
    double progress{0.0};  // 0..1，遥测用
    double peak_yaw_acceleration{0.0};
    double peak_pitch_acceleration{0.0};
    bool acceleration_limited{false};
    // 提交时切板已经近到来不及完整减速，过渡终点落在切板之后。不压缩时长
    // 去硬凑，那正好是本类要避免的超加速度。
    bool late{false};
    // 过渡终点比切板时刻晚了多少秒。late 是它超过 commit_margin 的布尔化，
    // 但调参时要看的是这个连续量：它稳定在一个帧周期附近就是正常的。
    double late_by{0.0};
    bool committed{false};
    bool search_attempted{false};
    BlendSolution candidate;
  };

  explicit AimSmoother(BlendLimits limits = {}) noexcept;

  // shoot_yaw / shoot_pitch 是本帧射击轨迹的原值，即不做平滑就该下发的角。
  // forecast 为空表示前视窗口内没有可预见的切板。
  Output update(
    TimePoint now,
    double shoot_yaw,
    double shoot_pitch,
    const std::optional<Forecast>& forecast);

  // 丢弃当前过渡段。目标丢失、跟踪对象换车、估计器重置时必须调用，否则会
  // 继续按已经失效的预测往下走。
  void reset() noexcept;

  bool blending() const noexcept { return active_; }
  const BlendLimits& limits() const noexcept { return limits_; }
  const BlendSolution& solution() const noexcept { return solution_; }
  TimePoint startTime() const noexcept { return start_time_; }

private:
  BlendLimits limits_;
  bool active_{false};
  bool late_{false};
  double late_by_{0.0};
  TimePoint start_time_{};
  BlendSolution solution_;
};

}  // namespace L4Planning
