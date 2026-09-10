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
};

struct BlendSolution {
  bool valid{false};
  double duration{0.0};
  // 两条多项式给的是**相对切板后轨迹的偏差**，不是下发角本身：起点等于
  // 两条轨迹之差，终点连同一、二阶导恒为零。下发角 = 本帧的切板后轨迹 + 偏差，
  // 所以过渡终点恒等于真实轨迹，不会因为预测漂移而在收尾处甩一下。
  Quintic yaw;
  Quintic pitch;
  // 提交那一帧的切板后轨迹状态。只在拿不到本帧采样器时用来外推，正常帧上
  // 基准每帧现取。
  AimState base;
  double peak_yaw_acceleration{0.0};
  double peak_pitch_acceleration{0.0};
  // 到 max_duration 仍然超加速度限。过渡段照发，但要让遥测看得见：这时
  // 重合度上不去是云台能力的物理限制，不是参数没调好。
  bool acceleration_limited{false};
};

// 在给定过渡时长下拟合过渡段。start 是切板前轨迹在本帧（t = 0）的状态，
// after 只在 t = 0 处取一次，用来算偏差的初值——终点不需要预测，因为求值时
// 基准每帧现取。
//
// 起点固定在 t = 0 而不是"切板时刻减去时长"，有两个好处：过渡段起点必然与
// 本帧正在下发的角三阶连续，不需要额外对接；而且永远不会去采样过去的时刻。
// "提前减速"体现在**提交时机**上——距切板只剩最小可行时长时才提交，过渡因此
// 在切板时刻或紧随其后结束。
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
    // 相对本帧的切板时刻，s。**非有限值表示这份预报只用来续供 after 基准
    // 轨迹，不作为提交依据**——过渡进行中调用方必须这样传，否则过渡走完的
    // 那一帧会立刻拿它再提交一段，永远结束不了。
    double switch_time{0.0};
    AimState before;           // 切板前那块板的射击轨迹在本帧的状态
    TrajectorySampler after;   // 切板后那块板的射击轨迹
    // 射击轨迹是否已经真的切到 after 那块板上。过渡进行中由调用方每帧告知：
    // 为假时不能交还控制权——那时把输出交回 shoot_yaw 等于把云台从已经奔到的
    // 新板拽回旧板，一个阶跃变成两个。切板时刻是预测出来的，预测偏晚就会撞上
    // 这种情况，回放里 417-420 和 593-596 两段正是如此。
    bool destination_selected{false};
  };

  struct Output {
    double yaw{0.0};
    double pitch{0.0};
    bool blending{false};
    // 以下三项只供遥测，不参与任何判定，语义见 L4Planning::BlendStatus。
    double peak_yaw_acceleration{0.0};
    double peak_pitch_acceleration{0.0};
    bool acceleration_limited{false};
    // 过渡终点比切板时刻晚了多少秒。提交条件是"距切板已不足最小可行时长"，
    // 一帧就能跨过等号，所以总会晚一点点；晚得明显说明前视窗口不够长。
    double late_by{0.0};
  };

  explicit AimSmoother(BlendLimits limits = {}) noexcept;

  // shoot_yaw / shoot_pitch 是本帧射击轨迹的原值，即不做平滑就该下发的角。
  // forecast 为空表示前视窗口内没有可预见的切板。
  //
  // **过渡进行中调用方必须继续提供 forecast.after，且必须指向提交时选定的
  // 同一块板。** 过渡输出 = 该轨迹本帧的值 + 衰减到零的偏差，基准每帧现取，
  // 终点因此恒等于真实轨迹。切板真的发生之后"当前板"就是它，若改用"下一块"
  // 重新采样，基准会整块跳掉。缺席时退化成从上一次基准匀加速外推。
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

private:
  // 过渡进行中的基准轨迹，见 .cpp 的说明。
  AimState liveBase(const std::optional<Forecast>& forecast, double tau);

  // 把当前过渡段的遥测量填进 Output。过渡开始那一帧和过渡进行中的帧都要填，
  // 抽出来避免两处各写一遍、改一处漏一处。
  void fillStatus(Output& output) const noexcept;

  BlendLimits limits_;
  bool active_{false};
  double late_by_{0.0};
  // 时长走完但射击轨迹还没切过来时，停在基准轨迹上多等的秒数。
  double held_for_{0.0};
  TimePoint start_time_{};
  BlendSolution solution_;
  // 最近一次拿到的基准轨迹及其对应的 tau。采样器缺席的那些帧从这里外推，
  // 保证退化路径也不引入阶跃。
  AimState last_base_{};
  double last_base_tau_{0.0};
};

}  // namespace L4Planning
