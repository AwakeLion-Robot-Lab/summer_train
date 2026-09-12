已知信息：
目标车辆状态
struct TargetState
{

 int robot_id;   车辆编号

int armor_count=4；装甲板数量

 Eigen::Vector3d center;  车辆中心在世界坐标系中的位置

 Eigen::Vector3d velocity;  车辆中心线速度

 double yaw;  编号为0的装甲板法线在世界坐标系中的朝向角

 double yaw_rate;  车体旋转角速度

 double radius;  车中心到一组装甲板的半径

 double radius_offset;  两组装甲板半径差

 double height_offset;  两组装甲板高度差

 Eigen::MatrixXd covariance;  11 × 11
（
   对角线：每个状态量的方差，即不确定程度

   非对角线：不同状态量误差之间的相关性

   数值越小：通常表示该状态估计越确定

   数值越大：表示跟踪器对该状态越不确定
 ）

 TimePoint timestamp;时间戳

};


基础信息
struct PlannerContext {
  TimePoint planning_time{};
  LatencyConfig latency;
  Eigen::Isometry3d T_barrel_world{Eigen::Isometry3d::Identity()};
  PlannerConfig config;
  ArmorScoreWeights armor_score_weights;
  double facing_angle_good{5.0};  // degree
  double facing_angle_bad{25.0};  // degree
};


车辆状态
 struct RobotState 
{
   // 当前云台姿态
   gimbal rpy;

   // 当前云台运动状态
   double yaw_rate = 0.0;
   double pitch_rate = 0.0;
   double yaw_acceleration = 0.0;
   double pitch_acceleration = 0.0;

   // 当前弹丸速度
   double bullet_speed = 0.0;

   // 时间戳
   TimePoint timestamp{};

 };
--------------------------------------------------------------------------------------------------------------------------
l4_planning目标：延迟补偿、预测、弹道、轨迹规划
一、通过l3滤波器->得到车辆的运动状态->预测dt后装甲板的位姿
二、装甲板的选择->在众多装甲板中，锁定一个最适合击打的装甲板进行追踪，同时负责装甲板的切换
三、发布最佳装甲板的信息
四、枪口的运动->规划枪口的运动，使得枪口在追踪时运动更加合理，尤其是在装甲板切换时，提前减速，最大化增加射击窗口
--------------------------------------------------------------------------------------------------------------------------
人工调参入口：

- `config/planner_config.yaml` 统一保存延迟、预测、弹道、装甲板选择和 TinyMPC 参数；
- 字段名中的 `_s`、`_us`、`_m`、`_deg`、`_rad`、`_rad_s`、`_rad_s2` 表示单位；
- 程序启动时加载并校验，缺失字段沿用 C++ 默认值，非法值会阻止自瞄运行时启动；
- 修改 YAML 后需要重启程序生效。

--------------------------------------------------------------------------------------------------------------------------
实现：
一、预测dt后装甲板的位姿：  使用迭代拦截法预测装甲板的未来位置

1  latency_compensator 延迟补偿器计算系统延迟：
系统出枪延迟 =（命令发布时间戳 - 图像时间戳）+ 命令发布到弹丸离开枪口的标定时间。
struct Delay
{
  TimePoint camera_timestamp;
  TimePoint command_timestamp;
  double fire_delay{0.0};
}

delay=(command_timestamp-camera_timestamp)+fire_delay

其中 `fire_delay` 存放在 `LatencyConfig` 中，由标定参数提供。规划器向
`LatencyCompensator` 传入图像时间戳和命令时间戳，补偿器负责构造并校验
本次使用的 `Delay`。
  struct LatencyResult
  {
      Delay delay;
      bool valid{false};
  };

2.1  BallisticSolver  通过装甲板的位置计算pitch和弹丸射中的时间fly_time；

支持两种可以切换的弹道模型：

- `enable_air_resistance=false`：使用真空抛物线弹道，只考虑初速度和重力；
- `enable_air_resistance=true`：使用线性空气阻力模型 `a_drag=-k*v`。

线性阻力系数 `linear_drag_coefficient=k` 的单位是 `s^-1`，需要通过实弹
落点数据标定。速度微分方程为：

```
dv/dt = -k*v + [0, 0, -g]
```

令 `A(t)=(1-exp(-k*t))/k`，目标水平距离 `d` 和高度 `h` 满足：

```
d = v0*cos(pitch)*A(t)
h = v0*sin(pitch)*A(t) - g/k*(t-A(t))
```

实现中根据给定 pitch 由第一式解析计算飞行时间，再对第二式的高度残差
使用黄金分割和二分求根，优先选择飞行时间较短的低弹道；低弹道求解或
校验失败时再尝试高弹道。在线性阻力模型下，
水平位移极限为 `v0/k`，超过该距离时弹道无解。

输入：
struct BallisticRequest
{

  Eigen::Vector3d target_position_barrel;

   double bullet_speed;  弹速

   double gravity;   重力加速度

   bool enable_air_resistance;  是否启用线性空气阻力

   double linear_drag_coefficient;  线性阻力系数k，单位s^-1

 };

 输出：
struct BallisticSolution
{
   double pitch;

   double yaw;  

   double fly_time;  飞行时间

   bool valid; 是否有解

   bool used_air_resistance; 实际是否使用了空气阻力模型
};

Planner通过PlannerConfig统一管理重力加速度、线性阻力系数和弹道模型选择；
PlannerContext通过config字段携带该配置：
```
struct PlannerConfig
{
  double gravity{9.80665};
  bool enable_air_resistance{false};
  double linear_drag_coefficient{0.0};  // s^-1
  double switch_dead_zone{5.0};         // degree
  double rotation_rate_dead_zone{0.05}; // rad/s
  int lock_stable_frames{3};
  double aim_cost_good_angle{5.0};      // degree
  double aim_cost_bad_angle{30.0};      // degree， 这两个变量是把“云台需要转多少角度”归一化为 Q_aim_cost 的软评分阈值
};

struct PlannerContext
{
  PlannerConfig config;
};
```


2.2  predictor  通过l3给出的滤波预测t0时刻装甲板的位置

输入：
struct PredictionRequest
{
   TargetState target;  目标车辆状态

   TimePoint target_time;  预测到的绝对命中时刻
};

输出：
  struct ArmorPose
  {
      int robot_id{-1};
      int armor_id{-1};
      ArmorType armor_type{ArmorType::Small};

      Eigen::Vector3d position_world{
          Eigen::Vector3d::Zero()};

      Eigen::Vector3d velocity_world{
          Eigen::Vector3d::Zero()};

      double yaw_world{0.0};

      TimePoint timestamp{};
      bool valid{false};
  };

  struct PredictionResult
  {
      TargetState predicted_vehicle;

      std::vector<ArmorPose> armor_candidates;

      bool valid{false};
  };

  
 
struct ArmorCandidate {
  ArmorPose armor;                 // 收敛后的装甲板位姿
  BallisticSolution ballistic;    // 对最终位置求得的弹道
  TimePoint impact_time{};        // 预计绝对命中时刻
  double delta_angle{0.0};        // 装甲板法向与目标方位角之差，rad
  int iteration_count{0};         // 实际固定点迭代次数
  double fly_time_error{0.0};     // 相邻两次飞行时间误差，s
  double position_error{0.0};     // 相邻两次位置误差，m
  double angle_error{0.0};        // 相邻两次瞄准角合成误差，rad
  double aim_angle_error{0.0};    // 当前云台到候选弹道角的合成角差，rad
  double relative_yaw_rate{0.0};  // 装甲板法线相对目标方位的角速度，rad/s
  double phase_angle{0.0};        // 沿旋转方向递增的窗口相位，rad
  double remaining_window_time{0.0}; // 到离开窗口的预计时间，s
  bool entering_firing_window{false}; // 是否正在转入正面窗口
  bool converged{false};          // 是否满足时间和位置/角度收敛条件
  bool within_firing_window{false}; // 命中时刻是否仍可射击
  bool valid{false};              // 身份、跟踪、预测和迭代收敛条件有效
  ArmorScore score;               // 多装甲板选择评分
};

3  对全部装甲板使用predictor+BallisticSolver迭代计算直至收敛，预测锁定装甲板的准确位置
先根据识别到的装甲板预测时间，再重新预测位置，不断迭代直至收敛
收敛条件：
- 相邻两次飞行时间差小于 0.2～1 ms；
- 相邻两次命中位置差小于约 5～10 mm，或瞄准角变化小于允许误差的一小部分；
- 最大迭代 20 次，未能迭代成功，则重新选择装甲板
--------------------------------------------------------------------------------------------------------------------------
二、装甲板的选择标准：  最好使用plotjugger跑仿真，得到装甲板选择模型
！！！！引入装甲板评分体系！！！！
评分只用于装甲板的切换，暂时不考虑枪口运动（交给MPC），目的是打击最佳的装甲板，让射击目标更加合理
装甲板评分主要考虑以下因素：
  - 正对程度：优先选择法线更朝向枪口、投影面积更大的装甲板。判断时应使用预测命中时刻的姿态，而不是当前姿态
  - 剩余射击窗口：不仅要判断命中时是否仍能击打，还要估计装甲板还能保持可击打状态多久。即将转出视野的装甲板应降低评分
  - 转向代价：优先选择云台可以更快转到的装甲板

  总代价函数
    Q = 0.40 * Q_facing
      + 0.40 * Q_window
      + 0.20 * Q_aim_cost

    Q(i)=[0,1]
  最终质量为：quality(i) = Q(i)

quality 只表示候选装甲板的相对质量，不再乘硬条件 flag，也不直接决定能否
开火。身份、预测、弹道有效性和迭代收敛等条件由 candidate.valid 负责；
within_firing_window 只在最终开火门控中使用。这样即使候选当前位于
射击窗口外，仍保留其真实评分并可以成为更优的跟踪目标，但不会触发开火。

  1. 正对程度 Q_facing
 把“正对”定义为枪口当前方向与装甲板朝向的夹角，
 delta_angle = armor_yaw - center_yaw，即 装甲板朝向角－方位角
  设置两个阈值：
  facing_angle_good = 5 度
  facing_angle_bad  = 25 度

  归一化：
  x = clamp
  (
      (angle - facing_angle_good)
      / (facing_angle_bad - facing_angle_good),
      0,
      1
  )
  Q_facing = 1 - x*x*(3 - 2*x)
  结果：
  - angle 小于facing_angle_good，Q_facing 接近 1
  - angle 大于facing_angle_bad，Q_facing 等于 0
  - 中间平滑下降


 2. 剩余窗口 Q_window
使用目标旋转角速度减去目标中心方位角速度，得到相对旋转角速度
relative_yaw_rate。对旋转方向归一化后，phase_angle 始终沿装甲板
转动方向递增：  phase_angle = sign(relative_yaw_rate) * delta_angle

普通车辆的窗口为：  -normal_enter_angle <= phase_angle <= normal_leave_angle

前哨站的窗口为：  -outpost_enter_angle <= phase_angle <= outpost_leave_angle

剩余窗口时间：   remaining_window_time = (leave_angle - phase_angle) / abs(relative_yaw_rate)


Q_window 在进入边界为1，在离开边界平滑下降到0。角速度小于
rotation_rate_dead_zone 时按近似静止处理，窗口内 Q_window=1。

 3. 转向代价 Q_aim_cost
使用当前云台姿态和候选装甲板最终弹道角计算合成角差：
yaw_error = abs(normalize_angle(candidate.ballistic.yaw - robot_state.rpy.yaw))
pitch_error = abs(candidate.ballistic.pitch - robot_state.rpy.pitch)
aim_angle_error = hypot(yaw_error, pitch_error)


使用 PlannerConfig 中的两个角度阈值平滑归一化：
good_angle_rad = aim_cost_good_angle * pi / 180
bad_angle_rad = aim_cost_bad_angle * pi / 180
x = clamp(
    (aim_angle_error - good_angle_rad)
    / (bad_angle_rad - good_angle_rad),
    0,
    1)
Q_aim_cost = 1 - x*x*(3 - 2*x)

Q_aim_cost 越大表示云台转向距离越短、转向代价越小。云台 yaw、pitch
和候选弹道角的单位统一为 rad，两个配置阈值的单位为 degree。

  struct ArmorScoreWeights 
{
  double facing_weight{0.40};
  double window_weight{0.50};
  double aim_cost_weight{0.10};
};

struct ArmorScoreComponents
{
  double Q_facing{0.0};
  double Q_window{0.0};
  double Q_aim_cost{0.0};
};

struct ArmorScoreHardConditions 
{
  bool identity_consistent{false};
  bool stable_tracking{false};     当前 stable_tracking 只验证数值合法，并没有真正判断跟踪是否稳定
  bool prediction_valid{false};
  bool within_firing_window{false};
  bool ballistic_valid{false};
  bool iteration_converged{false};
};

struct ArmorScore 
{
  ArmorScoreComponents components;
  ArmorScoreHardConditions hard_conditions;
  double quality{0.0};
};


  未锁定时：
  首次选择以及当前目标丢失后的恢复保留 prefer_entering 强优先：
  先从正在进入射击窗口的有效候选中选最高分；没有此类候选时，再从
  其余有效候选中选最高分。

  锁定后：
  每周期比较当前装甲板与其他有效候选的评分。只有满足best.score.quality > current.score.quality + config.score_switch_threshold
  才主动换板，避免评分微小波动造成频繁切换。Tracking 状态下的主动
  换板不使用射击窗口作为强制切换条件；窗口外的高质量候选可以击败
  窗口内候选。当前装甲板失效时，仍使用 prefer_entering 规则恢复。

  PlannerConfig 中对应的配置为：double score_switch_threshold{0.10};

锁定过程的主要阶段：
  - 当前锁定（Tracking）：稳定跟踪当前装甲板；只有命中时刻处于
    射击窗口内才允许开火。
  - 切换中（Switching）：枪口转向下一块装甲板，此时禁止开火。
  - 稳定确认（Stabilizing）：下一块装甲板已对准，连续确认若干帧，
    此时仍禁止开火；观测不新鲜会清零连续确认帧数。
  - 未锁定（Unlocked）：当前没有可跟踪装甲板。

只有目标时间戳严格晚于上一帧观测时间戳时才视为新鲜观测；重复或倒退
的时间戳按丢帧处理，不参与连续稳定确认，也不允许开火。

状态转换如下：
  Unlocked
     │ 选出候选
     ▼
  Switching ──瞄准误差进入死区──▶ Stabilizing
     ▲                              │
     │ 瞄准误差重新变大             │ 连续稳定足够帧
     └──────────────────────────────┤
                                    ▼
                                 Tracking
                                    │
                                    │ 当前板失效、出窗
                                    │ 或其他板持续明显更优
                                    ▼
                                 Switching

Tracking 主动换板由同一候选连续 score_switch_stable_frames 帧满足
评分收益阈值触发 Switching，默认需要连续 3 帧。候选变化、评分优势
消失、观测不新鲜或当前候选失效都会清零确认计数。current_armor_id
和 next_armor_id 分别保存当前装甲板与切换候选。
切换或稳定确认期间若 next_armor_id 对应候选失效，原装甲板仍有效时
取消切换并恢复 Tracking；原装甲板也失效时改选其他有效候选；全部候选
均失效时等待 max_lost_frames，超过阈值后解锁。



TJU选择标准（以下角度皆为  delta_angle = armor_yaw - center_yaw，即 装甲板朝向角－方位角）：
判断顺序如下

已锁定：

1.1  普通车辆可射击角度窗口：
  - 装甲板进入角：60°
  - 装甲板离开角：20°


1.2  前哨站可射击角度窗口：
  - 装甲板进入角：70°
  - 装甲板离开角：30°

2  保持与上一周期装甲板 ID 的连续性。当前候选有效时继续跟踪；其他
候选只有在评分高出 score_switch_threshold 后才触发主动换板。射击
窗口不强制换板，仅控制 fire_permitted。当前候选短暂失效时，在
max_lost_frames 宽限期内保持当前装甲板 ID 并禁止开火；连续失效超过
阈值后，才按 prefer_entering 强优先规则选择其他候选并切换。

3   entering_firing_window 是正式进入角之前的 10° 预进入区：[-(enter_angle + 10°), -enter_angle]

  - 当前装甲板仍在射击窗口：
      - 按综合价值进行比较；
      - 要求超过 score_switch_threshold；
      - 连续满足 score_switch_stable_frames 后换板。

  - 当前装甲板离开射击窗口：
      - 优先选择窗口内价值最高的其他有效装甲板；
      - 其次选择预进入区内价值最高的其他有效装甲板；
      - 直接开始切换，不进行三帧价值优势确认；
      - 都不存在时执行 resetTracking()，清除目标缓存并进入 Unlocked；
      - 返回的 AimPlan 同时设置 tracking=false、target_id=-1。

未锁定：

3  根据角速度的正负选择即将转入正面的装甲板（装甲板未到车辆中心）

4  装甲板的正对程度：
  优先选择相对朝向角绝对值最小的装甲板，即优先选择更正对枪口、投影面积更大的装甲板

5  命中时刻的可射击性：
  - 使用预测命中时刻，而不是当前时刻的装甲板位姿
  - 窗口外的有效装甲板仍可成为跟踪候选并参与评分
  - 只有预测命中时刻仍处于射击窗口内时，最终才允许开火

6  弹道可解性：
  - 弹道必须可解
  - 预测瞄准点必须有效
  - 弹丸到达时装甲板仍需满足可击打条件

7  目标身份一致性：
  - 装甲板的车辆编号和装甲板类型应与当前跟踪目标一致
  - 防止将相邻车辆的装甲板错误关联到当前目标



  根据全部装甲板的信息选择跟踪
  接收：
  struct SelectionRequest
  {
      std::vector<ArmorCandidate> candidates;
      std::optional<int> preferred_armor_id;
      bool observation_fresh{true};
  };

发布：
  struct SelectionResult
  {
      std::optional<ArmorCandidate> selected;

      ArmorTrackingPhase phase{ArmorTrackingPhase::Unlocked};
      bool tracking_ready{false}; // 稳定锁定且本周期观测可支持跟踪
      bool valid{false};

      SelectionReason reason{
          SelectionReason::NoCandidate};
  };
--------------------------------------------------------------------------------------------------------------------------
三、发布最佳装甲板的信息
分为是否使用MPC，为方便l5调用，这里将追踪装甲板的信息和MPC控制信息放在同个结构体里。
`AimPlan` 直接继承 `AimReference`；
由 `using_MPC` 明确告诉L5应读取哪组控制量：
- `using_MPC=false`：L5直接读取AimPlan继承的 `yaw`、`pitch` 等参考量；
- `using_MPC=true`：L5读取 `AimPlan.samples.front()`；
--------------------------------------------------------------------------------------------------------------------------
四、MPC产生平滑角加速度控制枪口运动/若不使用MPC则将使用传统方案（跟踪装甲板）

同济MPC：
状态 = [角度, 角速度]
输入 = 角加速度
加速度代价函数
  Uref[k] =
      Q_angle    * angle_error[k]^2
    + Q_velocity * velocity_error[k]^2
    + R_acc      * acceleration[k]^2

角度误差权重Q_angle= 9,000,000
角速度误差权重Q_velocity= 0
加速度大小权重R_acc= 1

yaw 加速度范围= [-50, 50] rad/s^2
pitch 加速度范围= [-100, 100] rad/s^2

求解器非常重视角度跟踪，基本不要求规划角速度贴近参考角速度，但限制加速度不能超过最大值
代价函数只惩罚加速度大小，没有惩罚相邻加速度的变化

 三阶 MPC
  状态 = [角度, 角速度, 角加速度]
  输入 = jerk
  同时设置：
  1. 加速度硬限制
  2. jerk 硬限制
  3. 较小的 jerk 代价
  4. 较高的角度跟踪权重

MPC接收AimPlan继承的AimReference理想瞄准目标，求解后仍输出同一个AimPlan协议：

  struct AimReference  // 非MPC模式的瞄准参考
  {
      int target_id{-1};           // 当前锁定的目标车辆ID，-1表示未锁定
      int armor_id{-1};            // 当前输出装甲板ID，-1表示未锁定
      TimePoint impact_time{};     // 预计弹丸命中装甲板的时刻
      bool tracking{false};        // 是否在追踪

      Eigen::Vector3d aim_point_barrel{
          Eigen::Vector3d::Zero()}; // 枪管坐标系下的瞄准点，单位m
      Eigen::Vector3d aim_point_world{
          Eigen::Vector3d::Zero()}; // 世界坐标系下的预测命中点，单位m

      double yaw{0.0};             // 目标yaw角，单位rad
      double pitch{0.0};           // 目标pitch角，单位rad
      double yaw_rate{0.0};        // 目标yaw角速度，单位rad/s
      double pitch_rate{0.0};      // 目标pitch角速度，单位rad/s
      double yaw_acceleration{0.0};   // 目标yaw角加速度，单位rad/s^2
      double pitch_acceleration{0.0}; // 目标pitch角加速度，单位rad/s^2
      double fly_time{0.0};        // 从出膛到命中的预计飞行时间，单位s
  };

  struct AimSample  // MPC模式的瞄准参考
  {
      TimePoint execute_time{};     // 该控制点应被执行的绝对时刻
      double yaw{0.0};             // 规划yaw角，单位rad
      double pitch{0.0};           // 规划pitch角，单位rad
      double yaw_rate{0.0};        // 规划yaw角速度，单位rad/s
      double pitch_rate{0.0};      // 规划pitch角速度，单位rad/s
      double yaw_acceleration{0.0};   // 规划yaw角加速度，单位rad/s^2
      double pitch_acceleration{0.0}; // 规划pitch角加速度，单位rad/s^2
      double yaw_jerk{0.0};        // yaw角加加速度，单位rad/s^3
      double pitch_jerk{0.0};      // pitch角加加速度，单位rad/s^3
  };

  struct AimPlan : AimReference
  {
      TimePoint generated_at{};    // 本次AimPlan生成完成的时刻

      std::vector<AimSample> samples; // MPC轨迹，按execute_time升序排列
      bool using_MPC{false};       // false直接使用继承的参考量，true使用samples

      ArmorTrackingPhase tracking_phase{ArmorTrackingPhase::Unlocked};
      bool fire_permitted{false};  // 稳定跟踪、位于射击窗口内且弹道有效
      bool valid{false};           // 规划结果是否有效
  };

给l5的火控为：
plan.fire_permitted =
    selection.tracking_ready
    && selected.within_firing_window;


协议约束：
- 非MPC规划器设置`using_MPC=false`、保持`samples`为空，并填充AimPlan
  继承的AimReference成员；
- MPC规划器设置`using_MPC=true`，填充按execute_time排序的samples，
  并保证samples.front()是当前周期应执行的控制点；
- L5始终接收AimPlan，不再接收独立的参考点向量。
--------------------------------------------------------------------------------------------------------------------------
