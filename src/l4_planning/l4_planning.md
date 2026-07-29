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
  bool valid{false};              // 所有硬条件是否满足
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

暂时先通过硬性标准来判断追踪目标的切换，后期考虑通过评分来切换装甲板

  总代价函数
    Q = 0.40 * Q_facing
      + 0.40 * Q_window
      + 0.20 * Q_aim_cost

    Q(i)=[0,1]
  最终评分为：Score(i) = flag(i) × Q(i)
  其中flag只有0和1：
  - 身份一致、稳定帧满足要求、预测有效、弹道有解且迭代收敛时，flag=1。
  - 任一硬条件不满足时，flag=0。

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
转动方向递增：
```
phase_angle = sign(relative_yaw_rate) * delta_angle
```

普通车辆的窗口为：
```
-normal_enter_angle <= phase_angle <= normal_leave_angle
```

前哨站的窗口为：
```
-outpost_enter_angle <= phase_angle <= outpost_leave_angle
```

剩余窗口时间：
```
remaining_window_time =
    (leave_angle - phase_angle) / abs(relative_yaw_rate)
```

Q_window 在进入边界为1，在离开边界平滑下降到0。角速度小于
rotation_rate_dead_zone 时按近似静止处理，窗口内 Q_window=1。

 3. 转向代价 Q_aim_cost
使用当前云台姿态和候选装甲板最终弹道角计算合成角差：
```
yaw_error = abs(normalize_angle(candidate.ballistic.yaw - robot_state.rpy.yaw))
pitch_error = abs(candidate.ballistic.pitch - robot_state.rpy.pitch)
aim_angle_error = hypot(yaw_error, pitch_error)
```

使用 PlannerConfig 中的两个角度阈值平滑归一化：
```
good_angle_rad = aim_cost_good_angle * pi / 180
bad_angle_rad = aim_cost_bad_angle * pi / 180
x = clamp(
    (aim_angle_error - good_angle_rad)
    / (bad_angle_rad - good_angle_rad),
    0,
    1)
Q_aim_cost = 1 - x*x*(3 - 2*x)
```

Q_aim_cost 越大表示云台转向距离越短、转向代价越小。云台 yaw、pitch
和候选弹道角的单位统一为 rad，两个配置阈值的单位为 degree。


  struct ArmorScoreWeights 
{
  double facing_weight{0.40};
  double window_weight{0.40};
  double aim_cost_weight{0.20};
};

struct ArmorScoreComponents
{
  double Q_facing{0.0};
  double Q_window{0.0};
  double Q_aim_cost{0.0};
};

//flag
struct ArmorScoreHardConditions 
{
  bool identity_consistent{false};
  bool stable_tracking{false};
  bool prediction_valid{false};
  bool within_firing_window{false};
  bool ballistic_valid{false};
  bool iteration_converged{false};
};

//score（i） = flag(i) * Q(i)
struct ArmorScore 
{
  ArmorScoreComponents components;
  ArmorScoreHardConditions hard_conditions;
  bool flag{false};
  double quality{0.0};
  double score{0.0};
};


  未锁定时：
  分数越高，优先锁定

  锁定后：
  当前装甲板失效时，需从新寻找锁定目标时，直接切换到有效候选中评分最高的装甲板。

锁定过程分为四个阶段：
  - 当前锁定：继续跟踪并射击当前装甲板。
  - 预切换：提前确定下一块候选装甲板，但枪口仍主要跟踪当前装甲板。
  - 切换中：枪口开始减速并转向下一块装甲板，此时一般暂时禁止开火。
  - 装甲板锁定（Stabilizing）：下一块装甲板进入稳定射击窗口并连续满足
    lock_stable_frames 后完成锁定。

代码额外使用 Unlocked 表示当前没有装甲板锁定。四阶段状态由
ArmorTrackingPhase 保存，current_armor_id 和 next_armor_id 分别保存
当前装甲板与预切换候选。

### 锁定需要考虑的核心条件
  1. 目标身份一致
      - robot_id 与当前目标一致。
      - armor_id 连续，不能仅凭最近距离关联。
      - 装甲板类型与目标车型匹配。
      - 需要防止切换到相邻车辆的装甲板。

  2. 时序连续性
      - 当前装甲板与上一周期锁定装甲板 ID 相同。
      - 预测位置、速度和朝向变化合理，不能发生不符合车辆运动模型的跳变。
      - 时间戳有效，观测和规划时间差不能过大。

  3. 短时丢失容忍
      - 文档规定可以连续丢失最多约 5 帧。
      - 5 帧以内继续用运动模型预测，并保持锁定状态。
      - 超过 5 帧，或预测不确定度过大，解除锁定并重新选择。

  4. 预测结果有效
      - PredictionResult.valid == true。
      - 对应 ArmorPose.valid == true。
      - 迭代拦截计算已经收敛。
      - 飞行时间误差、位置误差和迭代次数处于允许范围。
      - 协方差或预测置信度不能过差。

  5. 命中时刻的装甲板朝向

     应判断预测命中时刻的：
     delta_angle = armor_yaw - center_yaw;
     而不是只判断当前观测角度。已经锁定时采用“离开窗口”，未锁定时采用“进入窗口”，形成滞回，避免装甲板在边界附近反复切换。

  6. 旋转方向和可持续跟踪性
      - 根据 yaw_rate 判断装甲板正在转入还是转出正面。
      - 优先锁定即将转入正面、预计射击窗口更长的装甲板。
      - 如果当前装甲板即将快速转出，而下一块装甲板即将转入，可以提前进入切换状态。


TJU选择标准（以下角度皆为  delta_angle = armor_yaw - center_yaw，即 装甲板朝向角－方位角）：
判断顺序如下

已锁定：

1.1  普通车辆可射击角度窗口：
  - 装甲板进入角：60°
  - 装甲板离开角：20°


1.2  前哨站可射击角度窗口：
  - 装甲板进入角：70°
  - 装甲板离开角：30°

2  保持与上一周期装甲板 ID 的连续性，当前锁定装甲板仍处于可射击窗口时，优先选择上一周期锁定的装甲板
若5帧内未能找到已锁定的装甲板，则重新锁定



未锁定：

3  根据角速度的正负选择即将转入正面的装甲板（装甲板未到车辆中心）

4  装甲板的正对程度：
  优先选择相对朝向角绝对值最小的装甲板，即优先选择更正对枪口、投影面积更大的装甲板

5  命中时刻的可射击性：
  - 使用预测命中时刻，而不是当前时刻的装甲板位姿
  - 预测到命中时刻仍处于可射击角度窗口的装甲板才能成为候选

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
      bool switching{false};
      bool fire_permitted{false};
      bool valid{false};

      SelectionReason reason{
          SelectionReason::NoCandidate};
  };
--------------------------------------------------------------------------------------------------------------------------
三、发布最佳装甲板的信息
分为是否使用MPC，为方便l5调用，这里将追踪装甲板的信息和MPC控制信息放在同个结构体里
由AimPlan中成员；由using_MPC明确告诉L5应读取哪组控制量：
- `using_MPC=false`：L5读取AimPlan.reference；
- `using_MPC=true`：L5读取AimPlan.samples.front()；
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

MPC接收AimPlan.reference中的理想瞄准目标，求解后仍输出同一个AimPlan协议：

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

  struct AimPlan
  {
      TimePoint generated_at{};    // 本次AimPlan生成完成的时刻

      AimReference reference;      // 弹道和目标预测得到的理想瞄准参考
      bool using_MPC{false};       // false使用reference，true使用samples
      std::vector<AimSample> samples; // MPC轨迹，按execute_time升序排列

      ArmorTrackingPhase tracking_phase{ArmorTrackingPhase::Unlocked};
      bool armor_switching{false}; // 切换中或等待新装甲板稳定
      bool ballistic_valid{false}; // 最终瞄准点是否存在有效弹道解
      bool fire_permitted{false};  // L4是否允许进入后续射击判断
      bool valid{false};           // 规划结果是否有效
  };

协议约束：
- 非MPC规划器设置`using_MPC=false`、保持`samples`为空，并填充AimPlan
  的reference成员；
- MPC规划器设置`using_MPC=true`，填充按execute_time排序的samples，
  并保证samples.front()是当前周期应执行的控制点；
- L5始终接收AimPlan，不再接收独立的参考点向量。
--------------------------------------------------------------------------------------------------------------------------
