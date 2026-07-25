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
  Eigen::Vector3d gimbal_center_world{Eigen::Vector3d::Zero()};
  GimbalExtrinsics gimbal_extrinsics;
  double gravity{9.80665};
  PlannerConfig config;
  ArmorScoreWeights armor_score_weights;
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
  struct LatencyResult
  {
      Delay delay;
      bool valid{false};
  };

2.1  BallisticSolver  通过装甲板的位置计算pitch和弹丸射中的时间fly_time；
丐版使用真空抛物线低弹道；后续再加入空气阻力和弹道标定参数
输入：  
struct BallisticRequest
{

  Eigen::Vector3d target_position_world;

   double bullet_speed;  弹速

   double gravity;   重力加速度

 };

 输出：
struct BallisticSolution
{
   double pitch;

   double yaw;  

   double fly_time;  飞行时间

   bool valid; 是否有解
};

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
  ArmorPose armor;
  BallisticSolution ballistic;
  TimePoint impact_time{};
  double delta_angle{0.0};
  int iteration_count{0};
  double fly_time_error{0.0};
  double position_error{0.0};
  bool converged{false};
  bool within_firing_window{false};
  bool valid{false};
  ArmorScore score;
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
  - 跟踪可信度：根据l3的对装甲板观测质量进行评分
  - 弹道可靠性：弹道必须有解，预测的俯仰角、飞行时间和命中位置必须合理。迭代误差越小、收敛越稳定，评分越高
  - 目标身份一致性：装甲板必须属于当前跟踪车辆，编号和类型应保持一致。身份不一致的候选应直接评分为0（跟预测滤波的稳定性有关，只考虑稳定帧）

  总代价函数：
  Q(i)=Q_facing_value（0.30） * 正对程度Q_facing(i)
    +  Q_window_value（0.30）* 剩余窗口Q_window(i)
    +  Q_prediction_confidence_value（0.25） * 预测可信度Q_prediction_confidence(i)
    +  Q_ballistic_value（0.15） * 弹道可靠性Q_ballistic(i)

    Q(i)=[0,1]
  最终评分为：Score(i) = flag(i) × Q(i)
  其中flag只有0和1：
  - 身份一致、稳定帧满足要求、预测有效、弹道有解且迭代收敛时，flag=1。
  - 任一硬条件不满足时，flag=0。

  struct ArmorScoreWeights 
{
  double facing{0.30};
  double window{0.30};
  double prediction_confidence{0.25};
  double ballistic{0.15};
};

struct ArmorScoreComponents
{
  double Q_facing{0.0};
  double Q_window{0.0};
  double Q_prediction_confidence{0.0};
  double Q_ballistic{0.0};
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
  - 当前装甲板有效时，默认保持当前ID。
  - 新候选分数必须明显高于当前装甲板，例如高出0.08～0.15。
  - 分数优势需要连续保持3～5帧。
  - 建立锁定后至少保持约80～150毫秒。
  - 当前装甲板硬失效时，立即选择有效候选中的最高分。
  - 切换完成后设置短暂冷却时间，避免立刻切回。

当前装甲板失效时，需从新寻找锁定目标时，直接切换到有效候选中评分最高的装甲板。

锁定过程分为四个阶段：
  - 当前锁定：继续跟踪并射击当前装甲板。
  - 预切换：提前确定下一块候选装甲板，但枪口仍主要跟踪当前装甲板。
  - 切换中：枪口开始减速并转向下一块装甲板，此时一般暂时禁止开火。
  - 新装甲板锁定：下一块装甲板进入稳定射击窗口后完成锁定。

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
  };

发布：
  struct SelectionResult
  {
      std::optional<ArmorCandidate> selected;

      bool switching{false};
      bool valid{false};

      SelectionReason reason{
          SelectionReason::NoCandidate};
  };
--------------------------------------------------------------------------------------------------------------------------
三、发布最佳装甲板的信息

理想轨迹
  struct AimReferenceSample
  {
      TimePoint command_time{};
      TimePoint fire_time{};
      TimePoint impact_time{};

      double yaw{0.0};
      double pitch{0.0};

      double yaw_rate{0.0};
      double pitch_rate{0.0};

      double yaw_acceleration{0.0};
      double pitch_acceleration{0.0};

      double fly_time{0.0};
      double confidence{0.0};

      int target_id{-1};
      int armor_id{-1};

      Eigen::Vector3d aim_point_world{
          Eigen::Vector3d::Zero()};

      bool valid{false};
  };
--------------------------------------------------------------------------------------------------------------------------
5.MPC产生平滑角加速度控制枪口运动/若不使用MPC则将使用传统方案（跟踪装甲板）
当目前枪口与装甲板位姿误差在某范围内则fire_permitted=1，进入初步射击窗口，由l5火控判断最终开火


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

接收：
理想轨迹
struct AimReferenceSample





输出：
可执行轨迹
  struct AimPlan
  {
      uint64_t sequence{0};

      int target_id{-1};
      int selected_armor_id{-1};

      TimePoint generated_at{};
      TimePoint valid_until{};

      AimPlanStatus status{
          AimPlanStatus::NoTarget};

      PlannerType planner_type{
          PlannerType::Direct};

      std::vector<AimSample> samples;
      std::vector<AimReferenceSample> reference;

      PlanningDiagnostics diagnostics;

      double confidence{0.0};

      bool ballistic_valid{false};
      bool fire_permitted{false};
      bool valid{false};
 }; 

   struct AimSample
  {
      TimePoint execute_time{};

      double yaw{0.0};
      double pitch{0.0};

      double yaw_rate{0.0};
      double pitch_rate{0.0};

      double yaw_acceleration{0.0};
      double pitch_acceleration{0.0};

      double yaw_jerk{0.0};
      double pitch_jerk{0.0};
  };
  
 若不经过MPC，直接将struct AimReference传给下位机
--------------------------------------------------------------------------------------------------------------------------

