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
struct PlannerContext 
{

   TimePoint planning_time;   本次规划开始时间

   LatencyConfig latency;  延迟

   Eigen::Vector3d gimbal_center_world;  云台旋转中心在世界坐标系的位置

   GimbalExtrinsics gimbal_extrinsics;  云台到枪口的变换矩阵

   double gravity{9.80665};  重力加速度

   PlannerConfig config;  储存MPC信息
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

1  latency_compensator  延迟补偿器计算系统延迟->得到系统延迟：system_delay（图片识别->弹丸离开枪口）
  struct LatencyResult
  {
      double system_delay{0.0};  
      double confidence{0.0};    // [0, 1]
      bool valid{false};
  };

如果有必要可以将延迟分为
- 高转速延迟high_speed_system_delay
- 低转速延迟low_speed_system_delay

2.1  BallisticSolver  通过装甲板的位置计算pitch和弹丸射中的时间fly_time；
丐版使用真空抛物线低弹道；后续再加入空气阻力和弹道标定参数
输入：  
struct BallisticRequest
{

  Eigen::Vector3d target_position_muzzle;

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

  
 
   struct ArmorCandidate
  {
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
  };

3  对全部装甲板使用predictor+BallisticSolver迭代计算直至收敛，预测锁定装甲板的准确位置
先根据识别到的装甲板预测时间，再重新预测位置，不断迭代直至收敛
收敛条件：
- 相邻两次飞行时间差小于 0.2～1 ms；
- 相邻两次命中位置差小于约 5～10 mm，或瞄准角变化小于允许误差的一小部分；
- 最大迭代 20 次，未能迭代成功，则重新选择装甲板
--------------------------------------------------------------------------------------------------------------------------
二、装甲板的选择标准：  最好使用plotjugger跑仿真，得到装甲板选择模型

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




