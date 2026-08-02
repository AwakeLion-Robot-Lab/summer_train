# 目标优先级与整车滤波设计

本文记录 `newvision` 自瞄链路关于装甲板 PnP、整车滤波、装甲面关联和目标优先级的设计结论。模型推理属于 L2；本文从 L2 的检测输出开始，约束 L3、L4 的职责和数据接口。

## 1. 总体链路

推荐流程如下：

```text
L2 全图检测
  -> 颜色、置信度、角点、NMS 等低成本过滤
  -> 对剩余装甲板逐个 PnP，并计算重投影误差
  -> 按车辆编号分组，关联到具体物理装甲面
  -> L3 更新所有可维护的车辆 Tracker
  -> L4 从所有 VehicleState 中选择车辆
  -> 选择该车当前适合击打的物理装甲面
  -> 预测、弹道和 L5 开火判断
```

不能在 PnP 前仅保留“优先级最高”的检测。图像中心距离不能代替真实距离、车体朝向、云台转动代价和弹道可达性；同车双板也会因此丢失。

## 2. 分层职责

- **L2 Perception**：输出所有有效的 `ArmorDetection`，包含四角点、类别、颜色和置信度，不理解车体运动。
- **L3 Estimation**：完成 PnP、坐标变换、同车分组、装甲面关联、Tracker 生命周期和整车状态估计；输出所有车辆状态，不决定最终打谁。
- **L4 Planning**：先选车辆，再选物理装甲面；完成命中时刻预测、延迟补偿和弹道规划。
- **L5 Control**：根据跟踪质量、瞄准误差、数据新鲜度等条件决定是否开火。

## 3. 目标优先级

优先级不应是一个写死的 `switch`。建议分三步：

1. **硬过滤**：敌方颜色、类别有效、PnP 成功、重投影误差、距离、数据新鲜度、无敌状态和弹道可达性。
2. **战术等级**：操作手指定目标、基地/前哨站任务、威胁程度、剩余血量等；同等级目标再参与评分。
3. **连续评分**：综合跟踪质量、位姿质量、可击打性、云台转动量、飞行时间和图像中心距离。

初始权重可采用：

```text
score = 0.25 * track_quality
      + 0.20 * pose_quality
      + 0.20 * fire_feasibility
      + 0.15 * gimbal_score
      + 0.10 * tof_score
      + 0.10 * image_center_score
```

为防止目标来回切换，应设置切换迟滞：新目标分数至少高出 `switch_margin`，连续保持 `switch_confirm_ms` 后才切换，并设置最短锁定时间和短暂丢失保持时间。L4 应记录候选目标的过滤原因、分项得分和切换原因，供 L6 回放调参。

## 4. sp_vision 的整车 EKF

`sp_vision` 为当前选中的一辆车维护一个 EKF，而不是每块装甲板一个 EKF。典型状态为：

```text
x = [xc, vx, yc, vy, zc, vz, yaw, yaw_rate, r1, r2-r1, z2-z1]
```

EKF 根据整车状态预测所有物理装甲面：

```text
face_angle = yaw + face_id * 2pi / armor_count
face_position = center - radius(face_id) * direction(face_angle)
```

因此四板车相邻装甲面自然相差 `90°`，三板前哨站自然相差 `120°`。每块检测先匹配 `face_id`，再通过该装甲面的非线性观测函数更新同一个整车 EKF。

`sp_vision` 同帧双板的实际顺序是“预测一次，分别 PnP，连续校正两次”，参见 [`tracker.cpp`](../../sp_vision_25-main/tasks/auto_aim/tracker.cpp) 和 [`target.cpp`](../../sp_vision_25-main/tasks/auto_aim/target.cpp)。它没有先将双板合成为一个车体中心观测。

## 5. 单板、双板与观测噪声

同一个 EKF 表示同一辆车的后验状态，不代表所有观测共用固定的 `H` 和 `R`。每次校正都可以使用不同的观测函数、雅可比和协方差。

单板模式下，车体中心、yaw 和半径存在耦合。不要利用 EKF 历史半径反推出中心，再把该中心当成独立测量喂回 EKF，否则会重复使用先验信息。应直接使用装甲板 PnP 结果，通过 `h_face(x, face_id)` 让滤波器利用历史约束。

双板有两种更新方式：

- **顺序更新**：`predict(t)` 一次，然后分别执行左右板校正。实现简单，但默认两块板误差条件独立。
- **联合更新**：堆叠两块板的 `z` 和 `H`，在 `R` 的非对角块表达共享的相机内参、外参和云台姿态误差。它更适合 `newvision` 的最终实现。

```text
z = [z_left; z_right]       H = [H_left; H_right]
R = [R_left, C; C^T, R_right]
```

双板信息增益来自两个不同装甲面的几何约束，不应简单地把每块板的 `R` 除以二。每块板的 `R` 应根据距离、倾角、检测置信度、角点质量和重投影误差动态生成。第一版可以使用块对角 `R` 并适度膨胀，采集日志后再估计相关项 `C`。

单板 `R` 的正式定义、像素噪声模型、投影 Jacobian、从 6D PnP 位姿传播到
`[x_world, y_world, z_world, yaw_world]` 的方法，以及四角点残差自由度和
IPPE 双峰歧义的限制，见
[`pnp_observation_noise_and_covariance.md`](pnp_observation_noise_and_covariance.md)。

## 6. rm.cv.fans 的可借鉴部分

`rm.cv.fans` 会对所有有效装甲板分别 PnP，再按车辆编号聚合。其旧 `top4_model` 使用同车两块装甲板的八个角点联合拟合朝向，通过两块板法向延长线的交点估计车体中心，并得到两种半径；参见 [`top4_model.cpp`](../../rm.cv.fans-main/aimer/auto_aim/predictor/motion/top4_model.cpp)。

该双板几何适合用于：

- Tracker 初始化；
- 双板相邻关系和半径门限检查；
- 生成调试用车体中心；
- 联合观测的初值。

正式滤波仍推荐使用整车状态到装甲面的观测模型。旋转速度不能由单帧双板直接获得；双板提高 yaw 的观测精度，`yaw_rate` 仍由跨帧滤波估计。

## 7. 装甲面关联要求

关联必须在一帧内整体完成，而不是让每块检测独立选择最近装甲面。至少保证：

- 一块检测只能匹配一个物理装甲面；
- 一个物理装甲面最多接收一块检测；
- 双板必须满足相邻面夹角和左右顺序；
- yaw 残差使用角度归一化，跨越 `+-pi` 时不产生大残差；
- 装甲面切换时正确切换长短半径和高度差；
- 匹配代价超出门限时拒绝更新，而不是强行关联。

`sp_vision` 的逐板关联没有强制同帧一对一分配，可能让两块检测匹配到同一装甲面。`newvision` 第一版可枚举最多四个装甲面的排列求最小总代价，不必立即引入复杂的匈牙利算法。

## 8. newvision 数据契约

L3 至少需要以下中间数据：

```cpp
struct ArmorObservation
{
  int robot_id;
  ArmorType armor_type;
  Eigen::Vector3d position_world;
  double yaw_world;
  double reprojection_error;
  double confidence;
  TimePoint timestamp;
};

struct VehicleState
{
  int robot_id;
  Eigen::Vector3d center;
  Eigen::Vector3d velocity;
  double yaw;
  double yaw_rate;
  double radius;
  double radius_offset;
  double height_offset;
  Eigen::MatrixXd covariance;
  TimePoint timestamp;
};
```

`TargetEstimator` 应接收 `ArmorDetection[]`、图像时间对应的云台姿态和标定参数，返回 `VehicleState[]`。每辆车由独立 `VehicleTracker` 维护；L4 的 `TargetSelector` 从数组中选择最终目标。

## 9. 推荐实现顺序

1. 完成 L2 `ArmorDetection` 契约和角点顺序测试。
2. 完成逐板 IPPE PnP、坐标变换、重投影误差和门限。
3. 实现单车辆、单板观测的整车 EKF。
4. 加入 yaw 角度残差处理和装甲面切换。
5. 加入同帧一对一装甲面关联和双板顺序更新。
6. 升级为双板联合观测，并记录创新、NIS 和协方差。
7. 扩展为多车辆 Tracker，由 L4 实现评分和切换迟滞。

模型推理代码只需要稳定地产生 L2 `ArmorDetection[]`。ONNX、OpenVINO XML 或 TensorRT engine 的差异应封装在推理后端中，不能泄漏到本文描述的 L3/L4 接口。
