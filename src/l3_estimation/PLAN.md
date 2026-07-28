# L3 可读基线与分阶段升级计划

## 1. 当前目标

先实现一版容易阅读、容易调试、效果接近 `tongjiceshi` 的 L3，再逐项增加高级优化。

第一版不追求一次解决所有 IPPE 歧义、关联和协方差问题。每轮只增加一个主要能力，并用相同回放数据和上一轮比较，确保能够明确判断效果变化来自哪里。

基地目标暂不进入 L3，不处理 L4。当前目标范围是：

- `FourArmorVehicle`：普通四板车辆，4 面、90°、两组半径和高度。
- `ThreeArmorOutpost`：三板前哨站，3 面、120°、单一半径和高度。

二板平衡车辆暂不支持，后续根据实测需求单独增加模型。

当前实现状态（2026-07-28）：

- 第 0 轮已完成：接口、模型、配置和坐标链已经固定。
- 第 1 轮已完成：单 IPPE、真实 PnP RMSE 和固定步长 yaw 重投影搜索已经接入。
- 第 2 轮已完成：单假设 11 维 EKF、简单物理面匹配、前哨站约束和生命周期已经接入。
- 第 3 轮进行中：201 帧回放中 182 个检测全部生成 L3 观测，176 帧发布目标，未出现非法状态；长回放、同济对照和实测调参尚未完成。

## 2. 第一版数据流

```text
ArmorDetection[]
→ 单个 IPPE PnP
→ 简单 yaw 角度遍历
→ camera→barrel→world
→ 单假设整车 EKF
→ 简单物理面匹配
→ TargetState[] + 基本诊断
```

第一版保留当前项目的清晰接口和坐标命名，但内部算法尽量接近 `tongjiceshi`，避免一次引入过多难以理解的算法。

## 3. 第一版保留的基础能力

### 3.1 接口与坐标系

- 保留 `T_barrel_camera + R_world_barrel` 坐标链。
- 保留 `FrameContext{timestamp, image_size}`。
- 校验输入图像尺寸是否与相机标定尺寸一致。
- 保留 `TargetModel`、`TargetState` 和集中式 `l3_config.yaml`。
- barrel 与云台姿态坐标轴刚性对齐，输入姿态解释为 `R_world_barrel`。

坐标变换固定为：

```text
position_barrel =
    T_barrel_camera * position_camera

position_world =
    R_world_barrel * position_barrel
```

### 3.2 整车状态

普通车辆和前哨站复用 11 维状态：

```text
[xc, vx,
 yc, vy,
 zc, vz,
 yaw, yaw_rate,
 radius,
 radius_offset,
 height_offset]
```

第一版继续使用容易理解的世界系笛卡尔观测：

```text
z = [x_world, y_world, z_world, yaw_world]
```

不照抄 `tongjiceshi` 的球坐标观测 `[bearing, pitch, distance, yaw]`，避免额外引入球坐标转换和复合 Jacobian。

### 3.3 物理面模型

第 `face_id` 个物理面的预测状态为：

```text
face_yaw =
    vehicle_yaw + face_id * face_angle_interval

armor_x =
    center_x - radius * cos(face_yaw)

armor_y =
    center_y - radius * sin(face_yaw)

armor_z =
    center_z + height_offset
```

四板车辆的 1、3 号面使用第二组半径和高度。

前哨站固定：

```text
radius_offset = 0
height_offset = 0
```

对应状态、过程噪声、观测 Jacobian 和协方差行列都保持为零。

## 4. 第 0 轮：回退高级实验并整理基线

- 主流程回退到可读基线；双候选、Gauss-Newton 和多假设只保留为本文后续升级项。
- 主线只保留第一版需要的数据结构和接口。
- 删除第一版不会使用的复杂分支，不通过大量配置开关同时维护两套算法。
- 修正配置单位：
  - `pitch_rad` 必须填写弧度。
  - `+15° = +0.2617993878 rad`。
  - `-15° = -0.2617993878 rad`。
- 普通车辆和前哨站使用独立初始参数。

第一版参考参数：

```yaml
four_armor_vehicle:
  pitch_rad: 0.2617993878
  initial_radius_m: 0.20
  linear_acceleration_variance: 100.0
  angular_acceleration_variance: 400.0

three_armor_outpost:
  pitch_rad: -0.2617993878
  initial_radius_m: 0.2765
  linear_acceleration_variance: 10.0
  angular_acceleration_variance: 0.1
```

这些值只作为复现 `tongjiceshi` 效果的起点，最终必须通过实测重新标定。

### 第 0 轮验收

- 所有角度配置单位明确为 rad。
- 坐标链名称和方向在接口、代码和文档中一致。
- 普通车辆和前哨站可以读取各自独立参数。
- `.deps` 不进入 Git。
- 不修改 L4。

## 5. 第 1 轮：单 IPPE 与简单 yaw 遍历

### 5.1 单 IPPE PnP

第一版使用：

```cpp
cv::solvePnP(..., cv::SOLVEPNP_IPPE)
```

只接收一个 PnP 解，不使用：

- `solvePnPGeneric` 双候选；
- 六自由度 LM 精化；
- IPPE 歧义判断；
- 多候选输出。

保留最低限度的安全检查：

- 四个角点必须为有限值；
- 四边形面积必须大于门限；
- PnP 返回成功；
- `rvec/tvec` 必须为有限值；
- `tvec.z > 0`；
- 距离在配置范围内；
- 重新计算并记录四角点像素重投影误差。

### 5.2 同济式 yaw 遍历

先使用容易阅读的固定步长遍历，不使用连续数值优化。

```text
搜索中心：当前 barrel/world yaw
搜索范围：中心附近 ±70°
搜索步长：1°
```

每个 yaw 按固定 pitch 构造：

```text
R_world_armor =
    Rz(yaw) * Ry(configured_pitch)
```

保持 PnP 位置不变，将四个装甲角点重新投影到图像，并计算：

```text
pixel_error =
    sum(norm(observed_corner - projected_corner))
```

遍历中选择像素误差最小的 yaw。

第一版只输出：

```text
yaw_raw_world
yaw_optimized_world
raw_reprojection_error_px
optimized_reprojection_error_px
```

如果所有遍历结果都非法，直接拒绝该观测。yaw 搜索的重投影误差只用于选择最优角度和输出诊断，不作为丢弃观测的门限；单 IPPE 自身的重投影误差仍由 PnP 门限检查。第一版不做 yaw 不可信时的位置-only 更新。

### 第 1 轮验收

- 合成数据中可以从单个装甲板恢复有限的位置和 yaw。
- 普通车辆和前哨站使用不同固定 pitch。
- yaw 遍历结果是固定 pitch 搜索范围内的最小像素误差。
- `±π` 附近使用统一角度归一化，不出现数值跳变。
- 非法角点、负深度和超距离观测被安全拒绝。
- 同一组数据可以和 `tongjiceshi::Solver` 输出直接对比。

## 6. 第 2 轮：单假设整车 Tracker

### 6.1 Tracker 数量

第一版使用：

```text
robot_id → 一个 EkfTracker
```

不建立 IPPE 双假设，不累计候选评分，不做假设剪枝。

### 6.2 预测

- 位置使用匀速模型。
- yaw 使用匀角速度模型。
- 普通车辆和前哨站使用独立过程噪声。
- 时间倒退或帧间隔超过门限时安全重置。

### 6.3 简单物理面匹配

对于一块观测，枚举所有物理面并计算：

```text
position_error =
    norm(observed_position - predicted_face_position)

yaw_error =
    abs(normalized_angle(
        observed_yaw - predicted_face_yaw))

match_cost =
    position_weight * position_error
    + yaw_weight * yaw_error
```

选择代价最低且通过位置、yaw 门限的物理面。

第一版不做：

- 检测—候选—物理面全局分配；
- IPPE 候选一对一约束；
- NIS 参与关联；
- 多检测组合穷举。

如果同帧存在多个同机器人检测，按匹配代价从小到大依次处理；第一版优先保证代码直观，并在回放中观察重复更新问题。

### 6.4 EKF 更新

使用：

```text
z = [x_world, y_world, z_world, yaw_world]
```

第一版使用固定四维更新，不实现动态三维位置-only 更新。

保留：

- Joseph 形式协方差更新；
- yaw 残差归一化；
- 状态和协方差有限性检查；
- 半径范围检查；
- 前哨站 offset 锁零。

NIS 只记录到诊断中，不作为复杂候选分配条件。确认计算和门限正确后，再决定是否启用更新前拒绝。

### 6.5 生命周期

生命周期在整帧观测处理结束后更新：

```text
Lost
→ Detecting
→ Tracking
→ TemporaryLost
→ Lost
```

避免在正常有观测帧中产生：

```text
Tracking→TemporaryLost→Tracking
```

普通车辆和前哨站允许配置不同的临时丢失时间。

### 第 2 轮验收

- 静止单目标能够完成 Detecting 并进入 Tracking。
- 匀速目标的位置和速度保持有限。
- 旋转目标可以关联到不同物理面。
- 前哨站三个物理面正常工作，两个 offset 始终为零。
- 有观测帧不会产生虚假的 TemporaryLost 生命周期跳变。
- 半径发散、时间异常或协方差非法时安全重置。

## 7. 第 3 轮：同济基线回放与调参

沿用数据格式：

- `<base>.avi`
- `<base>.txt`：`time_s qw qx qy qz`
- YAML：相机内参、畸变、`T_barrel_camera` 和 L3 参数

同一组数据同时记录当前 L3 和 `tongjiceshi`：

```text
PnP position
raw yaw
optimized yaw
reprojection error
associated face
center
velocity
vehicle yaw
yaw rate
radius
tracker state
processing time
```

优先完成以下场景：

- 静止正视装甲板；
- 静止斜视装甲板；
- 图像中心和边缘；
- 多个距离；
- 普通车辆低速旋转；
- 前哨站匀速旋转；
- 短时遮挡和重新出现。

### 第 3 轮验收

- 10 分钟回放无 NaN、协方差失效或状态爆炸。
- 可见帧连续发布率不低于 `95%`。
- 静止点中位位置误差不超过 `max(0.10 m, 距离×2%)`。
- 稳态位置标准差不超过 `max(0.03 m, 距离×0.5%)`。
- 稳态 yaw 标准差不超过 `0.08 rad`。
- 除重新初始化外，不出现超过半个装甲面间隔的 yaw 跳变。
- 目标机器 L3 单帧 p95 不超过 `5 ms`。
- 对比结果明确标注哪些指标优于、接近或差于 `tongjiceshi`。

## 8. 基线完成后的升级顺序

基线通过回放和实测后，按以下顺序升级。每次只合入一个主要能力：

1. 增加更完整的 PnP 输入检查和真实重投影 RMSE。
2. 将 `solvePnP` 升级为 `solvePnPGeneric`，保留两个 IPPE 候选。
3. 已确认目标使用预测 face yaw 选择 IPPE 候选。
4. 将 1° yaw 遍历升级为连续 `[tx, ty, tz, yaw]` 重投影优化。
5. 增加 Huber loss、阻尼和优化可信度判断。
6. yaw 不可信时增加三维位置-only EKF 更新。
7. 增加检测—候选—物理面全局一对一分配。
8. 新目标增加最多两个 Detecting 假设和连续帧评分剪枝。
9. 从像素 Jacobian 传播正式四维观测协方差。
10. 增加姿态时间插值、标定误差和同步误差模型。

每次升级必须回答：

- 新增算法解决了哪个已复现的问题？
- 相比上一版，哪项回放或实测指标变好？
- 是否引入新的失败场景或运行耗时？
- 关闭该功能后能否恢复上一版结果？

## 9. 第一版明确暂缓的功能

以下代码不进入第一版主流程：

- IPPE 双候选和歧义判断；
- 六自由度 LM 精化；
- `[tx, ty, tz, yaw]` Gauss-Newton；
- Huber loss 和阻尼策略；
- yaw 不可信时的位置-only 更新；
- 检测—候选—物理面全局分配；
- IPPE 双假设；
- 连续帧假设评分和剪枝；
- 正式像素协方差传播；
- Monte Carlo 覆盖率验收；
- 基地和二板平衡车辆；
- L4 联调。

## 10. 代码可读性要求

- 每个主要算法块前写一段简短中文注释，说明输入、输出和目的。
- 公式紧邻实现，不把关键数学关系藏在辅助函数深处。
- 第一版优先使用固定维度 Eigen 类型和直接循环。
- 避免为了未来扩展提前加入复杂模板和多层抽象。
- 一个函数只负责一个清晰步骤。
- 每轮升级同步增加一个针对该能力的独立测试。
- 先保证代码能够从入口顺序读完，再考虑复用和性能优化。

推荐阅读顺序：

```text
types.hpp
→ config.hpp/config.cpp
→ pnp_solver.cpp
→ yaw optimizer 的角度遍历实现
→ ekf_tracker.cpp
→ target_estimator.cpp
→ 对应 smoke test
```

## 11. 不照抄的 tongjiceshi 问题

基线目标是接近 `tongjiceshi` 的效果，不是逐行复制所有实现：

- 不使用错误的 `0.711` 四维 NIS 95% 上门限。
- 不在 EKF 更新后使用后验状态计算标准创新 NIS。
- 不保留二板模型中固定访问三个面的越界风险。
- 不将相机分辨率写死为 `1440×1080`。
- 不把角度值 `15°` 直接写入 `pitch_rad`。
- 不将装甲尺寸、初始半径和过程噪声硬编码在 `.cpp` 中。

## 12. 最终原则

第一阶段的目标不是让 L3 算法最多，而是让负责 L3 的人能够：

1. 从输入到输出顺序读懂全部代码。
2. 在回放中看到每一步的中间结果。
3. 用同一数据和 `tongjiceshi` 做直接比较。
4. 每次只修改一个算法并验证收益。
5. 出现问题时能够快速退回上一版。
