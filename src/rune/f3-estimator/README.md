# 能量机关状态滤波与大小符运动预测

参考 [Alliance-Algorithm/rmcs_auto_aim_v2](https://github.com/Alliance-Algorithm/rmcs_auto_aim_v2) 的 `src/module/tracker/model/rune.cpp`、`rune_energy_fitter.cpp/.hpp` 和 `src/utility/math/hungarian.hpp`，适配本项目的 `Rune::Frame / Observation / Pose / Motion / Processor`、Eigen、OpenCV 与 xmake。上游 MIT 许可保存在 [ALLIANCE-LICENSE](ALLIANCE-LICENSE)。没有引入 ROS、RMCS executor 或上游的机器人坐标变换类型。

2026-10-16 下载 `main` 源码快照，SHA-256：

- 压缩包：`36ee27497e722382ffc7662bb80dddb4687737043601f2b3f5889ac5992dca6b`
- `rune.cpp`：`fc5f37177652a4daf85a26f33f52c5bf2600a53fcf4224be8e53895e9de71f88`
- `rune_energy_fitter.cpp`：`d0084477589671e12f9d7518c9d043cbda47c0e92cb5dfc8d105778688727260`

## 数据流与初始化

`Processor` 依次调用：检测/轮廓 → `BuildingStage` 候选 PnP 与联合几何校验 → `FeatureFilter` 像素观测 EKF → `EstimatorStage` 按大小符模式分发运动预测 → 原有弹道和开火门控。

`estimator.cpp` 负责公共的相位展开、历史维护、质量门限和模式切换重置。小符恒速预测位于 `motion/small_rune_motion.cpp`；大符的正弦扫频拟合及恒速/正弦模型选择位于 `motion/big_rune_motion.cpp`。`motion/motion_fitter.cpp` 保留共享的样本窗口、权重和线性拟合。切换模式会清空拟合历史，重新预热。

初始候选由每个有效未激活叶片的四个靶心锚点（有标定轮廓时可使用原有 PCA 锚框）与每个去重的 R 标组合生成。每个组合使用原有 IPPE 多解、联合轮廓/重力优化及残差门限。按未激活叶片内点数、总布局内点数、重投影/轮廓误差、置信度选择解，最多评估 64 对。每个候选从同一份平面滑窗出发，仅提交胜出候选的滑窗。跟踪中优先保持原有可用物理目标；目标丢失后重新选叶。

## 状态、坐标与关联

`FilterState::x` 为七维：`[cx, cy, cz, normal_yaw, normal_elevation, phase, angular_speed]`，协方差为 7×7。相比上游只保留水平朝向的六维状态，这里额外保留法向仰角，以兼容现有三维 PnP 平面；自转相位单独展开，不把平面朝向和自转角混在一起。

提供完整 `T_world_barrel` 时，中心、法向与相位在固定世界系下估计，投影使用当前曝光时刻的 `T_barrel_camera`、内参及畸变；输出仍统一为枪口系。未提供完整位姿时退回枪口系，依赖中心/法向过程噪声及跳变重置，**不能保证运动平台的自运动补偿**。单独 IMU 四元数用于现有重力先验，不冒充平移定位。世界原点变化必须递增 `world_frame_id`。

模型有 6 个特征：1 个 R 标和 5 个固定叶片身份。靶心由四个网络锚点的均值观测。重复 R 在 2 px 内去重，每帧最多更新一次共享 R。代价为平方马氏距离 `innovationᵀ (H P Hᵀ + R)⁻¹ innovation`，类型不匹配、不可见或超过门限的配对被拒绝。匈牙利算法提供一对一全局分配，每行保留拒绝列，方阵和矩形阵均可拒绝任意观测。

每个匹配依次更新，更新前重新计算当前投影、雅可比和创新协方差，再次检查门限。雅可比对实际含畸变的投影做中心差分；协方差使用 Joseph 形式并对称化。默认像素方差 4 px²、平方马氏门限 9.21。

大符已激活叶片只有灯臂时，不存在可用靶心观测，因此只参与原有几何层的状态专属轮廓约束，不把灯臂质心套入靶心模型。小符已激活靶心仍可关联，当前可见未激活靶心才可被选为瞄准目标。

五叶身份不随检测顺序变化。换叶只改变 `target_sector`，拟合历史继续使用同一 `x[5]`，避免 72° 相位跳变。丢失目标、时间乱序/断层、上下文变化及几何明显突变会清空历史；没有观测时不输出预测开火。下游 f5 默认使用周期单发时序，可选 `require_rearm` 保留外部确认后 `rearm()` 的闩锁模式。

## 指数加权与模型选择

拟合展开后的相位，而非噪声更大的逐帧差分速度：

- 小符模型：`theta(t) = C + v*t`，加权线性最小二乘。
- 大符模型：`theta(t) = C + v*t + A*cos(omega*t) + B*sin(omega*t)`。默认在 1.80～2.20 rad/s 扫 41 个频率，用带秩检查的 QR 解每个频率下的四个线性参数。
- 权重为 `2^(-age / half_life)`，同时用于求解和残差代价。默认窗口 6 s、半衰期 3 s、至少覆盖 1.5 s、最多 1200 点。时间以最新样本为零点，避免长时间运行的数值精度问题。

`Frame::mode` 决定预测路径：小符只拟合恒速模型，正弦代价诊断为无穷大、正弦频率为零；大符同时拟合恒速和正弦模型，保留按代价选择的机制。大符正弦模型需满足幅值 0.6～1.5 rad/s、不反转、速度上限及代价改进条件；默认代价比线性低 10% 才切入，已经选中正弦时用 2% 改进门限保持，降低抖动。两条路径均使用加权相位 RMS 0.06 rad 和速度范围作就绪门限，预热期间返回 `MotionNotReady`。幅值与频率范围是可调的部署参数，需按实际规则和录像标定。

预测使用拟合相位的解析差值，支持正反方向；不依赖数值积分，也不把拟合截距误差直接加到当前靶心。`Motion::phase_delta_rad` 捕获参数值副本，后续拟合不会改变已返回的预测。f4 的延迟/飞行时间迭代使用这个回调。另提供同一参数快照的 `phase_delta_signed_rad`，供居中 MPC 窗口回推历史相位，原回调仍拒绝负时间。原 `IBigMotionModel` 显式注入接口仍可覆盖大符默认拟合。

## 接入与诊断

原 `Processor` 构造接口可直接使用默认实现，无需再注入大符模型。配置示例：

```cpp
Rune::EstimatorConfig estimator;
estimator.fit.window_s = 6.0;
estimator.fit.weight_half_life_s = 3.0;
estimator.fit.min_fit_s = 1.5;
estimator.filter.gate_squared = 9.21;
estimator.filter.observation_variance_px2 = 4.0;
Rune::Processor processor(geometry, estimator, planning);
const auto command = processor.process(frame, control);
if (const auto& motion = processor.lastMotion()) {
    // motion->model: Unready / Linear / Sine / External
    // motion->linear_cost_rad2 / sine_cost_rad2 / fit_cost_rad2
    // motion->sine_frequency_rad_s / angular_speed_rad_s
    // motion->pose.filter_state: x、covariance、world_frame、target_sector
    // motion->pose.feature_matches: observation_index、feature_id、马氏距离平方
}
```

`Pose` 的重投影/Chamfer 误差保留几何层输入的质量指标；EKF 的逐特征创新质量查看 `feature_matches`。

`lastMotion()` 只表示本帧有效几何及滤波结果；`ready` 才表示运动预测已通过质量门限。输入失败时清空。各实例由单线程持有。

## 验证

```bash
xmake build rune_estimator_smoke
xmake run rune_estimator_smoke
xmake build rune_joint_building_smoke
xmake run rune_joint_building_smoke
xmake build rune_pipeline_smoke
xmake run rune_pipeline_smoke
xmake build rune_building_smoke
xmake run rune_building_smoke
xmake build newvision
```

测试覆盖匈牙利与穷举最优解对照、全拒绝/矩形关联、正反向正弦拟合、扫频、指数衰减、窗口上限、大符模型双向自动切换、小符仅使用恒速模型、大小符切换后重新预热、预测快照、世界系移动相机、叶片换序/换片、异常/重复特征、协方差半正定、初始化布局优于置信度、跨 R 候选和 Processor 到默认大符预测/弹道的完整链路。所有新增检查在 release 模式也执行。合成投影验证不等同于实机精度，当前尚未完成真实录像闭环与实弹验证。
