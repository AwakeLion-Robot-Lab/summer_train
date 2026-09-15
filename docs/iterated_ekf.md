# 迭代扩展卡尔曼滤波 (IESKF) 与半径约束

## 动机

整车 EKF 的观测模型是两层复合：先由 11 维状态展开出指定物理装甲板的
`[x, y, z]`（`h_armor_xyz`），再经 `xyz2ypd` 映射到球坐标 `[方位角, 俯仰角, 距离]`。
第二层是强非线性的。普通 EKF 只在先验点线性化一次，先验离真实后验越远，Jacobian
失真越大。整车状态中 `r1`、`r2-r1`、`z2-z1` 观测最弱，受这个误差影响最明显。

## 算法

`L3Estimation::IteratedKalmanFilter`（`include/l3_estimation/ieskf.hpp`）继承
`ExtendedKalmanFilter`，只增加一个带 Gauss-Newton 迭代的 `update` 重载。它对如下
MAP 代价求极小：

```
J(x) = ||x ⊟ x_pri||²_{P_pri⁻¹} + ||z - h(x)||²_{R⁻¹}
```

迭代式（Bell & Cathey, 1993）：

```
x_{i+1} = x_pri ⊞ K_i [ z - h(x_i) - H_i (x_pri ⊟ x_i) ]
K_i     = P_pri H_iᵀ (H_i P_pri H_iᵀ + R)⁻¹
```

### 移植时最容易踩的四个坑

1. **末项符号。** 是**减去** `H_i (x_pri ⊟ x_i)`。参考实现
   `SHtech_auto_aim-ax650-dev-2026/mathutils/IESEKF.hpp:226` 写成了加号。第 0 次迭代
   `dx_pri` 为零看不出差别，从第二次起会收敛到有偏的不动点。
2. **NIS 必须取先验线性化点的创新量。** 迭代后的残差被压缩过，不再服从自由度等于
   观测维数的卡方分布。若按迭代后的残差记账，`Tracker::badRecentNis` 的 40% 失败率
   门限会失配，复位保护形同虚设。本实现固定用 `x_pri` 处的 `residual`/`S`。
3. **协方差保持 Joseph 形式**，且只从先验传播一次——迭代只移动线性化工作点，不能
   把同一次观测重复吸收进状态。
4. **状态减法要走流形。** `x_minus` 必须对 yaw 取最短圆周差，否则跨越 ±π 时会产生
   2π 的伪残差，把迭代推向错误的工作点。

`max_iterations = 1` 时与基类 `update()` 逐位等价，这一点由 `tests/ieskf_smoke.cpp`
的第一个用例锁定，作为回归保护。

## 半径投影：迭代能生效的前提

半径的初始协方差 `P0[8] = 1.0 m²`（σ 达 1 米），而物理范围只有 5~50 厘米，先验极松；
`Q` 在 `r1`、`r2-r1`、`z2-z1` 三维上又是零。单次观测就足以把 `r1` 拽成负数，而迭代
会把这个过冲放大。

因此 `TrackedTarget` 的 `x_add` 在归一化 yaw 之外，把两个半径投影回
`[kMinRadius, kMaxRadius] = [0.05, 0.5]`。这是最简单的约束卡尔曼形式（投影法），
对迭代路径同样逐次生效，等价于投影 Gauss-Newton。

投影之后越界不再是发散信号，`diverged()` 改判**持续贴边**：连续
`kMaxRadiusPinnedCount = 10` 次更新被夹在边界上，说明观测与整车模型长期矛盾，
此时放弃当前目标。偶发一两帧被夹住则视为观测噪声，不再触发复位。

## 回放实测

`tests/data/sp_auto_aim/demo.avi`，620 帧：

| 配置 | tracking 帧 | lost 帧 | 复位次数 | NIS 均值 |
| --- | --- | --- | --- | --- |
| 无投影，iter=1（改动前基线） | 444 | 16 | 14 | 0.2419 |
| 无投影，iter=5 | 422 | 26 | 20 | 0.2274 |
| 投影，iter=1 | 462 | 15 | 10 | 0.2625 |
| 投影，iter=2 | 467 | 11 | 9 | 0.2689 |
| 投影，iter=5 | 467 | 11 | 9 | 0.2843 |

三个结论：

- **只上迭代、不加约束会更差**（复位 14 → 20）。迭代更紧地拟合观测，把 `r1` 更频繁
  地推出物理范围。这是迭代暴露了既有整定问题，不是迭代本身的错。
- **投影 + 迭代是净收益**：相比原基线，tracking 帧 444 → 467（+5.2%），复位
  14 → 9（−36%）。
- **迭代在第 2 次就收敛**，iter=2/3/5 的跟踪结果完全一致。实测 tracker 平均耗时
  0.289 ms → 0.300 ms（+4%），相对 detector 的 11 ms 可忽略。

## 当前默认：主干路走普通 EKF

`TrackerConfig::ekf_max_iterations` 默认为 **1**，即 runtime 主干路是单次线性化的
普通 EKF，与引入迭代前的行为逐位一致。迭代实现保留在代码里、随时可开，但在实车
验收前不作为默认路径——上面的收益只在一段离线回放上验证过，样本量不足以支撑改
默认。

注意**半径投影和 `diverged()` 的贴边判据不受这个开关影响**，它们是独立于迭代的修正，
默认生效。单看这一项相对改动前基线就是净收益（tracking 444 → 462，复位 14 → 10）。

要开启迭代：把 `ekf_max_iterations` 调到 2~5，或用 `auto_aim_test --ekf-iterations`。

NIS 均值 0.24~0.28 远低于 4 维观测的理想值 4，说明 `R` 整体偏大、滤波器过于保守。
这是独立于本次改动的既有整定问题，见 `pnp_observation_noise_and_covariance.md`。

## 使用

```bash
xmake run ieskf_smoke                                    # 单元行为测试，无需硬件
xmake f --use_openvino=y
xmake run auto_aim_test -- --ekf-iterations=1            # 默认，单次线性化普通 EKF
xmake run auto_aim_test -- --ekf-iterations=5            # 开启 Gauss-Newton 迭代
xmake run auto_aim_test -- --ekf-iterations=5 --show=true  # 附带回放窗口和 yaw-cost 图
```

`--show` 默认 `false`，且 OpenCV 的 `CommandLineParser` 对 bool 只认 `true` 和 `1`，
其它字符串一律静默当作 false——打错字不会报错，只是不弹窗。

运行时通过 `TrackerConfig::ekf_max_iterations` 和 `ekf_step_threshold` 配置。

## 未做的事

- `r1`、`r2-r1`、`z2-z1` 的过程噪声仍为零，收敛后这三维会被锁死。投影只防越界，
  不解决"锁死"。给这三维一个小的随机游走 `Q` 是下一步，但会改变基线行为，应当
  单独整定和验证。
- 收敛判据用的是状态差的普通范数，11 维里混了米、米每秒和弧度，偏保守：达不到
  阈值只会多跑几次迭代，由 `max_iterations` 兜住。
