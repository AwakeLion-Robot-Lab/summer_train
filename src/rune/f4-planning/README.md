# 能量机关决策、弹道与轨迹规划

参考 [Alliance-Algorithm/rmcs_auto_aim_v2](https://github.com/Alliance-Algorithm/rmcs_auto_aim_v2) 的 `RuneModel::State::get_aimpoints()` 和 `FireController::evaluate_blade()/aim()`。上游 MIT 许可保存在 [ALLIANCE-LICENSE](../f3-estimator/ALLIANCE-LICENSE)。本层沿用已有三维状态和运动预测，弹道调用本工程 `L4Planning::BallisticSolver`，MPC 调用现有 `solveTinyMpc()` 与 `makeTinyMpcSamples()`。

| 文件 | 职责 |
| --- | --- |
| `planning.cpp` | 生命周期、预热、叶片有效期、目标保持及规划结果组装 |
| `intercept.cpp` | 延迟与飞行时间的固定点迭代 |
| `ballistic_solver.cpp` | 现有装甲板弹道求解器的薄适配层 |
| `trajectory.cpp` | 角度前馈、居中参考轨迹、现有 TinyMPC 的调用与回退 |

## 选叶与预热

使用 `FilterState::target_sector` 将观测中的相对扇区转换为固定叶片身份。首次按固定编号选择未激活叶片；当前叶片仍有效时继续保持，避免检测顺序变化导致切换。优先叶片弹道不可达时，再按编号尝试其他有效未激活叶片。新观测确认激活立即撤销候选，短时遮挡保留最多 `inactive_timeout_s`（默认 100 ms）。重复规划同一采集帧不会刷新有效期。没有滤波身份时，相对扇区仅在当前跟踪段内有效，发生换片会重新预热。

小符默认预热 3 s，大符 6 s，与参考仓库的瞄准预热相对应；还必须满足 f3 的 `Motion::ready`。大小符的运动模型仍分别位于 f3 的 `motion/` 目录。跟踪断层、模式变化或历史重置会清空选叶状态并重新预热。切换同一机构的叶片不重置机构运动历史。

有有效三维位姿而运动未就绪、预热未完成或无可射叶片时，默认输出 `center_hold=true` 的符心跟随，禁止开火。位姿非法、观测过期、弹速无效或弹道失败时输出无效规划。Processor 在没有可重建的未激活叶片时仍遵循上游几何层的失败返回，调用者可使用 f5 的 `safeHold()` 保持最后方向。

## 坐标与提前量

`Pose` 输入保持曝光时刻的枪口坐标。提供 `q_world_barrel` 时，将中心、法向和半径旋转到重力对齐的绝对坐标，再使用现有弹道求解器。f2 会从曝光时刻的四元数或完整世界位姿填充该字段。没有姿态时按枪口轴已经与重力对齐处理；运动云台应提供实测姿态。

输出 yaw 为绝对方位角，pitch 向上为正，均为弧度。`aim_point_world` 表示绝对坐标轴下、以曝光时刻枪口为原点的相对位置，不能当作带世界原点平移的定位点。规划假设预测区间内机构中心和枪口平移保持不变，尚未外推车体平移。

每次求解使用 `图像年龄 + 发射延迟 + 飞行时间` 预测靶心，然后重新解弹道直至飞行时间收敛。大符调用 f3 的解析相位增量，小符可用恒速外推。所有 MPC 采样点也分别求解飞行时间，避免整段轨迹套用同一个飞行时间。

真空/线性空气阻力模型直接复用 `src/armor/l3_planning/ballistic_solver.cpp`，没有另写弹道算法。阻力配置为 `enable_air_resistance` 与 `linear_drag_coefficient`；后者单位 s^-1。`yaw_offset_rad`、`pitch_offset_rad` 作用于直接瞄准和整段轨迹。

## MPC

`PlanningConfig::mpc` 沿用 `L4Planning::PlannerConfig` 中的 TinyMPC 权重、ADMM 门限及加速度约束，默认启用。参考窗口为 100 点、间隔 10 ms，以当前执行时刻为第 50 点，输出第 50～99 点，共 50 个可执行样本，覆盖当前至 +490 ms。每个样本的目标仍是该指令对应的弹丸命中位置。

内置拟合器通过独立的 `Motion::phase_delta_signed_rad` 快照支持窗口前半段回推；原 `phase_delta_rad` 仍拒绝负时间。外部大符模型若只支持向前预测，无法构造完整居中窗口时自动使用直接瞄准。yaw 参考逐点展开，避免跨 ±π 产生虚假的高速跳变。

MPC 延用项目现有的参考轨迹初态，属于带加速度约束的参考优化，不将实测云台状态强行注入求解器。实际云台到位由 f5 独立检查。求解不收敛、参考缺失或配置不被求解器接受时，保留直接弹道和前馈；`using_MPC=false`、样本清空，原因见 `mpc_diagnostics`。中心跟随直接输出，不运行 MPC。

规划不产生开火脉冲。`AimPlan::fire_permitted` 只代表目标、预热与弹道条件满足，最终由 [f5-control](../f5-control/README.md) 判断。
