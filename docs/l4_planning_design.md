# L4 规划层设计

## L4 在解什么

要打的不是"目标现在在哪",而是"子弹落地那一刻目标在哪"。命中时刻取决于飞行时间,
飞行时间又取决于命中点的位置——这是一个不动点方程:

```
t_hit     = t_image + delay.beforeFire() + fly_time
aim_point = 整车模型外推到 t_hit 后选中那块板的位置
fly_time  = 弹道( aim_point 的水平距离 d 和高度 h )
           ↑________________________________________|
```

参考实现:sp_vision 的 `Aimer`。整个 L4 只有一个类 `Planner`(`l4_planning/armor/planner.hpp`),
选板是它的私有方法 `choose()`,真空弹道是 `planner.cpp` 里的 `solveTrajectory()`。

## 坐标系:弹道直接在枪管系解

`PnpSolver` 把相机系观测先经 `T_barrel_camera` 变到枪管系,再用纯旋转 `R_world_barrel`
变到世界系。world 系原点**就是**枪管原点,枪口平移已经被外参吸收,所以
`d = hypot(x, y)`、`h = z` 直接可用,不需要再加枪口偏置。

## 一、外推必须同时推进中心和 yaw

装甲板位置由 (旋转中心, 整车 yaw, 半径) 共同决定。只推中心不推 yaw 时,小陀螺目标会被
算成原地不动:`v_yaw` 取 10 rad/s 时,100 ms 延迟对应 57° 偏差,这恰恰是延迟补偿要解决的
主要误差。`Planner` 在目标副本上调 `EskfTarget::predict()`,它按整车运动模型同时推进两者,
`planner_smoke` 的 `testPredictAdvancesYaw` 钉住这一点。

## 二、命中时刻迭代

与 sp 相同:先把目标外推到预计发射时刻(`delay.beforeFire()`),选板、解一次弹道得到
初始飞行时间;之后每轮都从**同一个发射时刻状态**重新外推 `fly_time`、重新选板、重新解弹道,
直到相邻两次飞行时间之差小于 `fly_time_tolerance`,或跑满 `max_iterations`。

选板在循环内部,窗口边界附近换板会让飞行时间来回跳。双板同时在窗口内时的锁定迟滞
(见下一节)压住了这种来回切换;`testSharedIterationProducesFiniteCommands` 扫过
4 种转速 × 120 个 yaw 构型,要求每帧都有有限的命令角。

## 三、选板

输出的 `Plan::aim` 和 `Plan::fire` 都对应一块真实装甲板,不生成车辆中心代理点:

- 目标尚未发生跳板(`EskfTarget::jumped` 为假)时固定用 0 号板:其余板的位置还只是初值。
- 常规车:取与车心方向夹角在 `coming_angle` 内的板;两块都在时锁住其中一块,直到它离开
  窗口才切换,只剩一块时解除锁定。
- 前哨站(或半径异常的目标):按旋转方向用 `coming_angle` / `leaving_angle` 排除即将转走
  的板,前哨站用单独的一组角度。
- 窗口里一块都没有时规划失败(`PlanError::OutOfWindow`),云台保持上一条命令、关火。
  高速小陀螺下这是正常的击发间歇。

进自瞄后的头一次选板是例外:`reset()`(runtime 在 Idle 每帧调)标记新一轮自瞄,下一次成功
规划若有两块候选板,锁离枪口最近的那块而不是更正对的那块,之后照常沿用锁、等它离开窗口再
切。需要 `PlanInput::q_world_barrel`,缺省时退回原规则;前哨站的转向分支不受影响。回放统计
(3 m,6 段录像):15~27% 的帧两条规则选的板不同,这些帧第一条命令离枪口的 yaw 差中位数
少 5~8°,距离越近差得越多。

## 四、弹道

真空闭式解:把 `tanθ` 当未知量的一元二次方程,两条解里取飞行时间短的低弧。判别式为负
(打不到)时规划失败(`PlanError::BallisticFailed`)。

空气阻力暂不建模。以前有过一套 `BallisticSolver` + 阻力模型,但 `Planner` 从没接上它,
已删除;要加阻力时在 `solveTrajectory()` 里换模型,并重跑 `planner_smoke` 的往返用例。

## 五、弹速

下位机回传的弹速不可信(没发时是 0,或低于 `min_valid_bullet_speed`)时,用
`default_bullet_speed`(23 m/s)解弹道,**计划照常有效、照常可开火**。切换到缺省值和切回
实测值时各打一条日志,不是每帧都打。

## 六、规划结果只有一个状态字段

`Plan::error` 为 `PlanError::None` 时 `valid()` 为真,`aim` 与 `fire` 都已填好;否则本帧
没有新的瞄准角,L5 保持上一条命令并关火。不再有"可以跟随但不许开火"的中间态:是否开火
完全由 L5 判定,L4 只回答"有没有角度可打"。

| `PlanError` | L5 的拒绝原因 |
| --- | --- |
| `None` | — |
| `NoTarget` | `no_target` |
| `OutOfWindow` | `out_of_window` |
| `BallisticFailed` | `ballistic_failed` |

L5 的每条拒绝原因只对应一种情况:规划失败时只记上表那一条,不会再叠一条 `aim_error`。

## 七、`plan_time` 由调用方传入

sp_vision 的 `Aimer::aim` 在 `to_now` 分支里直接读 `steady_clock::now()`,这让 `aim()` 变成
非纯函数,同一段回放跑两次结果不同。newvision 有离线回放 harness,`plan()` 必须保持纯函数:
运行时传 `now()`,回放传录制的时间戳并置 `to_now = false`(`image_to_plan` 固定按 5 ms)。

## 延迟

按 `Delay` 的五段拆分记录,不塌缩成标量:

| 段 | 来源 |
| --- | --- |
| `image_to_plan` | 曝光时刻到规划时刻,runtime 实测 |
| `plan_to_send` | 规划结束到串口发出,runtime 实测,用上一帧的值 |
| `send_to_control` | 实车标定,`planning.send_to_control_ms`,不写按 0 |
| `control_to_fire` | 按整车 v_yaw 分高低速两档,`high_speed_delay_ms` / `low_speed_delay_ms` |
| `fire_to_hit` | 弹道飞行时间 |

## 给 MPC / 五次多项式留的位置

`Plan` 目前只提供位置参考。轨迹规划器接入时,速度和加速度应作为一组可选 reference 扩展,
而不是重新铺成多个独立字段;起点状态和云台约束作为一个完整配置组加入 `PlanConfig`。
`plan/mpc` 和 `plan/quintic-blend` 两条分支就是这样接的。

## 验证

```bash
xmake run planner_smoke     # 无需硬件
xmake run fire_decision_smoke
```

估计和规划数值的回归用 `track_diag` 的 `aim.csv` 逐字节比对。

## 未做的事

- **多目标加权选择**。`docs/target_selection_and_vehicle_tracking.md` 描述的打分与切换
  迟滞需要 L3 维护多个候选目标,目前只有一个。
- 空气阻力。
- `FireConfig::shoot_enable` 保持 `false`,实车验收前不解锁开火。
