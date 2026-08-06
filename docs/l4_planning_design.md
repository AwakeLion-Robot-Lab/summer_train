# L4 规划层设计

## L4 在解什么

要打的不是"目标现在在哪",而是"子弹落地那一刻目标在哪"。命中时刻取决于飞行时间,
飞行时间又取决于命中点的位置——这是一个不动点方程:

```
t_hit     = t_image + delay.beforeFire() + fly_time
aim_point = 整车模型外推到 t_hit 后第 i 块板的位置
fly_time  = 弹道( aim_point 的水平距离 d 和高度 h )
           ↑________________________________________|
```

参考实现:sp_vision 的 `Aimer`、Climber_Vision 的 `Aimer`/`Planner`、talos 的 `L4_planning`。
结构对标 sp 的 `Aimer`(选板内聚为私有方法,不单独立类),关键算法取 talos 的做法。

## 组件

| 组件 | 职责 |
| --- | --- |
| `Predictor` | 整车模型外推 + 展开物理装甲板,纯函数 |
| `IBallisticModel` | 正向弹道:给定发射角算落点高度和飞行时间 |
| `BallisticSolver` | 反向弹道:给定 (d, h, v0) 解 pitch 和 fly_time |
| `AimPhaseTracker` | 随转速升降瞄准档位,带双层迟滞 |
| `Planner` | 编排逐板不动点迭代、选板、火控门控 |
| `IPlanner` | 规划器接口,供后续 MPC / 五次多项式实现 |

## 坐标系:弹道直接在枪管系解

`pnp_solver.cpp:249-251`:

```cpp
const Eigen::Vector3d xyz_in_barrel = R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
const Eigen::Vector3d xyz_in_world  = R_barrel2world_ * xyz_in_barrel;   // 纯旋转
```

world 系原点**就是**枪管原点,枪口平移已经被 `t_camera2barrel_` 吸收。所以
`d = hypot(x, y)`、`h = z` 直接可用,不需要再加枪口偏置。这正是
`docs/pure_cpp_auto_aim_route.md:113` 要求的"弹道在枪口坐标解算",结构上已经满足。

---

## 一、外推必须同时推进中心和 yaw

装甲板位置由 (旋转中心, 整车 yaw, 半径) 共同决定。改造前的 `Predictor::predict()`
只有 `position += velocity * dt`,**没有推进 yaw** —— 小陀螺目标会被算成原地不动。
`v_yaw` 取 10 rad/s 时,100 ms 延迟对应 57° 偏差,这恰恰是延迟补偿要解决的主要误差。

为此 `TargetState` 新增 `armor_num`、`second_radius`、`height_diff`,让 L4 能自行展开
全部装甲板并外推,不必反向依赖 `Tracker`(那只能给当前帧的装甲板位置)。这保持了
"跨层只暴露不可变数据快照"的约定,`Planner` 也因此可以脱离硬件单测。

## 二、不动点迭代逐板跑,收敛后再选板

**这是与 sp / Climber 最重要的一处差异,取自 talos。**

sp 和 Climber 把选板放在迭代循环**内部**:每次迭代重新外推、重新选板、重新解弹道。
问题是这个循环可能不收敛:

```
飞行时间变了 → 外推的 yaw 变了 → 选中的板变了 → 距离变了 → 飞行时间又变了
```

在窗口边界附近两块板交替胜出,`fly_time` 在两个吸引子之间来回跳,10 次迭代跑完仍不
满足收敛判据 —— 而边界附近恰恰是最需要出解的时刻。

talos 的 `aim_generic` 反过来:对**每块板**各跑一次 `refine_flying_time`(板号在迭代
中固定,目标不变,必然收敛),得到 N 组收敛解,然后在这 N 组里选板。代价是 N 倍的弹道
求解,而弹道是闭式解,可以忽略。

`Planner::refineArmor()` 实现前者,`Planner::selectArmor()` 实现后者。
`testPerArmorFixedPointAlwaysConverges` 扫过 4 种转速 × 120 个 yaw 构型,要求 480 帧
全部收敛。

## 三、选板与火控分离

改造前:反陀螺档在 coming/leaving 窗口里找不到板时返回"无瞄准点",`Plan::valid` 为
false。后果是**高速小陀螺的正常击发间隙里,云台会停止跟随** —— 等窗口回来时枪口已经
指偏了。

talos 的 `ControlIntent` 把这件事拆成三态 variant:`TrackCommand`(跟随轨迹)、
`ShotCommand`(直接瞄准,带 `degradation_reason`)、`HoldCommand`(无目标)。newvision 的
L5 目前是桩,不值得为此上 variant,取其语义即可:

| 字段 | 含义 |
| --- | --- |
| `Plan::valid` | 云台该不该跟随这个角度。**有目标就该跟**,哪怕当前打不中 |
| `Plan::fire_admissible` | 这一帧允不允许考虑开火 |
| `Plan::error` | `valid == true` 时是降级原因,`false` 时是拒绝原因 |

于是 coming/leaving 窗口从**选板判据**改成**火控判据**。这也是它本来该在的位置:
coming/leaving 回答的是"子弹飞到时这块板还正对枪口吗",那是开火问题不是指向问题。
`testFireWindowGatesWithoutDroppingAim` 断言整圈 60 帧云台**帧帧有角度**,而可开火的
只有 36 帧。

选板改用 talos 的三级回退:**锁定板 → 前置窗口内夹角最小 → 全局夹角最小(标记降级)**。
最后一级保证永远有输出。

## 四、瞄准档位阶梯

单一 `spin_threshold` 有两个毛病:`v_yaw` 是 EKF 估计量,在阈值附近会抖,每抖一次换一次
策略;而且转速再高时"瞄某块板"本身就失效了。

talos 的 `ArmorAimPhase` 是一个带双层迟滞的阶梯,这里实现为 `AimPhaseTracker`:

| 档位 | 触发 | 瞄准策略 |
| --- | --- | --- |
| `SingleArmor` | 低速 | 盯正对枪口的板,允许锁定迟滞 |
| `WholeCarArmor` | `> 1.5` rad/s | 仍瞄实体板,选板跟着旋转走 |
| `WholeCarCenter` | `> 16.5` rad/s | 瞄旋转圆上离枪口最近的点,等板扫过来 |

两层迟滞:

1. **施密特触发** —— 上行 1.5 / 下行 1.0,上行 16.5 / 下行 15.0,中间是死区;
2. **计数确认** —— 条件连续成立 `transfer_count`(默认 30 帧,200 fps 下约 150 ms)才换档。

talos 在中间还有一档 `WholeCarPair`。查过它的 `predict_aim_point` 和 `select_armor_id`:
两处**都只对 `WholeCarCenter` 分支**,`WholeCarPair` 的瞄准行为与 `WholeCarArmor` 完全
一致,只是迟滞阶梯上的一级。这里合并成三档,迟滞由计数器本身保证。

`WholeCarCenter` 的瞄准点由 `projectCenterAim()` 给出:从枪口指向旋转中心的射线上退回
一个半径,即旋转圆上离枪口最近的那个点。高度取实体板的高度而不是中心高度。
**火控在这一档仍然按实体板判定**(`fire_armor_id` / `fire_delta_angle`),否则中心档会在
两块板之间的空档里照样开火 —— 这对应 talos 的 `build_fire_reference_trajectory` /
`fire_aim_phase()`。

## 五、可观测性门禁:整车几何没被观测过就不许整车瞄准

只见过一块板时,整车 yaw、`r2-r1`、`z2-z1` 几乎不可观测 —— 其余板的位置完全由初值猜
出来。此时瞄别的板等于**拿伪造的几何去开火**,与"缺失标定保持缺失,绝不用单位阵顶替"
是同一条约定。

sp 没有这道门禁。Climber 有 `Target::jumped`,talos 有 `TrackerOutput::target_jumped`,
两者都用它决定"能不能相信整车模型"。newvision 的 `TrackedTarget` 已经有 `jumped`,但语义
是**逐帧**的("这一帧关联到的是哪块板"),不是需要的那个。因此新增粘滞版本:

```cpp
// TrackedTarget
bool jumped{false};                 // 这一帧看的是哪块板
bool multi_armor_observed{false};   // 整车几何到底可不可观测(只增不减)
```

`multi_armor_observed` 导出到 `TargetState`,`Planner` 据此:

- `AimPhaseTracker::update()` 强制留在 `SingleArmor`;
- `selectArmor()` 直接返回 0 号板,不参与窗口比较。

这个改动对 L3 是**只写不读**的,`auto_aim_test` 回放的 Tracking 帧数保持 462 不变。

## 六、弹道:模型与求解器拆开

talos 的 `core/trajectory` 把正向物理(`BallisticModel::compute_impact`)和反向求解
(`TrajectorySolver::solve`)拆成两层。这里照搬:

```cpp
struct Impact { double z; double fly_time; };
class IBallisticModel {
  virtual std::optional<Impact> impact(double range, double pitch, double v0) const = 0;
};
```

| 模型 | 正向解 |
| --- | --- |
| `VacuumModel` | `t = d/(v0 cosθ)`,`z = v0 sinθ·t - gt²/2` |
| `LinearDragModel` | `t = (e^{kd}-1)/(k v0 cosθ)`,z 同上 |

`BallisticSolver` 用真空闭式解(把 `tanθ` 当未知量的一元二次方程,取飞行时间短的低弧)
作为初值:真空模型直接返回,有阻力时走 talos `DirectSolver` 的高度补偿迭代 —— 把实际落
点与目标的高度差累加回瞄准高度,重新求角,直到落差小于 `height_tolerance`。这个迭代天然
收敛到低弧,因为起点就在低弧一侧。

**这样就不需要 Ceres。** Climber_Vision 的 `AirResistTrajectory` 在两个方向都加阻力,
反解隐式方程只能上 Ceres;talos 的模型只在水平方向加阻力,换来正向闭式解。17mm 弹丸在
10 m 内竖直速度远小于水平速度,这一项的影响比水平衰减小一个量级 —— 用一点物理精度换掉
一个重型依赖,对 newvision 是划算的。

`k = 0` 时 `LinearDragModel` 严格退化成真空解(用 `expm1` 处理 0/0),
`testLinearDragDegradesToVacuum` 断言两者逐位一致。`k = 0.02` 时 6 m 处需要多抬 0.21°。

## 七、弹速门限只留一处

裁判系统上电初期回传 0 是正常的。此时用 `fallback_bullet_speed`(23 m/s)仍然解算并输出
瞄准角,但把 `PlanError::BadBulletSpeed` 记进 `Plan`,由 L5 拒绝开火。

门限**只在 `PlanConfig::min_valid_bullet_speed` 一处**。`BallisticSolver` 不再自带业务
门槛,只拒绝数学上无解的输入(`v0 < 1e-3`)。两处各设一道且数值不一致的话,落在夹缝里的
弹速会既不触发兜底、又被求解器拒绝,最后报成 `BallisticFailed` 而不是 `BadBulletSpeed`,
把真正的原因藏掉。`testPlannerFlagsBadBulletSpeed` 专门盯住这个回归。

## 八、`plan_time` 由调用方传入

sp_vision 的 `Aimer::aim` 在 `to_now` 分支里直接读 `steady_clock::now()`(`aimer.cpp:47`),
这让 `aim()` 变成非纯函数,同一段回放跑两次结果不同。newvision 有离线回放 harness,
`plan()` 必须保持纯函数:运行时传 `now()`,回放传录制的时间戳。

## 延迟

仍按 `Delay` 的五段拆分记录,不塌缩成标量:

| 段 | 来源 |
| --- | --- |
| `image_to_plan` | 曝光时刻到规划时刻,直接可测 |
| `plan_to_send` | 规划到下发,由 L5 回填 |
| `send_to_control` | 需实车标定,来自 `PlanConfig`,未标定时为 0 |
| `control_to_fire` | 需实车标定,来自 `PlanConfig`,未标定时为 0 |
| `fire_to_hit` | 弹道解算填入 |

未标定的段保持 0 并由 `PlanConfig::fireDelayReady()` 拦住开火——**绝不用猜测值填补**,
否则火控门禁会被静默绕过。

Climber 按转速分了 `high_speed_delay_time` / `low_speed_delay_time` 两档常量延迟。那是
把未标定的段用经验值顶上,与上面的约定冲突,**不采用**。

## 给 MPC / 五次多项式留的接口

`IPlanner`(`planner_interface.hpp`)定义统一契约,三种实现共用同一份输入输出:

| 实现 | 状态 | 说明 |
| --- | --- | --- |
| `Setpoint` | **已实现**(`Planner`) | 只解命中点,速度和加速度保持 0 |
| `QuinticSwitch` | 未实现 | 只在切板造成的轨迹断点处插入五次多项式过渡段 |
| `TinyMpc` | 未实现 | 全程用 MPC 约束云台角加速度 |

三者差别只在 `Plan` 的 `yaw_vel` / `pitch_vel` / `yaw_acc` / `pitch_acc` 是否被填充,
以及 `type` 字段。**L5 不需要知道用的是哪一种。**

输入用 `PlanInput` 结构体而不是参数列表,后续加字段不必改动已有实现的签名。已经预留:

- `PlanInput::gimbal_yaw_vel` / `gimbal_pitch_vel` —— 轨迹起点的速度边界条件。下位机尚未
  回传时为 `nullopt`,实现方应据此退化而不是假定为 0。
- `PlanConfig::max_yaw_acceleration` / `max_pitch_acceleration` —— 五次多项式靠它决定过渡
  段时长,MPC 用作硬约束。实车标定前为空值,`trajectoryLimitsReady()` 供实现方判断是否必须
  退化成定点输出。

真要上 MPC 时,talos 和 Climber 走的是同一条路,可以直接抄:
`ReferenceTrajectory`(4×horizon 的 `[yaw, yaw_rate, pitch, pitch_rate]` 状态矩阵,由
`Aimer` 在 horizon 上逐点采样、有限差分出速度)+ TinyMPC 求解器。Climber 的
`tasks/auto_aim/planner/tinympc/` 是可直接移植的实现。

## 验证

```bash
xmake run planner_smoke     # 15 个用例,无需硬件
```

覆盖:yaw 外推、平动外推、真空弹道往返一致性、`k=0` 退化、阻力抬头量与正向回代、弹道非法
输入拒绝、求解器不含业务门限、档位阶梯的死区与计数迟滞、不动点收敛、弹速夹缝报错、
无目标/丢失拒绝、可观测性门禁锁 0 号板、选板迟滞、火控门控不掉瞄准、中心代理点几何、
480 构型逐板收敛。

## 未做的事

- `plan_to_send` 需要 L5 回填后才有值。
- **多目标加权选择**。talos 的 `armor_target_decider` 按 image_center / track_state / tof /
  gimbal_effort / armor_name 五项加权打分,并用 `switch_margin` 做切换迟滞;
  `docs/target_selection_and_vehicle_tracking.md` 描述的正是同一套设计。但 newvision 的
  `Tracker` 目前只维护**一个** `TrackedTarget`,没有多候选可选,做这件事要先改 L3。
- `Planner` 尚未挂进 `runtime::AutoAimRuntime::run()` 的主循环。
- 空气阻力系数 `k` 需要实车打靶标定,默认 0(真空)。
- `FireConfig::shoot_enable` 保持 `false`,实车验收前不解锁开火。
