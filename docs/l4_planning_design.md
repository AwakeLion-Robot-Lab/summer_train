# L4 规划层设计

## L4 在解什么

要打的不是"目标现在在哪",而是"子弹落地那一刻目标在哪"。命中时刻取决于飞行时间,
飞行时间又取决于命中点的位置——这是一个不动点方程:

```
t_hit     = t_image + delay.beforeFire() + fly_time
aim_point = 选板( 整车模型外推到 t_hit )
fly_time  = 弹道( aim_point 的水平距离 d 和高度 h )
           ↑________________________________________|
```

`Planner::plan()` 以 `fly_time = 0` 起步反复代入,直到相邻两次飞行时间之差小于
`PlanConfig::fly_time_tolerance`(默认 100 μs),上限 `max_iterations`(默认 10)。
未收敛按 `BallisticFailed` 拒绝,不输出可疑解。

结构对标 sp_vision 的 `Aimer`,选板内聚为 `Planner` 的私有方法而不单独立类。

## 组件

| 组件 | 职责 |
| --- | --- |
| `Predictor` | 整车模型外推 + 展开物理装甲板,纯函数 |
| `BallisticSolver` | 给定 (d, h, v0) 解 pitch 和 fly_time |
| `Planner` | 编排不动点迭代;私有 `chooseAimPoint()` 负责选板与迟滞 |
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

## 三个关键点

### 1. 外推必须同时推进中心和 yaw

装甲板位置由 (旋转中心, 整车 yaw, 半径) 共同决定。改造前的 `Predictor::predict()`
只有 `position += velocity * dt`,**没有推进 yaw** —— 小陀螺目标会被算成原地不动。
`v_yaw` 取 10 rad/s 时,100 ms 延迟对应 57° 偏差,这恰恰是延迟补偿要解决的主要误差。

为此 `TargetState` 新增 `armor_num`、`second_radius`、`height_diff` 三个字段,让 L4
能自行展开全部装甲板并外推,不必反向依赖 `Tracker`(那只能给当前帧的装甲板位置)。
这保持了"跨层只暴露不可变数据快照"的约定,`Planner` 也因此可以脱离硬件单测。

### 2. `plan_time` 由调用方传入

sp_vision 的 `Aimer::aim` 在 `to_now` 分支里直接读 `steady_clock::now()`
(`aimer.cpp:47`),这让 `aim()` 变成非纯函数,同一段回放跑两次结果不同。newvision 有
离线回放 harness,`plan()` 必须保持纯函数:运行时传 `now()`,回放传录制的时间戳。

### 3. 选板迟滞是必须项,不是优化项

两块装甲板都接近 45° 时会逐帧互换,命令抖动到云台根本跟不上。`chooseAimPoint()`
用 `locked_id_` 做锁定,新候选必须好过 `switch_hysteresis`(默认 5°)才允许换板。

选板分两档,沿用 sp 的策略:

- **低速档**(`|v_yaw| ≤ spin_threshold`,默认 2 rad/s):挑法线夹角最小且在
  `max_face_angle`(60°)窗口内的板,带锁定迟滞。
- **反陀螺档**:只打正在**转入**视野的一侧。`v_yaw > 0` 时 `delta_angle` 递增,尚未
  越过 `leaving_angle` 的板才是转入的那块;反向旋转时判据镜像。转出侧的板等子弹飞
  到时已经背对枪口。高速旋转下窗口内可能一块板都没有,这是正常间歇,不算错误。

**修正了 sp 的一个真 bug**:`aimer.cpp` 的档位判据写的是 `ekf_x()[8]`,那是**半径**
而不是 `v_yaw`(索引 7)。半径恒在 0.05~0.5 之间、永远小于阈值 2,导致 sp 的反陀螺
分支对非前哨站目标是**死代码**。同一函数后面用 `ekf_x[7]` 判旋转方向,说明 `[8]` 是笔误。

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

弹速为 0 是裁判系统上电初期的正常值:此时用 `fallback_bullet_speed`(23 m/s)仍然解算
并输出瞄准角,但把 `PlanError::BadBulletSpeed` 记进 `Plan`,由 L5 拒绝开火。

## 给 MPC / 五次多项式留的接口

`IPlanner`(`planner_interface.hpp`)定义统一契约,三种实现共用同一份输入输出:

| 实现 | 状态 | 说明 |
| --- | --- | --- |
| `Setpoint` | **已实现**(`Planner`) | 只解命中点,速度和加速度保持 0 |
| `QuinticSwitch` | 未实现 | 只在切板造成的轨迹断点处插入五次多项式过渡段,跟随段仍贴合原射击轨迹 |
| `TinyMpc` | 未实现 | 全程用 MPC 约束云台角加速度 |

三者差别只在 `Plan` 的 `yaw_vel` / `pitch_vel` / `yaw_acc` / `pitch_acc` 是否被填充,
以及 `type` 字段。**L5 不需要知道用的是哪一种。**

输入用 `PlanInput` 结构体而不是参数列表,后续加字段不必改动已有实现的签名。已经预留:

- `PlanInput::gimbal_yaw_vel` / `gimbal_pitch_vel` —— 轨迹起点的速度边界条件。下位机
  尚未回传时为 `nullopt`,实现方应据此退化而不是假定为 0。
- `PlanConfig::max_yaw_acceleration` / `max_pitch_acceleration` —— 五次多项式靠它决定
  过渡段时长(逐步增大直到峰值加速度落在限内),MPC 用作硬约束。实车标定前为空值,
  `trajectoryLimitsReady()` 供实现方判断是否必须退化成定点输出。

## 验证

```bash
xmake run planner_smoke     # 9 个用例,无需硬件
```

覆盖:yaw 外推、平动外推、弹道往返一致性(解出的 pitch 代回抛体方程还原目标高度)、
弹道非法输入拒绝、不动点收敛、弹速异常标记、无目标/丢失拒绝、选板迟滞(20 帧扰动
零跳变)、反陀螺占空比。

## 未做的事

- `plan_to_send` 需要 L5 回填后才有值。
- 线性空气阻力 `dv/dt = -k*v - g`。`BallisticSolver` 的构造函数已接受
  `drag_coefficient`,`k = 0` 时严格退化真空解,升级不必改接口(参考
  `Climber_Vision_26` 的 `AirResistTrajectory`,用 Ceres 解隐式方程)。
- `Planner` 尚未挂进 `runtime::AutoAimRuntime::run()` 的主循环。
- `FireConfig::shoot_enable` 保持 `false`,实车验收前不解锁开火。
