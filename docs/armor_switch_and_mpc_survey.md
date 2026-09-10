# 切板抖动：MPC 系开源的做法，以及本项目的差距

调研时间 2026-09-10。起因是实机回放里反复出现"换板之后又甩回原来那块"，
本文记录四套带 MPC 的开源自瞄怎么处理这件事，以及本项目 `chooseAimPoint`
的对应缺陷。

结论先行：**没有一套 MPC 方案是靠"把切板时刻预测准"来解决的，它们全都
取消了"预测切板时刻"这个概念。**

## 一、被调研的四套实现

| 项目 | 位置 | 求解器 | 备注 |
| --- | --- | --- | --- |
| sp_vision 2025 | `sp_vision_25-main/tasks/auto_aim/planner/` | TinyMPC | 同济 SuperPower 开源，本项目的架构原型 |
| jlu_vision 2026 | `jlu_vision_26-master/src/auto_aim/armor_tracker/` | TinyMPC | README 自述"使用同济大学开源的MPC规划器" |
| awakening | `awakening-main/src/tasks/auto_aim/armor_control/very_aimer.cpp` | 自带 dual_small_mpc | 唯一带完整降级状态机的 |
| auto_aim_rps26 | `auto_aim_rps26-master/src/rm_auto_aim/rm_angle_solver/` | TinyMPC | 不走 MPC 轨迹，但选板策略最完整 |

## 二、共同骨架

前三套的 MPC 骨架几乎逐字相同：

```cpp
// 参考轨迹：-0.5s 走到 +0.5s，每 DT=10ms 一个点，共 HORIZON=100
for (int i = 0; i < HORIZON; i++) {
  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed);   // 每个点都重新选板
  traj.col(i) << ...;
}
tiny_set_x0(solver, traj.col(0));               // 起点是 0.5 秒前
solver->work->Xref = traj;
tiny_solve(solver);
plan.yaw = solver->work->x(0, HALF_HORIZON);    // 取中点输出
```

三个要害：

1. **切板不是一个决策，是参考轨迹里的一个台阶。** 选板函数在 horizon 的每个
   点上重算，未来某刻要换板，台阶就自然出现在 `Xref` 里。没有"切板时刻"这个
   变量，也就无所谓预测错。
2. **加速度限幅 `u ∈ [−a_max, a_max]` 是唯一的缓冲。** 台阶来了 QP 爬不动，
   输出就是一条受限斜坡。
3. **输出取 horizon 中点——一半在过去，一半在未来。** 非因果的零相位滤波，
   中点的解已被前半秒的参考带跑起来了。

合起来的效果是：**预测错了不用管。** 下一帧整条 `Xref` 重算重解，错误参考
最多存活一帧。一帧 34ms 内 QP 能跑出的位移是 ½·a_max·τ² ≈ ½·50·0.034²
≈ 0.029 rad ≈ **1.6°**。甩不回来，因为没甩出去。

jlu 的 README 把这点说得最直白：

> 至于斜对着抑制开火避免打舵的火控就交给**切板跳变时 MPC 撞到加速度限制**
> 来完成了

## 三、六个具体机制

### 1. 加速度限幅当低通（四家全有）

| 项目 | max_yaw_acc | max_pitch_acc |
| --- | --- | --- |
| sp_vision | 50 | 100 |
| jlu | 50 | — |
| awakening | 40 | 50 |
| 本项目（继承自 sp） | 50 | 100 |

本项目现有的 50/100 落在同行区间内。仍应向机械/电控要真值，但不是悬空的。

MPC 权重上 sp 和 jlu 都用 `Q_yaw: [9e6, 0]`、`R_yaw: 1`——只罚位置误差，
不罚速度误差。

### 2. 相对死区，而不是绝对门限（jlu、rps26）

本项目比的是单块板的角度对绝对门限 `coming_angle = 60°`。这两家比的是
**在任板 vs 挑战者**：

```cpp
// rps26 getClosestArmor，switch_threshold = 10°
if (abs(angle_i) + D2R(switch_threshold) < abs(angle_incumbent)) index = i;

// jlu selectArmor
auto facing_angle_diff = getArmorFacingAngleAbs(last) - min_facing_angle_abs;
if (facing_angle_diff > config_.armor_switch_facing_degree_diff_thres)
  return selected_index;      // 新板明显更正对才切
return last_armor_index_.value();
```

两块板的角度随车身旋转和 EKF 噪声是**一起动**的，差值远比任一绝对值稳定。
"外推越过 60° 又被拉回来"这个失效模式对差值量不成立。

注意 jlu 出厂配置是 `armor_switch_facing_degree_diff_thres: 0.0`，即**没有
启用**，实际靠机制 1 兜底。rps26 出厂 10°。

### 3. 结构性锁：只在歧义时生效（awakening）

```cpp
候选 = 所有 |delta_angle| <= 60° 的板（最多 2 块）
if (candidate_count == 1) { pick = candidates[0]; lock_id = -1; }
else {  // 两块都在窗口内 = 歧义区
    if (lock_id 不在候选里) lock_id = 更正对的那块;
    pick = lock_id;         // 否则死保持
}
```

切板由"某块板**离开**窗口"这个单调事件触发，而不是由一个可以来回翻转的
比较触发。本项目 `chooseAimPoint` 主分支已经是这套逻辑（见第四节）。

### 4. 跟不上就降级（awakening 四级状态机、rps26 锁中心）

awakening `auto_aim_fsm.hpp`：

```
SINGLE_ARMOR → WHOLE_CAR_ARMOR → PAIR(只在 0、2 号板里选) → CENTER
```

由 `|v_yaw|` 驱动，双门限 + 积分停留时间，死区内计数器清零，且必须
`target_jumped` 才允许升级：

```yaml
single_whole_up: 1.5 / down: 1.0
whole_pair_up:   7.5 / down: 6.5   # 1s内至少换2次板子 2pi 否则轨迹规划无意义
pair_center_up: 16.5 / down: 15.0
transfer_time: 0.5
```

`overflow_time_` 是个**双向积分器**：超上限累加、低于下限递减、在死区内归零。
短暂越界不算数，需要持续 0.5s 的证据。

PAIR 模式只在对角的 0、2 号板里选——切板频率减半，且这两块板的瞄准 yaw
几乎相同（都在中心方位角附近），云台基本不用动。源码注释：
"4选2,本质提升控制轨迹与目标轨迹重合窗口"。

rps26 更直接：`switch_track_center_palstance_threshold = 15 rad/s`，超了锁中心。

### 5. 提前量按角度而不是按时间（rps26）

```cpp
double switch_advanced_time = std::min(0.2, D2R(switch_threshold) / abs(palstance));
TargetArmors armors = predictArmors(msg, predict_time + switch_advanced_time);
// ... 在 t+advanced 上选板 ...
return predictArmors(msg, predict_time)[index];   // 但取 t 时刻的位姿
```

按**恒定角度**提前切板，不是恒定时间。转得越快提前量越短，提前量永远不会
冲过头。且提前量只影响"选哪块"，不影响"瞄哪里"。

### 6. 弹道迭代中锁死选板（jlu、awakening 各自独立做了）

jlu README：

> 锁定这块装甲板后进行迭代飞行时间计算（不在每次迭代时重新选板，避免迭代时
> 反复横跳无法收敛）

awakening 留了废弃的尝试：`// auto iter_select = select_armor(i_target, fsm); //不知道哪个最好`。

若锁定板在迭代中转到背对，jlu 是**直接放弃该帧**
（`iterative_max_facing_angle: 75°`），而不是改选别的板——
"这种状况已经违背物理常识了"。

### 附：火控自动免疫切板

sp_vision 的开火判据比的是**参考轨迹 vs MPC 解**：

```cpp
plan.fire = std::hypot(traj(0, HALF+2) - yaw_solver_->work->x(0, HALF+2),
                       traj(2, HALF+2) - pitch_solver_->work->x(0, HALF+2))
            < fire_thresh_;
```

切板瞬态时 MPC 撞加速度限幅、跟不上参考，误差大 → 自动禁火。不需要显式的
"过渡段不开火"门，它是从跟踪误差里掉出来的。

## 四、本项目的差距

### 已经有的

`chooseAimPoint` 主分支（`|ekf_x[8]| <= 2.0`）**已经实现了机制 3**，与
awakening 的 `select_armor` 逐字对应。之前笔记里"没有进出迟滞"的说法不准确。

真正死掉的是 `leaving_angle` 分支：`diverged()` 把半径锁在物理范围内，
`|ekf_x[8]| <= 2.0` 对普通车辆恒成立，那段代码走不到。但那是另一套（面向
旋转目标的）方案，**四家 MPC 都没用类似的东西，救它不是正确的修法**。

### 缺陷一：锁在单候选帧上被销毁

`src/l4_planning/armor/planner.cpp`：

```cpp
// 只剩一块候选时无需迟滞，退出双板锁定。
locked_id_ = -1;          // ← 病灶
return pointAt(ids[0]);
```

复现路径：

| 帧 | 窗口内 | `locked_id_` | 结果 |
| --- | --- | --- | --- |
| 1 | {A} | 被清成 −1 | A |
| 2 | {A, B} | −1，既不是 A 也不是 B → **重新选举** | 谁更正对选谁，**可能直接是 B** |
| 3 | {A}（B 被噪声挤出窗口） | 清成 −1 | **A，甩回去了** |

锁在每个单候选帧上被销毁，于是第二块板一进窗口，锁是空的，立刻改选更正对的
那块。同文件上方的注释"短暂中断不清锁，避免恢复后立即切板"写的是正确意图，
这一行把它推翻了。

**awakening 的 `select_armor` 有同一个洞**（`candidate_count == 1` 时也写
`lock_id = -1`）。区别在下游：它挂的是逐帧重解 + 加速度限幅的 MPC，选错一两帧
只值 1.6°，看不见；本项目挂的是**一次性提交的 200ms 五次多项式**，同一个抖动
被放大成一整条奔向错误板的轨迹。

> 本项目抄了 awakening 的选板逻辑，但没抄它的减震器。那套选板逻辑只有在下游
> 逐帧重解且限加速度时才是安全的。

这也解释了实测中"提交后 1568ms 才真换板"的段落——`nextSwitchTime` 拿着同样
会自毁的锁副本去前视，预测的是一次由噪声触发的伪切换。

### 缺陷二：弹道迭代里重新选板

`chooseAimPoint` 在飞行时间迭代循环内被调用（每帧 1 + `max_iterations` 次），
每次都改写 `locked_id_`——正是 jlu 和 awakening 都明确拒绝的做法。

### 不要照抄的地方

jlu 自己也没解决跨帧缓存：`buildReferenceTrajectory` 每帧调 `solveTarget`
上百次、每次都写 `last_armor_index_`，跨帧留下的是 horizon 末端（t=+0.5s）的
状态而不是当前时刻的。其 README 承认"需要判断下什么时候重置这个缓存"。

## 五、参考

- 同济 SuperPower 2025 自瞄开源 <https://bbs.robomaster.com/article/803315>
- TinyMPC <https://tinympc.org/> / 论文 <https://arxiv.org/pdf/2310.16985>
