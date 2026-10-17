# 能量机关火控与单发输出

参考 [Alliance-Algorithm/rmcs_auto_aim_v2](https://github.com/Alliance-Algorithm/rmcs_auto_aim_v2) 的 `src/kernel/fire_control.cpp` 中能量机关的 idle/shoot 时序，MIT 许可见 [ALLIANCE-LICENSE](../f3-estimator/ALLIANCE-LICENSE)。适配现有 `Rune::Command` 和 `L1Sensor::SerialCommand`，不引入上游 ROS 或 executor。

`ControlStage` 独立消费 f4 的 `AimPlan`：先检查规划时效和有效性，再判断实测云台对准、稳定时间、idle/shoot 窗口与射击冷却，最后输出一次处理周期的 `fire` 脉冲。默认总开火开关关闭。

## 状态与时间

- 取得可射叶片后开始节拍：默认 idle 0.8 s、shoot 0.12 s。每个 shoot 窗口最多一个脉冲，之后进入下一轮；错过窗口不补发。
- 云台需连续对准 `aim_settle_s`，默认 30 ms。失准、反馈无效或控制调用间隔超过 `max_plan_age_s` 会重新计时。
- 同时检查实际角与可执行命令、实际角与直接弹道解的误差，避免 MPC 轨迹有滞后时错误放行。参考仓库的符分支跳过角误差检查，本适配保留项目已有的到位门控。
- 相邻发射默认至少间隔 0.3 s；换叶、失去目标、自动切换大小符模式不会清除这项冷却。换目标和恢复跟踪会重新等待 idle 与对准稳定。
- `require_rearm=false` 为默认时序模式，可对同一未激活叶片在后续窗口重试。设为 true 时发射后一直闭锁，由外部确认后调用 `rearm()`。rearm 仍遵守冷却。
- `fire_feasible` 不考虑总开火开关和本周期的 `shoot_requested`；`fire` 为两者放行后的实际单发脉冲。`single_shoot=true` 表明上层应使用单发语义。主程序使用 `SerialWorker::updateCommand(command, true)`，一次完整写入后清除 shoot 位，避免按串口发送频率重复触发；协议帧格式不变。

`ControlInput::actual_yaw_rad/actual_pitch_rad` 必须是实测的绝对角，pitch 向上为正；默认 NaN，缺失时仍输出跟随角但不开火。可填写 `feedback_at` 检查反馈时效；零值表示调用者保证反馈与本周期同步。不要用上一条命令代替反馈。若反馈为右手 Ry 的向下正俯仰，需要取反后填入。

## 接入

```cpp
Rune::PlanningConfig planning;
planning.mpc.enable_mpc = true;
// 沿用现有 MPC 权重与约束；可从既有规划配置复制后调整。
// planning.mpc = armor_planner_config;

Rune::FireControlConfig fire;
fire.shoot_enable = false;
fire.idle_duration_s = 0.8;
fire.shoot_duration_s = 0.12;
Rune::Processor processor(geometry, estimator, planning, {}, fire);

Rune::ControlInput input;
input.command_at = std::chrono::steady_clock::now();
input.bullet_speed_m_s = measured_bullet_speed;
input.fire_delay_s = measured_send_to_muzzle_delay;
input.actual_yaw_rad = measured_absolute_yaw;
input.actual_pitch_rad = measured_upward_positive_pitch;
input.feedback_at = measured_feedback_timestamp;
input.shoot_requested = operator_or_navigation_shoot_request;
const auto command = processor.process(frame, input);
const auto serial_command = command.found
    ? Rune::ControlStage::makeSerialCommand(command)
    : processor.safeHold();
// serial_command 使用现有 L1Sensor::SerialCommand 和串口发送接口。
// 调用者负责实际下发；本模块不创建相机/串口线程。
```

`makeSerialCommand()` 将向上正的 pitch、pitch 角速度与角加速度一起取反，yaw 保持不变；复用既有协议的前馈字段。`safeHold()` 返回最后有效方向，清零发射位和全部前馈，尚无有效命令时返回空。无效的 Command 不能被组装成新串口命令。

`Processor::lastMotion()` 和 `lastPlan()` 提供当前帧诊断。没有显式提供第五参数时，兼容旧的 `PlanningConfig::shoot_enable/max_yaw_error_rad/max_pitch_error_rad`，用于构造默认 f5 配置。显式第五参数优先。`PlanningStage::plan()` 现在返回规划数据，独立调用者须再交给 `ControlStage::update()`。

## 验证

```bash
xmake build rune_planning_control_smoke
xmake run rune_planning_control_smoke
xmake build rune_pipeline_smoke
xmake run rune_pipeline_smoke
xmake build rune_joint_building_smoke
xmake run rune_joint_building_smoke
xmake build newvision
```

合成测试覆盖正反转与大符相位积分、延迟/飞行时间一致性、重力坐标与绝对 yaw、现有阻力模型、固定身份换叶、超时/激活撤销、模式预热、失效输入、MPC 时刻/动力学/加速度界/回退、idle/shoot 单发、稳定门限、冷却、手动闩锁、反馈缺失、MPC 滞后门控及串口符号。实车时序、标定和命中率仍需实测。
