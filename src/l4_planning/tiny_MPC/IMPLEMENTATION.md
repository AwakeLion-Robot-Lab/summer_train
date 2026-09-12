# TinyMPC 实现组织说明

## 当前状态

本目录已经实现 README.md 描述的二阶 TinyMPC 云台控制器：

- `get_trajectory.cpp` 生成居中的 100 点 yaw/pitch 参考轨迹；
- `track_planner.cpp` 使用 Riccati 递推和 ADMM 求解两个单轴问题；
- `result.cpp` 从第 50 点开始生成 50 个 `AimSample`；
- `Planner` 在直接瞄准有效且 MPC 收敛时设置 `using_MPC=true`，失败时
  保留安全的直接瞄准输出。

公共接口位于 `include/l4_planning/tiny_mpc.hpp`，输出继续复用
`include/l4_planning/types.hpp` 中的 `AimReference`、`AimSample` 和
`AimPlan`。

## 文件职责

| 文件 | 建议职责 |
| --- | --- |
| `get_trajectory.cpp` | 根据目标预测和弹道解生成 yaw、pitch 角度及角速度参考，处理角度环绕和采样时间 |
| `track_planner.cpp` | 单轴优化迭代，包括反向递推、正向展开、盒约束投影、对偶更新和残差检查 |
| `result.cpp` | 将求解结果转换为 `AimSample`，填写执行时间并更新 `AimPlan` |

实现增加了以下文件：

| 建议新增文件 | 职责 |
| --- | --- |
| `include/l4_planning/tiny_mpc.hpp` | 对规划层暴露 MPC 配置、调用接口和求解状态 |
Riccati 预计算和内部配置校验均与 ADMM 实现在同一翻译单元中；残差平衡
改变 `rho` 时会重新计算缓存。所有对外类型和函数声明统一放在
`include/l4_planning/tiny_mpc.hpp`。

先保持单轴求解器可同时服务 yaw 和 pitch，无需为两个轴复制目录。
`xmake.lua` 已通过 `src/**/*.cpp` 收集源文件；新增实现时需检查依赖与
测试目标的链接配置，避免在该目录添加带 `main()` 的演示程序。

## 已采用的实现约定

1. **模型阶数**：采用 `[角度, 角速度]` 状态和角加速度输入的二阶模型；
   `AimSample` 中的 jerk 只是相邻加速度差分，不是优化约束。
2. **时间与初值**：采用 README 的居中窗口和参考轨迹第 0 点初值，输出从
   第 50 点开始；首项执行时间为本次规划时间。
3. **权重约定**：Q/R 保持配置原值，ADMM 只在控制 Hessian 中增加一次
   `rho`，不复现参考源码的重复增广。
4. **求解状态**：`TinyMpcSolveInfo` 报告迭代次数、原始/对偶残差和收敛状态；
   每次调用独立初始化，不跨目标热启动。
5. **输出有效性**：用投影后的加速度重新正向展开状态，从而同时保证盒约束
   和离散动力学；只有残差收敛才发布 MPC，否则回退到直接瞄准。
6. **协议接入**：成功时 `samples` 含第 50～99 点并按执行时间排序；当前点的
   角度、角速度和角加速度经 `SerialCommand` 写入串口 `TxPayload`。该协议
   布局需要下位机同步为 6 个 `float` 加 1 个 `uint8_t shoot`。

## 后续实现的验证重点

- 静止参考和匀速参考的跟踪结果，以及离散动力学残差。
- yaw 跨越 ±π 时的参考连续性。
- 加速度饱和与迭代未收敛时的状态和回退行为。
- 输出控制点的索引、绝对执行时间及 `AimPlan` 协议一致性。

上述验证已由 `tiny_mpc_smoke` 覆盖，串口字段编码由
`serial_protocol_smoke` 和 `serial_worker_smoke` 覆盖。



  注意：下位机必须同步采用以下字段顺序，否则协议无法兼容：

  float yaw;
  float pitch;
  float yaw_rate;
  float pitch_rate;
  float yaw_acceleration;
  float pitch_acceleration;
  uint8_t shoot;