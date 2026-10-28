# 固定起点、不同步长的 yaw 响应延迟示例

示例代码：`tools/diagnostics/gimbal_latency_example.py`，调用现有 C++ 工具；参数集中在文件开头。
默认 yaw 起点 0 rad，偏转量为 0.25、0.50、0.75 rad，各 500 次，共 1500 次。
每次都先回到起点，稳定并采集基线，再发目标命令。顺序按固定种子打乱，
各步长交错执行，而非把上一次终点当作下一次起点。

```bash
xmake build gimbal_step_csv
# 先将示例文件的 REPEAT 改成 5，检查参数；默认不打开串口。
python3 tools/diagnostics/gimbal_latency_example.py
# 实测时使用一个新的输出前缀；同名前缀会覆盖旧结果。
python3 tools/diagnostics/gimbal_latency_example.py --output /tmp/yaw_latency_pilot --arm
```

默认限位 ±0.80 rad 只是示例，必须按实机修改。其他串口程序应退出。
此示例只控制 yaw 起点；pitch 沿用每次试验开始时的反馈角，roll 没有控制字段。
因此不会自动把三轴都归零。若要测零姿态，先将另外两轴调整到所需位置。

## 改哪里、往哪个方向调

| 参数 | 示例 | 调整方向 |
| --- | --- | --- |
| `INITIAL_RAD` | 0 | 全部试验的共同 yaw 起点。 |
| `OFFSETS_RAD` | `[0.25, 0.50, 0.75]` | 改为所需步长；可加负值测反方向，分析时正负方向分开。所有目标必须落在限位内。 |
| `REPEAT` | 500 | 先用 5～10 排查流程，再改为 500。 |
| `LIMITS_RAD` | `(-0.80, 0.80)` | 按实机运动范围设置，不要为了通过检查随意放宽。 |
| `DURATION_MS` | 100 | 仅控制发命令后的观测窗口。若存在 `not_observed`，先延长到 200 或更长排查；缩短窗口不会提高采样率。 |
| `ONSET_THRESHOLD_RAD` | 0.005 | 统一角位移阈值，约 0.286°。查看基线噪声和回零后的漂移，阈值应高于这些扰动；太低会误触发，太高会增加跨越阈值的运动时间。比较不同步长时固定同一阈值。 |
| `SETTLE_TIMEOUT_MS` | 5000 | 允许回零的最长等待时间；回零失败先检查实际轨迹、单位和限位，再考虑延长。 |

更细的回零判定在 `tools/diagnostics/gimbal_step_csv.cpp` 的 `Rig::settle()`：当前是进入
初始角 ±1° 并持续 300 ms，随后 `run()` 再记录 500 ms 基线。±1° 是位置条件，
不保证速度为零；如果原始基线还在漂移，应收紧位置容差、延长稳定时间，或增加
角速度/基线波动判据。否则残余运动可能被误认为新命令的响应。

C++ 的 `kTxPeriod` 为 5 ms（200 Hz）。保持此值与后续实际控制场景一致；
下位机反馈频率和发送频率是两回事，响应计时精度受反馈间隔影响。

## 数据与拟合

- `_samples.csv`：原始反馈，角度单位为度；可重新按不同阈值离线计算延迟。
- `_trials.csv`、`_model.csv`、`_curve.csv`：原工具逐次指标、分组中位数和曲线。
- `_config.json`：本轮步长、阈值、窗口等参数。
- `_latency.csv`：示例另外导出的拟合输入，位置和步长单位 rad，延迟单位 ms。

`_latency.csv` 每行一次试验：`trial_id, repeat, initial_rad, initial_actual_rad,
step_rad, target_rad, onset_ms, status, sample_count, threshold_rad, window_ms`。
`observed` 表示测到；`not_observed` 表示有反馈但未达到阈值；`no_feedback` 表示窗口
内没有状态反馈。未测到的延迟留空，不要补 0。若工具中断，弧度版导出不会执行，
已写出的原始 `_samples.csv` 和 `_trials.csv` 仍可分析。

先画每个步长的延迟分布，统计有效数量/总次数、中位数、P95，并看延迟随
`trial_id` 是否漂移，再决定拟合形式。只用成功试验可能低估延迟，必须同时报告
未测到比例。若要研究曲线形状，应增加不同步长的数量，而不只是增加同一步长的次数。

本工具计时从上位机开始写目标帧，到解析到沿命令方向超过阈值的状态帧。
它包含通信、控制调度、机械起动、跨越阈值和反馈传输，不是纯 MCU 执行延迟。
状态帧没有命令编号回显或采样时间戳，无法严格关联某次反馈与某条命令，也无法
排除反馈积压的影响；基线稳定性和原始轨迹需要一并检查。
