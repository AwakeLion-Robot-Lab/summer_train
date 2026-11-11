# auto_aim 实机性能采样（2026-11-11）

直接采样用户已经运行的进程，没有启动第二路相机或推理，没有连接串口或发送控制命令。
测试硬件为 i7-13620H；当前装甲板模型为 `szu-v5n.xml`，OpenVINO GPU，
图像 1440×1080，`armor_pipeline: true`，每帧显示，`waitKey(1)`。
日志确认 GPU 加载成功。采样时无串口反馈，实际运行检测与显示，未运行跟踪、规划或 MPC。

## 接电后的结果

15 秒被动采样收到 7 个两秒统计窗口，共覆盖 1021 个已显示帧、约 14.05 秒。
窗口按帧数加权，实际吞吐率为 **72.67 FPS**。CPU 总占用约 225.7%，相当于 2.26 个核心。

| 项目 | 平均耗时 |
|---|---:|
| 检测总计（视觉线程） | 7.723 ms |
| 其中预处理 | 0.671 ms |
| 其中 GPU 推理 | 6.940 ms |
| 其中解码 | 0.106 ms |
| 绘制叠加层（主线程） | 0.818 ms |
| `imshow + waitKey(1)`（主线程） | 12.931 ms |
| 主线程等待视觉结果 | 0.008 ms |
| 完整主循环 | 13.759 ms |
| 已完成视觉结果等待主线程消费 | 5.962 ms |

视觉线程与主线程重叠，不能把检测总计与显示耗时相加。
显示和事件处理约占主循环 94%，主线程几乎不等待视觉结果，
而视觉结果平均等待约 6 ms 才被消费；因此本次约 73 FPS 主要受显示侧限制。
`waitKey(1)` 的 1 ms 是等待参数，不能代表整个显示与事件处理耗时。
后续应优先对比降低显示刷新频率或关闭显示，而不是只优化网络推理。
这里没有测量关闭显示后的吞吐率，不将检测耗时的倒数作为实测整链路 FPS。

原始记录：`build/auto_aim_live_perf_ac.json`、`build/auto_aim_power_state_ac.json`。

## 未接电时的异常慢速

前两轮每轮采样 15 秒，吞吐率分别约 10.41 / 10.01 FPS；
检测约 81.4 ms、GPU 推理约 76.3 ms、显示约 70.5 ms。
连续 5 次频率读取显示 16 个逻辑 CPU 均在约 400 MHz，核显实际频率读数为 100 MHz，
交流供电为 0、电池为 Discharging；CPU 温度约 44～55°C。
系统电源模式仍为 performance。接电后交流供电变为 1，CPU 恢复动态调频，
检测和显示耗时明显下降。该慢速状态不能与接电后的优化结果直接比较，
也没有据此断言具体的固件限频机制。

记录：`build/auto_aim_live_perf_instrumented.json`、`build/auto_aim_live_perf_repeat.json`、
`build/auto_aim_power_state.json`。

## 复测

`auto_aim` 在 `debug.overlay` 或 `debug.plot` 开启时，每两秒输出一条
`runtime performance` JSON，包含各段均值、中位数和 P95。
无串口的检测模式同样统计，不会因为不进入规划而缺少性能记录。
`debug.plot: true` 时，还会向已配置的 UDP 端口发送 `runtime_perf` 汇总。
关闭两个 debug 开关则不做窗口统计。

```bash
python3 tools/collect_auto_aim_perf.py --seconds 15 --output build/auto_aim_live_perf.json
```

脚本自动查找唯一的 `auto_aim` 进程，也可通过 `--pid` 指定。
脚本只读进程统计并监听本机 UDP，不发送数据、不触碰相机串口、不停止或重启进程。
如果 UDP 端口已有接收器，脚本不会抢占，只记录 CPU 负载。
使用 `runtime_summary.fps` 作为处理/显示吞吐率；普通 pipeline 消息可能只对应
拒绝帧，因此其 UDP 包速率不能代表 FPS。

构建 `auto_aim` 与 `pose_overlap_smoke` 通过，姿态等待和调度回归通过。
