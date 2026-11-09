# 延迟检查与测量

## 时间补偿是否重复

装甲板的出膛时刻为 `target.timestamp + (planning_time - target.timestamp) + fire_delay`，
即 `planning_time + fire_delay`。候选生成先把状态推到出膛时刻，再预测到
`出膛时刻 + 飞行时间`；后一次使用的是绝对时间，不会重复加图像年龄。
每轮固定点迭代也都从同一出膛状态出发，不会累计前一轮预测时间。

能量机关的预测跨度为 `command_at - captured_at + fire_delay + flight_time`。
MPC 的额外 offset 是采样点相对当前命令的时间，不是再加一次系统延迟；
居中 100 点、步长 10 ms 也不表示程序实际等待 0.5 s。

`planner_config.yaml` 的 `*_fire_delay_s` 是命令发布到出膛的标定量。
如果现场把它标成“曝光到出膛总时间”，就会与代码已测量的图像年龄重复。
仅从代码无法判断现场标定是否包含了这一段。

当前时间补偿以规划开始时刻为基准；规划本身的耗时与异步串口排队不会自动
作为独立项补进去。这是需要测量的剩余误差，不能通过删减图像年龄解决。

## 已调整的执行路径

- 装甲板稳定跟踪时支持受门限约束的提前检测，详见文末“姿态等待与检测重叠”。
- 能量机关图像检测不依赖姿态，先执行检测，让 RX 线程同时接收姿态。
  姿态等待使用从取到图像时起算的绝对截止时间；检测结束后只等剩余预算。
  保留曝光时刻姿态插值，不使用当前姿态替代曝光姿态。
- 装甲板初始化的第二遍检测在 ROI 与第一遍相同时跳过；不同 ROI 仍保留精检。
- 能量机关 letterbox 输入直接初始化为 114，去掉先清零再整图填充的重复写入。
- `debug.overlay` 默认关闭，避免逐帧绘图、`imshow`、`waitKey` 占用主循环。
  需要看图时可重新打开；即使显示发生在命令入队后，也会推迟下一帧处理。
- UDP 遥测采用非阻塞发送；发送队列拥塞时丢弃该调试包，不等待网络恢复。
- 查询当前姿态使用 `latestGimbalPose()`，避免每帧执行一次必然越过历史上界的查询。

相机驱动已经有独立采集线程与单槽 `LatestBuffer`，新图覆盖尚未读取的旧图；
没有再增加图像队列。SDK 内部缓存和 USB 传输仍需实机测量。
串口接收已采用“等待可读后只读当前可用字节”，不会等满 256 字节；
`read_timeout_ms` 也不是每包固定要等的时间。

## 上机查看指标

保留 `debug.plot: true`，在 PlotJuggler 接收配置中的 UDP 端口。
装甲板和能量机关均输出 `pipeline`，时间单位均为 ms：

| 字段 | 含义 |
| --- | --- |
| `frame_age_at_read_ms` | 曝光时间戳到主循环取得图像的年龄 |
| `pose_wait_ms` | 本帧实际阻塞等待姿态的时间 |
| `detect_ms` | 本帧所有检测调用的总耗时 |
| `detection_passes` | 检测次数，可识别初始化双遍推理 |
| `estimate_ms` | 装甲板跟踪估计耗时；能量机关记在下项 |
| `plan_control_ms` | 装甲板规划和控制；能量机关包含轮廓、PnP、估计、规划和控制 |
| `read_to_queue_ms` | 主循环取得图像到控制命令入队的总时间 |
| `image_to_queue_ms` | 曝光时间戳到控制命令入队的总时间 |
| `serial_last_queue_ms` | 最近一条已发送新命令，从入队到开始写串口的时间 |
| `serial_last_write_ms` | 上述命令完整写入系统串口接口的耗时 |
| `serial_last_sample_age_ms` | 上述串口测量距今多久，避免把旧数据当成本帧数据 |

`image_to_queue_ms` 已包含 `frame_age_at_read_ms` 和处理时间，不要再次相加。
装甲板旧遥测中的 `delay.before_fire` 已包含 `delay.image_to_plan`，也不能再次相加。
不同阶段未覆盖的 ROI 构造、适配和调度开销可由总时间与阶段时间之差观察。
没有命令入队的帧不输出 `*_to_queue_ms`。

串口计时只记录每条新命令的首次完整写入，周期重发不会使其不断变大；
它来自异步 TX 线程，可能属于前一帧，不能当作本帧精确的端到端时间。
系统 `write()` 完成也不代表最后一个字节已到达 MCU，更不代表电机或拨弹机构已执行。
当前 200 Hz 发送周期会引入一个发送节拍内的等待，通常约 0–5 ms，
还可能受到线程调度和写入耗时影响。此次保留发送频率及出膛标定值。

应先记录实际运行中的中位数及 P95/P99，再根据占时最多的阶段继续优化。
仅靠离线测试无法给出相机到 MCU 执行的真实延迟下降量。

## 2026-11-08：保留显示的装甲板优化

本轮保留 `debug.overlay: true`、GPU、szu-v5n、640×640 网络输入及现有线程结构。

### 实现与适用范围

- 冷启动/Lost 时，先整图检测，再等待曝光姿态。串口 RX 在检测期间照常收包，
  等待沿用读帧时设定的绝对截止时间。初始化 ROI 第二遍检测仍在获得姿态后执行。
- Detecting/Tracking/TempLost 仍先取曝光姿态，再生成原来的 ROI 与侧灯条预测。
  没有用近似姿态替换滤波观测，也没有改变网络裁剪尺度。
- YOLOv5 的 logit ≤ 0 对应置信度 ≤ 0.5；当初筛阈值 > 0.5 时直接拒绝，
  跳过大量背景候选的 exp。可能保留的候选仍使用原 sigmoid、阈值与 NMS。
- `pipeline.filter_input_age_ms` / `filter_output_age_ms` 分别记录进入/完成
  tracker.track 时距离图像时间戳的时间，只在实际调用跟踪器时提供。
  `preprocess_ms`、`infer_ms`、`decode_ms`、`refine_ms`、`side_light_ms`
  累加该帧所有装甲板检测调用，含初始化双遍。现有 `plan_control_ms` 用于判断
  是否值得拆规划线程。年龄仍受相机时间戳近似的限制，不是真实曝光延迟的独立标定。

### 本机录像对照

Release，OpenVINO GPU，szu-v5n；每段取前 120 帧，预热 30 次，再测 5 遍
共 600 次。先解码视频并存入内存，计时仅覆盖检测。关闭窗口的基准没有模拟实际
显示竞争，因此不能用它宣称实机端到端延迟下降。旧可执行文件在改动前保存于
`/tmp/armor_latency_before`，原始结果在 `/tmp/latency-ab-{before,after}-*.txt`
和 `/tmp/latency-high-{before,after}.txt`，不加入 Git。

`3m_run_mid.avi` 交替运行三组：

| 组 | 解码均值：前 → 后（ms） | 检测总均值：前 → 后（ms） |
| --- | --- | --- |
| 1 | 0.194842 → 0.084891 | 7.42432 → 7.48404 |
| 2 | 0.188621 → 0.081664 | 7.23456 → 7.12471 |
| 3 | 0.193818 → 0.083712 | 7.85872 → 7.59208 |

解码均值平均从 0.1924 降到 0.0834 ms，减少约 0.109 ms（57%）。
总耗时受 GPU/系统波动影响，第一组反而上升，不能宣称整条链路稳定降低 57%。
三组每侧均为 1200 个检测，角点、置信度、类别、颜色及检测数哈希均为
`2593492760354216620`。

另一段 `3m_high.avi`：解码 0.182905 → 0.078488 ms；总检测
7.90336 → 7.19387 ms。每侧 710 个检测，结果哈希均为 `6605244387664411753`。
这些是同模型输出一致性检查，不是标注集精度评估。

另外，在同为 OpenCV 1 线程的对照下，v5n 总检测均值 7.55427 ms，v8n 为
9.24491 ms，故保留 v5n。OpenCV 线程限制也未显示收益，未调整生产配置。
输入尺寸会改变模型契约和角点精度，本轮未在缺乏精度验证时切换。

### SDK 与验证边界

随仓库的 `MvCameraControl.h` 明确说明 `MV_CC_SetGrabStrategy` 仅支持 Windows；
Linux 不能直接使用 LatestImagesOnly。`MV_CC_SetImageNodeNum` 的文档默认值为 1，
当前驱动未增加缓存节点，应用层又已有单槽 LatestBuffer，因此未添加重复设置。
是否存在设备/USB 内部积压仍需联机验证，不能仅凭单槽认定没有全部传输延迟。

已通过主程序编译，以及 pose_detection_schedule、armor_decoder_threshold、
armor_runtime_no_serial、armor_refiner、eskf_tracker 五项 smoke。
调度测试用确定性时钟覆盖姿态已到达、延迟到达和截止超时：例如推理 7 ms、
姿态 6 ms 后到达时，冷启动调度为 7 ms，原串行调度为 13 ms；这是调度模型验证，
并非实机节省 6 ms 的测量。尚未进行相机与下位机联机端到端 A/B。

复现本轮检测基准：

```bash
xmake build armor_latency_smoke
build/linux/x86_64/release/armor_latency_smoke video/3m_run_mid.avi GPU
# 可选：显式指定模型及 OpenCV 线程数
build/linux/x86_64/release/armor_latency_smoke video/3m_run_mid.avi GPU model/armor_model/szu-v8n-fp16.onnx 1
```

## 2026-11-08：装甲板有界流水线

`runtime.armor_pipeline: true` 启用视觉工作线程。检测器、ROI 预测、姿态等待、
初始化二次检测和跟踪更新全部由该线程顺序执行；规划、控制、遥测和 HighGUI
仍在主线程。设为 `false` 可恢复同步调度，两种模式共用 `processArmorVision`。
能量机关仍使用原执行路径，模型、输入大小、角点精修和显示开关不变。

相机继续使用原来的单槽 LatestBuffer。新的结果交接槽不覆盖结果：生产者发布
一帧后，必须等消费者取走，才允许读取下一帧。因此主线程处理 N 时，视觉线程
最多处理 N+1；慢消费者不会导致结果 FIFO 增长。交接包拥有该帧图像、曝光姿态、
机器人状态、检测结果、目标快照和诊断数据；主线程不再查询活跃 tracker。
OpenVINO 同步 infer 仍只有一个请求，输出在视觉线程内完成解码并释放租约，
不把网络输出缓冲区传给主线程。

新增 `pipeline.result_wait_ms`（视觉结果完成到主线程接收）和
`pipeline.stale_results`（累计过期结果数）。`runtime.pipeline_max_result_age_ms`
默认为 100 ms：流水线结果进入规划前，若距图像时间戳超过此值，丢弃并执行
safe hold；开启 plot 时额外输出 `result_rejected: true`。该年龄仍受当前
相机时间戳近似影响。它是结果接收时的检查，不是规划执行时间的硬实时保证。
取帧/交接等待超时也执行 safe hold。生产者异常传回主线程；退出先停止并 join
视觉线程，再复位跟踪器、关闭串口和相机，避免悬空访问。

### 离线 A/B

新增 `armor_pipeline_replay`，复用生产视觉处理和单槽交接。每段预先解码前
240 帧；每次加载同一 szu-v5n、OpenVINO GPU，预热 30 次，按串行、流水线、
流水线、串行顺序运行。所有运行使用相同录像时间戳、固定单位姿态，以及
图像时间戳 +10 ms 的固定规划时刻，以比较调度前后的输出。
消费者执行真实规划、控制、遥测 JSON 序列化和叠加绘图。

| 录像 | 串行 FPS（两次） | 流水线 FPS（两次） | 流水线交接等待 P95（ms） |
| --- | --- | --- | --- |
| 3m_run_mid.avi | 71.98 / 71.35 | 114.84 / 113.24 | 0.224 / 1.317 |
| 3m_high.avi | 59.16 / 58.93 | 97.09 / 98.95 | 2.886 / 2.743 |

每段的四次运行，检测、跟踪状态、规划角度和控制输出哈希一致：
mid 为 `660644938002286637`（480 个检测），high 为
`18153036800644883774`（334 个检测）。每次均有 240 帧目标和有效规划。
这些是同一输入序列下的回归结果，不是标注集精度评估。

**上述 FPS 不包含相机采集、姿态等待、串口传输、UDP 发送或 imshow/waitKey。**
录像逐帧消费，不模拟实机 LatestBuffer 丢帧；真实调度会改变被选中的图像序列。
固定规划时间仅用于输出一致性检查，不能由这些数据推算实机端到端延迟。
以上表格保留当次测量汇总；临时目录中的原始日志不作为长期存档。

复现：

```bash
xmake build armor_pipeline_replay
build/linux/x86_64/release/armor_pipeline_replay video/3m_run_mid.avi GPU
build/linux/x86_64/release/armor_pipeline_replay video/3m_high.avi GPU
```

主程序构建及 single_frame_pipeline、armor_runtime_no_serial、
pose_detection_schedule、eskf_tracker 四项 smoke 已通过。
并发 smoke 覆盖慢消费者只能提前一帧、快照所有权、顺序/不丢结果、读取超时、
EOF、生产者异常、满槽退出及计算中退出。运行时 smoke 覆盖两种调度下缺失/禁用
串口、取帧超时恢复、相机异常，以及过期结果拒绝。
ThreadSanitizer 在本机启动失败（`unexpected memory mapping`），未能完成
动态竞态检测。仍需相机、下位机和保留显示条件下的端到端 A/B，重点观察
`result_wait_ms`、`image_to_queue_ms`、过期计数和实际跟踪误差。

## 2026-11-09：姿态等待与检测重叠

`runtime.pose_detection_overlap: true` 启用，设为 false 恢复跟踪阶段先等姿态的
调度。同步与有界流水线两条路径共用 `processArmorVision`，均适用。

仅在 Tracking 状态、没有补检冷却时尝试。串口线程锁内取最近两包姿态：图像比
最新姿态晚 0–8 ms、采样间隔 0.5–20 ms、四元数角距离对应转速不超过 1 rad/s
才提供临时 ROI 姿态。采样异常、快速转动、Detecting/TempLost 均回退原流程。
如果历史已覆盖图像，也直接走原有插值路径，无需扩大 ROI。临时查询不增加
`pose_after_history` 计数，返回值仅用于 ROI，不进入滤波更新。

ROI 由旧跟踪状态外推到本帧，再按临时姿态投影；宽高每侧增加约 2%（至少
2 px），裁到图像边界。这个余量是启发式的，不是运动误差的严格上界。
先执行同步推理、解码和角点精修，串口 RX 线程同时接收后续姿态；没有增加
推理线程或在途请求，推理输出当次立即解码成自有角点数据。

检测后仍使用 `ready_at + pose_wait_ms` 的原绝对截止时间，等待只消费剩余预算。
随后查询图像时刻姿态，重算原尺寸 ROI；若它未被首遍裁剪范围完整包含，或者
首遍没有敌方整板，最多补检一次。连续两次提前检测尝试都需要补检，后续
30 帧暂停提前检测。初始化二次 ROI 检测与跟踪补检互斥，每帧至多两遍推理。

侧灯条搜索从检测器拆为 `completeSideLights`，在最终检测结果确定后，用图像
时刻姿态生成 hints，再搜索、去重。去重使用所有颜色/类别的整板，之后才筛敌方。
推理失败时不生成侧灯条观测。最终跟踪使用查询到的曝光姿态；原有超时后退回
边界姿态的行为保留，并以 `pose_timeout` 明确记录，不能把超时结果称为精确插值。

新增逐帧遥测（与 `pose_wait_ms` 在同一组）：

- `pose_overlap`：本帧是否提前检测；
- `pose_roi_repair`：本帧是否补检；
- `pose_timeout`：曝光姿态等待是否超时；
- 原有 `detection_passes`、`detect_ms`、`filter_input_age_ms` 继续用于评估总代价。

验证：`pose_overlap_smoke` 调用生产处理函数，检查阶段次序、原截止时间、最终
姿态、ROI 越界补检、空检测补检、关闭开关、超时、门限和冷却恢复。
`light_detector_smoke` 比较合并/拆分搜索结果，覆盖重复灯条、清空提示和推理失败。
`pose_detection_schedule_smoke` 使用确定性时钟验证先检测时消耗的是剩余等待预算。
`armor_runtime_no_serial_smoke` 验证无串口及运行时退出路径。以上测试通过。

真实图像对照命令：

```bash
xmake build pose_overlap_smoke
build/linux/x86_64/release/pose_overlap_smoke video/3m_high
build/linux/x86_64/release/pose_overlap_smoke video/3m_run_mid
```

每段前 360 帧，当前 szu-v5n / OpenVINO GPU / 相机标定；真实录像四元数用作最终
姿态。**可用性是模拟的**：用相邻录像姿态插值构造 200 Hz 样本，最新样本比曝光
早 4 ms，并假定检测后最终姿态已到齐。这只能检查裁剪/关联的变化，不是相机或
串口延迟实测。日志保存在 `build/pose_overlap_audit/`。

| 片段 | Tracking 帧：关→开 | 整板检测总数：关→开 | 完整板关联数：关→开 | 提前检测 / 补检 | 端点纵向创新绝对值 P90（px）：关→开 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3m_high | 355→355 | 496→492 | 487→482 | 354 / 0 | 2.283→2.246 |
| 3m_run_mid | 355→355 | 720→720 | 718→718 | 354 / 0 | 1.287→1.307 |

两种配置每段都是 361 次推理，含一次初始化精检。跟踪可用帧没有减少，但高速
片段少了 4 个检测、5 个关联，不能宣称无漏检或检测完全等价；没有标注真值，
上述数量也不是检测精度指标。ROI 放宽改变网络缩放，尤其可能影响远距小目标。
没有修改 `roll_pitch_noise`（仍为用户要求恢复的 0.1）或滤波响应。

理想情况下等待 W 与检测 D 的串行耗时 W+D 变为 max(W,D)，收益上限是
min(W,D)；ROI 校验、补检以及改变的检测输入会抵消收益。实际接相机应在相同
串口、显示和目标条件下切换配置，比较图像年龄、实际等待、补检率与跟踪情况。
本次没有报告实机毫秒数或 FPS 提升。
