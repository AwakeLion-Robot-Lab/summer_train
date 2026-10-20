# MV-CS016-10UM/UC 图像与云台姿态硬同步方案

## 结论和同步点

方案可行。现有相机资料确认它是全局快门，支持硬触发，有 Line0 光耦输入、Line1 光耦输出及 Line2 非隔离双向 IO。若保持相机当前的自由运行采图方式，建议以**相机曝光开始**为同步事件：相机 Line1 输出曝光信号，下位机用定时器输入捕获/外部中断锁存该时刻的姿态或 IMU 时间，下位机发一条带独立曝光序号的姿态快照，上位机把它与对应图像绑定。普通周期姿态包继续用于控制状态和“当前姿态”。

“下位机发姿态包的同时翻转 GPIO”不能确定图像何时曝光；GPIO 和发包同刻只同步了串口发送动作。相机输出的事件必须是 `ExposureStart` / `ExposureStartActive`（以实机 MVS 节点枚举为准），不是 `FrameEnd`、图像到达、USB 传输完成或任意 Strobe 延时脉冲。首版仍可保持自由运行，不必立即改成外触发拍照。

## 接线和相机配置

提供的两页产品规格书只给出了 IO 种类，没有给引脚、电平与时序。用户提供的 P7 线缆图给出了**这条线缆**的引脚与线色；海康 [USB3.0 面阵相机用户手册](https://www.hikrobotics.com/en2/source/vision/document/2023/8/18/UD31198B_USB3.0%20Area%20Scan%20Camera%20User%20Manual_V2.4.0_20230307.pdf) 也给出相同的引脚信号定义。接线时以实物管脚编号和万用表通断核对线色，因为不同 P7 线缆的颜色可能不同。

| P7 管脚 | 用户图中的线色 | 信号 | 本方案用途 |
| --- | --- | --- | --- |
| 1 | 红 | `DC_PWR` | 相机外部电源；USB 供电时本方案不接 |
| 2 | 黄 | `OPTO_IN` / Line0+ | 相机硬触发输入；首版不用 |
| 3 | 棕 | `GPIO` / Line2+ | 非隔离双向 IO；首版不用 |
| 4 | 白 | `OPTO_OUT` / Line1+ | 接下位机定时器捕获输入 |
| 5 | 绿 | `OPTO_GND` / Line0-/Line1- | 接下位机侧信号地 |
| 6 | 黑 | `GND` / Line2- | 相机电源地；本方案不作 IO 回路地 |

推荐接法：P7-4 白线 `OPTO_OUT` 接下位机定时器捕获输入，该输入用外部约 1 kΩ 电阻上拉到下位机 3.3 V；P7-5 绿线 `OPTO_GND` 接下位机侧信号地。光耦输出按下拉脉冲处理，配置边沿以示波器测得的曝光起始有效边沿为准。P7-6 黑线是相机本体 GND，属于 Line2 回路，不拿来代替 P7-5；P7-1 红线是外部供电，若用 USB 供电不要误接到 MCU GPIO。通用手册在 3.3 V、1 kΩ 条件下给出输出低电平典型 575 mV、上升延迟约 15～60 μs；该数值不能直接当作这台相机的实测保证，触发极性与延迟必须联调确认。

在 MVS 里先确认 `LineSelector=Line1`、输出模式/`LineSource=ExposureStartActive`（或实机等价节点）、Strobe 使能、无额外 Delay/PreDelay；再用示波器同时看 Line1、电控捕获脚、图像触发/曝光事件，确认**每张图正好一条边沿**，包括 0.8 ms 曝光、目标帧率及相机重连后。生产代码在每次 `MV_CC_OpenDevice` 后、`MV_CC_StartGrabbing` 前配置和读回节点；配置失败时不能标记为硬同步成功。当前 `tools/camera_sdk/hikrobot/hikrobot.cpp` 只设曝光、增益、165 fps，尚未配置 IO。

## 下位机采样与串口协议

1. Line1 边沿进入 MCU 定时器捕获或高优先级中断；先锁存单调计数器 `edge_tick` 和本次 `exposure_seq`。中断里不阻塞串口发送，也不做耗时姿态解算。使用捕获计数器可避免中断调度抖动变成时间误差。
2. 在同一个 MCU 时间域给 IMU 原始样本/姿态解算结果留短历史。用 `edge_tick` 对应的姿态；若算法有滤波群延迟，需要从历史重建该时刻姿态。当前配置曝光 0.8 ms，几何算法通常应使用曝光中点：`t_image = t_start + exposure_us / 2`。如果下位机只在边沿读取“最新姿态”，必须量出采样、融合与中断延迟并补偿，不能把它称为无误差锁存。
3. 建议单独新增 `ExposurePose` 消息：`boot/session_id`、`exposure_seq`（至少 32 位）、`edge_tick_us`（建议 64 位）、`q_world_imu` 或现有 RPY、采样有效标志与 CRC；定义字节序、单位、序号溢出/重启语义。它不复用现有 8 位串口 `HeaderFrame.seq`，因为该序号对所有帧逐帧递增，用于丢包统计，不能充当图像序号。周期 `RobotState` 可以保留；若带宽紧张，可让曝光快照替代部分周期姿态上报。
4. 串口接收线程按 `cmd_id` 分别解析周期状态和曝光快照，两个通道各自建有界队列。不要把曝光快照简单写入当前按**串口接收时刻**索引的 `gimbal_history_`，否则又引入传输抖动。MCU 复位、相机重连、序号跳变时清空绑定状态。

串口带宽需要实测预算。115200 8N1 理论上限约 11.5 kB/s；相机代码设 165 fps。若曝光消息 32 字节，每秒约 5.3 kB，再叠加当前约 103 Hz 的周期状态和现场其他帧，可能逼近上限。用实际线上的每类帧长度和频率算总量并留突发余量；必要时提高波特率或减少重复状态帧。

## 图像与曝光快照配对

当前海康驱动 `read()` 只返回 `cv::Mat` 和一个用 `GetImageBuffer` 返回时刻减半个曝光估计出的时间戳；SDK 的 `raw.stFrameInfo.nFrameNum` 目前被丢弃。应在相机层增加 `FramePacket {image, camera_frame_id, exposure_us, arrival_time, ...}`，在进入 `LatestBuffer` 前就连同帧号保存。上位机同时保存 `ExposurePose` 的 `exposure_seq`。

两套计数器**不能天然认为相等**。进入硬同步模式时先建立一个经验证的偏移 `camera_frame_id - exposure_seq`，可用相机曝光事件回调提供的 FrameID 校对，或停流、清队列后单帧/低速启动并核验第一帧与第一条 IO 边沿。是否支持带 FrameID 的相机曝光事件，以这台相机的实际节点为准。偏移确认后按帧号与独立序号一对一匹配；图像丢帧可由相机帧号跳变识别，串口丢包可由曝光序号跳变识别。若相机事件不可用、启动时无法证明第一帧和第一边沿对应，单靠两条异步数据流“最近到达”只能算软件推测，不能宣称已实现可靠硬同步。

对每帧设短等待上限和队列容量。超时、重复序号、相机重连、MCU 重启或配对歧义时，标记该帧 `sync_invalid`；验收阶段禁止把邻近姿态冒充硬同步结果。正式运行可配置安全降级为旧软同步，但要显式上报 `sync_mode=soft_fallback` 并统计次数。失去有效图像姿态时不要把该帧送入需要 `world<-barrel` 的跟踪/开火计算。

## 工程改动位置与迁移

| 位置 | 所需改动 |
| --- | --- |
| `tools/camera_sdk/hikrobot/hikrobot.cpp/.hpp`、`include/l1_sensor/camera/camera.hpp` | 配置 Line1，读回确认，传递 `nFrameNum`、曝光长度与帧元数据；每次相机重连重新配置并重建会话。 |
| `include/l1_sensor/serial/serial_protocol.hpp`、`src/l1_sensor/serial/serial_protocol.cpp` | 新增曝光快照 `cmd_id`，严查长度、CRC、序号及字段合法性；保留现有周期状态消息。 |
| `src/l1_sensor/serial/serial_worker.cpp`、`src/runtime/runtime_io.cpp` | 维护曝光快照队列并与相机帧号配对；成功后沿用 `R_imu_barrel` 转换为 `world<-barrel`。 |
| `src/runtime/auto_aim_runtime.cpp`、配置、遥测 | 用配对姿态进入 L3；保留 `latestGimbalPose()` 给规划/控制使用。增加硬同步有效率、帧号缺口、IO/串口缺口、等待耗时与回退状态。 |

切换成功的帧不再套用 `config/serial_config.yaml` 的 `pose_delay_ms: 22`、`pose_wait_ms: 20` 和相机当前的 `now()-half_exposure` 时间戳来选择**图像姿态**；这些只服务于旧软同步路径。22 ms 是当前录像拟合出来的经验补偿，包含传输与 IMU 融合等因素，不能直接认定硬同步会把全部误差消掉。硬同步帧的算法时间基准仍需明确：可以用上位机稳态时间轴与 MCU `edge_tick` 做时钟映射来测端到端延迟，或先以到达时刻估计规划延迟；不要直接把 MCU tick 当成主机 `steady_clock`。

## 上位机需要完成

1. **相机驱动与配置**：在 `camera_config.yaml` 增加显式开关（默认保持现状）；仅海康硬件后端尝试配置 Line1 曝光输出，逐项读回并输出成功/失败原因。相机自动重连时重新配置 IO，给新的相机会话分配代号并清除旧匹配。将 SDK `nFrameNum`、曝光设置值、接收时间和图像一起放进帧结构；`LatestBuffer` 丢掉旧图时保留新图原来的 FrameID，并统计帧号缺口。曝光热更时更新以后帧的曝光中点计算，不能让半曝光补偿沿用旧值。
2. **串口解包**：与下位机约定未占用的 `cmd_id`，新增 `ExposurePose` 解析结果，不改变现有 `0x0002` 状态帧和控制帧格式。现有 `SerialProtocol::feed()` 只返回 `RobotState`，且未知帧会被静默丢弃；必须扩展为能返回两种消息，或提供两个明确的输出队列。现有帧头 `seq` 只承担串口总帧丢包统计，曝光配对用消息内独立序号。对畸形长度、CRC、非有限姿态、重复/倒退序号分别计数。
3. **配对器**：在 `HardwareIo` 侧持有相机帧与曝光快照的有界缓存，先建立经过核验的 FrameID↔曝光序号偏移，再按 ID 精确查找。遇到相机重连、MCU 会话变化、首帧不确定、队列溢出或序号矛盾时撤销已建立的偏移。等待时长、缓存长度和允许降级策略做成配置；先把带 FrameID 的图像交给 L2 检测，检测后再以同一个 FrameID 查询配对姿态，使等待与检测重叠。返回 `PoseMatch {camera_frame_id, image_pose, exposure_time, sync_status}`，在运行时校验它与当前 `FramePacket` 的 ID 一致，避免两次独立查询错位。
4. **算法与时钟**：装甲板分支和打符分支都改用配对姿态；规划/控制的“当前姿态”继续使用周期状态。硬同步并未自动让 MCU 时钟等于主机 `steady_clock`，而 L3 跟踪速度、预测和 L4 延迟都依赖时间戳。需实现并验收 MCU tick→主机单调时钟映射，或提供经过测量的曝光时间估计与误差上限；映射失效则不能宣称整条链路完成硬同步。保持旧软同步模式可切换，硬同步失败时按配置选择显式软回退或本帧安全保持，禁止静默使用最近姿态。
5. **诊断与回归**：记录每帧 `camera_frame_id`、`exposure_seq`、会话代号、配对结果、姿态等待时间、相机丢帧、曝光消息丢包、超时及软回退次数。用模拟相机帧和串口字节流验证正常配对、两边独立丢帧、乱序、序号回绕和两端重启；确认 MindVision、Talos 等其他后端接口仍可用。完成构建及现有装甲板、打符回归。

## 下位机需要完成

1. **硬件与中断**：按上表核对 P7-4/5 实物连线，3.3 V 上拉和 MCU 输入阈值由示波器实测。选有输入捕获能力的定时器通道，配置与曝光开始对应的边沿和必要的去抖/毛刺过滤；计数器溢出后仍能单调扩展。中断只锁存 `edge_tick`、递增 `exposure_seq` 并写入无阻塞环形队列；统计捕获溢出，不在中断里做姿态解算或串口阻塞发送。
2. **姿态时间基准**：明确 IMU 原始采样时间、融合输出时间及滤波延迟都使用同一 MCU 时基。保存足够长的姿态/角速度历史，根据曝光开始 `edge_tick` 取曝光中点对应姿态；若插值/积分所需样本未到，稍后补齐再发送，并标记是否有效。将当前协议的 RPY 单位、坐标系、欧拉角顺序与上位机 `rpyToQuaternion` 对齐；若改为四元数，双方约定 `w,x,y,z` 顺序、归一化条件和 `world<-imu` 方向。
3. **协议与调度**：输出独立 `ExposurePose` 消息，带版本、MCU 启动/会话号、32 位以上曝光序号、64 位扩展 tick、姿态、有效位及 CRC；明确所有字段字节序和 tick 单位。串口发包任务按捕获顺序发出，即使姿态无效也发送含同一序号的无效记录，避免上位机误将下一次姿态配给上一张图。周期状态可保留，但要检查最坏帧率下 RX 带宽、TX 命令接收、队列积压和火控实时性；必要时与上位机一起改波特率/状态频率。
4. **可观测性与自测**：提供 `edge_count`、中断/队列溢出、`ExposurePose` 已发/发送失败数、IMU 历史不足数、tick 回绕及 MCU 会话号。先用信号发生器代替相机脉冲验证输入捕获、序号与时间戳，再接相机实测。断线/重启后序号从何值开始，应在协议里固定并可从日志辨认。

## 双方先冻结的接口约定

| 约定 | 联调前需要写定的内容 |
| --- | --- |
| 曝光事件 | 相机节点名、有效边沿、Strobe Delay/PreDelay、一个脉冲对应一帧的实测结果 |
| `ExposurePose` | 独立 `cmd_id`、协议版本、完整字段偏移/长度/字节序、CRC、姿态单位和坐标方向；附至少一帧十六进制样例供两端互验 |
| 序号与会话 | `HeaderFrame.seq`、`exposure_seq`、`camera_frame_id` 各自含义；溢出、相机重连和 MCU 重启后的清缓存规则 |
| 时间与降级 | tick 频率、曝光中点、姿态采样延迟、主机时钟映射、等待上限，以及硬同步失败时回退还是安全保持 |

## 联调顺序与交付判据

1. **相机单独验 IO**：MVS 配置曝光输出，示波器量白/绿线和 MCU 输入；逐项记录空闲/有效电平、脉宽、上升/下降时间、曝光到边沿延迟。在 0.8 ms 曝光和目标 165 fps 下连续运行；没有稳定的“一帧一边沿”关系就先解决相机节点/极性。
2. **下位机单独验采样与协议**：输入信号发生器脉冲，记录脉冲数、捕获数、发包数、tick 差分、IMU 历史命中率；上位机用协议样例验证 CRC 与字段。断开/接回 IO、重启 MCU，确认会话变化和无效快照可见。
3. **接线后只跑采集，不开火**：同步记录 `camera_frame_id`、`exposure_seq`、MCU tick、两条数据在主机的到达时间及配对结果。先低帧率单帧/慢速建链，再恢复 165 fps；手动遮挡/拔插 USB、丢图、丢曝光消息、重启任一端，确认重新建链前不会出现“有效硬同步”标记。
4. **动云台验姿态**：静止、恒速和往复摆动分别录制软同步/硬同步数据，比较相同目标的重投影误差、方位角新息、姿态相位差与端到端预测误差。同步用示波器确认硬件边沿误差，用日志确认图像/串口到达抖动不再影响所配姿态；若仍有系统性相位差，先排查 IMU 融合延迟、曝光中点和时钟映射。
5. **恢复控制并压测**：确认装甲板、打符在掉同步时保持安全；测试串口最坏吞吐、CPU、缓存积压、相机重连和长时间运行。验收记录至少包含每种故障的预期动作与实际结果、丢帧/丢包计数、同步有效率、最大等待时间、软回退次数，以及与旧软同步对照的精度数据。达到这些条件后再将硬同步设为默认。

## 对照当前仓库的上位机改动清单

下面是实际代码路径，不需要改 L2 检测器、PnP 求解器和 L4 规划公式；但两条 L3 跟踪链路接收的曝光时刻必须正确。

| 顺序 | 当前代码位置 | 具体改法与完成标志 |
| --- | --- | --- |
| 1 | `config/camera_config.yaml`、`src/l1_sensor/camera/camera.cpp` | 增加例如 `hardware_sync.enable: false`、Line1 配置与等待参数；构造 `HikRobot` 时传入。旧配置缺字段仍走当前路径，MindVision 不误启用。 |
| 2 | `tools/camera_sdk/hikrobot/hikrobot.cpp/.hpp` | `capture_start()` 在 `MV_CC_OpenDevice` 后、`MV_CC_StartGrabbing` 前设置并读回 Line1 的曝光输出节点；失败时上报不可用。`MV_CC_GetImageBuffer` 后立刻拷出 `stFrameInfo.nFrameNum`，与 BGR 图像、曝光长度、主机收到时间写入同一 `CameraData`。`capture_stop()`/重新打开设备时递增相机会话号，防止旧帧进入新会话。不能继续只写 `{img, timestamp}`。 |
| 3 | `include/l1_sensor/camera/camera.hpp`、`src/l1_sensor/camera/camera.cpp` | 新增通用 `FramePacket` 或并行的 `readFrame()`，把 `camera_frame_id`、`camera_session` 一路传到 `RuntimeIo`。保留旧 `read()` 适配其他调用方；若直接改虚函数签名，需要同步更新 MindVision 和 Talos。 |
| 4 | `include/l1_sensor/serial/serial_protocol.hpp`、`src/l1_sensor/serial/serial_protocol.cpp` | 当前 `feed()` 返回 `vector<RobotState>`，只有 `cmd_id=0x0002` 会产出状态。新增曝光姿态数据类型和解析分支，保留已有 CRC8/CRC16 检查与所有帧共用的 8 位 `HeaderFrame.seq` 统计；曝光序号作为 payload 字段独立解析。写一条样例消息的双端编解码测试。 |
| 5 | `include/l1_sensor/serial/serial_worker.hpp`、`src/l1_sensor/serial/serial_worker.cpp` | `rxLoop()` 把曝光消息放进独立有界队列并通知等待者。保留当前按串口解析时刻给 `RobotState` 打时间戳的周期姿态历史，因为 `latestGimbalPose()` 还用于当前角度；曝光消息按自身序号和 MCU tick 索引，不进入 `gimbal_history_`。增加 MCU 会话变化、队列溢出、重复/跳号计数。 |
| 6 | `include/runtime/runtime_io.hpp`、`src/runtime/runtime_io.cpp` | `HardwareIo::read()` 当前只给图像和时间戳。加 `readFrame()` 与 `matchExposurePose(frame_id, camera_session, timeout)`（名称可调整）两个语义明确的接口：先取图做检测，后按帧号等快照。`HardwareIo` 对配对结果应用现有 `R_imu_barrel`，输出 `world<-barrel`。Talos 后端可直接返回它已有的同帧姿态，不走串口匹配；其他后端维持旧模式。 |
| 7 | `src/runtime/auto_aim_runtime.cpp` | 主循环 `io->read(frame,timestamp)` 改为读取帧元数据。装甲板约第 543 行、打符约第 661 行的 `waitPose(timestamp)`/`gimbalPoseAt(timestamp)` 改为查询**该帧 ID**的姿态；`pose_ready=false` 或匹配失败时不要再取历史边界姿态并继续正常开火。`latestGimbalPose()` 给规划/控制的用法保留。打符分支也要修改，不能只改装甲板。 |
| 8 | 时间与遥测 | 当前图像 `timestamp=GetImageBuffer 返回时的 now()-半曝光` 不是实测曝光时刻，`Planner::plan()` 用目标 timestamp 计算 `delay.image_to_plan`。硬同步模式要把 MCU 曝光 tick 映射成主机 `steady_clock`，或记录经过验收的估计和误差界；仍将 `arrival_time` 单独保留用于诊断。`RuntimeIoStats` 和 `telemetryFrame` 增加匹配状态、两侧缺口、超时、重建链路和时钟映射质量，供录像回放对比。 |

上位机可先独立完成 1～4 和模拟数据测试；5～8 的正式联调依赖下位机发出的 `ExposurePose` 字节布局、姿态单位与 MCU tick 定义。尤其不要把现有 `pose_delay_ms: 22` 直接设成 0 就宣称完成：它只改变旧历史插值查询，既没有帧号配对，也没有解决曝光时间基准。

## 后续可选方案

若后续希望由下位机控制帧率，可改成 MCU 定时器输出至相机 Line0 做外触发，MCU 在发触发脉冲时锁存姿态，并同时用 Line1 曝光反馈确认实际曝光。触发边沿到真正曝光有相机内部延迟，因此姿态基准仍以 Line1 的曝光反馈为准。此方案还需处理相机忙时拒触发、帧率上限和事件序号，建议放在上述自由运行方案验收之后。
