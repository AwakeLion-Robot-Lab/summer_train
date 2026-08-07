# L3 录制视频回放

`l3_video_replay` 读取 `record_capture` 生成的同名 AVI/TXT，按 TXT 的真实时间戳执行
L2 检测、L3 估计、画面叠加和 PlotJuggler 数据发送。

默认的 `realtime` 模式模拟在线相机：严格跟随 TXT 时间轴；当 L2/L3 慢于录制帧率时，
跳过已经过期的帧，只处理当前最新帧。因此画面时长与原录像一致，L3 收到的时间戳也仍是
真实采样时间。优化检测精度或需要逐帧复现时，使用 `offline` 模式处理全部帧。

## 编译和运行

```bash
source .deps/l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64/setupvars.sh
xmake f -m release --use_openvino=y
xmake build l3_video_replay
xmake run l3_video_replay recorder/3m_high
```

输入参数是不带扩展名的公共路径；也可以直接传 `.avi` 或 `.txt`。常用选项：

```text
--start-index=0       起始帧
--end-index=0         结束帧，0 表示文件末尾
--mode=realtime       原速实时回放，落后时跳到最新帧（默认）
--mode=offline        不限速处理每一帧
--prediction-ms=100   绘制多少毫秒后的整车预测，0 仅关闭未来层
--robot-id=-1         PlotJuggler 目标，-1 自动锁定首个目标
--no-geometry-constraints  临时关闭半径、夹角和物理面一对一约束
--no-ippe-dual-candidates  关闭 IPPE 双候选，恢复单候选基线
--no-predicted-face-yaw-selection  关闭预测 face yaw 选解，恢复 yaw 误差选解
--show-armor-text     启动时显示装甲类别、置信度、XYZ、RPY 和物理面编号
--no-gui              不打开 OpenCV 窗口
--no-plotjuggler      不发送 UDP
--plotjuggler-port=9870
```

装甲板附近的详细文字默认隐藏，只保留框线和整车轮廓。播放时按 `L` 可立即显示或
隐藏文字；空格暂停或继续，`N`/右方向键单帧前进，`Q`/Esc 退出。

## 整车预测叠加

Tracker 进入 `Tracking` 后，视频会反投影 EKF 整车模型：中心点、3/4 块装甲中心、
装甲四角和各物理面编号 `P0...`。颜色含义如下：

- 绿色：本帧观测成功更新后的 EKF 整车状态。
- 橙色：本帧没有匹配观测，仅由 EKF 预测得到的状态。
- 紫色：按 `center += velocity * dt`、`yaw += yaw_rate * dt` 外推到
  `--prediction-ms` 后的整车轮廓。

这部分表示 L3 匀速/匀角速度模型预测，不包含弹道飞行时间等 L4 预测。

## PlotJuggler

1. 打开 PlotJuggler 的 UDP Server，端口设为 `9870`，消息协议选择 JSON。
2. 加载 `config/l3_plotjuggler.xml`。
3. 启动回放；程序默认发送 UDP，不要求 ROS2。

曲线单位统一为 rad、m、m/s 和 ms。JSON 中没有有效 L3 观测时只发送
`valid=false`，不会插入伪造的零姿态。

`/performance/realtime_lag_ms` 是处理完当前帧时相对录像时间轴的滞后；
`/performance/skipped_frames` 是 `realtime` 模式的累计跳帧数。窗口标题信息和程序退出
摘要也会显示这两个指标。`offline` 模式不会跳帧，滞后固定报告为 0。

`Vehicle geometry` 页显示 EKF 的 `radius_1_m`、`radius_2_m` 和四装甲中心连线的
`minimum_corner_angle_rad`。当前关联与后验状态要求该最小内角不低于 50°；两半径
相等时角度为 90°。`geometry_constraints_enabled` 标明本次回放是否启用约束，视频
顶部和退出摘要也会显示开关状态。

用同一帧区间进行 A/B 对比：

```bash
# 开启（默认）
xmake run l3_video_replay recorder/3m_run_fast --end-index=780 --mode=offline

# 关闭，恢复位置+yaw关联基线
xmake run l3_video_replay recorder/3m_run_fast --end-index=780 --mode=offline \
  --no-geometry-constraints
```

不要在一次滤波过程中途切换；开关会改变此前的关联和状态历史，两次从同一帧重新运行
才具有可比性。持久默认值可由 `tracker.enable_vehicle_geometry_constraints` 设置。

IPPE 双候选同样支持 A/B：`pnp.enable_ippe_dual_candidates` 为持久默认值，
回放命令行可用 `--no-ippe-dual-candidates` 临时关闭：

```bash
# 双候选选解（默认）
xmake run l3_video_replay recorder/3m_run_fast --end-index=780 --mode=offline

# 恢复单候选基线
xmake run l3_video_replay recorder/3m_run_fast --end-index=780 --mode=offline \
  --no-ippe-dual-candidates
```

已确认目标的预测 face yaw 选解也可用 `--no-predicted-face-yaw-selection` 关闭，
恢复按 yaw 优化误差选解；持久默认值由 `pnp.enable_predicted_face_yaw_selection`
设置。

## 标定与文件检查

`config/carmera_config.yaml` 当前保留本工程内参，并临时复用参考工程的
`T_barrel_camera`。若相机安装发生变化，必须替换为新的手眼标定结果。

程序启动时会检查视频与 TXT 是否存在、TXT 是否为空、时间戳是否严格递增、四元数
是否合法、视频帧数是否等于姿态行数，以及视频分辨率是否匹配标定。当前
`recorder/3m_run_no.txt` 为空，因此该组输入会明确报错。

当前 6 个录像文件的 AVI 都在录制过程中损坏：有效 MJPEG 帧只到某个下标，之后是
截断帧和无法解析的数据，而 AVI 头部仍声明完整的帧数（与 TXT 行数一致），因此
回放必须用 `--end-index` 停在最后可用帧。2026-08-07 检查的可用区间：

| 文件 | TXT 行数 | 最后可用帧 | 回放参数 |
|---|---|---|---|
| 3m_run_fast | 3181 | 780 | `--end-index=780` |
| 3m_run_mid | 3977 | 568 | `--end-index=568` |
| 3m_high | 2960 | 1917 | `--end-index=1917` |
| 3m_mid | 2955 | 1099 | `--end-index=1099` |
| 3m_low | 3008 | 1046 | `--end-index=1046` |
| 3m_run_no | 0（TXT 为空） | 421 | 本就无法回放 |

各文件损坏位置互不相同，说明是录制过程写入链路的问题而非单个文件偶发；重新录制
前建议先检查 recorder 的写入链路。
