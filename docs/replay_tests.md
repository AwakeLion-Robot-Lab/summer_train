# 离线回放

装甲板、小符、大符三个入口独立运行，不连接相机或串口，回放中强制关闭开火。
需先启用 OpenVINO：

```bash
xmake f --use_openvino=y
xmake build auto_aim_test_armor
xmake build auto_aim_test_small_tune
xmake build auto_aim_test_big_tune
```

## 装甲板：auto_aim_test_armor

原 `auto_aim_test` 已更名，检测、PnP、整车跟踪、规划、火控及代价曲线保持原有流程。
默认输入为 `video/3m_high.avi` 与同名 `.txt`。

```bash
xmake run auto_aim_test_armor
xmake run auto_aim_test_armor video/3m_mid.avi --view=full
xmake run auto_aim_test_armor --end-index=100
```

## 能量机关

两个入口复用 `config/auto_aim.yaml` 的 `rune` 参数与
`config/planner_config.yaml` 的弹道、延迟和 MPC 参数，模式由目标名称固定，不受 YAML
顶层装甲板 `mode: 1` 影响，无需 `--mode` 参数。默认蓝色目标，模型与推理设备读取
`rune.detector`。

默认视频分别为：小符 `video/rune/2026-10-28_00-05-41.avi`，大符
`video/rune/2026-10-28_00-03-18.avi`。小符录像目前缺少同名姿态文本，默认仅做
检测回放；大符录像自动读取配套 `.txt`。

```bash
# 小符；默认播放 00-05-41，目前仅检测。
xmake run auto_aim_test_small_tune
# 大符；默认播放 00-03-18，读取同名姿态文本。
xmake run auto_aim_test_big_tune
# 也可传入路径，覆盖所选入口的默认视频。
xmake run auto_aim_test_small_tune /path/to/another_small_rune.avi
# 保存最后一帧的标注图；是否显示窗口由 debug.overlay 决定。
xmake run auto_aim_test_small_tune --end-index=100 --output=build/rune_replay.png
```

输入接受视频完整路径或省略 `.avi` 的路径前缀。`--pose=auto` 默认寻找同名 `.txt`，
每行格式为 `t w x y z`，与视频按帧配对；有文本时使用其中的时间戳，避免视频标称
FPS 与实际采集间隔不一致影响运动拟合。文本提前结束时停止回放。`--pose=none`
强制仅检测，`--pose=/path/to/poses.txt` 指定其他**确实与视频逐帧配对**的文本。

有有效姿态且标定匹配时，运行关键点检测 → 轮廓/三维重建 → 运动估计 → 规划 →
火控。缺少姿态、外参或标定分辨率不符时，仅展示检测并说明原因。默认标定来自
`config/camera_config.yaml`；录像的实际标定应通过 `-c=...` 指定，几何尺寸应在
`rune.geometry` 中填写实测值。当前相机配置的机械名义外参不等于录像精度已验证。

`--convention=imu` 默认复用 `config/serial_config.yaml` 的 `R_imu_barrel`，也支持
与装甲板回放相同的 `--convention=sp`。`--enemy=red|blue` 选择轮廓颜色；
`--model=...`、`-d=CPU` 和 `--conf=...` 可临时覆盖检测配置。

三个入口均支持 `--config=...`、`--start-index=N`、`--end-index=N`（包含结束帧，
0 为直到结尾）和 `--wait=N`。窗口中空格暂停/继续，`q` 或 ESC 退出；
`--wait=0` 每次按键前进一帧。能量机关窗口显示五关键点、检测诊断、处理状态及
有效命令角度；结束时输出检测帧数、发现目标帧数和各状态计数。

## 统一调试配置

实机与三个回放入口均读取 `config/auto_aim.yaml` 的同一节点：

```yaml
debug:
  overlay: true
  # 每 N 帧画一次。
  overlay_every: 1

  # PlotJuggler UDP 遥测保留为独立开关。
  plot: true
  plot_host: "127.0.0.1"
  plot_port: 9870
```

`overlay: false` 不创建窗口、不等待键盘；`overlay_every` 只影响绘制，不跳过算法
处理。`plot` 独立控制 UDP 发送，目标地址和端口使用 `plot_host` / `plot_port`，
关闭后不创建 UDP 套接字。无窗口验证可在配置中将 `overlay` 设为 `false`；不需要
遥测时同时关闭 `plot`。不再使用回放命令行的 `--headless` 和 `--plot` 开关。
装甲板 `--view=full` 的 PnP 代价曲线属于叠加窗口，仍受 `debug.overlay` 及绘制间隔
控制，与 PlotJuggler 开关独立。能量机关遥测包含时间、模式、检测数、处理状态、
有效瞄准角度和火控状态，所有帧均发送，不受绘制间隔影响。

## 2026-10-28 验证

三个目标均构建通过。分别验证了 `overlay` / `plot` 的四种开关组合：
每个入口处理 6 帧，`overlay_every: 2` 时绘制 3 帧；UDP 开启时向配置的回环
地址及自选端口收到全部 6 帧遥测，关闭窗口不影响发送，关闭 UDP 不影响绘制。
大小符入口无需传模式参数，遥测模式分别为 2 和 3。

装甲板默认录像前 31 帧回放通过，其中 26 帧进入 Tracking。
能量机关第一段在缩放修正前（`input_scale: 1.0`）完整回放 1843 帧，390 帧检出目标：1453 帧为
`no_detection`，16 帧为 `no_contour`，374 帧为 `motion_not_ready`，尚未进入稳定
运动预测。回放接入完成不代表当前模型在该录像上已能持续跟踪。

第二段在大符模式下验证前 100 帧，64 帧检出目标，因缺少姿态文本仅运行检测。
第一段大符模式的第 90～99 帧验证了视频与姿态同步跳转及标注图保存。
反向帧范围、缺失显式姿态路径和损坏姿态文本均返回明确错误。

后续已加入 `rune.detector.input_scale` 尺度补偿，当前配置为 `0.75`；上面的
390 帧是修正前基线。修正后结果及统计口径见
[尺度诊断与验证](rune_detection_scale.md)。
