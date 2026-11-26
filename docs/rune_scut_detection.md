# SCUT 能量机关传统视觉识别

实时程序、离线检测和大小符回放均通过 `ScutContourDetector` 处理每帧图像。

识别步骤包括红蓝通道差、轮廓层级分析、靶圈椭圆与缺口／同心圈筛选、
靶圈—灯臂—R 匹配，以及缺口点和灯条端点提取。
灯臂使用独立的颜色纯度门限排除近白反光。首轮无匹配时，提高颜色阈值重试。
缺少匹配证据时返回空结果。

`ScutContourDetector::detectGroup` 返回一帧的 `RuneGroup`。组内每片扇叶关联自己的
靶心节点，并共享同一个 R 标节点。完整的已激活扇叶输出 0～5 号实测角点，部分遮挡时
使用 0～2 号实测外端点。若靶环与灯臂已经配对，遮挡时也保留能单独确认身份的外端点。
遮挡恢复得到的 R 标节点只有推算位置。大符仅看到灯臂时，扇叶节点允许缺少靶心节点。

识别源码和多角点提取代码分别位于
`src/rune/f1-observation/scut_contour_detector.cpp` 和
`scut_feature_points.cpp`。参考提交为
`5c8085ae4b5f1f8f75e7c7d157ea60395f118429`。 [参考实现](https://github.com/scutrobotlab/rm_vision_core)

## 配置与调用

`rune.detector.scut` 包含颜色阈值、面积、形状和匹配门限。
`close_kernel` 使用 1～19 的奇数，当前录像配置为 15。
几何匹配比例由 `geometry.armor_half_height_m / geometry.radius_m` 提供。

`RuneRuntime::detect` 将当前图像、大小符模式和颜色传入检测器。
测量轮廓随检测结果送至 `ObservationStage`，后续沿用项目现有的位姿和云台控制接口。
已标定三维坐标的实测编号角点可参与位姿解算。配置见 [编号角点联合解算](../src/rune/f2-building/README.md#编号角点联合解算)。

`rune.gimbal_only: true` 启用云台跟随，要求 `fire.shoot_enable: false`。R 标被遮挡时，检测器可用至少两组独立的靶圈和灯臂推算符心，推算结果只用于匹配与编号。大符靶圈被遮挡时，可以用实测 R 标与有清晰外端的已激活灯臂参与解算。位姿求解将图像方位角和透视校正分别用于建立五叶布局候选，用联合 SQPnP 与逐叶重投影误差选择结果。

云台根据曝光时刻的姿态直接跟随未激活靶心，有可靠的已激活叶片位姿时可以跟随符心。短时漏检时沿用滤波后的符心与叶片相位，超过有效期后保持最后角度。相机只有姿态输入时，滤波使用固定朝向的坐标轴，完整平移补偿仍需提供 `T_world_barrel`。

`ImageRefiner` 对椭圆、缺口点和灯臂端点做图像运动补偿。当前启用
`rune.image_refiner.track_numbered_points`，按叶片为每个编号维护局部坐标和观测方差。
共享 R 标的叶片复用同一配准结果，同一帧对的各次光流计算复用图像金字塔。
使用近期残差较小、位置分布有间距的最多四个编号点校正叶片位置，再推断已确认但
短时漏检的其他编号。只有实测能够刷新各点的有效期，预测期限取 `max_gap_s`。
窗口以黄色显示预测，离线审计分别记录在 `fan_display_predictions` 和
`gap_display_predictions`。位姿解算使用当前帧通过校验的观测点。
某片叶片短时漏检时，窗口可根据同一机构上已匹配的叶片和 R 标延续其外框。
缺少其他叶片时，需要当前画面有光流和发光圈证据。预测外框以黄色绘制，
沿用 `max_gap_s` 限制，并在离线审计的 `display_predictions` 中单独记录。

`rune.image_refiner.reobserve_target_rings` 启用实时识别和回放共用的靶环重检。
短时缺少靶环时，跟踪外框限定当前图像中的搜索范围。重检从多方向的实际色彩边缘
拟合靶心，并校验边缘覆盖、拟合残差和环内对比度。通过校验的中心单独记录为
`reobserved_rings`。已有平面状态的云台跟随模式可用经过扇区匹配的靶心降权校正相位。
重检不产生编号角点，也不延长最近一次完整观测的有效期。
光流和同组运动都无法提供搜索位置时，可在有效期内沿用近期靶圈外框限定搜索范围。
这一入口仍须通过当前帧边缘校验，成功时以 `history_roi` 记录来源。

`rune.image_refiner.refine_inactive_arm_edges` 启用未激活灯臂边线精修。
沿粗框两侧在原图采样色差梯度，用多行有效边缘拟合两侧长边。
调试画面仅对已激活灯臂连线画框。未激活灯臂显示编号点，当前观测用绿色，缺失点的短时预测用橙色。
审计在 `stable[].inactive_arm_outline` 记录显示四角和当前帧有边缘支持的长边。
`rune_detect --arm-refinement on|off` 可用于同录像对照。

`rune.image_refiner.registration_backend: opencl` 按帧上传灰度图，先批量配准
R 标和靶板，再根据靶板结果批量配准灯臂。`opencl_device` 使用 OpenCV 的设备选择
格式，当前工程指定 `NVIDIA CUDA:GPU:0`。默认后端为 `cpu`，适用于旧配置。
`gpu_ecc_batches` 统计批量提交次数，`gpu_ecc_accepted` 统计通过相关性和形状校验的区域数。
原有光流补充路径仍由 CPU 执行。`rune_detect --registration-backend cpu|opencl`
可以在同一配置和录像上对照两条路径。

装甲板与大小能量机关统一使用顶层 `fire.shoot_enable` 开火开关；
`rune.fire` 只配置能量机关专用的对准门限和射击节拍，旧的嵌套开关不再读取。

## 验证入口

离线时序验证使用 `rune_detect --replay-clock realtime`。第一帧映射到当前时间，
后续帧按配套姿态文本中的时间戳调度，没有配套文本时使用录像 FPS。
`replay_lag_ms` 记录处理开始时相对采集计划的延迟。处理慢于录像节奏时，
工具依次处理所有帧并记录延迟积累。默认 `unpaced` 连续处理，用于检查离线吞吐。

`--features-output` 保存每帧原始与防抖后的特征，包含 R 标、五点及有效标记、
缺口点编号、灯臂点编号、图像尺寸和耗时。

```bash
xmake run rune_detect --input video/rune/2026-11-15_15-05-03.mkv \
  --mode big --frames 590 --replay-clock realtime \
  --features-output build/rune_offline_validation/big_scut.jsonl
python3 tools/rune/summarize_rune_audit.py \
  --audit build/rune_offline_validation/big_scut.jsonl \
  --output build/rune_offline_validation/big_scut_summary.json
```

汇总输出特征可见数量以及平均、P50、P95、最大耗时。
编号正确率和绝对定位误差需要人工逐帧标注，靶心标注可交给
`tools/rune/score_rune_detection.py` 计算检出精确率与召回率。
`rune_detect --ring-reobservation on|off` 可覆盖配置进行 A/B 回放。
默认打分统计原始靶心，添加 `--include-reobserved` 时合计通过当前帧验证的重检靶心。
多角点的三维验证需要同编号的实物坐标或 CAD，平移补偿验证需要
与曝光时间同步、同一固定坐标系下的米制位姿轨迹。

```bash
xmake build auto_aim
xmake run rune_scut_detector_smoke
xmake run rune_scut_features_smoke
xmake run rune_backend_smoke
xmake run rune_image_refiner_smoke

xmake run rune_detect --input video/rune/2026-11-15_15-04-07.mkv \
  --mode small --enemy blue --frames 60 --output build/scut_restored_small.png

xmake run auto_aim_test_small_rune
xmake run auto_aim_test_big_rune
```

大小符回放沿用 `debug.overlay` 决定是否显示窗口。
纯离线检测工具只读取图像或录像，可用 `--features-output` 保存测量审计。
`rune_detect` 分别汇总检测和防抖耗时。回放汇总的 `normal_jump_frames` 统计配有姿态的相邻有效帧之间补偿视角变化后法向差超过 5° 的帧数。

## 回放记录
完整小符录像处理 569 帧，其中 459 帧检出目标；大符处理 590 帧，其中 474 帧检出目标。
这些计数描述检出覆盖情况。识别准确率需要逐帧标注后评估。
大小符主线回放各验证前 120 帧，分别有 88 帧和 96 帧产生有效目标状态。
硬件云台运动尚待实机验证。
