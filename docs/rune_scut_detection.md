# SCUT 能量机关传统视觉识别

SCUT 检测器已恢复到主线。实时程序、离线检测和大小符回放通过
`runtime::makeRuneDetector` 使用同一选择逻辑。当前
`config/auto_aim.yaml` 的 `rune.detector.backend` 为 `scut`。

识别步骤包括红蓝通道差、轮廓层级分析、靶圈椭圆与缺口／同心圈筛选、
靶圈—灯臂—R 匹配，以及缺口点和灯条端点提取。
灯臂使用独立的颜色纯度门限排除近白反光。首轮无匹配时，提高颜色阈值重试。
缺少匹配证据时返回空结果。

识别源码和多角点提取代码从删除前的本地修改记录恢复，分别位于
`src/rune/f1-observation/scut_contour_detector.cpp` 和
`scut_feature_points.cpp`。参考提交为
`5c8085ae4b5f1f8f75e7c7d157ea60395f118429`。 [参考实现](https://github.com/scutrobotlab/rm_vision_core)

## 配置与调用

`rune.detector.scut` 包含颜色阈值、面积、形状和匹配门限。
`close_kernel` 使用 1～19 的奇数，当前录像配置为 15。
几何匹配比例由 `geometry.armor_half_height_m / geometry.radius_m` 提供。
`backend: openvino` 可切换到现有模型，省略后端字段的旧配置也使用 OpenVINO。

`RuneRuntime::detect` 将当前图像、大小符模式和颜色传入检测器。
测量轮廓随检测结果送至 `ObservationStage`，后续沿用项目现有的位姿和云台控制接口。
多角点用于图像显示，位姿接口使用五个兼容锚点。
`ImageRefiner` 恢复了椭圆、缺口点和带语义编号的灯臂端点的运动补偿。

装甲板与大小能量机关统一使用顶层 `fire.shoot_enable` 开火开关；
`rune.fire` 只配置能量机关专用的对准门限和射击节拍，旧的嵌套开关不再读取。

## 验证入口

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

## 恢复验证

主程序、检测工具和大小符回放均已重新构建。
SCUT 检测、多角点、后端加载、图像防抖、OpenVINO 检测和靶圈精修共六项冒烟测试通过。

完整小符录像处理 569 帧，其中 459 帧检出目标；大符处理 590 帧，其中 474 帧检出目标。
这些计数描述检出覆盖情况。识别准确率需要逐帧标注后评估。
大小符主线回放各验证前 120 帧，分别有 88 帧和 96 帧产生有效目标状态。
硬件云台运动尚待实机验证。
