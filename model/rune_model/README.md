# RuneDetectionModel 五关键点模型

- 来源：[SZURPVision/RuneDetectionModel](https://github.com/SZURPVision/RuneDetectionModel)。
- 上游位置：`model/model-0624.onnx`（`master` 分支）。
- SHA-256：`e6e6bddf1f793ece1b9b351b8db061dbb026d32b8a6d71cb39a8fd21ce9cb917`。2026-10-16 直接下载该仓库模型核对，与本地文件完全一致。
- 模型及部署思路采用 MIT 许可，见 `LICENSE` 和 `UPSTREAM_NOTICE.md`。
- 当前适配代码参考 RP-26Rune 的 `NNDetector.cpp`，其 MIT 许可见 `RP-26Rune-LICENSE`。

模型输出为 `[1,18,A]`（适配器同时支持 `[1,A,18]`）：前 3 个通道是类别分数，之后是五个 `(x,y,confidence)`。类别 `0` 未激活、`1` 小符已激活、`2` 大符已激活。原始关键点顺序为 `top / left / R / right / bottom`；适配器转换为工程的 `top / left / right / bottom / center_r`，并按当前大小符模式把已激活类别映射为 `1`。

预处理采用保持比例缩放、四舍五入尺寸、居中补边 `114`、BGR 转 RGB、除以 `255`。
后处理以 SZU 官方 `infer/fivepoints_infer.cpp` 为准：默认至少三个关键点达到阈值，
NMS 使用原图像素下**过阈值关键点**的平均中心，质量排序为类别分数乘这些点的
平均置信度；低分点保留原预测及置信度，负坐标或非有限数仍使整片拒绝。
`min_valid_keypoints` 可设为 `1..5`，设为 `5` 恢复旧筛选规则。
无配置时类别/关键点阈值均为官方缺省 `0.8`；项目 YAML 显式保留已验证的
`0.65 / 0.5` 门限与 `input_scale: 0.75`。默认配置见 `config/auto_aim.yaml`。

本地 `Keypoints::point_confidences / point_valid` 按本地五点顺序保存。弱点只作
ROI 搜索初值，不作为 PnP 锚点或滤波测量；当前平面 IPPE 初始化需要至少四个
可靠锚点。靶心使用可靠的相对两点或完整四点计算，弱 R 不作为中心测量。
仅三个可靠点可显示检测，单片不能据此初始化三维位姿。调试图只画可靠点及
两端均可靠的连线。验证记录见 [SZU 后处理对齐](../../docs/rune_szu_alignment.md)。

`rune.detector.input_scale` 是本项目增加的可选尺度补偿，取值 `(0, 1]`，缺省
`1.0` 完全保留上游预处理。当前录像配置使用 `0.75`：先把原图缩至网络画布的
75% 适配尺寸，再居中补边；缩小时使用面积采样以减少 LED 点阵混叠。网络尺寸
保持 `640x480`，图像内容为 `480x360`。解码使用同一缩放与补边偏移恢复原图
五点坐标，标定、PnP 与 NMS 继续使用原图像素。此参数针对本机录像验证，
更换拍摄距离或镜头后应重新评估。

运行及接入示例见 `src/rune/README.md`。
