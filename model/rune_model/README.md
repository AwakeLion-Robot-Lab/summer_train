# RuneDetectionModel 五关键点模型

- 来源：[SZURPVision/RuneDetectionModel](https://github.com/SZURPVision/RuneDetectionModel)。
- 上游位置：`model/model-0624.onnx`（`master` 分支）。
- SHA-256：`e6e6bddf1f793ece1b9b351b8db061dbb026d32b8a6d71cb39a8fd21ce9cb917`。2026-10-16 直接下载该仓库模型核对，与本地文件完全一致。
- 模型及部署思路采用 MIT 许可，见 `LICENSE` 和 `UPSTREAM_NOTICE.md`。
- 当前适配代码参考 RP-26Rune 的 `NNDetector.cpp`，其 MIT 许可见 `RP-26Rune-LICENSE`。

模型输出为 `[1,18,A]`（适配器同时支持 `[1,A,18]`）：前 3 个通道是类别分数，之后是五个 `(x,y,confidence)`。类别 `0` 未激活、`1` 小符已激活、`2` 大符已激活。原始关键点顺序为 `top / left / R / right / bottom`；适配器转换为工程的 `top / left / right / bottom / center_r`，并按当前大小符模式把已激活类别映射为 `1`。

预处理采用保持比例缩放、四舍五入尺寸、居中补边 `114`、BGR 转 RGB、除以 `255`。NMS 使用原图像素下五点平均中心的距离，质量排序为类别分数乘五点平均置信度。五点均需达到关键点阈值；未达到时整片丢弃。默认配置见 `config/rune_detector.yaml`。

运行及接入示例见 `src/rune/README.md`。
