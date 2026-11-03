# 能量机关：SZU 后处理对齐

2026-10-29 以 [SZURPVision/RuneDetectionModel 的官方部署代码](https://github.com/SZURPVision/RuneDetectionModel/blob/master/infer/fivepoints_infer.cpp)
为依据，将本地五点全部过阈值的规则改为可配置的最少有效关键点数。

## 检测行为

本文记录 2026-10-29 的基线策略。2026-10-31 工程配置在 NMS 分组后增加了可关闭
的同叶片关键点融合；筛选、类别及 NMS 分组规则保持本文约定，坐标可有变化。
最新行为与 A/B 验证见 [关键点抖动优化](rune_keypoint_fusion.md)。

- `min_valid_keypoints` 默认 `3`，范围 `1..5`；设为 `5` 恢复原来的严格筛选。
- NMS 中心取过关键点门限的点的平均坐标，质量为类别分数乘这些点的平均置信度。
  使用原图像素距离，先做跨类别抑制，再按大小符模式过滤。
- 保留五个原始预测及逐点置信度；负坐标或非有限坐标、置信度使整片拒绝，
  超出右/下边界的坐标按官方实现裁至图像边界。
- `Keypoints::point_confidences / point_valid` 使用本地顺序：上、左、右、下、R。
  调试图只画可靠点及两端均可靠的边；NMS 标签中心也只用可靠点。
- 无配置时类别和关键点阈值均为官方默认 `0.8`。工程 YAML 保留已验证的
  `confidence_threshold: 0.65`、`keypoint_confidence_threshold: 0.5` 和
  `input_scale: 0.75`，本次不混入阈值或输入尺度实验。

## 与几何链路衔接

检测允许三个可靠点，当前平面 IPPE 初始化仍需要至少四个可靠锚点。
弱点坐标可作为轮廓搜索初值，但不作为 PnP 或滤波测量。缺一个靶板点时，
靶心由可靠的相对两点估计；弱 R 不参与中心测量或跨帧目标延续判定。
四个靶板点可靠而 R 低分时，可直接用四点解算。

共享 R 假设借用其他检测的可靠 R 时，同步复制坐标、置信度、可靠性及轮廓。
因此三个可靠靶板点也可由其他检测补充可靠 R；孤立的三点检测不能凭空补点
初始化位姿。PCA 包络仍要求五个网络点可靠，避免弱点干扰包络方向。
轮廓、重投影和运动拟合质量检查继续生效。

## 验证

以下两个测试通过：

```bash
xmake build rune_detector_smoke
xmake run rune_detector_smoke
xmake build rune_partial_keypoints_smoke
xmake run rune_partial_keypoints_smoke
```

检测测试覆盖两种输出布局、三点接收/两点拒绝、五点严格模式、有效点 NMS
中心和质量排序、坐标恢复、负数/非有限值、门限等号、边界裁剪和弱点绘制。
几何测试逐个将五点中的一个标为弱点并偏移 120/90 像素，验证剩余四点的
解算不被污染、滤波忽略弱 R、孤立三点不能初始化以及共享可靠 R 的补充。

使用同一权重、CPU、项目 YAML 门限和 `input_scale: 0.75`，修改前后完整
运行 `rune_detect`，统计至少有一个检测的帧：

| 录像 / 模式 | 总帧数 | 修改前 | 修改后 |
| --- | ---: | ---: | ---: |
| `2026-10-28_00-03-18.avi` / 小符 | 1843 | 1534 | 1534 |
| `2026-10-28_00-05-41.avi` / 大符 | 798 | 581 | 581 |

两段录像在当前输入尺度和门限下，过类别门限的候选没有因关键点门限被拒绝，
最终检测也没有不足五点的样本，因此这次放宽规则**没有增加这两段的检出帧数**。
不能据此宣称召回率提高，也不能由检出数量推断误检率。未检出帧在这次实验中
均没有候选通过类别门限；这仍需结合叶片熄灯状态和人工标注分析。

第一段有配套姿态的完整回放，三点和五点配置均得到：
`no_detection=309`、`no_contour=1`、`motion_not_ready=1533`，
`found_frames=1533`，无可开火帧。第二段缺少姿态，完整大符回放只执行检测。
本次对齐不解决已有的运动拟合未就绪问题。

原始日志、三点/五点无窗口配置和汇总保存在 `build/rune_szu_alignment/`：
`before_first.log`、`after_first.log`、`before_second.log`、`after_second.log`、
`comparison.json`、`pipeline_three.log`、`pipeline_five.log`、`pipeline_big.log`。
性能耗时受并行验证负载影响，本次不作速度对比。
