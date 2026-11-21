# 能量机关录像索引

2026-11-15 导入，类型由采集者指定。导入时位于 `video/dafusp/`，当前位于
`video/rune/`；以下路径均相对于项目根目录。

| 类型 | 视频 | 帧数 | 姿态时间范围（秒） |
| --- | --- | ---: | --- |
| `const_rune` | `video/rune/2026-11-15_15-03-12.mkv` | 583 | 0～24.183041437 |
| `small_rune` | `video/rune/2026-11-15_15-04-07.mkv` | 569 | 0～23.601942930 |
| `big_rune` | `video/rune/2026-11-15_15-05-03.mkv` | 590 | 0～24.432344520 |

三段视频均为 MJPEG 编码、1440×1080，标称帧率为 289/12（约 24.083 FPS）。
每段均有同名 `.txt`，格式为 `t w x y z`；已核对记录数与视频帧数一致、数值有限、
四元数非零且时间戳严格递增。回放应使用文本中的实际时间戳。
这些检查仅验证文件结构和数量，不代表姿态坐标约定及标定精度已经验证。

`const_rune` 为素材标签，当前算法模式只有 `small` 和 `big`，此处不将其自动映射为
某一模式。

小符和大符录像已分别接入以下测试入口的默认输入，并自动读取同名姿态文本：

```bash
xmake run auto_aim_test_small_rune
xmake run auto_aim_test_big_rune
```

## 传统视觉离线预览

`tools/rune/rune_contour_preview.py` 使用 Python 和 OpenCV（`cv2`），接受本地图片或录像。预览左侧按颜色标出环形、条形和其他轮廓，右侧显示颜色分割掩膜。终端汇总帧数和各类候选数量，候选按帧累计。

```bash
python3 tools/rune/rune_contour_preview.py video/rune/2026-11-15_15-04-07.mkv \
  --color blue --frames 120 --output build/rune_contour_preview/small.mp4
```

图片使用 `.png` 输出，录像使用 `.mp4` 输出并沿用标称帧率。省略 `--frames` 会处理整段录像。可用 `--color`、`--threshold` 和 `--min-area` 调整显示的候选。轮廓面积以预览图像素为单位。

预览参考纯色分割与轮廓分析思路，输出独立的标注画面。候选按可见形状分类，反光和背景中的同色图形也可能被标出。 [rm_vision_core](https://github.com/scutrobotlab/rm_vision_core)
