先拍照
xmake build camera_capture
xmake run camera_capture -- config/carmera_config.yaml \
  --output-dir=calibration_images \
  --fps=1.0

按q结束

接着调用cv标定
xmake run camera_calibrator -- \
  calibration_images/2026-07-29_18-50-23
  
文件名记得改成刚拍下来那个




# 相机标定完整流程

本文介绍如何使用项目中的 `camera_capture` 和 `camera_calibrator`，从相机自动拍照开始，生成可以直接用于项目的相机内参和畸变参数。

## 1. 标定内容

当前工具完成的是单目相机内参标定，输出：

- 图像分辨率 `image_width`、`image_height`
- 相机内参 `fx`、`fy`、`cx`、`cy`
- 畸变参数 `k1`、`k2`、`p1`、`p2`、`k3`

工具不会生成 `T_barrel_camera`。该参数是相机到枪管坐标系的机械外参，需要通过独立的外参标定流程获得。

标定结果只适用于标定时的相机状态。修改以下任意设置后都应重新标定：

- 图像分辨率
- ROI
- binning
- 镜头焦距或对焦位置
- 更换镜头或相机

## 2. 标定板定义

当前使用的棋盘格为：

- 方格数量：`11 × 8`
- OpenCV 内角点数量：`10 × 7`

方格数量和内角点数量不要混淆：

```text
11 × 8 个方格 = 10 × 7 个内角点
```

因此标定程序参数是：

```bash
--cols=10 --rows=7
```

如果只需要相机内参和畸变参数，棋盘格真实边长不会影响标定结果。程序默认使用：

```bash
--square-size=1.0
```

如果需要具有真实尺度的标定板位姿，可以将它设置为实际单格边长，并保证所有长度使用同一单位。

## 3. 拍摄前准备

### 3.1 固定相机设置

先确认 [config/carmera_config.yaml](../../config/carmera_config.yaml) 中的相机配置正确，例如：

```yaml
camera_name: "hikrobot"
exposure_ms: 4
gain: 10.0
vid_pid: "2bdf:0001"
```

拍摄过程中不要修改曝光、增益、ROI、分辨率、镜头焦距或对焦位置。

建议：

- 关闭自动对焦，使用固定对焦。
- 避免过曝，黑白方格应保留明显灰度差。
- 缩短曝光时间，减少移动棋盘格时产生的拖影。
- 保持镜头和相机安装牢固。
- 优先使用平整、刚性、哑光的实体棋盘格。

显示器上的棋盘格也可以检测，但屏幕像素、摩尔纹、反光和平面误差可能限制最终精度。

### 3.2 检查相机连接

可以先检查 USB 设备：

```bash
lsusb | grep 2bdf
```

如果程序提示：

```text
Unable to open usb
MV_CC_EnumDevices failed
```

通常应依次检查：

1. 相机是否被 `lsusb` 正确识别。
2. 当前用户是否拥有 USB 设备访问权限。
3. Hikrobot SDK 的 udev 规则是否已经安装并重新加载。
4. 相机是否被其他程序独占。
5. USB 数据线、接口和供电是否可靠。

## 4. 自动拍照

在项目根目录运行：

```bash
xmake run camera_capture -- config/carmera_config.yaml \
  --output-dir=calibration_images \
  --fps=1.0
```

参数说明：

- `config/carmera_config.yaml`：相机配置文件。
- `--output-dir`：照片保存根目录。
- `--fps`：每秒保存的照片数量，默认 `1.0`。

每次运行会创建独立目录：

```text
calibration_images/YYYY-MM-DD_HH-MM-SS/
```

照片按顺序保存：

```text
000001.png
000002.png
000003.png
...
```

保存的是相机原始 PNG：

- 不缩放
- 不叠加文字
- 不覆盖旧照片
- 使用无损 PNG 编码

预览窗口中的文字只绘制在预览副本上，不会写入原始照片。

按以下任意按键退出：

- `q`
- `Q`
- `Esc`

## 5. 正确的拍摄方法

高精度标定依赖的是视角多样性，不是相同姿态下的照片数量。

建议拍摄 50～150 张，覆盖以下情况：

### 5.1 画面位置

让棋盘格中心依次覆盖：

- 图像中心
- 左上、右上、左下、右下
- 左边缘、右边缘
- 上边缘、下边缘

畸变主要在图像边缘体现，只拍摄画面中心会导致畸变参数不稳定。

### 5.2 距离和大小

同时拍摄：

- 棋盘格占画面较大的近距离照片
- 中等距离照片
- 棋盘格占画面较小的远距离照片

棋盘格必须完整出现在画面中，不能缺少外侧角点。

### 5.3 倾斜角度

同时覆盖：

- 正对相机
- 向左、向右倾斜
- 向上、向下倾斜
- 绕相机光轴旋转
- 两个方向同时倾斜

不要只让棋盘格始终平行于相机成像面。

### 5.4 图像质量

每张有效照片应满足：

- 棋盘格完整可见。
- 黑白方格对比明显。
- 没有明显运动模糊。
- 没有大面积过曝或反光。
- 棋盘格没有弯曲。
- 棋盘格角点附近没有遮挡。

自动保存会产生连续相似照片，这是正常的。标定程序会进行清晰度筛选和多样视角选择，不会让大量重复帧对结果产生过高权重。

## 6. 检查照片

拍摄完成后先确认生成的会话目录：

```bash
find calibration_images -maxdepth 2 -type f -name '*.png' | sort
```

也可以检查照片数量：

```bash
find calibration_images/会话目录 -maxdepth 1 \
  -type f -name '*.png' | wc -l
```

建议人工打开若干照片，重点检查：

- 第一张和最后一张
- 棋盘格位于四角的照片
- 大倾角照片
- 近距离照片
- 远距离照片

标定程序只扫描输入目录中的 `.png`、`.jpg` 和 `.jpeg`，不会读取 ZIP 文件。

## 7. 运行高精度标定

假设照片保存在：

```text
calibration_images/2026-07-29_18-50-23/
```

运行：

```bash
xmake run camera_calibrator -- \
  calibration_images/2026-07-29_18-50-23
```

完整写法：

```bash
xmake run camera_calibrator -- \
  calibration_images/2026-07-29_18-50-23 \
  --cols=10 \
  --rows=7 \
  --square-size=1.0 \
  --max-views=1000 \
  --min-views=20 \
  --max-sharpness=3.0 \
  --min-contrast=50.0 \
  --min-area=0.005 \
  --bootstrap=20 \
  --output-dir=calibration_results
```

查看全部参数：

```bash
xmake run camera_calibrator -- --help
```

### 7.1 参数说明

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--cols` | `10` | 水平方向内角点数量 |
| `--rows` | `7` | 垂直方向内角点数量 |
| `--square-size` | `1.0` | 单格边长，内参标定可使用任意一致单位 |
| `--max-views` | `1000` | 最多使用的多样视角数量 |
| `--min-views` | `20` | 标定所需的最少有效视角 |
| `--max-sharpness` | `3.0` | 最大边缘过渡宽度，单位为像素 |
| `--min-contrast` | `50.0` | 黑白区域最小灰度差 |
| `--min-area` | `0.005` | 内角点凸包占图像面积的最小比例 |
| `--bootstrap` | `20` | 参数重采样次数，设为 `0` 可关闭 |
| `--output-dir` | `calibration_results` | 标定结果根目录 |
| `--preview` | `false` | 显示角点检测预览 |

快速检查数据时可以关闭 bootstrap：

```bash
xmake run camera_calibrator -- \
  calibration_images/会话目录 \
  --bootstrap=0
```

bootstrap 只用于评估参数稳定性。关闭它不会改变最终标定算法、交叉验证或输出格式，但报告中不会包含 bootstrap 标准差。

需要人工查看角点时使用：

```bash
xmake run camera_calibrator -- \
  calibration_images/会话目录 \
  --preview=true
```

预览状态下按 `q`、`Q` 或 `Esc` 会终止标定。

## 8. 程序内部流程

标定程序按以下顺序处理数据。

### 8.1 输入检查

- 自然排序读取照片。
- 检查图片是否能够解码。
- 检查所有图片是否具有相同分辨率。
- 拒绝不支持的位深和通道格式。

### 8.2 亚像素角点检测

使用 OpenCV `findChessboardCornersSB` 检测 `10 × 7` 个内角点，并启用：

- 归一化
- 穷举检测
- 高精度模式

该接口直接返回亚像素角点。

### 8.3 图像质量筛选

每张照片都会计算：

- 水平和垂直边缘清晰度
- 黑白区域对比度
- 棋盘格占画面比例
- 棋盘格中心位置
- 旋转和透视变化

检测失败、模糊、低对比度或棋盘格过小的照片不会进入标定。

### 8.4 多样视角选择

程序根据棋盘格的：

- 画面位置
- 大小
- 旋转角度
- 透视倾斜
- 四个外角位置

选择差异最大的视角，默认最多使用 100 张。连续重复照片会被降低优先级。

### 8.5 鲁棒异常剔除

初次标定后计算每张照片的重投影 RMS。异常阈值为：

```text
max(0.25 px, median + 3 × 1.4826 × MAD)
```

超过阈值的最差视角会被逐步剔除并重新标定。最多剔除初始候选视角的 15%，且不会低于 `--min-views`。

### 8.6 畸变模型选择

程序比较：

- 固定 `k3=0` 的四参数模型
- 允许 `k3` 的五参数模型

两种模型都进行固定分组的五折交叉验证。只有五参数模型满足以下条件时才会启用 `k3`：

1. 验证误差至少改善 2%。
2. `k3` 相对于其不确定度足够稳定。

否则输出：

```text
brown_conrady_4_fixed_k3
```

此时 YAML 仍输出五个畸变参数，第五项写为 `0`，与项目运行时接口保持一致。

程序不会自动使用 8 参数 rational 模型，避免高阶参数过拟合。

### 8.7 质量验收

默认通过条件包括：

- 总体 RMS 不超过 `0.25 px`
- 每视角 RMS 的 p95 不超过 `0.35 px`
- 五折验证 RMS 不超过 `0.30 px`
- 焦距为有限正数
- 主点位于图像范围内
- 焦距相对标准差不超过 `0.5%`
- 主点标准差不超过 `2 px`

棋盘格位置覆盖不足或尺度变化不足会记录为警告。

## 9. 输出文件

每次标定都会创建独立目录：

```text
calibration_results/YYYY-MM-DD_HH-MM-SS/
├── calibration.yaml
├── report.yaml
├── used_images.txt
├── coverage.png
├── reprojection_errors.png
└── undistorted_preview/
```

### 9.1 calibration.yaml

这是可以提供给项目运行时的标定参数：

```yaml
calibration:
  image_width: 1440
  image_height: 1080
  camera_matrix:
    - [fx, 0.0, cx]
    - [0.0, fy, cy]
    - [0.0, 0.0, 1.0]
  distortion_coefficients: [k1, k2, p1, p2, k3]
```

各参数含义：

- `fx`：水平方向焦距，单位为像素。
- `fy`：垂直方向焦距，单位为像素。
- `cx`：主点横坐标。
- `cy`：主点纵坐标。
- `k1`、`k2`、`k3`：径向畸变。
- `p1`、`p2`：切向畸变。

### 9.2 report.yaml

完整质量报告，包括：

- 输入、检出、筛选和最终使用的照片数量
- 标定板规格
- 自动选择的畸变模型
- 总体和每视角重投影误差
- 五折交叉验证误差
- 内参标准差
- bootstrap 标准差
- 每张照片的使用状态
- 被拒绝照片的具体原因
- 最终质量是否通过

应确认：

```yaml
quality_passed: true
```

如果为 `false`，即使生成了 `calibration.yaml`，也不建议直接投入运行。

### 9.3 used_images.txt

记录最终参与标定的照片路径，便于复查和复现实验。

### 9.4 coverage.png

显示棋盘格中心在画面中的覆盖情况：

- 圆心表示棋盘格中心位置。
- 圆的大小表示棋盘格相对面积。

理想情况下，圆点应覆盖画面中心、边缘和四角，并具有多种大小。

### 9.5 reprojection_errors.png

显示每个最终视角的重投影 RMS。少数柱子明显高于其他视角时，应在 `report.yaml` 中定位对应照片。

### 9.6 undistorted_preview

包含若干组原图和去畸变图对比。重点观察：

- 图像边缘的直线是否变直。
- 去畸变后是否出现不合理拉伸。
- 不同位置照片的校正方向是否一致。

## 10. 将参数写入相机配置

确认标定质量通过后，将 `calibration.yaml` 中的 `calibration:` 节点复制到：

[config/carmera_config.yaml](../../config/carmera_config.yaml)

例如：

```yaml
camera_name: "hikrobot"
exposure_ms: 4
gain: 10.0
vid_pid: "2bdf:0001"

calibration:
  image_width: 1440
  image_height: 1080
  camera_matrix:
    - [1636.201, 0.0, 707.105]
    - [0.0, 1635.813, 555.920]
    - [0.0, 0.0, 1.0]
  distortion_coefficients:
    [-0.129744, 0.141441, -0.000549, -0.000072, 0.0]
```

建议复制工具输出的完整精度数值，不要使用本文示例中的截断值。

不要删除已有且经过独立标定的 `T_barrel_camera`。如果尚未完成相机到枪管的外参标定，则保持该节点不存在，不要用单位矩阵冒充有效外参。

## 11. 当前实拍数据示例

使用：

```text
calibration_images/2026-07-29_18-50-23/
```

得到：

```text
输入照片                 263
成功检测棋盘格           238
图像质量通过             191
多样视角选择             100
重投影异常剔除             2
最终使用                  98
总体 RMS             0.119094 px
五折验证 RMS         0.119248 px
每视角 p95           0.201800 px
质量结果                  PASS
```

自动选择的模型：

```text
brown_conrady_4_fixed_k3
```

对应完整输出位于：

```text
calibration_results/2026-07-30_19-55-34/
```

## 12. 常见问题

### 棋盘格始终检测失败

检查：

- 使用的是 10 × 7 内角点，而不是 11 × 8。
- 棋盘格是否完整出现在画面中。
- 图像是否严重过曝、欠曝或模糊。
- 黑白方格边界是否清晰。
- 输入目录是否正确。

### 有效图片少于 20 张

重新拍摄更多清晰、完整并具有不同位置和倾角的照片。不要通过降低 `--min-views` 来强行输出低可信度参数。

### 大量图片因 sharpness 被拒绝

优先调整拍摄条件：

- 缩短曝光时间。
- 固定相机或减慢棋盘格运动。
- 重新对焦。
- 增加环境照明。

只有确认图像确实清晰时，才适当增大 `--max-sharpness`。

### RMS 很低但结果不稳定

低 RMS 不等于数据覆盖充分。检查：

- `coverage.png` 是否覆盖四角和边缘。
- 是否同时存在近距离和远距离照片。
- 是否存在多个倾斜方向。
- `report.yaml` 中的参数标准差是否过大。
- 五折验证误差是否明显高于总体 RMS。

### 标定后自瞄结果仍有固定偏差

内参只能修正相机成像模型。固定瞄准偏差还可能来自：

- `T_barrel_camera` 未标定或方向错误。
- 相机、枪管安装发生移动。
- 弹道模型误差。
- 时间同步误差。
- 相机分辨率或 ROI 与标定时不同。

### 修改曝光或增益后是否需要重新标定

单独修改曝光或增益通常不会改变几何内参，但可能影响角点检测质量。修改焦距、对焦位置、ROI、binning 或分辨率后必须重新标定。

## 13. 推荐操作顺序

完整流程可以归纳为：

```text
固定相机和镜头
    ↓
确认 11×8 方格 / 10×7 内角点
    ↓
运行 camera_capture 自动保存原始 PNG
    ↓
拍摄覆盖中心、边缘、四角、远近和多种倾角的照片
    ↓
人工抽查照片
    ↓
运行 camera_calibrator
    ↓
检查 quality_passed、RMS、交叉验证和 coverage.png
    ↓
查看去畸变预览
    ↓
将 calibration 节点写入 carmera_config.yaml
    ↓
保持相机分辨率、ROI、焦距和安装状态不变
```
