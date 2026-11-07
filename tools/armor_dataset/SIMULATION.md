# 参数化点阵屏仿真数据集 v3：仅小装甲板

这版将视频字形转为轮廓，再用新的点阵、辐射亮度、板面和相机参数渲染。
没有复制录屏的 RGB 数字贴片，也没有使用上一版合成图作为训练图。
它是 CPU 上的三维平面投影和简化成像仿真，不是对未知硬件的精确复刻或照片级赛场仿真。

## 尺寸来源和显示假设

`simulation_config.json` 仅使用 **135×56 mm 小装甲板**，不生成大装甲板。
尺寸沿用项目 `config/auto_aim.yaml` 的 `armor.small_width_m: 0.135` 和 `armor.height_m: 0.056`。
配置内记录了来源字段和源文件 SHA256。这里沿用的是现有 PnP 特征点之间的跨度，
不是从视频测得的新屏幕外沿尺寸，也不能据此声称新屏幕已经确认具有同样的边界。
此前 160×80 mm 的 v2 数据保留在独立目录；当前默认配置不再使用那个占位尺寸。

数字居中、字高为板高的 62%～82%，仍是可修改的显示布局假设。
边缘反射、LED 形状/间距、数字占比、行亮度变化也都是仿真假设。
点阵发光由程序生成，字形轮廓来自用户视频，不代表掌握了实际点阵编码或字体设计文件。

不同于数字可见笔画，屏幕四角是该假设平面的投影。
即使暗处看不清板边，也保存完整模拟四角，方便验证几何训练链路。
这不使四角成为真实装甲板的测量真值。数字布局或尺寸假设错误会引入实机系统误差。

## 运行

仅依赖 Python 3.9+、NumPy、OpenCV。项目根目录执行：

```bash
python3 tools/armor_dataset/simulate.py \
  --reference out/armor_digits_v1/reference \
  --output out/armor_led_sim_v3_small

python3 tools/armor_dataset/validate_simulation.py out/armor_led_sim_v3_small

python3 tools/armor_dataset/preview_simulation.py out/armor_led_sim_v3_small
```

默认 2400 张训练、300 张合成验证、300 张合成测试图，640×640。
`--train/--val/--test/--size/--seed/--config` 可调整生成参数。
输出目录必须为空，避免覆盖已有数据。相同参数和依赖环境下可复现。
相机畸变系数为零，焦距、主点、板面偏航/俯仰/滚转、距离和屏内数字大小随机变化。
保存曝光、点阵密度、发光形状、模糊、光晕、噪声及压缩参数。
每 15 张有一张无目标负样本，其余图有 1～3 块不相交、完整入画的模拟板面。
当前版本不包含遮挡或画外截断，四个角点的可见性值均为 2。

训练/验证用 32、40 行点阵；合成测试保留 48 行点阵，作为未用于训练的模拟密度变化。
所有集合仍共享同一视频重建的字形轮廓。这个划分只检验模拟分布变化，不是独立真实测试。
没有训练神经网络，也没有声称真实场景准确率。

## 输出

- `data.yaml`、`images/`、`labels/`：红蓝数字 1～7 的 14 类 YOLO 检测数据，框为发光字形支持区域。
- `panel_pose/data.yaml`、`panel_pose/images/`、`panel_pose/labels/`：独立 YOLO Pose 数据；
  单类 `small_panel`，四角顺序为板面局部坐标的左上、右上、右下、左下；`kpt_shape: [4,3]`。
- `masks/`、`panel_pose/masks/`：字形和模拟板面的实例掩码，0 为背景，1～3 为实例。
- `classification/{train,val,test}/{1..7,negative}/`：64×64 三通道分类小图。
- `manifest.jsonl`：每张图的 K、每块板的三维角点、旋转向量、平移向量、四角像素、尺寸假设和外观参数。
- `classification_manifest.jsonl`：分类小图与父图的关联。
- `reference/silhouettes.jpg`：从视频模板重建的七种字形轮廓。
- `reference/source_templates/`：重建轮廓所需的源裁剪及模板说明；源裁剪仅用于提取轮廓，不贴入生成图。
- `assumptions.json`、`summary.json`：完整假设、源裁剪 SHA256、数量、参数和依赖版本。
- `preview.jpg`：黄色为数字框，绿色为假设板面的轮廓和四角；标注线不在训练原图中。
- `digit_preview.jpg`：重新渲染的 14 种数字/颜色组合。
- `simulation_preview.mp4`：运行预览脚本后生成的七秒连续运动示意；叠加框线仅用于展示。
- `generator/`：本次生成与验证使用的脚本及参数。
- `validation_report.json`：完成验证后生成的完整性和几何一致性报告。

两个任务的 images 在本机通过硬链接共享字节；压缩包中是可独立读取的两份文件。
两个 YAML 使用各自生成目录的绝对路径，移动目录后应修改 `path`。
分类文件夹排序后的类别下标为 0～6 对应 1～7，下标 7 为 negative。
数字身份没有自动映射到现有项目的 `ArmorClass`。

校验包括图像/标注/掩码/分类裁剪一致性、集合内类别覆盖、图像无字节级重复、
用 OpenCV 独立重投影三维角点，以及通过 PnP 反算模拟平移。
PnP 校验使用同一套假设几何，验证的是实现的一致性，不是真实测距精度。

解压后也可以用包内脚本继续生成，无需原始视频：

```bash
python3 out/armor_led_sim_v3_small/generator/simulate.py \
  --reference out/armor_led_sim_v3_small/reference/source_templates \
  --output out/armor_led_sim_v4_small
```

要采用新的长宽和布局，先编辑一份 `simulation_config.json`，再通过 `--config` 传入。
正式实物明确后，更新几何与显示参数，并用独立真实录像验证模型迁移效果。
