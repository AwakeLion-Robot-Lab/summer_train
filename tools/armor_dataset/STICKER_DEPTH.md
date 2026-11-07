# 旧贴纸数字：灯条教师生成距离标签，数字学生学习测距

仅处理旧贴纸录像，不混用点阵屏和仿真数据。离线教师用现有整板网络找到候选、
复用 `ArmorRefiner` 确认左右灯条，再用与工程一致的 135×56 mm 小板几何做 IPPE。
保存同一帧、同一候选中的数字裁剪和距离标签，无跨帧/跨录像编号配对。
只有传统灯条精修成功才接受，网络角点回退不会生成距离训练标签。

距离是 **PnP 伪标签**，不是真实测距仪真值。相机标定必须属于该录像；
分辨率相同不足以证明标定适用。当前按工程标定生成，尚未独立确认录像与标定的对应关系。
教师的系统误差也会进入训练，学生与教师的一致性不等于真实距离精度。

## 导出

```bash
xmake build armor_depth_export
xmake build armor_depth_dataset_smoke
build/linux/x86_64/release/armor_depth_dataset_smoke

python3 tools/armor_dataset/export_sticker_depth.py \
  --output out/armor_sticker_depth_v2
```

`sticker_recordings.json` 定义视频、模型、标定和按整段录像划分的 train/val/test。
当前三个静止/转动录像用于 train，`3m_run_mid` 用于 val，`3m_run_fast` 用于 test。
它们来自同一场景、主要在约 3 米，并不是跨场景、全量程的真实测试集。
默认逐帧顺序解码前 1800 帧，每三帧推理一次。`--max-frames=0` 可处理至解码结束，
`--stride` 调整采样间隔。顺序解码可避免依赖有问题的 AVI 随机跳帧索引；
解码结束不代表文件完整，逐视频日志与 summary 会保留实际处理帧数。

旧录像中可能存在旧大板；导出仅保留教师识别为小板的数字类别 2～5。
不会把旧英雄大板按 135 mm 求深度，也不会生成大板样本。
当前素材主要只有数字 2，不足以验证其他数字的识别。

质量门：灯长至少 10 px、重投影 RMSE <=1.5 px、轴向深度 0.3～15 m、
IPPE 两候选相对深度差不超过 10%。这些门限仅筛除明显异常，不保证厘米级准确。
数字通过灯条之间的引导区域搜索白色笔画，使用中央区域的亮度确定阈值，
避免板外亮海报控制二值化。检查的是实际笔画是否被搜索区域截断，
并为灯条中心线留出宽度余量；倾斜数字外接矩形的空角落不会被误判为笔画越界。
保存未做透视矫正的原图裁剪。
裁剪包含原图位置、原始宽高和 K，避免统一缩放后丢失尺度信息。
`review/` 包含带裁剪框的上下文图，仅供人工检查，不进入学生训练。

输出包含：

- `manifest.jsonl`：数字类别、原图框、原始分辨率、K/畸变、PnP Z/欧氏距离/XYZ、
  教师角点/旋转/质量、视频与帧号、原始数字裁剪路径。
- `sequences/*/crops/`：旧贴纸数字原图裁剪。
- `sequences/*/review/`：教师上下文检查图。
- `sequences/*/review_rejected/`：前几个被拒绝的数字搜索区域，供复查裁剪门限。
- `summary.json`：采样参数、源文件/标定/模型哈希、拒绝统计与真实处理帧数。
- `validation_report.json`：裁剪、源帧、标签、尺寸、重投影和集合分离校验。
- `preview.jpg`：不同集合的数字和教师距离预览。
- `generator/`：本次代码与说明。

数字类别也来自旧整板模型，是伪类别标签，应通过预览人工复查。
多块板即使数字相同，也使用各自的框、角点和 PnP 结果，不按数字编号查找距离。

## 训练

需要 Python OpenCV、NumPy、PyTorch；不需要 torchvision 或 GPU。
可在项目独立环境安装 CPU 版 PyTorch：

```bash
python3 -m venv --system-site-packages out/armor_depth_env
out/armor_depth_env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

out/armor_depth_env/bin/python tools/armor_dataset/train_sticker_depth.py \
  out/armor_sticker_depth_v2 --output out/armor_sticker_depth_model_v2
```

学生只输入数字裁剪和数字检测框可获得的五个相机特征：
`log(w/fx), log(h/fy), (u-cx)/fx, (v-cy)/fy, log(w/h)`。
不输入教师距离、灯长、灯条角点、装甲板像素宽高或 IMU。
图像预处理取 `min(B,G,R)` 抑制彩色灯光溢出，再等比例留边到 64×64。
模型用小 CNN 编码字形，以训练集拟合的尺寸先验预测初始深度，再学习乘法修正。
尺寸先验、归一化和固定距离中位数全部只从 train 估计。

训练目标为沿相机光轴的 **Z**，并非 `sqrt(X²+Y²+Z²)` 的斜距。
训练/验证损失与指标只表示对教师伪标签的一致性。
报告同时包含固定输出 train 距离中位数、逆数字高度估计、网络三种方法，
防止窄量程数据下“总输出约三米”造成误判。按 val 选择 checkpoint，test 不参与选模型。

当训练数据只有一个数字时，关闭分类损失，推理返回 `digit: null`。
这表示当前训练只验证数字图像的测距，不能宣称完成了多类别识别。
补齐多个旧贴纸数字录像后，脚本会启用类别分支，并要求测试数字在 train 出现。

输出 `best_weights.pt`、可独立推理的 `student.ts`、预处理配置、
逐轮 history、相对教师的 metrics 和逐样本 predictions。
推理方法按 val 选择：如果网络 MAE 未优于尺寸先验，就保留网络输出供研究，
默认使用尺寸先验作为当前距离结果，并明确返回 `used_method` 及两种估计值。
这个选择不使用 test。网络若未胜过基线，不应把本次训练视为神经网络测距方案已成立。

## 无灯条/PnP的学生推理入口

```bash
out/armor_depth_env/bin/python tools/armor_dataset/predict_sticker_depth.py \
  out/armor_sticker_depth_model_v2 --image /path/to/original_frame.png \
  --bbox 580 410 620 480 --camera config/camera_config.yaml
```

框必须来自**数字检测器或人工数字标注**，而不是现场再运行灯条教师。
推理入口只读原图、数字框和相机内参，不调用灯条、PnP 或跟踪器。
本轮验证裁剪由教师引导得到，尚未验证学生数字检测器的端到端定位表现。
没有把模型接入现有实机或替换当前测距；模型是否可用由独立实测距离数据决定。
