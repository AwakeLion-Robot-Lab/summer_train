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

## 连续帧时序滤波

`StickerDepthPredictor` 会缓存模型，并可为每个数字实例维护独立的距离/径向速度 Kalman 状态。
滤波作用于验证集选择的原始距离结果，目前是尺寸基线；网络估计仍单独保留。
单图 `infer(...)` 和原来的 `--image` 用法保持无状态，不对单张图伪造平滑效果。

```python
from predict_sticker_depth import StickerDepthPredictor
from temporal_depth import DepthFilterConfig

predictor = StickerDepthPredictor(model_directory, DepthFilterConfig())
result = predictor.predict(
    frame, digit_box, camera_matrix,
    timestamp_s=capture_time, target_id=digit_track_id,
    sequence_id="camera_1",
)
depth = result["depth_z_m"]  # 没有新鲜有效估计时为 None
# 本帧未检出该目标：
missing = predictor.missing(digit_track_id, next_capture_time, "camera_1")
```

`target_id` 必须是数字检测/跟踪器给出的**实例编号**，不能直接用数字类别“2”。
多个同号装甲板使用不同实例编号。更换录像或相机时使用新的 `sequence_id` 或调用 `reset()`。
同一实例的相机内参或图像尺寸发生变化会重置状态。

默认配置位于 `depth_filter_config.json`：

- 用捕获时间差计算状态转移，支持不均匀帧间隔，不硬编码 FPS。
- 4σ 创新门抑制孤立跳点；连续三个一致的新距离会重新初始化，避免真实变化后一直停在旧值。
- 连续无有效观测最多预测 100 ms，之后输出 `None`；超过 250 ms 后的新观测重新初始化。
- 时间戳重复或回退会重置该实例。短时预测不是新的检测结果。
- 过程噪声、测量噪声和时间阈值都可配置；输出的 `depth_std_m` 是模型内的不确定度，未经实测校准。

JSONL 流式命令行输入示例（图片相对路径以 JSONL 所在目录为基准）：

```json
{"image":"frame_000.png","bbox":[580,410,620,480],"timestamp_s":0.00,"target_id":"plate_A"}
{"image":"frame_001.png","bbox":[581,410,621,480],"timestamp_s":0.02,"target_id":"plate_A"}
{"timestamp_s":0.04,"target_id":"plate_A","missing":true}
```

```bash
out/armor_depth_env/bin/python tools/armor_dataset/predict_sticker_depth.py \
  out/armor_sticker_depth_model_v2 --stream /path/to/frames.jsonl \
  --camera config/camera_config.yaml --output out/depth_stream.jsonl
```

流式模式默认开启滤波，`--no-filter` 可关闭，`--filter-config` 可指定参数。
输出包括原始/滤波后距离、观测是否被接收、速度、估计新鲜度、重置原因与计算耗时。
可选输入 `available_time_s` 与 `timestamp_s` 必须属于同一时钟，输出二者之差作为观测传递年龄。
该年龄、CPU 计算时间与滤波引起的动态滞后含义不同。

## 滤波评估与延迟

```bash
python3 tools/armor_dataset/test_temporal_depth.py
python3 tools/armor_dataset/evaluate_temporal_depth.py \
  --dataset out/armor_sticker_depth_v2 --model out/armor_sticker_depth_model_v2 \
  --timestamps paired --output out/armor_depth_temporal_v2
```

现有 AVI 标称 90 FPS，但对应 `.txt` 的捕获时间戳间隔约为 20 ms。
评估优先读取同名配套时间戳并核对递增性、帧号与长度，`--timestamps paired` 要求其存在；
`auto` 在没有配套文本时退回标称时间并标注来源，`nominal` 明确只用标称时间。
直接按 90 FPS 解释现有采样会误算速度和延迟。

回放不调用灯条或 PnP：原始估计来自学生模型保存的输出，或数字框/相机参数计算的尺寸基线。
PnP 标签只参与事后误差统计，不参与状态更新、异常门或目标关联。
因为数据没有稳定实例 ID，离线评估用数字框位置和大小做近似关联；这一关联不是实例真值，
也不等于已验证完整学生检测器。每段录像独立重置，绝不按数字类别混合多块板。

`report.json` 分开记录：

- 原始与滤波后相对 PnP 伪标签的误差，以及有效输出数量，不能据此宣称真实距离精度提升。
- 同一近似轨迹中相邻有效观测的变化 RMS，包含真实运动，不能直接等同于测量噪声。
- 滤波 CPU 时间，不能当作传感器到输出的总延迟。
- 30 FPS 已知真值合成序列的静止噪声、0.5 Hz 运动拟合滞后和真实距离阶跃的半响应延迟。
  合成动态滞后只代表相应运动与配置，不是实机固定延迟承诺。

同时保存逐观测 `predictions.jsonl`、合成轨迹；安装 matplotlib 时输出 `synthetic_response.png`。
滤波参数不依据测试集调优。这一层用于抑制抖动，不消除已有标定或尺寸系统偏差。
