# 上游识别与滤波改进适配

对比日期：2026-10-21。本地基线：`2f35321`。参考仓库：
[AwakeLion-Robot-Lab/summer_train](https://github.com/AwakeLion-Robot-Lab/summer_train)。

## 分支与改动来源

默认 `main` 为 `59f79bb`（2026-09-06），早于本地已有的识别和 ESKF 实现。
本次主要适配 `eskf-uvl` 的以下提交：

- [f729229：精简识别流程](https://github.com/AwakeLion-Robot-Lab/summer_train/commit/f729229)：
  删除数字二次分类和轮廓侧灯条，保留整板网络与预测引导的剖面搜索。
- [b97db72：跟踪状态管理](https://github.com/AwakeLion-Robot-Lab/summer_train/commit/b97db7224cd66e4ddf527d686fa88948593aece6)：
  把单个目标及其生命周期、更新时刻、采纳灯条封装成槽位，减少重复管理的数据。
- [f504ac2：TensorRT 类型作用域修复](https://github.com/AwakeLion-Robot-Lab/summer_train/commit/f504ac20415cc99621ecc8d7b2c7eb215177e3ac)：
  `TensorRtBackend::load()` 中使用 `Impl::OutputBinding`，修复开启后端后的编译错误。

另检查了 `rune_v1` 的 `bdf0845`（2026-10-20）。该分支新增另一套 Buff、Talos、Web
支持，并沿用不同的装甲板 PnP/EKF 路线。提交日期较新不足以证明替换本地整车 IESKF
会更好，因此此次保持本地估计模型与运行接口。

## 本地已有的算法

忽略注释和包含路径后，`eskf-uvl` 中的 `ErrorStateEkf`、`VehicleModel` 和
`LightMeasure` 与本地数学实现一致。以下功能此前已经具备：

| 功能 | 本次处理 |
| --- | --- |
| 13 维整车误差状态、SO(3) 注入、对数半径 | 保留 |
| 迭代时重新锚回先验、Joseph 协方差更新 | 保留 |
| 先验点 NIS、像素端点观测、马氏关联门 | 保留 |
| 饱和感知颜色差分、半高质心、剖面拟合与端点插值 | 保留 |
| 冷启动 ROI、暂丢后停止运动外推、双槽重捕获 | 保留 |

本次没有更换网络权重，也没有依据上游配置覆盖本地标定与噪声参数。

## 具体适配

### 识别

`ArmorDetector` 直接采用网络输出的车辆类别，侧边灯条只沿 `lightHints()` 搜索。
不再计算用于轮廓搜索的整块 `lightRoi()`，也不再让独立 ROI 的有无成为剖面搜索的额外开关。
没有预测灯条时跳过搜索；已有整板内部的灯条继续剔除，防止相同端点重复用于更新。

同步删除数字分类器及其配置、`LightSearch`、`findLights()`、轮廓阈值和长宽比配置。
原 `number_classifier` 默认关闭，原配置已使用 `search: profile`，所以不能将代码精简
直接等同于现用配置下识别精度或整帧延迟的提升。

新的检测接口为：

```cpp
auto frame = detector.detectFrame(image, net_roi, enemy_color, light_hints);
auto target = tracker.track(frame.armors, frame.lights, image_pose, captured_at);
```

### 滤波与跟踪

新增 `TrackSlot`，把目标、生命周期、上次更新时刻和本帧实际采纳灯条放在一起。
主槽与备用槽交换时，这些数据一起交换；丢弃时统一清理采纳清单。

`lastMatchCount()` 和 `lastMatchedIds()` 由主槽的实际采纳清单生成。原实现使用全局
逐帧成员，备用槽更新会覆盖它们，可能出现当前目标仍在 TempLost、诊断却显示已经
关联上另一辆车的情况。新增回归用例覆盖了这一过程。

参考上游减少重复协方差管理的做法，活目标的 `covariance()` 直接引用滤波器中的 P，
去掉每次预测、更新之后的重复矩阵赋值。本地规划器仍通过 `toL4TargetState()` 消费 P，
因此保留无滤波器快照的协方差，并在 `snapshot()` 时复制当前 P。测试覆盖初始化、
预测、更新后三个阶段的快照一致性和本地规划接口。

## 验证

编译检查包括 `auto_aim`、`auto_aim_test`、`track_diag` 和三个新增 smoke 目标。

- `light_detector_smoke`：过曝、细灯条、倾斜、边界、重复预测；额外覆盖网络 ROI
  坐标恢复、预测直接触发搜索、整板灯条去重、推理异常后清空上一帧数据。
- `eskf_target_smoke`：初始化、关联、闭环收敛、快照隔离、协方差和本地规划适配。
- `eskf_tracker_smoke`：确认、暂丢、恢复、超时、断流、冷启动 ROI、侧灯条与双槽诊断。

使用相同 CPU 模型和参数，对修改前后各回放 SP demo 的 0–100 帧：

```bash
build/linux/x86_64/release/track_diag tests/data/sp_auto_aim/demo \
  -c=tests/data/sp_auto_aim/camera_calibration.yaml --convention=sp \
  -m=model/armor_model/szu-v8n-fp16.onnx --enemy=any --end-index=100 \
  --scan-step=5 --out=/tmp/summer_train_recognition_filter_adapted
```

| 指标 | 修改前 | 修改后 |
| --- | --- | --- |
| 回放帧数 | 101 | 101 |
| 有检测的帧数／检测总数 | 91／91 | 91／91 |
| Tracking 帧数 | 81 | 81 |
| 跟踪丢失／重置 | 0 | 0 |

`frame.csv`、`obs.csv`、`overlay.csv`、`pred.csv` 逐字节一致；有效规划行也保持一致。
无效规划行的 `before_fire` 使用空命中时间减本次进程的单调时钟起点，因此不用于 A/B 比较。
这项回放验证原检测与估计结果未退化；该工具当前走整图检测，侧灯条路径另由合成集成测试验证。

本机未启用 TensorRT，类型作用域修复仅完成当前 OpenVINO 构建下的兼容检查，
CUDA/TensorRT 分支仍需在相应 SDK 与硬件上验证。
