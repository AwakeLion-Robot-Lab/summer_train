# 能量机关多叶三维重建

参考 [SZURPVision/RP-26Rune](https://github.com/SZURPVision/RP-26Rune) 的 `PowerRunePlane.cpp/.hpp`，本地参考提交 `7181abb35ffa282356324266427f9cc295f1218b`，MIT 许可见本目录 `RP-26Rune-LICENSE`。2026-10-16 下载上游 `main` 的 `PowerRunePlane.cpp` 与本地参考版本逐字节核对一致，SHA-256 为 `b55394cb06f146e3e1561e457abd962a2bd99196cdb7f485c7de8aecf728fd74`。

实现复用 OpenCV 与 Eigen，不增加 Ceres、Sophus 依赖。`BuildingStage` 现在持有跨帧状态，`run()` 为非 const 方法，每个处理线程使用独立实例。原有单片调用仍可用；新增 `run(const std::vector<Observation>&, const Frame&)`。`Processor` 已接入多片提取与联合求解。

## 文件组织

| 文件 | 职责 |
| --- | --- |
| `builder.cpp` | 对外入口、历史重置、目标保持与候选组合择优 |
| `model_preparation.cpp` | 模型边界缓存与图像 PCA 锚框 |
| `builder_geometry.cpp` | 能量机关参数、相机和观测校验，五叶扇区旋转与相位计算 |
| `pose_solver.cpp` | 单个候选的多叶匹配、联合 IPPE 求解、位姿输出与滑窗结果复核 |
| `contour_refiner.cpp` | 锚点、轮廓和重力残差，以及 Huber/LM 联合优化 |
| `plane_window.cpp` | 固定公共坐标系下的跨帧平面拟合 |
| `include/rune/f2-building/builder_detail.hpp` | 重建层内部数据结构与函数声明 |

通用投影、刚体变换校验、轮廓距离场和均匀轮廓采样放在 [tools/geometry](../../../tools/geometry/README.md)，这些工具不依赖 `Rune` 类型。能量机关专用的五叶布局、状态模型及质量门限保留在本层。`BuildingStage` 公共接口不变。独立 building/pipeline 测试通过 xmake 统一收录本目录和 `tools/geometry` 的源文件。

## 五叶联合匹配

1. `DetectionStage::runAll()` 同时保留未激活与已激活叶片，按置信度排序，最多生成 10 个独立 ROI。旧 `run()` 仍只返回最佳未激活叶片。
2. 每片分别提取装甲板、灯臂和 R 标轮廓。大符激活状态允许只有灯臂和 R 标；不能凭空构造装甲板约束。
3. 选择未激活叶片作为 0 号参考。提供完整公共坐标系位姿时，优先保留上一帧仍可见的物理目标，避免大符两片未激活目标因置信度顺序改变而跳转。
4. 初始化时枚举未激活叶片与去重 R 标的组合，按全体布局内点数和误差择优，最多 64 对；跟踪时优先保持有效参考叶片。用候选参考叶片做 IPPE 初始化。把其他叶片的图像方向通过当前平面反投影到模型系，再按 72° 间隔匹配 1～4 号扇区。这比直接量化图像夹角更能适应透视压缩。
5. 每个扇区最多一片，角度门限必须小于 36°；按几何误差和置信度解决重复检测，检查 R 中心一致性、径向范围。每片保留检测到的激活状态，未激活叶片可参与布局评分。
6. 旋转模型锚点与语义边界，汇总可见叶片的锚点做联合 IPPE，再联合优化锚点、Chamfer 残差和可用的姿态先验。不同初始化得到相同匹配时只求解一次。

允许部分叶片不可见。无可用未激活叶片时，Processor 不输出目标。保留的单片小符激活状态调用仅用于诊断，输出 `activated=true`，不作为开火目标。目标实际切换时输出 `target_switched`；上下文被清空时输出 `history_reset`，Processor 下游以固定叶片身份维护连续相位，换片不清空拟合；历史失效时重置，避免把 72° 换片误当成转速。开火闩锁仍需外部确认后显式 `rearm()`。

## PCA 锚框与实测模型

所有物点单位为米：R 原点 `(0,0,0)`，参考靶心 `(0,radius,0)`，右侧 NN 点为 `(+half_width,radius,0)`。上游物点的负 y 方向和毫米单位不能直接套用。

```cpp
Rune::Geometry geometry{measured_radius, measured_half_width, measured_half_height};
geometry.armor_boundary_m = measured_inactive_armor;
geometry.arm_boundary_m = measured_inactive_arm;
geometry.center_r_boundary_m = measured_r_boundary; // 单片时可用
geometry.small_activated_model = Rune::BladeModel{
    measured_small_active_armor, measured_small_active_arm, {}, {}};
geometry.big_activated_model = Rune::BladeModel{
    {}, measured_big_active_arm, {}, {}};
Rune::BuildingStage building(geometry);
```

边界数组为 `std::vector<cv::Point3d>`，应沿实际发光边界均匀采样，不是实心区域内部点。同类可包含多个不相连边界，模型数组的相邻关系不用于连线。模型必须与激活状态对应；大符激活灯臂不能套用未激活灯臂。未提供激活模型时，小符仍可提供原有五个装甲板语义点；大符灯臂不会参与模型轮廓优化。

`use_pca_anchors` 默认 true。提供装甲板和灯臂模型后，可从它们的模型系 x/y 包围范围生成四个物理锚框点。也可在 Geometry 或对应 BladeModel 的 `pca_box_anchors_m` 显式配置四角，顺序为 **外侧左、内侧左、内侧右、外侧右**，锚框必须共面 z=0、沿模型 x/y 轴对齐。

图像侧对装甲板与灯臂轮廓按周长均匀采样，去畸变后做 PCA，用网络方向消除轴向正负歧义，提取包围框，再将角点映射回原始畸变像素。PCA 四角与网络五点是不同的物理点，使用各自的物点求解，不混用。没有可用锚框模型或 PCA 退化时回退网络点。轮廓被截断时 PCA 包围范围可能偏小，需要结合联合轮廓残差和质量门限；它不保证任意遮挡下都能恢复完整轮廓。

联合模式中 R 标只提供共享中心锚点，不把 R 字形当成五个随扇区旋转的轮廓。单片模式仍支持显式 R 边界模型。

每个叶片/语义轮廓分别构建 ROI 距离场；投影保留相机畸变。Huber 损失和距离截断限制异常点影响。每类按采样数归一化，轮廓总权重随锚点数量缩放，避免多叶观测时轮廓约束被锚点数量淹没。除总体 RMS 外，还检查每片锚点和每个语义距离场的残差。

## 复用 armor 标定与姿态

```cpp
#include "sensor/camera/camera_calibration.hpp"

frame.bgr = image;
frame.captured_at = capture_time;
const auto& calibration = camera.calibration(); // 与 armor 共用的 Camera
if (!calibration || !Rune::applyCameraCalibration(frame, *calibration)) return;
frame.q_world_barrel = serial.gimbalPoseAt(frame.captured_at); // 曝光时刻姿态
// 若有里程计/定位，额外提供同一时刻、同一固定参考系的完整位姿：
frame.T_world_barrel = localized_barrel_pose; // optional<Isometry3d>，含米制平移
frame.world_frame_id = localization_epoch;   // 重定位或原点变化后递增
```

`Rune::applyCameraCalibration` 统一在 `include/sensor/camera/camera_calibration.hpp` 声明，在 `src/sensor/camera/camera_calibration.cpp` 实现，复用 armor 的 `L1Sensor::CameraCalibration`：同一份内参、畸变和 `T_barrel_camera`，来源仍为 `config/camera_config.yaml`。图像尺寸不匹配或没有外参时清空旧标定并返回 false。固定分辨率、标定不变时可绑定一次；姿态和定位必须逐帧更新，缺失时显式置空，不能沿用旧值。

车体姿态约束在这里具体实现为 **重力方向约束**：世界系 z 向上，把 `normal_world.z()` 作为软残差，鼓励符平面竖直。只需要曝光时刻的 `q_world_barrel`；若给出完整 `T_world_barrel`，使用其旋转。两者同时存在时检查一致性。权重 `normal_vertical_weight_px` 默认 20，设 0 可关闭。该约束只限制平面倾斜，不固定法向的水平 yaw，也不把符强行拉到枪口正前方。对于倾斜安装的符或不以重力为 z 轴的参考系，应调整/关闭该先验。

求解后统一按 `T_barrel_model = T_barrel_camera * T_camera_model` 转到枪口系。位置包含平移，法向只乘旋转；枪口系 x 前、y 左、z 上。`pose.T_barrel_model * point_model` 可转换任意模型点。外参平移仍用 armor 原有配置，需标定到实际枪口原点，不另加 rune 偏移。

## 跨帧平面滑窗

上游当前 `update_plane_points_window()` 每帧 clear，实际只使用单帧；保留的跨帧逻辑处于注释状态。本实现恢复了受门限控制的跨帧拟合：

- **必须提供完整 `T_world_barrel`**，把各帧平面样本变换到固定公共坐标系后累计。仅有 IMU 旋转无法补偿车体平移，因此只有四元数时不启用滑窗。
- 每帧贡献五个均匀分布的平面样本，以等帧权重累积质心与协方差，用 3×3 特征分解求平面，避免分配高维 SVD 矩阵。
- 通过刚体旋转对齐法向，并把符心投影到拟合平面；整个模型一起调整，保持旋转半径、五叶间隔和 `T_barrel_model` 一致。
- 平滑结果重新检查当前帧全部锚点、每个语义轮廓、正深度、最大修正量及重力偏差；不过关则恢复当前原始解并重建窗口。
- 时间倒退/重复、时间断层、模式/颜色切换、相机标定变化、世界坐标原点变化、世界位姿缺失、明显符心/法向突变或失败帧都会清空对应历史。跨帧样本不持有原图。

默认窗口 0.5 s、最多 30 帧、至少 3 帧；最大帧间隔 0.2 s，符心跳变 0.3 m，法向角变化 0.35 rad，拟合平面 RMS 0.03 m，最大模型修正 0.05 m。把 `plane_window_max_frames` 设为 0 可关闭。不能用零平移冒充运动平台的定位；已确认平台静止时才可以显式采用固定原点的姿态。

`Pose::phase_rad` 使用曝光时刻重力在枪口系中的方向定义上方，避免云台俯仰直接改变相位参考；缺少姿态时沿用枪口系 z。下游弹道仍在枪口系计算。下游特征滤波器在提供完整位姿时于世界系跟踪，并在输出时转回枪口系，详见 [f3-estimator](../f3-estimator/README.md)。

## 诊断与资源限制

`Pose` 提供 `reprojection_error_px`、`chamfer_error_px`、`contour_refined`、`blades`（观测索引、扇区、激活状态、是否使用 PCA、枪口系靶心）、`plane_window_frames`、`plane_smoothed`、`attitude_constrained`、`normal_vertical_component`、`target_switched` 和 `history_reset`。平滑后的残差会重新计算。输出法向按枪口系 x 非负定向，这不会改变完整变换矩阵。

默认每类最多 96 个模型样本、15 次轮廓/姿态联合迭代、距离场 padding 24 px、单场最多 1048576 像素、单帧全部场最多 3145728 像素。模型在构造时缓存；每个距离场在一次种子求解内只构建一次，由该种子的 IPPE 候选解共享；初始化枚举其他种子时会重新构建。错误输入返回 nullopt，由 Processor 转成 PoseFailed 并重置运动历史，不复用上帧结果。轮廓模型缺省时仍支持网络锚点联合 PnP。

## 验证

```bash
xmake build rune_joint_building_smoke
xmake run rune_joint_building_smoke
xmake build rune_building_smoke
xmake run rune_building_smoke
xmake build rune_pipeline_smoke
xmake run rune_pipeline_smoke
```

两项 building 测试使用运行时检查，release 构建中也生效。联合测试覆盖大透视五叶匹配、输入顺序变化、重复检测、错误 R 中心、部分叶片缺失、大小符激活模型、PCA 锚框、真实图像分割到 Processor 的链路、目标保持/切换、机体旋转和平移补偿、滑窗降噪、姿态先验，以及历史失效条件。单片测试保留精确投影、畸变、armor 配置复用与异常输入检查。

2026-10-16 本机合成测试：大符四片（1 片未激活、3 片已激活）联合 PCA/轮廓求解约 8.4 ms；模拟交替 ±15 mm 深度观测误差时，滑窗后的世界系深度平均绝对误差约 1.1 mm；单片带噪网络点的 PCA 对照从约 267 mm 降到 133 mm；重力约束对照的法向竖直分量从约 0.157 降到 0.00021。它们都是特定合成夹具的对照，不代表相对上游的性能提升，也不代表实机精度。未进行实车录像或实弹验证。
