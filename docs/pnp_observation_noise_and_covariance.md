# PnP 观测噪声与 EKF 协方差设计

本文规定单装甲板 PnP 观测的噪声语义，并给出从四个二维角点传播到
EKF 观测协方差的理论依据和实现边界。目标输出为：

```text
z = [x_world, y_world, z_world, yaw_world]
R = Cov(z) ∈ R^(4×4)
```

`R` 的变量顺序必须与 `ArmorCovariance` 的约定一致。前三个分量使用米，
`yaw` 使用弧度，因此 `R` 的对角线单位依次为
`m², m², m², rad²`，位置与 yaw 的非对角项单位为 `m·rad`。

当前 `PnpSolver` 已完成 IPPE 双候选、几何检查和像素 RMSE 计算，但尚未
实现下面的协方差传播。因此在实现和验证完成前，
`ArmorQuality::covariance_ok` 必须保持 `false`，观测不能进入 EKF 正式
更新。`Armor::R` 中的默认单位阵不是有效测量噪声。

## 1. 必须区分的噪声来源

PnP 输出的不确定性不只来自重投影误差。至少要区分以下几类：

1. **角点随机噪声**：检测器输出的四个像素角点会受模糊、曝光、灯条形变、
   网络量化和遮挡影响。这是第一版通过投影 Jacobian 传播的主要噪声。
2. **标定不确定性**：相机内参、畸变和 `camera -> barrel` 静态外参存在误差。
   它通常跨帧相关，不能严格地当成每帧独立白噪声。
3. **姿态同步不确定性**：图像曝光时刻与云台姿态时间不一致，或 IMU/编码器
   姿态本身有噪声。运动越快，它对世界系位置和 yaw 的影响通常越大。
4. **模型误差**：装甲板物理尺寸、板面非平面、角点顺序或大小装甲板类别
   错误。这些首先应通过模型和几何检查排除，而不是只靠增大 `R` 掩盖。
5. **IPPE 模态歧义**：近正视平面目标可能存在两个都能解释像素观测的位姿。
   这是双峰分布，不等价于单个高斯噪声。

`reprojection_error` 只描述选中位姿对本帧四个角点的拟合程度。它不能单独
代表上述全部误差，也不能直接当作 EKF 的 `R`。

## 2. 像素观测模型

把四个角点按 PnP 使用的相同顺序堆叠为：

```text
u = [u1, v1, u2, v2, u3, v3, u4, v4]^T ∈ R^8
```

OpenCV 位姿参数定义为：

```text
θ = [rvec_x, rvec_y, rvec_z, t_x, t_y, t_z]^T ∈ R^6
```

使用原始相机内参、畸变参数和装甲板三维模型建立投影函数：

```text
h(θ) = projectPoints(model_points, rvec, tvec, K, distortion)
e(θ) = u - h(θ)
```

第一版可以假设每个像素坐标独立、同方差：

```text
e ~ N(0, S)
S = σ_px² I8
σ_px = ArmorConfig::corner_noise
```

`corner_noise` 表示单个像素坐标方向的标准差，单位为 pixel；它不是四点
RMSE。当前默认值 `0.8 px` 是待数据标定的工程初值，不应被解释为已经验证
过的传感器指标。

后续若检测器能给出角点协方差，应使用：

```text
S = diag(S1, S2, S3, S4),  Si ∈ R^(2×2)
```

这样可以表达每个角点不同的置信度，以及沿灯条方向和垂直灯条方向不同的
定位精度。检测置信度本身不是方差，必须通过重复标注数据或覆盖率实验完成
从 confidence 到 covariance 的标定。

## 3. 从像素噪声得到 6×6 位姿协方差

在最终位姿附近对投影模型线性化：

```text
h(θ + δθ) ≈ h(θ) + J δθ
J = ∂h / ∂θ ∈ R^(8×6)
```

在角点高斯噪声、数据关联正确、相机模型固定且所选位姿模态正确的条件下，
加权非线性最小二乘的局部位姿协方差为：

```text
Λθ = J^T S^(-1) J
Σθ = Λθ^(-1)
```

若 `S = σ_px² I8`，则：

```text
Σθ = σ_px² (J^T J)^(-1)
```

这就是高斯牛顿 Hessian/Fisher 信息的局部协方差近似。Ceres 官方文档给出
了无权和带观测协方差的这两个形式；MLPnP 也通过加权信息矩阵估计 PnP
旋转和平移精度：

- [Ceres Solver: Covariance Estimation](https://ceres-solver.readthedocs.io/latest/nnls_covariance.html)
- [MLPnP: A Real-Time Maximum Likelihood Solution to the Perspective-n-Point Problem](https://arxiv.org/pdf/1607.08112)

OpenCV 的 `cv::projectPoints()` 可以直接返回大小为
`2N × (10 + numDistCoeffs)` 的 Jacobian。其前六列分别对应 `rvec` 和
`tvec`，所以四个角点时可直接取得 `8×6` 的 `J`，不需要在正式实现中
维护一套忽略畸变的手写投影导数：

```cpp
std::vector<cv::Point2d> projected;
cv::Mat jacobian;
cv::projectPoints(
  object_points,
  rvec,
  tvec,
  camera_matrix,
  distortion_coefficients,
  projected,
  jacobian);

const cv::Mat J_pose = jacobian.colRange(0, 6);  // 8×6
```

接口与 Jacobian 列定义参见
[OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

### 3.1 为什么建议先细化 IPPE 候选

上述协方差是在一个最小二乘最优点附近建立的局部近似。为了让它的统计
含义与实际输出一致，建议对每个几何有效的 IPPE 候选分别执行完整六自由度
`cv::solvePnPRefineLM()`：

1. 以该 IPPE 候选的 `rvec/tvec` 为初值；
2. 使用原始内参、畸变和相同角点顺序细化；
3. 重新执行有限性、正深度、正面朝向和 RMSE 检查；
4. 在最终选中候选的位置计算 `projectPoints()` Jacobian。

这一步优化完整的旋转和平移，不是固定位置搜索 yaw。若继续保留未经细化
的原始 IPPE 输出，上式仍可作为局部近似，但不能声称它就是该输出算法严格
达到的最大似然协方差。

计算协方差时应使用最终位置处的**未加阻尼**信息矩阵
`J^T S^(-1) J`。LM 中用于求步长的 `H + λI` 不能直接作为协方差信息矩阵，
否则阻尼会人为制造不存在的信息。

## 4. 从 6×6 位姿协方差传播到 4×4 观测协方差

记：

```text
A = R_world_barrel R_barrel_camera
p_world = A t_camera + R_world_barrel t_barrel_camera
R_world_armor = A Rodrigues(rvec)
yaw_world = atan2(R_world_armor(1, 0), R_world_armor(0, 0))
```

定义当前 EKF 所需观测映射：

```text
z = g(θ) = [p_world.x, p_world.y, p_world.z, yaw_world]^T
```

它对 PnP 参数的 Jacobian 为：

```text
G = ∂g/∂θ ∈ R^(4×6)

    [ 0_(3×3)       A       ]
G = [                         ]
    [ ∂yaw/∂rvec   0_(1×3)  ]
```

像素角点噪声对应的 EKF 观测协方差为：

```text
R_pixel = G Σθ G^T
```

必须保留完整矩阵，不能只取四个对角元素。PnP 的旋转和平移通常相关，传播
后位置与 yaw 也可能相关，这些非对角项会影响 EKF 的创新协方差和增益。

`∂yaw/∂rvec` 第一版可使用中心差分，并通过 `limit_rad()` 计算包角后的角差：

```text
∂yaw/∂rvec_k ≈
  wrap(yaw(rvec + ε e_k) - yaw(rvec - ε e_k)) / (2ε)
```

这里必须与 OpenCV Jacobian 使用相同的参数化：既然 `J` 表示对 Rodrigues
向量分量直接加法的导数，`G` 也应使用 `rvec ± εe_k`。如果未来改用李代数
左扰动或右扰动，投影 Jacobian 与输出 Jacobian 必须一起改，不能混用扰动
约定。相关状态估计约定可参考
[A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537)。

## 5. 标定和姿态同步误差

若 `R_calibration` 与 `R_pose_sync` 已经通过各自 Jacobian 传播到了同一个
`[x_world, y_world, z_world, yaw_world]` 空间，并且暂时假设三类误差独立，
可以采用首版工程近似：

```text
Armor::R = R_pixel + R_calibration + R_pose_sync
```

更完整的形式是：

```text
R = Gθ Σθ Gθ^T
  + Gκ Σκ Gκ^T
  + Gq Σq Gq^T
```

其中 `κ` 表示内参、畸变和静态外参，`q` 表示图像时刻枪管姿态。OpenCV
`projectPoints()` 还提供内参和畸变参数的投影导数，但需要考虑“PnP 最优位姿
也会随标定参数变化”，不能简单截取额外几列后直接相加。首版更适合通过
标定参数 Monte Carlo 或重复标定实验，先得到保守的 `R_calibration`。

还要注意：静态标定偏差通常在许多帧之间相关。每帧把它当独立噪声加入
`R` 只是一种保守接口近似，长期严格处理应考虑把关键外参偏差加入滤波状态，
或显式建模相关噪声，防止 EKF 错误地认为可通过大量帧把系统偏差平均掉。

当前 `FrameInfo` 只携带图像时刻 `q_world_barrel` 和有效标志，没有携带该
姿态的协方差。因此在扩展接口或完成实测标定之前，不能宣称
`R_pose_sync` 已被逐帧严格计算。

## 6. RMSE 不能直接代替像素噪声方差

本项目的重投影误差定义为四个角点的二维像素 RMSE：

```text
RMSE = sqrt(sum_i((du_i)^2 + (dv_i)^2) / 4)
RSS = sum_i((du_i)^2 + (dv_i)^2) = 4 RMSE²
```

四个角点提供 8 个标量观测，完整位姿有 6 个参数，所以用同一批点拟合位姿
后只剩：

```text
dof = 8 - 6 = 2
```

若所有模型假设均成立，残差方差的无偏估计形式是：

```text
σ_hat² = RSS / dof = 2 RMSE²
```

但这里只有 2 个自由度，单帧估计会剧烈波动。一个几何条件很差的平面位姿
也可能拥有很小的 RMSE，却沿某些位姿方向具有很大方差。因此：

- 不使用当前单帧 RMSE 直接设置 `corner_noise`；
- 优先通过静止目标、多距离、多倾角的重复数据标定 `corner_noise`；
- RMSE 主要用于异常值门限和模型失配检测；
- 若采用残差估计，只能作为带噪的保守膨胀依据，并应设置噪声下限；
- `max(corner_noise², RSS/2)` 可作为工程启发式，不能写成理论必然结论。

从简单针孔关系 `z ≈ fL/l_pixel` 还可以得到直观结论：若像素宽度误差近似
固定，则 `σ_z` 近似随 `z²` 增长，深度方差近似随 `z⁴` 增长。正式实现不必
手写这个距离四次方模型，因为投影 Jacobian 会自动反映距离、焦距、板尺寸
和倾角对各方向协方差的影响；该关系适合用于检查计算结果是否符合物理直觉。

## 7. IPPE 双解不是“大一点的 yaw 方差”

当两个几何有效 IPPE 解的 RMSE 都通过门限且足够接近时，真实后验更接近：

```text
p(z | u) ≈ w1 N(z1, R1) + w2 N(z2, R2)
```

单个 Hessian 协方差只能描述所选解附近的一个局部峰，不能覆盖两个分离的
yaw 候选。IPPE 原论文说明了弱透视、近正视条件下的二解歧义，并指出两个
候选甚至都可能满足正面朝向条件：

- [Infinitesimal Plane-Based Pose Estimation](https://3dvar.com/Collins2014Infinitesimal.pdf)

因此当前质量语义应保持：

- `yaw_ambiguous == false`：通过其他数值检查后，允许完整 4D 观测；
- `yaw_ambiguous == true`：不得仅靠膨胀 yaw 方差把完整观测送入 EKF；
- 若 Tracker 支持，可只使用 `R.topLeftCorner<3, 3>()` 更新位置；
- 可靠 yaw 需要 Tracker 先验消歧、多假设/高斯混合跟踪，或更多几何约束。

简单放大 `R(3,3)` 还可能忽略位置与 yaw 的交叉相关，因此只能作为明确记录
过局限性的临时策略，不能令 `covariance_ok` 变为真。

## 8. `covariance_ok` 的数值与质量门限

完整 4×4 `R` 只有满足以下条件时才可标记有效：

1. `pnp_ok && reprojection_ok && geometry_ok && finite`；
2. `yaw_ambiguous == false`；
3. `J`、`Σθ`、`G` 和 `R` 的全部元素有限；
4. 加权 Jacobian 数值秩为 6；
5. SVD 最小/最大奇异值比例超过配置门限；
6. `R_world_armor(0,0)^2 + R_world_armor(1,0)^2` 未接近 yaw 提取奇异点；
7. `R` 对称，且特征值在容差范围内非负；
8. 图像时刻枪管姿态有效，所需的标定和同步噪声已提供或明确采用经验证的
   保守近似。

不要直接求矩阵逆。应使用带秩检查的 SVD、QR 或 LDLT 解线性系统；接近奇异
时拒绝协方差，而不是输出极大或非有限数。Ceres 协方差文档同样强调秩亏和
近奇异 Jacobian 下普通逆不可靠。

数值误差可能破坏严格对称性，输出前执行：

```cpp
R = 0.5 * (R + R.transpose());
```

允许为浮点误差设置很小的负特征值容差，但不能用无条件夹正数的方法掩盖
错误的 Jacobian、坐标系或参数顺序。

## 9. 推荐验证方法

协方差代码通过编译并不表示统计上正确。至少需要以下测试：

### 9.1 Jacobian 数值检查

- 用中心差分检查 `projectPoints()` 前六列和输出映射 `G`；
- 覆盖多个距离、yaw、pitch、图像位置和畸变区域；
- 检查扰动参数化、角度包裹和矩阵列顺序。

### 9.2 合成 Monte Carlo

1. 设定已知装甲板位姿并投影四角点；
2. 按给定 `S` 给像素坐标添加高斯噪声；
3. 重复运行与正式代码完全相同的 IPPE、细化、选解和坐标变换；
4. 统计 `[x, y, z, wrapped yaw]` 的经验协方差；
5. 与预测 `R` 比较对角线、相关系数和 1σ/2σ 覆盖率。

局部协方差验证时应只统计落在同一个 IPPE 模态的样本。若样本自然分成两个
簇，应验证 `yaw_ambiguous` 能拒绝完整 4D 高斯观测，而不是强行让一个 `R`
覆盖两个簇。

### 9.3 实测静止目标

- 在多距离、多倾角和多曝光条件下采集重复观测；
- 使用独立参考位姿或高精度测量作为真值；
- 标定 `corner_noise`，检查 NIS/覆盖率是否随距离和倾角系统性偏小；
- 分开记录检测噪声、标定误差和时间同步误差，避免用一个经验倍率混淆来源。

## 10. 建议实现顺序

1. 对每个有效 IPPE 候选执行六自由度 LM 细化，并保持现有几何与 RMSE 检查；
2. 用 `projectPoints()` 前六列构造 `J`，由 `corner_noise` 得到 `Σθ`；
3. 构造 `G` 并得到 `R_pixel = GΣθG^T`；
4. 加入秩、条件数、有限性、对称性和半正定检查；
5. 用合成 Monte Carlo 验证后，才允许非歧义观测设置 `covariance_ok=true`；
6. 标定并接入 `R_calibration` 和 `R_pose_sync`；
7. EKF 实现观测函数、创新包角和 NIS 门限后，再开放正式更新；
8. 最后扩展逐角点异方差、位置单独更新及 IPPE 多假设处理。

在第 5 步完成前，当前“保留协方差接口但禁止进入 EKF”的策略是有意的安全
边界，而不是把单位阵当作已经可用的观测噪声。
