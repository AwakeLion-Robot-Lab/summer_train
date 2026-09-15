# 整车 ESEKF 与 UVL 图像观测：awakening 路线拆解与移植

本文逐函数拆解 `awakening-main` 的整车误差状态卡尔曼滤波（ESEKF）与图像平面灯条观测
（UVL），并给出移植到本仓库 L3 的对照表与陷阱清单。

对照的上游源码位于 `../awakening-main/`，无 git 历史，行号按当前工作副本标注，阅读时
以函数名为准。

覆盖范围：

| 文件 | 范围 | 内容 |
| --- | --- | --- |
| `src/utils/utils.hpp` | `:19-90`, `:343-400` | SO(3) 指数/对数映射、带畸变投影 |
| `src/tasks/auto_aim/armor_track/motion_model.hpp` | 全文 532 行 | 状态定义、流形运算、运动模型、UVL 观测 |
| `3rdparty/KalmanHyLib/error_state_extended_kalman_filter.hpp` | `:60-140`, `:330-430` | 滤波器本体 |
| `src/tasks/auto_aim/armor_track/armor_target.cpp` | `:83-170`, `:280-510` | 装配 |

阅读顺序是严格的依赖序：读到任何一节，它用到的东西前面都讲过了。

**与本仓库既有文档的关系。** `docs/iterated_ekf.md` 已经写过迭代 EKF 的 MAP 代价、
Bell & Cathey 迭代式和四个移植坑，本文第 6.2 节与之呼应——上游 awakening 的迭代式
恰好踩了其中一个。`docs/pnp_observation_noise_and_covariance.md` 讲的是"为什么
`reprojection_error` 不能直接当 R"，UVL 观测是对那个问题的一种廉价回答。


## 0. 读之前：两条约定

这两条如果搞反，后面所有符号都会读反。

### 0.1 装甲板系的 x 轴指向车心，不是朝外

`armor_pose()` 里第 i 块板的位置是 `-r·(cos yaw, sin yaw)`，而姿态绕 z 轴转了
`+yaw`，于是旋转矩阵第一列（板的 x 轴）约等于 `+(cos yaw, sin yaw)`。

两个方向恰好相反——板的 x 轴从板心指回车心。板的**可见面是它的 −x 侧**。这就是代码里
到处写 `front_normal = -axis_x` 的原因（`armor_target.cpp:196`、`:524`）。

### 0.2 角点顺序两边不同

```text
awakening : LEFT_TOP, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP   逆时针，先往下
newvision : 左上,     右上,        右下,          左下         顺时针
```

移植时必须重映射：

```text
左灯条 = newvision corners[0] (TL) + corners[3] (BL)
右灯条 = newvision corners[1] (TR) + corners[2] (BR)
```

### 0.3 三维物点约定两边一致（好消息）

两个项目的装甲板系都是 **x = 板面法向，y = 左，z = 上**，物点都是 `(0, ±W/2, ±H/2)`。
本仓库 `pnp_solver.cpp:43` 的 `armorPoints()` 已经是这个约定。相机光学系变换矩阵也
一样：awakening 的 `R_CV2PHYSICS` 与 CLAUDE.md 记的
`[[0,0,1],[-1,0,0],[0,-1,0]]` 是同一个矩阵。这两块不用动。


## 1. 李群工具

`../awakening-main/src/utils/utils.hpp:19-90`。三个函数，整套东西的地基。全部模板化在
`T` 上，因为要同时吃 `double` 和 `ceres::Jet`。

### 1.1 `so3_hat` — 反对称矩阵

hat 算子把 R³ 映到李代数 so(3)，作用是**把叉乘写成矩阵乘法**：

```text
        ⎡  0   -w3   w2 ⎤
w^  =   ⎢  w3   0   -w1 ⎥          w^ · v = w × v
        ⎣ -w2   w1   0  ⎦
```

两条要背下来的性质，后面推导反复用：

```text
(w^)ᵀ  = -w^                    反对称
(w^)³  = -‖w‖² · w^             三次方退回一次方
```

第二条是罗德里格斯公式只有两项的根本原因——幂级数被截断。

### 1.2 `so3_exp` — 指数映射（`utils.hpp:26`）

```cpp
const T theta2 = phi.squaredNorm();
const T theta  = ceres::sqrt(theta2);      // 不是 std::sqrt
if (theta2 < T(1e-12)) {                   // 小角度分支
    A = 1 - theta2/6 + theta4/120;
    B = 0.5 - theta2/24 + theta4/720;
} else {
    A = ceres::sin(theta) / theta;
    B = (1 - ceres::cos(theta)) / theta2;
}
R = I + A*W + B*W2;
```

**推导。** 令 φ = θa，θ = ‖φ‖ 是转角，a 是单位转轴。指数映射按定义是矩阵幂级数：

```text
exp(φ^) = I + φ^ + (1/2!)(φ^)² + (1/3!)(φ^)³ + ...
```

用 (a^)³ = -a^（单位向量），所有高次项塌回 a^ 和 (a^)² 两项，按奇偶次分组：

```text
exp(θa^) = I + ( θ - θ³/3! + θ⁵/5! - ... ) a^  +  ( θ²/2! - θ⁴/4! + ... ) (a^)²
             └──────── sin θ ────────┘            └───── 1 - cos θ ─────┘
```

即罗德里格斯公式 `R = I + sin θ · a^ + (1-cos θ)(a^)²`。代码存的是 φ 不是 (θ, a)，
代入 a^ = φ^/θ 后：

```text
R = I + (sin θ / θ) · φ^  +  ((1-cos θ) / θ²) · (φ^)²
        └─── A ───┘            └───── B ─────┘
```

**为什么必须有小角度分支。** θ→0 时 A 和 B 都是 0/0。极限存在（1 和 1/2），但浮点直接
算得到 NaN。所以用泰勒展开替代：

```text
A = sin θ / θ       = 1   - θ²/6  + θ⁴/120 - ...
B = (1-cos θ) / θ²  = 1/2 - θ²/24 + θ⁴/720 - ...
```

在误差状态滤波里 δ 永远接近 0，**这条分支才是常走的路径，不是边界情况**。

`ceres::` 而不是 `std::`：T 可能被实例化成 `ceres::Jet<double,13>`，`std::sqrt` 不认识
Jet。整份代码模板化的原因就在这。

### 1.3 `so3_log` — 对数映射（`utils.hpp:80`）

**第一步，由迹取转角。** 对罗德里格斯公式取迹，注意 tr(a^)=0、tr((a^)²)=-2：

```text
tr(R) = 3 + 0 - 2(1 - cos θ) = 1 + 2 cos θ
  ⟹   cos θ = (tr(R) - 1) / 2
```

**第二步，由反对称部分取转轴。** (a^)² 是对称的，所以 R 的反对称部分只来自中间那项：

```text
R - Rᵀ = 2 sin θ · a^
```

代码里的 `w` 正是从 `R - Rᵀ` vee 出来的向量，所以 w = 2 sin θ · a，于是：

```text
φ = θ·a = θ / (2 sin θ) · w        ← 代码里的 scale
```

θ→0 时 θ/(2 sin θ) → 1/2，对应早退分支 `return 0.5 * w`。

**θ→π 没有保护。** sin θ→0 时 scale 会爆，代码只保护了 θ→0 那一端。但 `so3_log` 只被
用在 `box_minus`（算两个接近姿态之差）和 `inject`（注入一个小 δ），两处输入都保证接近
单位阵，永远碰不到 π。

反过来说：**如果把旋转当普通向量存在状态里直接加减，就一定会撞上这个奇异点。**这正是
误差状态的价值。


## 2. 状态定义与整车几何

`../awakening-main/src/tasks/auto_aim/armor_track/motion_model.hpp:20-100`

### 2.1 `idx` 枚举与槽位复用（`:20`）

```cpp
enum { CX, VCX, CY, VCY, CZ, VCZ, C_ROT_Z, VYAW, LOG_R1, P1, P2, C_ROT_Y, C_ROT_X, X_N };
constexpr int LOG_R2      = P1;     // 同一个槽位
constexpr int H           = P2;
constexpr int OUTPOST01DZ = P1;     // 还是那个槽位
constexpr int OUTPOST02DZ = P2;
```

| 下标 | 名字 | 含义 | 单位 |
| --- | --- | --- | --- |
| 0,2,4 | `CX,CY,CZ` | 整车旋转中心在 odom 的位置 | m |
| 1,3,5 | `VCX,VCY,VCZ` | 整车中心速度 | m/s |
| 6 | `C_ROT_Z` | 整车姿态旋转向量的 z 分量 | rad |
| 7 | `VYAW` | 绕**车体** z 轴的角速度 | rad/s |
| 8 | `LOG_R1` | 第一组半径的**对数** | ln(m) |
| 9 | `P1` | 四板车 = `LOG_R2`；前哨 = `OUTPOST01DZ` | — |
| 10 | `P2` | 四板车 = `H`（奇偶板高差）；前哨 = `OUTPOST02DZ` | — |
| 11,12 | `C_ROT_Y, C_ROT_X` | 整车姿态旋转向量的 y、x 分量 | rad |

**陷阱。** `x[idx::LOG_R2]` 和 `x[idx::OUTPOST01DZ]` 是同一个 double。所有碰这两个槽位
的代码都必须先分支判 `armor_number`。

另外旋转三维**不连续**：z 在 6，y 和 x 在 11、12。这是历史演进痕迹（先只有 yaw，后来才
补 roll/pitch），不是设计。移植时建议改成连续的 `rot_x, rot_y, rot_z`。

### 2.2 `normalize_angle`（`:44`）

```cpp
return a - two_pi * ceres::floor((a + T(M_PI)) / two_pi);
```

无分支写法，比 `while (a > pi) a -= 2pi` 好在**对 Jet 友好**：`floor` 的导数是 0，Jet
穿过去后导数不变——加减 2π 不改变角度的物理意义，导数当然也不该变。

### 2.3 `car_rotation`（`:50`）

前哨站和基地**不估完整三自由度姿态**，只留 yaw：

```cpp
if (armor_number == OUTPOST || armor_number == BASE || !USE_WROT)
    return so3_exp(Vec3(0, 0, x[C_ROT_Z]));
return so3_exp(Vec3(x[C_ROT_X], x[C_ROT_Y], x[C_ROT_Z]));
```

原因很实际：前哨是固定装置绕竖直轴匀速转，基地根本不转，给它们估 roll/pitch 只会引入
不可观测自由度让滤波器漂。

`USE_WROT` 是文件顶部的 `constexpr bool`（当前 true），关掉它整体退化成"只估 yaw"，
也就是本仓库现在的模型。**这是个方便的对照开关，移植时可以照搬。**

### 2.4 `armor_radius`（`:60`）

```cpp
if (armor_number == BASE) return T(0.0);            // 基地：板就在中心
const bool is_r2 = (armor_num == 4) && (id & 1);    // 只有四板车才分两组
return ceres::exp(is_r2 ? x[LOG_R2] : x[LOG_R1]);
```

`is_r2` 的两个条件缺一不可：三板车（前哨）所有板共用 `LOG_R1`，因为它们到轴心的距离
物理上相同；四板车前后一组、左右一组，底盘不是正方形。

**为什么用 log 参数化。** 半径有硬约束 r > 0。三种处理：

- 不管 → 滤波器可能把 r 推成负数，几何整个翻转
- 每步 clamp → 破坏协方差与状态的一致性（状态被投影了，P 没有）
- 重参数化 ℓ = ln r → ℓ ∈ R 无约束，r = exp(ℓ) > 0 自动成立 ✓

代价是噪声要换算，由一阶传播：

```text
ℓ = ln r   ⟹   ∂ℓ/∂r = 1/r   ⟹   σ_ℓ² ≈ σ_r² / r²
```

这就是 `process_noise()` 里 `q(LOG_R1,LOG_R1) = q_r / (r1*r1)` 的来历（见 7.2）。
另一个副作用：log 参数化让半径的**相对**不确定性恒定。

### 2.5 `whole_car_pose`（`:69`）

六行包装，把状态的位置三维和姿态三维拼成一个 SE(3)。注意平移用的是 `CX/CY/CZ`，不含
速度——速度只在 `Predict` 里推进位置时用。

### 2.6 `armor_pose` — 核心（`:78`）

**整个整车估计思路的全部内容。**

```cpp
const T yaw = normalize_angle(T(id) * T(2.0 * M_PI / armor_num));   // ① 常量，不是状态
const T r   = armor_radius(x, id, armor_num, armor_number);         // ② 状态量
const T ax  = -ceres::cos(yaw) * r;                                 // ③ 注意负号
const T ay  = -ceres::sin(yaw) * r;
T az = outpost ? (id==1 ? x[OUTPOST01DZ] : id==2 ? x[OUTPOST02DZ] : 0)
               : (is_r2 ? x[H] : 0);
pose_in_car.translation() << ax, ay, az;

const T armor_pitch = outpost ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;   // ④
pose_in_car.linear() = rpy2matrix(Vec3(0, armor_pitch, yaw), ZYX);

return whole_car_pose(x, armor_number) * pose_in_car;               // ⑤
```

**数学。** 第 i 块板的方位角是常量，不是待估量：

```text
θ_i = 2πi / N,      N ∈ {3, 4}
```

板心在车体系的位置与姿态：

```text
p_i^car = [ -r_i cos θ_i,  -r_i sin θ_i,  Δz_i ]ᵀ,     r_i = exp(ℓ1 或 ℓ2)
R_i^car = Rz(θ_i) · Ry(α),                             α = ±15°
```

复合到世界系：

```text
                ⎡ Exp(φ)   c ⎤   ⎡ R_i^car  p_i^car ⎤
T_armor_i^odom= ⎢            ⎥ · ⎢                  ⎥
                ⎣   0      1 ⎦   ⎣   0         1    ⎦
                └ 整车，来自状态 ┘  └ 结构先验 + 两个状态 ┘
```

**"装甲板跳变"在这里消失了。** 无论当前看到的是正面板、侧面板还是相邻板的一条灯条，
它们都只是*同一个 13 维状态*在不同 i 上的确定性投影。只要能把观测关联到正确的 i，它们
就共同约束同一个状态向量——这是"整车估计"相对"每块板各自滤波"的全部收益。

**那个负号。** 位置是 `-r(cos θ, sin θ)`，而 `Rz(θ)Ry(α)` 的第一列约等于
`+(cos θ, sin θ)cos α`。两者反向，所以可见性判据要用 `-axis_x`：

```cpp
// armor_target.cpp:524
Vec3 front_normal = -pose_in_camera_cv.linear().col(0);     // 朝外那一面
score = front_normal.dot(-pose_in_camera_cv.translation()); // 越大越正对
```

**移植提示。** 本仓库已有 `armorPitchOf(name)`（常规车 +15°、前哨取负），语义与
`armor_pitch` 完全一致，直接用，不要再写一份。当前在 `pnp_solver.cpp:64`、
`aim_overlay.cpp:194`、`fire_decision.cpp:152` 三处被调用。


## 3. 流形运算 ⊞ / ⊟

`motion_model.hpp:104-140`。全篇最该逐字读的 30 行。

### 3.0 为什么需要它们

普通 EKF 的三个动作都假设状态住在向量空间：

```text
x += K·r ,      P = F P Fᵀ + Q ,      P = E[(x-x̄)(x-x̄)ᵀ]
```

旋转不住在向量空间，SO(3) 是流形。把旋转向量当普通向量加会出四种问题：

1. `φ1 + φ2 ≠ Log(Exp(φ1)·Exp(φ2))`，差一个 BCH 修正
2. ‖φ‖→π 处 Log 的雅可比发散，2π 处突变
3. 同一个物理姿态不确定性，在 φ≈0 和 φ≈π 附近算出的 3×3 协方差完全不同
4. 直接对旋转向量分量求 ∂f/∂x，差一个右雅可比 Jr(φ)

**解法：把状态劈成两半。**

| | 住在哪 | 大小 | 代码里 |
| --- | --- | --- | --- |
| 名义状态 x̌ | 流形上 | 任意大 | `x_nominal` |
| 误差状态 δ | x̌ 处的切空间 | **恒小，更新后归零** | `delta_x` |

关键：**滤波器的 P 是 δ 的协方差，不是 x 的。**切空间是货真价实的向量空间，所有线性
代数都合法；而 δ 恒在 0 附近，永远碰不到 π 的奇异点。

### 3.1 `state_rotation`（`:104`）

和 `car_rotation` 的区别：后者对前哨/基地会退化成只用 `C_ROT_Z`，`state_rotation`
**永远用完整三维**。这是对的——⊞/⊟ 是纯粹的流形运算，不该知道目标类型；类型退化由
`Predict` 那边保证 `C_ROT_X/Y` 始终为 0。

### 3.2 `inject_state` ⊞（`:111`）

```cpp
for (int i = 0; i < X_N; ++i)
    if (!是旋转分量) nominal[i] += delta[i];         // 欧氏分量：直接加

const Vec3T delta_rot(delta[C_ROT_X], delta[C_ROT_Y], delta[C_ROT_Z]);
const Vec3T injected = so3_log( state_rotation(nominal) * so3_exp(delta_rot) );  // R·Exp(δ) 右乘
nominal[C_ROT_X..Z] = injected;
```

形式定义：

```text
             ⎧ x_i + δ_i              i ∉ {rot}
x̌ ⊞ δ  =    ⎨
             ⎩ R · Exp(δ_rot)         i ∈ {rot}
```

**右乘 vs 左乘。** 两种扰动方式，通过伴随矩阵相联：

```text
R · Exp(δ)        右乘 · 局部/体坐标系   "这辆车绕自己的轴又转了一点"
Exp(δ) · R        左乘 · 全局/世界系     "整个世界绕世界轴转了一点"

Exp(δ_world)·R = R·Exp(Rᵀ δ_world)   ⟹   δ_body = Rᵀ δ_world
```

选右乘不是审美，是被下游两处逼的：

1. **过程噪声天然是体坐标系的。** 配置 `qxyz_common: [30, 30, 1]`——地面轮式车沿行进
   方向的加速度噪声远大于竖直方向。Q 只有在体系里表达才有物理意义，再旋到世界系。
2. **`vyaw` 是绕车体自身 z 轴的角速度。** Q 里 `rot_z`-`vyaw` 那个 2×2 常角加速度耦合
   块，只有在 yaw 误差也是体系时才自洽。

代码注释说得很直白（`armor_target.cpp:312`）：
*"这里将误差设计为相对自身的旋转误差，所以 q 就是相对自身的运动"*。

**实现细节。** 那个 for 循环**跳过**旋转三维，循环外再单独处理。这样写而不是"先全加再
覆盖"，是因为旋转分量参与加法会污染中间结果——`state_rotation(nominal)` 在循环后被
调用，读的必须是未被误加的旋转。下一个函数反而用了"先全减再覆盖"。**两个函数的写法
不对称，不是笔误。**

### 3.3 `box_minus_state` ⊟（`:131`）

```cpp
delta = value - nominal;                                            // 先整体相减
const Vec3T delta_rot = so3_log(
    state_rotation(nominal).transpose() * state_rotation(value) );  // Log(Řᵀ·R)
delta[C_ROT_X..Z] = delta_rot;                                      // 再覆盖掉
```

**互逆性证明（第一个该写的单测）。** 要求的性质：

```text
(x̌ ⊞ δ) ⊟ x̌ = δ        且        x̌ ⊞ (x ⊟ x̌) = x
```

验证旋转部分（欧氏部分显然）：

```text
Log( Řᵀ · (Ř · Exp(δ)) ) = Log( (ŘᵀŘ) · Exp(δ) ) = Log(Exp(δ)) = δ   ✓
```

**转置的位置必须和右乘配套。**如果 ⊞ 用右乘而 ⊟ 写成 `Log(R·Řᵀ)`（左乘形式），互逆性
就破了，而且**不会报错**——滤波器照样跑，只是雅可比全错、协方差没有意义、遇到大机动
就发散。

所以第一个单测就写这个：随机生成 x̌ 和小 δ，断言
`‖(x̌ ⊞ δ) ⊟ x̌ − δ‖ < 1e-10`。它能抓住 90% 的符号/左右乘错误。


## 4. 运动模型

`motion_model.hpp:176-258`。很朴素：匀速平移 + 绕车体 z 轴匀角速度，其余随机游走。
复杂度全在观测端。

### 4.1 `Predict::operator()`（`:183`）

```cpp
std::copy(x0, x0 + X_N, x1);              // 默认全部不变 —— 半径、高度都是随机游走
x1[CX] += x0[VCX] * dt;                   // 匀速平移
x1[CY] += x0[VCY] * dt;
x1[CZ] += x0[VCZ] * dt;

if (armor_number != BASE) {
    delta_rot << 0, 0, x0[VYAW] * dt;     // 只绕车体 z 轴
    R1 = so3_exp(Vec3(x0[ROT_X], x0[ROT_Y], x0[ROT_Z])) * so3_exp(delta_rot);  // 右乘
    x1[C_ROT_X..Z] = so3_log(R1);
}
clamp(x1);
```

状态转移：

```text
c⁺ = c + v·Δt ,        v⁺ = v
R⁺ = R · Exp([0, 0, ω·Δt]ᵀ) ,   ω⁺ = ω
ℓ1⁺ = ℓ1 ,  ℓ2⁺ = ℓ2 ,  h⁺ = h        随机游走：均值不变，方差靠 Q 增长
```

**为什么 `delta_rot` 只有 z 分量？**因为模型假设车只绕*自己的*竖直轴转。而"自己的竖直
轴"这个概念之所以能成立，正是因为用了**右乘**——右乘的扰动天然表达在体坐标系里。如果
用左乘，`[0,0,ωΔt]` 就变成"绕*世界* z 轴转"，车一旦有 roll/pitch 就错了。

这是 3.2 里"右乘是被下游逼出来的"的第二个具体兑现点。

**Voter**（`:142`）只服务前哨站：开机 1 秒内不投票，之后每次 yaw 变化超过 0.05 rad 就
给计数器 ±1，累计超过 10 票判定转向。方向一旦判明，角速度就钉死成规则常量
`OUTPOST_WZ = 2.51`，不再估计——把一个自由度换成已知量。第一版移植可整个跳过。

### 4.2 `Predict::clamp`（`:214`）

```cpp
x[LOG_R1] = fmax(log(0.05), fmin(log(1.0), x[LOG_R1]));   // 半径：饱和
if (abs(h) > 0.5)     h = 0.0;                            // 高度：归零
if (abs(vyaw) > 20.0) vyaw = 0.0;                         // 角速度：归零
if (OUTPOST) x[LOG_R1] = log(OUTPOST_R);                  // 前哨半径钉死 0.275
```

**半径用 clamp（饱和），高度和角速度用归零（复位）。**区别在于：半径超界通常是估计偏
了一点，拉回边界合理；而 h 或 vyaw 超界几乎一定是关联错误导致的发散，归零等于承认
"这一维已经没救了，重来"。

它在 f 内部，所以 Jet 会穿过这些分支。`fmin`/`fmax` 在边界上导数分段（不光滑），理论
上会让 F 在饱和瞬间不连续。实践中很少触发，但这是个值得知道的近似。


## 5. UVL 观测模型

`motion_model.hpp:260-341`、`utils.hpp:343-400`。整条链路的分水岭：观测不是 PnP 解出的
位姿，而是**图像平面上一条灯条的四个几何量**。

### 5.0 为什么不喂 PnP 位姿

传统做法：每块板 PnP 得到 `z = (x, y, z, ψ)` 当观测。两个问题。

**① PnP 误差高度各向异性且相关。**共面四点、远距小目标、斜视角下，协方差的真实形状是：

```text
        ⎡ σx²   ·    ·        ·       ⎤
        ⎢  ·   σy²   ·        ·       ⎥          σd  ~ 10-50 × σx
R_pnp = ⎢  ·    ·   σd²    ρ σd σψ    ⎥          |ρ| → 1
        ⎣  ·    ·  ρ σd σψ    σψ²     ⎦
```

深度和 yaw 强相关，量级差一两个数量级。在滤波器里给它一个*对角* R，就是在撒谎。本仓库
`docs/pnp_observation_noise_and_covariance.md` 讲的正是这件事，而且目前实现里确实还没做
协方差传播，`TargetConfig` 用的是手调常数。

**② PnP 是非线性优化的输出。**它已经替你"决定"了姿态。再去滤它，是在滤一个被处理过的
量——信息在 PnP 那一步就损失且被扭曲了。

**UVL 的做法是把观测退回到更接近传感器原始输出的地方。**像素误差近似各向同性，对角 R
是个诚实得多的近似；而且一块板拆成两条灯条 = 8 个约束，比 PnP 吐出的 4 维多一倍。

### 5.1 `UVCtx`（`:260`）

一个 UVL 观测的上下文："这是哪块板的哪条灯条，以及那一帧相机在哪"。

```cpp
struct UVCtx {
    int armor_num;                   // 这辆车几块板
    int id;                          // 这个观测对应第几块板
    ISO3 camera_cv_in_odom;          // 该帧相机位姿（TF 时间插值得到）
    CameraInfo camera_info;          // 内参 + 畸变
    ArmorClass armor_number;
    bool is_left;                    // 左灯条 or 右灯条
    bool normalized;
};
```

`camera_cv_in_odom` 是**按图像曝光时刻从 TF 里 slerp 出来的**，不是"当前"相机位姿。
这是图像与 IMU 时间对齐实际发生的地方，对应本仓库的 `SerialWorker::gimbalPoseAt()`。

`normalized` 切换观测在像素坐标还是归一化坐标下表达；当前 `MEASURE_NORMALIZED = false`
（`armor_target.hpp:69`），走像素路径。

### 5.2 `project_points_jets`（`utils.hpp:343`）

手写的针孔 + Brown-Conrady 畸变投影。**不能用 `cv::projectPoints`，因为要吃 Jet。**

```text
① 刚体变换      Pc = R · Pw + t
② 透视除法      xp = Xc/Zc ,  yp = Yc/Zc
③ 畸变          r² = xp² + yp²
                radial = 1 + k1 r² + k2 r⁴ + k3 r⁶
                xd = xp·radial + 2 p1 xp yp + p2 (r² + 2 xp²)
                yd = yp·radial + p1 (r² + 2 yp²) + 2 p2 xp yp
④ 内参          u = fx·xd + cx ,   v = fy·yd + cy
```

整条链 Pw → (u,v) 是**纯代数、处处可微**（除了 Zc = 0），所以 Jet 能一路穿过去，中心
差分也能稳定工作。**这是 UVL 观测模型可行的技术前提。**

**陷阱：`Zc ≤ 0` 没有保护。**板转到相机背面时 Xc/Zc 会翻符号甚至除零，投影出荒谬的
像素坐标，直接毒化 H 和残差。上游靠可见性筛选挡住（`match_armor` 按法向排序只取前 3
块），但那是启发式，不是保证。**移植时必须补 `if (Zc < eps) return failure`，并让观测
被丢弃而不是被使用。**

### 5.3 `UVLMeasure::project_points`（`:276`）

把前面所有层串起来：

```cpp
auto pose_in_odom      = armor_pose(x, ctx.id, ctx.armor_num, ctx.armor_number);  // §2.6
auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;
auto object_points     = getArmorLightKeyPoints3D(ctx.armor_number, ctx.is_left);
//   左灯条 = {(0, +W/2, +H/2), (0, +W/2, -H/2)}
//   右灯条 = {(0, -W/2, +H/2), (0, -W/2, -H/2)}
project_points_jets(object_points, pose_in_camera_cv, K, D, pts_jet);              // §5.2
return { pts_jet[0], pts_jet[1] };   // (上端点, 下端点)
```

完整的观测函数：

```text
h_{i,s}(x) = π( K, D, (T_cam^odom)⁻¹ · T_armor_i^odom(x) · P_s )
```

其中 i 是板编号，s ∈ {左, 右}，P_s 是该灯条两端点在板系的**常量**三维坐标。

**注意 x 只通过 `T_armor_i^odom` 进入。**其余全是常量（内参、外参、板尺寸、板编号）。
这就是为什么滤波器能用它反解状态——观测和状态之间只有一条通路。

**移植提示。** 本仓库 `pnp_solver.cpp:43` 的 `armorPoints()` 已经用了同一个板系约定，
只要写一个 `armorLightPoints(width, height, bool left)` 返回
`{(0, ±W/2, +H/2), (0, ±W/2, −H/2)}` 就够了，三维几何一行不用改。

### 5.4 `points_to_observation`（`:307`）

**整个 UVL 设计里最值得琢磨的函数。**

```cpp
const ImagePoint<T> delta  = top - bottom;
const ImagePoint<T> center = (top + bottom) / T(2);
z[UV_ANGLE]    = ceres::atan2(delta.x(), delta.y());   // x 在前！
z[UV_CENTER_X] = center.x();
z[UV_CENTER_Y] = center.y();
z[UV_LENGTH]   = ceres::sqrt(delta.squaredNorm());
```

同一个函数被用在两处：**预测值**（喂投影出来的点）和**观测值**（喂检测出来的点，见
`armor_target.cpp:32` 的 `get_uvl_measurement`）。**预测和观测必须走同一段代码**，否则
两边定义一旦漂移就是灾难性的隐蔽 bug。

**为什么是这四个量，而不是两个端点的四个坐标。**维度一样是 4，信息量理论上也一样（是个
可逆变换）。差别在**误差结构**。

设灯条上下端点的检测误差为 ε_top、ε_bot。真实的角点误差是**各向异性**的：沿灯条方向
（纵向）的定位误差远大于垂直方向（横向），因为灯条两端是渐变的亮度衰减，而两侧是陡峭
的边缘。记 σ∥ ≫ σ⊥。

如果用原始端点坐标 `z = (ut, vt, ub, vb)`：σ∥ 和 σ⊥ 会按灯条倾角混进*每一个*分量，R
必须是稠密的、且随倾角变化。用对角 R 就错了。

换成 (α, uc, vc, L) 之后，误差被自然分离：

| 分量 | 定义 | 主要吸收 | 反映 |
| --- | --- | --- | --- |
| `L` | ‖Δ‖ | σ∥（两端各一次） | 距离 / 尺度 |
| `uc, vc` | (pt + pb)/2 | σ∥ 被**平均掉一半** | 位置 |
| `α` | atan(Δx / Δy) | σ⊥ / L | 姿态 |

中心点尤其关键：纵向误差在两端反号时相消、同号时保留，取中点相当于做了一次平均，σ 降到
σ∥/√2。而角度只受横向误差影响，且被灯条长度归一化：

```text
σ_α ≈ √2 · σ⊥ / L
```

于是三类量可以**各给各的 sigma**，对角 R 才成立。这正是配置里三个独立参数
`r_sigma_px_by_length_ratio` / `r_sigma_length_by_length_ratio` / `r_sigma_angle`
的由来。

**陷阱：`atan2(Δx, Δy)` —— x 在前，不是笔误。**标准写法 `atan2(y, x)` 量的是"偏离水平
方向"的角。这里参数交换，量的是**偏离竖直方向**的角。原因很实际：装甲板灯条基本竖直，
|Δy| ≫ |Δx|，用标准写法 α 会在 ±π/2 附近工作，残差归一化和线性化都别扭；交换之后
α ≈ 0，干净得多。

### 5.5 `UVLMeasure::residual`（`:325`）

```cpp
Eigen::Matrix<T, UVLZ_N, 1> v = z - z_pred;
v[idx::UV_ANGLE] = normalize_angle(v[idx::UV_ANGLE]);   // 只有这一维要缠绕归一化
return v;
```

其余三维（中心 x、中心 y、长度）都是普通欧氏量，直接相减。只有角度活在 S¹ 上，
179° − (−179°) 必须是 2° 而不是 358°。

**记住这个函数。**`update_multi` 用中心差分求 H 时，差的不是 `z_pred` 而是 `residual`
——就是为了让缠绕归一化也进到导数里。见 6.2。


## 6. 滤波器

`3rdparty/KalmanHyLib/error_state_extended_kalman_filter.hpp`。只需要读两个函数。文件里
的 `update`（单观测版）和 `ObsImpl::evaluate` 在这条链路上**根本不会被调用**。

### 6.1 `predict`（`:69`）— F 用 ceres::Jet 自动微分

```cpp
f(x_prev.data(), x_pred.data());          // ① 名义状态推进
x_nominal = x_pred;

for (int i = 0; i < N_X; ++i) {           // ② 播种单位阵
    delta_jet[i] = Jet(0.0);
    delta_jet[i].v[i] = 1.0;
}
inject_state_jet(delta_jet, x_pert_jet);                              // ③ ⊞
f(x_pert_jet.data(), x_pert_pred_jet.data());                         // ④ f
box_minus_state_jet(x_nominal_jet, x_pert_pred_jet, delta_pred_jet);  // ⑤ ⊟
for (int i = 0; i < N_X; ++i)
    F.row(i) = delta_pred_jet[i].v.transpose();                       // ⑥ 读出导数

P_delta = F * P_delta * F.transpose() + update_Q();
P_delta = 0.5 * (P_delta + P_delta.transpose());     // 强制对称，抗数值漂移
```

**误差状态转移矩阵。**普通 EKF 里 F = ∂f/∂x。ESEKF 要的是**误差怎么传播**：F = ∂δ⁺/∂δ。

但 δ 和 δ⁺ 住在**不同点的切空间**（分别在 x̌ 和 f(x̌) 处），不能直接对状态求导。必须
绕道名义状态：

```text
δ  ──⊞──▶  x̌ ⊞ δ  ──f──▶  f(x̌ ⊞ δ)  ──⊟──▶  δ⁺
```

于是：

```text
        ∂
F  =  ──── [ f(x̌ ⊞ δ)  ⊟  f(x̌) ]        取 δ = 0
        ∂δ
```

**代码就是这个式子的逐字实现**，步骤 ②-⑥ 一一对应。链式法则由 Jet 自动完成——它把
∂⊞/∂δ、∂f/∂x、∂⊟/∂x 三段乘起来了。

**如果偷懒直接对旋转向量分量求导会怎样？**得到的是 ∂φ⁺/∂φ，它与真正的 F 相差右雅可比：

```text
Jr(φ) = I − ((1 − cos θ)/θ²) φ^ + ((θ − sin θ)/θ³) (φ^)²
```

φ≈0 时 Jr≈I，差别看不出来；‖φ‖ 一大就完全错，而且在 π 附近发散。**ESEKF 的写法让你
永远不必显式算 Jr**——它被吸收进 ⊞/⊟ 的自动微分里了。

`delta_x = F * delta_x` 那行：正常流程里 `delta_x` 在每次 update 结束时被清零，所以乘
的是零向量。它存在是为了支持"预测多次再更新一次"。`ArmorTarget` 的用法是严格交替，
所以这行实际是 no-op。

### 6.2 `update_multi`（`:333`）— 三个反直觉的地方

```cpp
MatrixX1 delta_iter = delta_x;      // = 0
MatrixXX P_iter     = P_delta;      // ② 迭代中不更新

for (int iter = 0; iter < iteration_num; ++iter) {          // esekf_iter_num: 5
    MatrixX1 x_eval = x_nominal;
    inject_state(delta_iter, x_eval);                       // 在当前迭代点线性化

    for (auto& obs : obs_list) {
        // ① H 用中心差分，不是 Jet
        constexpr double eps = 1e-6;
        for (int i = 0; i < N_X; ++i) {
            delta_plus[i] += eps;  delta_minus[i] -= eps;   // 扰动加在 δ 上，不是 x 上
            inject_state(delta_plus,  x_plus);
            inject_state(delta_minus, x_minus);
            obs->predict(x_plus,  z_plus);
            obs->predict(x_minus, z_minus);
            // ③ 差的是 residual，不是 z_pred
            obs->residual_and_R(z_minus, dz, _);
            obs->residual_and_R(z_plus,  r_plus, _);
            Hk.col(i) = -(r_plus - dz) / (2.0 * eps);
        }
        H.block(...) = Hk;  R.block(...) = Rk;              // R 块对角
    }
    S = H * P_iter * H.transpose() + R;
    K = S.ldlt().solve((P_iter * H.transpose()).transpose()).transpose();
    delta_iter.noalias() += K * residual;                   // ← 见下方"偏离教科书"
}

inject_state(delta_iter, x_nominal);       // 一次性注入
delta_x.setZero();                         // 误差状态复位
P_delta = (I - K*H) * P_iter * (I - K*H).transpose() + K*R*K.transpose();   // Joseph
P_delta = 0.5 * (P_delta + P_delta.transpose());
```

**多观测拼接与信息累加。**一帧里所有观测垂直拼接：

```text
    ⎡ z1 ⎤        ⎡ H1 ⎤
z = ⎢ ⋮  ⎥ ,  H = ⎢ ⋮  ⎥ ,  R = blkdiag(R1, ..., Rm)
    ⎣ zm ⎦        ⎣ Hm ⎦
```

一块完整板 = 2 条灯条 × 4 维 = **8 行**；每条孤立灯条再加 4 行。从信息形式看收益更清楚：

```text
P₊⁻¹ = P₋⁻¹ + Hᵀ R⁻¹ H = P₋⁻¹ + Σ_k Hₖᵀ Rₖ⁻¹ Hₖ
```

R 块对角 ⟹ 信息**严格相加**。每多一个观测，后验不确定性单调下降。所以同时用装甲板和
孤立灯条不是"多堆几个特征点"，而是在滤波框架内给同一个整车状态**增加独立约束**。

**① 中心差分求 H。**要求的是 H = ∂z/∂δ，不是 ∂z/∂x，所以扰动必须加在 δ 上再 ⊞ 进去：

```text
H[:,i] ≈ [ h(x̌ ⊞ (δ + ε eᵢ)) − h(x̌ ⊞ (δ − ε eᵢ)) ] / (2ε)
```

这一点错了（比如把 ε 直接加在 x 的旋转分量上），H 和 P 就不在同一个坐标系里，增益全废。

**③ 为什么差 residual 而不是 z_pred。**因为 r = z − ẑ，z 是常量，所以
∂r/∂δ = −∂ẑ/∂δ = −H，这解释了代码里的负号。而**真正的动机是角度缠绕**：如果直接差 ẑ，
当 ẑα⁺ 和 ẑα⁻ 分别落在 ±π 两侧时，差出来是 ≈2π 而不是 ≈0，H 的那一列会得到一个巨大的
假梯度。走 `residual` 就吃到了 `normalize_angle`（5.5）。

代价：每个观测 2 × 13 = 26 次完整投影。

**为什么这里不用 Jet（而 predict 用了）。**因为 `ObsBase` 是**类型擦除**的虚接口：

```cpp
virtual void predict(const Eigen::VectorXd& x, Eigen::VectorXd& z_pred) const = 0;
```

`VectorXd` 只认 `double`，Jet 流不过去。`ObsImpl::evaluate()`（`:277`）里其实*写了*完整
的 Jet 路径，但 `update_multi` 只在 `inject_state` 未设置时才调它——而 `ArmorTarget`
一定会设置。**所以那段 Jet 代码是死代码。**

**移植时这是个明确的改进点：**不做类型擦除（用 `std::variant` 或模板 visitor 组织观测
列表），Jet 就能一路走通，每个观测省掉 26 次投影，而且导数是精确的而非 O(ε²) 近似。

**迭代更新，以及此处偏离教科书之处。**标准 EKF 在*先验点*线性化 h。先验偏得远时线性化
点很糟。迭代版每轮在*当前估计*处重新线性化，本质是对 MAP 目标做高斯牛顿：

```text
min_δ  ‖δ‖²_{P⁻¹}  +  ‖z − h(x̌ ⊞ δ)‖²_{R⁻¹}
       └ 先验 ┘        └──── 观测 ────┘
```

对这条链路收益很大，因为 h 强非线性：透视除法 + 畸变多项式 + atan2 + sqrt。

**标准 IEKF 的迭代式**（Bell & Cathey，见本仓库 `docs/iterated_ekf.md`）：

```text
δ_{k+1} = K_k [ r_k + H_k δ_k ]
```

**awakening 的写法：**

```text
δ_{k+1} = δ_k + K_k r_k
```

两者相差 (I − K_k H_k) δ_k，而 KH ≠ I，**不等价**。前者每轮都把 δ 重新锚回先验，后者是
不断累加高斯牛顿步，理论上会偏离 MAP 解（先验项被反复计入）。

> **本仓库已经有正确答案。**`docs/iterated_ekf.md` 里写明了 Bell & Cathey 形式
> `x_{i+1} = x_pri ⊞ K_i [ z − h(x_i) − H_i (x_pri ⊟ x_i) ]`，并且专门指出末项符号是
> **减**、某参考实现（`SHtech_auto_aim-ax650-dev-2026/mathutils/IESEKF.hpp:226`）写成了
> 加号、第 0 次迭代看不出差别但第二次起会收敛到有偏不动点。移植时**按那份文档实现，
> 不要照抄 awakening 这一版**，然后两版对拍。
>
> 注意该文档描述的 `include/l3_estimation/ieskf.hpp` 和 `tests/ieskf_smoke.cpp` 目前
> 在树里**不存在**，和 `docs/` 里其他几处一样是被 revert 掉的实现。文档本身的结论仍然
>有效。

另外 `P_iter` 在迭代中**不更新**，最后用最后一轮的 K、H 走 Joseph form。

**Joseph form 为什么值得。**标准形式 `P₊ = (I − KH) P₋` 只在 K 恰为最优增益时成立，
浮点误差会让 P 失去对称性甚至正定性。Joseph form 对**任意** K 都成立：

```text
P₊ = (I − KH) P₋ (I − KH)ᵀ + K R Kᵀ
```

右边是两个正定二次型之和，**结构上保证半正定**。在迭代 EKF 里 K 本来就不是最优增益
（它是最后一轮的），所以这里用 Joseph 不是可选优化而是必需。后面那行强制对称
`0.5*(P + Pᵀ)` 是再上一道保险。


## 7. 装配

`../awakening-main/src/tasks/auto_aim/armor_track/armor_target.cpp`

### 7.1 `reset`（`:83`）— 从一块板反推整车

```cpp
// ① 初始协方差：完全靠先验知识给量级
p0[CX] = p0[CY] = p0[CZ] = 1;
p0[VCX] = p0[VCY] = p0[VCZ] = 10;
p0[C_ROT_X..Z] = 1;
p0[LOG_R1] = p0[LOG_R2] = p0[H] = 1;
p0[VYAW] = 100;                            // 角速度最不确定

// ② 半径先验：按目标类型给经验值
r_pre = OUTPOST ? 0.2765 : BASE ? 0.3205 : 0.26;

// ③ 关键：从板位姿反推车心位姿
armor_in_car.translation() << -r, 0, 0;    // 假设这块板是 0 号（yaw = 0）
armor_in_car.linear() = rpy2matrix(Vec3(0, armor_pitch, 0));
auto car_in_odom = armor_in_odom * armor_in_car.inverse();

// ④ 填状态；速度、vyaw、h 全部初始化为 0
target_state.set_pos(car_in_odom.translation());
target_state.x[LOG_R1] = target_state.x[LOG_R2] = std::log(r);
target_state.x[C_ROT_X..Z] = so3_log(car_in_odom.linear());
```

2.6 给的是正向 `T_armor^odom = T_car^odom · T_armor^car`。初始化时已知
`T_armor^odom`（PnP 给的），想求 `T_car^odom`，于是右乘逆：

```text
T_car^odom = T_armor^odom · (T_armor^car)⁻¹
```

但 `T_armor^car` 依赖两个未知量：板编号 i 和半径 r。代码的处理是**都用假设值**：

- 假设看到的是 0 号板（θ0 = 0），于是 `p_armor^car = (−r, 0, 0)ᵀ`，`R_armor^car = Ry(α)`
- 半径用经验先验 r_pre（常规车 0.26 m）

两个假设都会错，但都**不致命**：

```text
编号假设错  ⟹  整车 yaw 偏 2πk/N
半径假设错  ⟹  车心沿视线偏 Δr
```

因为编号只是**标签的循环平移**——把实际的 2 号板叫作 0 号，整个几何仍然自洽，只是整车
yaw 差 2πk/N。而 r 的偏差在后续观测中被修正（p0[LOG_R1] = 1 给了足够不确定性）。

**真正不可观测的是 `vyaw`**：单帧完全看不出车在不在转，所以 p0 = 100，等于告诉滤波器
"这一维我基本不知道，请大胆用观测改它"。

`jumped = false` 表示"还没关联到过 0 号以外的板"。在这个状态下整车 yaw、`LOG_R2`、`H`
几乎完全不可观测。所以 `match_armor` 用两套门限：`armor_match_gate_not_all_init`
（宽松，让第二块板更容易进来）和 `armor_match_gate`（收敛后收紧）。本仓库
`TrackedTarget::jumped` 语义完全一致，注释里也写明了是**粘滞**的。

### 7.2 `process_noise`（`:280`）— 右乘在这里兑现

```cpp
// ① 平移：体系加速度噪声 → 旋到 odom
const Mat3 car_in_odom_R = whole_car_pose(...).linear();
const Mat3 Q_acc_odom = car_in_odom_R * q_xyz_body.asDiagonal() * car_in_odom_R.transpose();
q(pos,pos) = 0.25*dt4*Q_acc_odom;   q(pos,vel) = 0.5*dt3*Q_acc_odom;
q(vel,pos) = 0.5 *dt3*Q_acc_odom;   q(vel,vel) =      dt2*Q_acc_odom;

// ② yaw：常角加速度，误差本来就在体系 → 不用旋转
q(VYAW,VYAW)     += dt2      * q_yaw;
q(C_ROT_Z,VYAW)  += 0.5*dt3  * q_yaw;
q(C_ROT_Z,C_ROT_Z) += 0.25*dt4 * q_yaw;

// ③ roll/pitch：随机游走，吸收地面坡度、外参残差
q(rot,rot) += dt * diag(q_wpr, q_wpr, 0);

// ④ 半径 / 高度：随机游走（log 状态要换算）
q(LOG_R1,LOG_R1) = cfg.q_r / (r1 * r1);
q(LOG_R2,LOG_R2) = cfg.q_r / (r2 * r2);
q(H,H) = cfg.q_h;
```

**常加速度模型的 Q 从哪来。**把未建模的加速度当白噪声 a ~ N(0, σa²)。在 Δt 内它对位置和
速度的影响：

```text
Δp = ½ a Δt²          ⎡ ½Δt² ⎤
Δv = a Δt      ⟹  G = ⎣  Δt  ⎦

                  ⎡ ¼Δt⁴   ½Δt³ ⎤
Q_pv = G σa² Gᵀ = ⎢             ⎥ σa²
                  ⎣ ½Δt³    Δt² ⎦
```

这就是代码里那四个系数的来历，平移和 yaw 用的是同一套。

**为什么 Q 必须先在体系建再旋转。**配置 `qxyz_common: [30.0, 30.0, 1.0]` 的物理含义是
*"地面轮式车可以突然前后左右加速，但不会突然上下加速"*。这句话只有在**车体坐标系**里
才成立——车头朝东和朝北时，世界系下的噪声椭球方向完全不同。所以要旋转：

```text
Q_a^odom = R_car^odom · Q_a^body · (R_car^odom)ᵀ
```

这是协方差在坐标变换下的标准传播律 Cov(Ra) = R Cov(a) Rᵀ。

**而 yaw 那一块不需要旋转**，因为选了右乘之后误差状态本来就定义在体系里。如果当初选了
左乘，这里就得反过来把 yaw 噪声旋进世界系，而且 `rot_z`-`vyaw` 的耦合块会变成三维稠密
的，因为"绕车体 z 轴"在世界系里是三个分量的组合。**右乘的选择让这段代码保持简单。**

**log 半径的噪声换算** `q_ℓℓ = q_r / r²`，由 2.4 的一阶传播 σ_ℓ ≈ σ_r / r 得到。直觉：
配置里的 `q_r` 描述的是*物理半径*每秒能漂多少（米²），而状态存的是 ln r，所以要除以 r²
换算到对数尺度。副作用是大半径目标的对数噪声更小——基地（r≈0.32）比步兵（r≈0.26）的
半径更"稳"，符合直觉。

> **对本仓库的启示。**CLAUDE.md 记着一个已知问题：`track_diag` 的 NIS 停在 0.12、
> 4 自由度期望是 4、一个 sweep 显示 **Q 主导这个差距**，而且降 Q 100 倍会让
> `records/3m_high` 的方位角创新降 41% 但让 `records/3m_run_fast` 差 3.6 倍。
>
> **体系 Q + 旋转到世界系，很可能正是这个矛盾的出路。**目前本仓库的 Q 是直接在世界系
> 给的对角阵，为了照顾机动目标只能整体抬高；体系 Q 允许"沿行进方向给大、竖直方向给
> 小"，在不牺牲机动响应的前提下降低总的协方差膨胀。**这是移植时值得优先单独验证的一
> 项，甚至可以脱离 ESEKF 先在现有 EKF 上试。**

### 7.3 `predict_ekf`（`:333`）

```cpp
auto dt = duration<double>(timestamp - target_state.timestamp).count();
esekf->set_predict_func(Predict{ .dt = dt, .armor_number = target_number, .voter = voter });
esekf->set_update_Q([&]() { return process_noise(dt); });
target_state.x = esekf->predict();
target_state.timestamp = timestamp;
this_id = GLOBAL_ID++;
```

- **`dt` 是实际帧间隔，不是标称值。**由曝光时间戳相减得到，掉帧、检测耗时抖动都被自然
  吸收。这是 `Predict.dt` 每帧都要重设的原因。
- **`Q` 是 lambda 而非值。**因为 Q 依赖当前姿态（要旋转）和当前半径（log 换算），必须
  在 predict 内部、用推进前的状态求值。
- **`this_id = GLOBAL_ID++`** 是给下游的"状态版本号"。`very_aimer` 靠它判断目标状态有
  没有变（`InputCtx::is_same`），没变就复用上一次构建的轨迹，只推进时间。

### 7.4 `update`（`:345`）

```cpp
// ── 单条灯条 → 一个 4 维观测 ──────────────────────
auto add_uvl_obs = [&](const cv::Point2f& top, const cv::Point2f& bottom,
                       int id, bool is_left) {
    const auto observation = get_uvl_measurement(top, bottom, camera_info);  // §5.4 同一函数
    const auto u_r = [&](const auto& z) {
        auto length = cv::norm(top - bottom);            // 用灯条长度描述误差强度
        const double sigma_px     = cfg.r_sigma_px_by_length_ratio     * length;
        const double sigma_length = cfg.r_sigma_length_by_length_ratio * length;
        const double sigma_angle  = cfg.r_sigma_angle;   // 常数
        r(UV_ANGLE,    UV_ANGLE)    = sigma_angle  * sigma_angle  / 2.0;
        r(UV_CENTER_X, UV_CENTER_X) = sigma_px     * sigma_px     / 2.0;
        r(UV_CENTER_Y, UV_CENTER_Y) = sigma_px     * sigma_px     / 2.0;
        r(UV_LENGTH,   UV_LENGTH)   = sigma_length * sigma_length / 2.0;
        return r;
    };
    obs.push_back(esekf->make_obs(observation, UVLMeasure{ctx}, u_r, cal_uvl_residual));
};

// ── 一块完整板 → 拆成左右两条灯条 ──────────────────
auto add_uva_obs = [&](Armor& a, int id) {
    add_uvl_obs(kp[LEFT_TOP],  kp[LEFT_BOTTOM],  id, true);
    add_uvl_obs(kp[RIGHT_TOP], kp[RIGHT_BOTTOM], id, false);
    if (matched_armors.size() == 1 && armor_pnp(...)) {
        // 单板退化时补一维 IPPE 深度差约束（DiffMeasure，不在本文范围）
    }
};

for (auto& [id, armor] : matched_armors) {
    jumped |= (id != 0);                              // 粘滞置位
    add_uva_obs(armor, id);
}
for (const auto& [id, is_left, light] : matched_lights)
    add_uvl_obs(light.top, light.bottom, id, is_left);   // 孤立灯条，同一个模型

target_state.x = esekf->update_multi(obs);
```

**R 为什么正比于灯条长度。**设角点检测误差在*像素空间*近似恒定（网络回归精度不随距离
变）。那为什么 σ 还要乘 L？因为我们关心的不是像素误差本身，而是它**折算到状态空间的
影响**。灯条长度 L 与距离 d 成反比：

```text
L ≈ f · H_armor / d      ⟹      d ≈ f · H_armor / L
```

同样 1 像素的横向误差，在距离 d 处对应的物理横移是 d/f 米——**远处放大得多**。若把 σ
设成常数，滤波器会误以为远处观测和近处一样可信。取 σ ∝ L 相当于：

```text
σ_px = κ·L ∝ 1/d      ⟹      σ_物理 = σ_px · d/f ∝ 常数
```

也就是说，**这个缩放让"物理尺度上的观测噪声"近似恒定**，距离权重不用手调。这是本仓库
`docs/pnp_observation_noise_and_covariance.md` 里那套完整协方差传播的一个廉价但方向正确
的替代品。

那个 `/ 2.0`：一块板被拆成两条灯条、信息量翻倍，所以每条方差减半，总信息量与"把整块板
当一个观测"相当。

**被注释掉的一行。**

```cpp
// sigma_angle *= std::cos(z(idx::UV_ANGLE));
//   越接近垂直上下点的垂直灯条方向偏移越大，这个实际还是需要考虑遮挡模型
```

作者知道 σ_α 应该随灯条倾角变化，但没有遮挡模型支撑就先按住了。对应 awakening README
TODO 里的"针对现实装甲板特征可见性的投影模型"，以及那个 1 行空文件
`armor_obstruction_model.hpp`。

**观测的唯一入口。**注意完整板和孤立灯条走的是**同一个 `add_uvl_obs`**。这是 UVL 设计
最漂亮的地方：配不成板的单根灯条不需要任何特殊处理，它就是少了一个搭档的普通观测。


## 8. 移植到本仓库

### 8.1 状态布局对照

| 下标 | awakening | 本仓库 `TrackedTarget` | 兼容 |
| --- | --- | --- | --- |
| 0-5 | `cx,vcx,cy,vcy,cz,vcz` | `xc,vx,yc,vy,z,vz` | ✔ 完全一致 |
| 6 | `C_ROT_Z` | `yaw` | ✔ 数值一致（退化时） |
| 7 | `VYAW` | `v_yaw` | ✔ 完全一致 |
| 8 | `LOG_R1` | `r1` | ✘ **log vs 线性** |
| 9 | `LOG_R2` / `OUTPOST01DZ` | `r2 − r1` | ✘ 语义不同 |
| 10 | `H` / `OUTPOST02DZ` | `z2 − z1` | ≈ 含义相近 |
| 11-12 | `C_ROT_Y, C_ROT_X` | `dz1, dz2` | ✘ 完全不同 |

**L4 只读四个下标。**`planner.cpp` 对 L3 的全部耦合是 `x[0]`、`x[2]`（`:64`）、`x[7]`
（`:84, :290, :293`）、`x[8]`（`:240`）加上 `armor_xyza_list()`。前三个直接兼容；
`x[8]` 让 `ekf_x()` 对外吐**线性**半径即可，L4 一行不改。

### 8.2 分阶段

| 阶段 | 内容 | 验证方式 | 动 L2/L4 |
| --- | --- | --- | --- |
| M0 | `so3_exp/log` + 手写 `Jet<double,N>` | 与中心差分交叉验证 | 否 |
| M1 | `inject_state` / `box_minus_state` | 互逆性单测（3.3） | 否 |
| M2 | ESEKF 壳：predict 求 F + 迭代 update_multi | 线性系统退化成标准 KF 对拍 | 否 |
| M3 | UVL 观测（由装甲板四角导出） | 单板重投影自洽 | 否 |
| M4 | 接进 `Tracker`，config 选后端 | `auto_aim_test` 同段 records A/B | 否 |
| M5 | 孤立灯条观测 | 遮挡场景对比 | L2 要输出 Light |

M3 之所以不用动 L2：UVL 观测并不依赖独立灯条检测，`add_uva_obs` 就是把一块装甲板的四角
拆成左右两条灯条。本仓库角点序是 TL,TR,BR,BL，直接对应 `corners[0]+corners[3]` 和
`corners[1]+corners[2]`。孤立灯条是增量，不是前提。

### 8.3 自动微分怎么选

awakening 用 `ceres::Jet` 求 F、中心差分求 H。本仓库 `xmake.lua` 没有 ceres，但
`/usr/include/ceres/jet.h` 在开发机上是有的。

| 方案 | 代价 | 学习价值 |
| --- | --- | --- |
| 引 Ceres | 加构建依赖，`jet.h` 还要拖 `ceres/internal/*` | 低，直接抄 |
| **自己写 `Jet<double,N>`** | **~150 行 header，零依赖** | **高** |
| 全用中心差分 | 最简单，但 13 维 × 每观测 26 次投影 | 中，适合当对照组 |

自己写只需要这些算子：`+ − × ÷`、`exp`、`sqrt`、`sin`、`cos`、`atan2`、`abs`、`fmin`、
`fmax`、`floor`。而且**真正跑 H 的 `update_multi` 用的是中心差分，Jet 只有 predict 求 F
时才需要**，规模比想象中小。建议把中心差分版一并写出来，在 smoke test 里交叉验证——
这能抓住 `inject`/`box_minus` 的符号和左右乘错误。

### 8.4 陷阱清单

| 项 | 说明 |
| --- | --- |
| 角点顺序 | awakening `TL,BL,BR,TR` vs 本仓库 `TL,TR,BR,BL`。左灯条 = `corners[0]+corners[3]` |
| 板法向朝内 | 可见性判据要用 `-axis_x`，搞反则全部选错板 |
| `x[8]` 是 log | `planner.cpp:240` 按线性半径判，必须在 `ekf_x()` 处转换 |
| `Zc ≤ 0` | awakening 没保护，移植时补，并让观测被丢弃而非使用 |
| 迭代式 | awakening 是 `δ += K·r`；按 `docs/iterated_ekf.md` 的 Bell & Cathey 形式实现，两版对拍 |
| 类型擦除 | 不要照抄 `ObsBase` 虚接口，否则 Jet 用不上 |
| 前哨站 | 11/12 维语义两边冲突，第一版只做常规四板车 |
| `armorPitchOf` | 本仓库已有，直接用，别再写一份 |
| `so3_log` θ→π | 上游无保护。误差状态下碰不到，但若在别处复用要补 |


## 9. 自检

能答上来就算读懂了，答案都在正文里。

1. 如果把 `inject_state` 里的 `R·Exp(δ)` 改成 `Exp(δ)·R`，`process_noise()` 里**哪两处**
   会变得不自洽？
2. 为什么 `so3_log` 不保护 θ→π 也没事，而 `so3_exp` **必须**保护 θ→0？
3. 求 H 时如果直接差 `z_pred` 而不走 `residual`，在什么情况下会出错？错成什么样？
4. `x[9]` 这一个 double，在四板步兵和前哨站上分别是什么？哪些代码必须先分支才能碰它？
5. 为什么观测要设计成 (α, uc, vc, L) 而不是两个端点的四个坐标？（提示：和对角 R 能否
   成立有关）
6. `reset()` 假设"看到的这块板是 0 号"。这个假设错了会怎样？为什么不致命？
7. 为什么 `p0[VYAW] = 100`，而位置只给 1？
8. R 里的 σ 为什么正比于灯条像素长度 L？如果设成常数会有什么后果？


## 参考

- 上游源码：`../awakening-main/`（无 git 历史，行号按当前工作副本）
- 本仓库相关文档：`docs/iterated_ekf.md`、`docs/pnp_observation_noise_and_covariance.md`
- 渲染版（含 LaTeX 公式）：https://claude.ai/code/artifact/04ff92fb-4efd-4664-94b2-82ed0fdd5e41
