`processNoise()` 构造的是 **ESKF 预测时使用的过程噪声协方差矩阵 $Q$**：描述经过 $dt$ 后，运动模型没有考虑到的加速、转速变化和几何漂移，会给状态增加多少不确定性。

对应实现位于 [vehicle_model.hpp](../include/l3_estimation/armor/vehicle_model.hpp)，滤波器在 [error_state_ekf.hpp](../include/l3_estimation/filter/error_state_ekf.hpp) 中这样使用它：

$$
P^- = F P^+ F^{\mathsf T} + Q
$$

其中，$P^+$ 是上一次更新后的误差协方差，$F$ 是误差状态转移雅可比，$P^-$ 是本次预测后的误差协方差。$FP^+F^{\mathsf T}$ 传播已有误差，$Q$ 补充这段时间新产生的误差。

这里没有随机生成噪声去修改预测位置，而是增加预测的不确定性，进而影响后续观测更新的权重。下面按代码的四部分解释。

**① 平移部分来自随机加速度经过积分。** 运动模型按恒速度预测：

$$
p_{k+1}=p_k+v_kdt,\qquad v_{k+1}=v_k
$$

真实目标可能突然加速。假设一个预测间隔内，存在近似恒定的随机加速度，且不同预测间隔的加速度样本相互独立：

$$
a_k\sim\mathcal N(0,\sigma_a^2)
$$

它给位置、速度造成的预测误差是：

$$
\Delta p=\frac12a_kdt^2,\qquad \Delta v=a_kdt
$$

写成矩阵：

$$
\begin{bmatrix}\Delta p\\\Delta v\end{bmatrix}
=
\underbrace{\begin{bmatrix}\frac12dt^2\\dt\end{bmatrix}}_G a_k
$$

**$G\sigma_a^2G^{\mathsf T}$ 表示把输入噪声的方差传播到输出误差上。** 这里 $G$ 是一个 $2\times1$ 的列向量，记录同一个加速度误差分别会造成多少位置误差、多少速度误差。

令误差向量 $w=[\Delta p,\Delta v]^{\mathsf T}=Ga_k$。因为 $E[a_k]=0$，所以 $E[w]=0$，协方差可以写成：

$$
Q_{pv}
=E[ww^{\mathsf T}]
=E[(Ga_k)(Ga_k)^{\mathsf T}]
=G\,E[a_k^2]\,G^{\mathsf T}
=G\sigma_a^2G^{\mathsf T}
$$

这一步中，$G$ 在当前预测间隔内是确定的，可以从期望中提出；$E[a_k^2]=\sigma_a^2$ 则用到了加速度噪声均值为零。

右侧乘 $G^{\mathsf T}$，会让每个输出的系数与所有输出的系数分别相乘：

$$
\begin{bmatrix}g_1\\g_2\end{bmatrix}
\sigma_a^2
\begin{bmatrix}g_1&g_2\end{bmatrix}
=
\begin{bmatrix}
g_1^2\sigma_a^2 & g_1g_2\sigma_a^2\\
g_2g_1\sigma_a^2 & g_2^2\sigma_a^2
\end{bmatrix}
$$

对角线是各个输出自身的方差，非对角线是两个输出之间的协方差。代入 $g_1=\frac12dt^2$、$g_2=dt$：

$$
Q_{pv}
=G\sigma_a^2G^{\mathsf T}
=\sigma_a^2
\begin{bmatrix}
\frac14dt^4 & \frac12dt^3\\
\frac12dt^3 & dt^2
\end{bmatrix}
$$

这就对应代码中的四个系数：

| 代码填入的位置 | 含义 | 系数 |
|---|---|---|
| `q(position, position)` | 位置误差方差 | $\frac14dt^4$ |
| `q(position, velocity)` | 位置与速度误差的协方差 | $\frac12dt^3$ |
| `q(velocity, position)` | 上一项的对称项 | $\frac12dt^3$ |
| `q(velocity, velocity)` | 速度误差方差 | $dt^2$ |

**位置和速度误差有关联，是因为它们来自同一个随机加速度。** 因此不能只给对角线加噪声。

也可以逐项计算。一个随机量乘上确定系数后，方差要乘系数的平方；同一个随机量产生的两个输出，其协方差要乘两个系数的乘积：

$$
\operatorname{Var}(ca)=c^2\operatorname{Var}(a),\qquad
\operatorname{Cov}(ca,da)=cd\operatorname{Var}(a)
$$

因此，位置方差中的系数是 $(\frac12dt^2)^2=\frac14dt^4$，速度方差中的系数是 $(dt)^2=dt^2$，位置与速度协方差中的系数是 $(\frac12dt^2)(dt)=\frac12dt^3$。

例如，一个随机量 $a$ 的方差为 4，两个输出分别为 $y_1=2a$、$y_2=3a$，则：

$$
Q=
\begin{bmatrix}2\\3\end{bmatrix}
\cdot4\cdot
\begin{bmatrix}2&3\end{bmatrix}
=
\begin{bmatrix}16&24\\24&36\end{bmatrix}
$$

16 是 $y_1$ 的方差，36 是 $y_2$ 的方差，24 是两者的协方差。两者都来自同一个 $a$，所以会一起增大、一起减小。

三维情况下，代码先在车体系定义加速度协方差：

$$
\Sigma_{a,b}
=\operatorname{diag}(\sigma_{a_x}^2,\sigma_{a_y}^2,\sigma_{a_z}^2)
$$

`body_acceleration` 存的就是这些**方差**，单位为 $\mathrm{m^2/s^4}$。例如 `[30, 30, 1]` 表示允许模型车体 $x$、$y$ 方向有更大的加速度变化，车体 $z$ 方向的变化较小；数值 30 对应的标准差为 $\sqrt{30}\,\mathrm{m/s^2}$。

**车体系噪声需要换成世界系描述，是因为状态量使用的坐标轴不同。** 噪声配置使用的三根轴跟着目标车身转，而滤波器中的 `CX、CY、CZ` 和 `VCX、VCY、VCZ` 都用世界坐标轴表示。随机加速度 $a_b$ 是描述未知加速的模型变量，与状态中的位置、速度是不同的量；计算它对状态的影响时，需要统一坐标系。最终 $Q$ 的位置、速度部分已经是世界系下的误差协方差。

这里的转换依据是状态及其误差使用的坐标系。当前代码的主要滤波观测是灯条端点的像素坐标，[LightMeasure](../include/l3_estimation/armor/light_measure.hpp) 通过观测函数把世界中的目标投影到图像上；观测的坐标系与过程噪声的坐标系需要分别区分。

例如，只考虑平面运动，假设沿车体 $x$ 方向的加速度方差为 9，沿车体 $y$ 方向的加速度方差为 1：

$$
\Sigma_{a,b}=\begin{bmatrix}9&0\\0&1\end{bmatrix}
$$

这表示沿模型车体 $x$ 方向的加速度变化更大。这个方向对应世界中的哪个方向，取决于模型的姿态：

| 车体 $+x$ 轴朝向 | 世界 $x$ 方向的加速度方差 | 世界 $y$ 方向的加速度方差 |
|---|---:|---:|
| 世界 $x$ 方向 | 9 | 1 |
| 世界 $y$ 方向 | 1 | 9 |

车体绕竖直轴转了 $90^\circ$ 后，世界系里的协方差应该变成：

$$
\Sigma_{a,w}=\begin{bmatrix}1&0\\0&9\end{bmatrix}
$$

车自身的运动特性没有改变，改变的是这份不确定性在世界坐标轴上的分布。如果车体 $x$ 轴与世界 $x$、$y$ 轴都有夹角，沿车体 $x$ 轴的一次加速会同时改变世界系的 $x$、$y$ 速度，两个方向的误差就会产生相关性，协方差矩阵的非对角项可能不为零。

**代码中的车体系由装甲板编号确定，$+x_b$ 不保证是真实车头方向。** 初始化时，程序把首次用于建模的装甲板设为 0 号板。对于普通四板车，`armorPose()` 定义的板心位置如下，其中 $h$ 是奇数板相对偶数板的高度差：

| 装甲板编号 | 车体系中的板心坐标 |
|---|---|
| 0 | $(-r_1,0,0)$ |
| 1 | $(0,-r_2,h)$ |
| 2 | $(r_1,0,0)$ |
| 3 | $(0,r_2,h)$ |

原点 $O$ 是模型旋转轴上的参考点；$+x_b$ 从原点指向 2 号板，即与指向 0 号板的方向相反；$+y_b$ 沿 3 号板相对原点的水平投影方向；$+z_b$ 沿模型车体向上的方向，与另外两轴构成右手系。“前、左、上”可以作为理解随车坐标轴的类比，但不能据此认定程序已经识别了目标真实的车头。

原点在车体系中的坐标始终为 $(0,0,0)$，其世界坐标才是状态中的 `(CX, CY, CZ)`。车体系随目标移动、旋转，而世界系用于描述它的位置和朝向。

**$R_{\mathrm{vehicle}}$ 是当前估计的目标车体到世界坐标系的旋转矩阵。** 它把一个向量的车体系坐标变成世界系坐标：

$$
a_w=R_{\mathrm{vehicle}}a_b
$$

**旋转矩阵的三列，就是三根车体轴在世界系中的单位方向向量。** 这是列向量左乘矩阵时的直接结果。设 $R=[u_1\ u_2\ u_3]$，其中 $u_i$ 表示矩阵列向量，则：

$$
R\begin{bmatrix}1\\0\\0\end{bmatrix}=u_1,\qquad
R\begin{bmatrix}0\\1\\0\end{bmatrix}=u_2,\qquad
R\begin{bmatrix}0\\0\\1\end{bmatrix}=u_3
$$

输入 $[1,0,0]^{\mathsf T}$ 代表沿车体 $x_b$ 轴的单位箭头，输出必须是这根轴在世界系中的方向，所以第一列必须放这个方向；另外两列同理。任意加速度向量都可以分解成三个单位方向的线性组合：

$$
a_w
=a_{bx}e_x^w+a_{by}e_y^w+a_{bz}e_z^w
=\begin{bmatrix}e_x^w&e_y^w&e_z^w\end{bmatrix}a_b
$$

这表示用每根车体轴在世界中的方向，乘上对应的加速度分量，再相加。物理加速度没有变化，变化的是描述它的坐标分量。加速度是向量，这里只需旋转；转换一个点的位置时，还需要加车心的世界坐标，即 $p_w=c_w+R_{\mathrm{vehicle}}p_b$。

**先明确角度零点，再写正弦和余弦。** 以下平面示例统一约定：图上向右是世界 $+x_w$，向上是世界 $+y_w$，世界 $+z_w$ 朝向读者；车身保持水平，车体 $+z_b$ 与世界 $+z_w$ 一致。令 $\alpha$ 表示从世界 $+x_w$ 轴逆时针转到车体 $+x_b$ 轴的角度。

车体 $x_b$ 轴的世界系投影是 $(\cos\alpha,\sin\alpha,0)$。车体 $y_b$ 轴比它再逆时针转 $\pi/2$，所以投影是 $(-\sin\alpha,\cos\alpha,0)$。按列排列得到：

$$
R_{\mathrm{vehicle}}=
\begin{bmatrix}
\cos\alpha&-\sin\alpha&0\\
\sin\alpha&\cos\alpha&0\\
0&0&1
\end{bmatrix}
$$

当 $\alpha=90^\circ$，沿车体 $+x_b$ 方向的加速度 $[1,0,0]^{\mathsf T}$ 转换为：

$$
\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&1
\end{bmatrix}
\begin{bmatrix}1\\0\\0\end{bmatrix}
=
\begin{bmatrix}0\\1\\0\end{bmatrix}
$$

也就是沿车体 $+x_b$ 加速，对应世界系沿 $+y_w$ 加速。

**当 0 号板在图上方、2 号板在下方时，车体 $+x_b$ 向下、$+y_b$ 向右。** 沿用上述世界轴约定，四块板的水平投影为：

```text
               0号板
                 |
    1号板 ------ O ------ 3号板 --> +y_b
                 |
               2号板
                 |
                 v
                +x_b

世界 +x_w 向右，+y_w 向上，+z_w 朝向读者。
```

此时三根车体轴在世界系中的方向分别为：

$$
e_x^w=\begin{bmatrix}0\\-1\\0\end{bmatrix},\qquad
e_y^w=\begin{bmatrix}1\\0\\0\end{bmatrix},\qquad
e_z^w=\begin{bmatrix}0\\0\\1\end{bmatrix}
$$

因此，这张图对应的矩阵是：

$$
R_{\mathrm{vehicle}}=
\begin{bmatrix}
0&1&0\\
-1&0&0\\
0&0&1
\end{bmatrix}
$$

在前面从世界 $+x_w$ 轴起算的角度约定下，这个姿态对应 $\alpha=-\pi/2$，即 $-90^\circ$。

也可以选择另一个角度零点：令 $\psi=0$ 就对应这张图的姿态，$\psi$ 表示从这里继续逆时针旋转的角度。此时两个角度满足：

$$
\alpha=\psi-\frac{\pi}{2}
$$

代入前面的投影关系：

$$
e_x^w=
\begin{bmatrix}\sin\psi\\-\cos\psi\\0\end{bmatrix},\qquad
e_y^w=
\begin{bmatrix}\cos\psi\\\sin\psi\\0\end{bmatrix}
$$

于是，同一个车体到世界系的旋转可以用新的角度变量写成：

$$
\boxed{
R_{\mathrm{vehicle}}=
\begin{bmatrix}
\sin\psi&\cos\psi&0\\
-\cos\psi&\sin\psi&0\\
0&0&1
\end{bmatrix}
}
$$

代入 $\psi=0$，恰好得到车体 $x_b$ 向下、$y_b$ 向右的矩阵。这与使用 $\alpha$ 的标准形式表示同一个旋转，区别在于角度的起点。

这个形式的第二行应为 $[-\cos\psi,\sin\psi,0]$。若误写成 $[-\sin\psi,\cos\psi,0]$，则在 $\psi=0$ 时第一列全为零，会把车体 $x_b$ 轴的单位向量变成零向量。旋转必须保持向量长度，因此这种写法不成立。一个旋转矩阵的三列必须是相互垂直的单位向量，并满足右手系方向约定。

**代码中的旋转矩阵由滤波状态里的姿态计算得到。** [vehicle_model.hpp](../include/l3_estimation/armor/vehicle_model.hpp) 中的 `vehicleRotation()` 对普通车辆的计算相当于：

```cpp
// 从滤波状态中取出姿态的旋转向量。
Eigen::Vector3d phi(
  x[idx::ROT_X],
  x[idx::ROT_Y],
  x[idx::ROT_Z]
);

// 把旋转向量转换成 3×3 旋转矩阵。
Eigen::Matrix3d R_vehicle = L6Telemetry::so3Exp<double>(phi);
```

这三个状态量存的是旋转向量：向量方向表示旋转轴，长度表示旋转角度。它们不是可以直接依次代入 roll、pitch、yaw 的三个欧拉角。`so3Exp()` 在 [so3.hpp](../include/l6_telemetry/so3.hpp) 中使用罗德里格斯公式，将旋转向量转换成旋转矩阵。对于前哨站和基地，`vehicleRotation()` 把旋转向量的 $x$、$y$ 分量按零处理，只保留绕竖直轴的姿态。

滤波状态中的姿态按下面的流程获得：

1. 初始化时，PnP 得到装甲板姿态，再结合装甲板相对车体的安装姿态，反推出整车姿态。具体在 [EskfTarget::reset()](../src/l3_estimation/armor/eskf_target.cpp) 中实现，第一块观测板按 0 号板初始化。
2. 跟踪过程中，根据角速度预测姿态，再用装甲板、灯条观测修正姿态。
3. 计算过程噪声时，从当前状态取出姿态，转换成 $R_{\mathrm{vehicle}}$，用它转换噪声协方差。

初始化时的位姿关系在代码中写成：

```cpp
vehicle_in_world = armor_in_world * armor_in_vehicle.inverse();
```

其中 `armor_in_world` 表示装甲板在世界系中的位姿，`armor_in_vehicle` 表示装甲板在车体系中的安装位姿。由此得到整车位姿，再通过 `so3Log()` 把旋转矩阵转换成旋转向量，存入 `ROT_X`、`ROT_Y`、`ROT_Z`。

**左右分别乘旋转矩阵和它的转置，可以从协方差定义直接推导。** 把 `vehicle_rotation` 简记为 $R$，在本次噪声计算中，将当前估计出来的 $R$ 视为已知量。设 $a_b$ 是车体系下的随机加速度噪声，其均值为零。

$$
\mathbb{E}[a_b]=0,\qquad
\Sigma_{a,b}=\mathbb{E}[a_ba_b^{\mathsf T}]
$$

其中 $\mathbb{E}$ 表示取统计平均。均值为零时，协方差就是误差向量与自身转置的乘积的期望。

同一个加速度噪声转换到世界系后：

$$
a_w=Ra_b
$$

因此，世界系下的协方差为：

$$
\begin{aligned}
\Sigma_{a,w}
&=\mathbb{E}[a_wa_w^{\mathsf T}]\\
&=\mathbb{E}[(Ra_b)(Ra_b)^{\mathsf T}]\\
&=\mathbb{E}[Ra_ba_b^{\mathsf T}R^{\mathsf T}]\\
&=R\,\mathbb{E}[a_ba_b^{\mathsf T}]\,R^{\mathsf T}\\
&=R\Sigma_{a,b}R^{\mathsf T}.
\end{aligned}
$$

这里用到两个规则：

- 乘积转置要反转顺序：$(Ra_b)^{\mathsf T}=a_b^{\mathsf T}R^{\mathsf T}$。
- 确定的矩阵可以移到期望外面：本次计算中的 $R$ 已经由当前估计姿态确定。

协方差涉及两份误差的乘积，所以两份误差都需要转换坐标。左边的 $R$ 对应第一份误差的转换，右边的 $R^{\mathsf T}$ 来自第二份误差转换后的转置。若噪声均值不为零，对减去均值后的误差进行同样的推导，转换结论不变。

在代码中，车体系三个方向的加速度噪声被设为互不相关，因此：

$$
\Sigma_{a,b}
=\operatorname{diag}(\sigma_x^2,\sigma_y^2,\sigma_z^2)
$$

这正是 `body_acceleration.asDiagonal()`。逐个元素展开，世界系协方差的第 $i,j$ 项为：

$$
(\Sigma_{a,w})_{ij}
=\sum_{k=1}^{3}R_{ik}R_{jk}\sigma_k^2
$$

- 当 $i=j$ 时，得到世界系第 $i$ 个方向的方差，每个转换系数要平方。
- 当 $i\ne j$ 时，得到世界系两个方向的协方差，需要把对应的两个转换系数相乘。

代码中的这一行，就是把这些方差和协方差一次性算出来：

```cpp
vehicle_rotation
  * body_acceleration.asDiagonal()
  * vehicle_rotation.transpose();
```

这里的 $\Sigma_{a,b}$ 和 $\Sigma_{a,w}$ 是加速度噪声的三维协方差。随后还要乘上 $dt$ 的积分系数，才能填入十三维过程噪声矩阵 $Q$ 的位置、速度子块。

它和前面的 $G\sigma_a^2G^{\mathsf T}$ 使用同一个原理，只是输入从一个随机加速度变成了三维随机加速度向量，变换矩阵从 $G$ 变成了旋转矩阵。

默认配置 `[30, 30, 1]` 的车体 $x$、$y$ 方向方差相同。因此车身保持水平、只有 yaw 改变时，旋转后的协方差不变；车身发生倾斜时，较小的车体 $z$ 方向噪声才会重新分配到世界各轴上。如果三个方向的方差都相同，则任意旋转都不会改变这个协方差矩阵。

然后用其中每个元素填入位置、速度的对应块。这就是双层 `for` 循环的作用，也保留了旋转后不同轴之间可能出现的相关性。

按位置、速度分组写，整个平移块为：

$$
Q_{\mathrm{translation}}=
\begin{bmatrix}
\frac14dt^4\Sigma_{a,w} & \frac12dt^3\Sigma_{a,w}\\
\frac12dt^3\Sigma_{a,w} & dt^2\Sigma_{a,w}
\end{bmatrix}
$$

实际状态中的位置、速度下标交错排列，因此代码通过 `position_index` 和 `velocity_index` 填入矩阵。

**② yaw 部分沿用上述积分关系。** 把位置、速度、加速度换成角度误差、角速度误差、角加速度：

$$
\Delta\theta_z=\frac12\alpha dt^2,\qquad
\Delta\omega_z=\alpha dt
$$

于是：

$$
Q_{\theta_z,\omega_z}
=\sigma_\alpha^2
\begin{bmatrix}
\frac14dt^4 & \frac12dt^3\\
\frac12dt^3 & dt^2
\end{bmatrix}
$$

`yaw_acceleration` 表示角加速度方差，单位为 $\mathrm{rad^2/s^4}$。

这里不再旋转到世界系，是因为代码的姿态误差采用**右乘扰动**：

$$
R_{\mathrm{true}}
=R_{\mathrm{nominal}}\operatorname{Exp}(\delta\theta)
$$

$\delta\theta$ 表达在车体系，`VYAW` 也表示绕车体 $z$ 轴的角速度，坐标系已经一致。严格说，$Q$ 中的 `ROT_Z` 对应这个局部小角度误差，不能直接把它当成世界系欧拉 yaw 的误差。

函数开头根据目标是否为前哨站，选择普通目标或前哨站的加速度、角加速度方差。两者使用相同的推导，仅噪声参数不同。

**③ roll/pitch 使用随机游走模型。** 代码没有为这两个方向单独维护角速度，而是允许姿态在预测期间产生小幅未知变化：

$$
\delta\theta_x,\delta\theta_y
\sim\mathcal N(0,q_{rp}dt)
$$

所以对应对角线增加：

```cpp
dt * config.roll_pitch
```

这里 `roll_pitch` 是方差增长率，单位为 $\mathrm{rad^2/s}$。方差与时间成正比，标准差与 $\sqrt{dt}$ 成正比。它可以吸收车身倾斜变化等未显式建模的影响。绕 $z$ 轴的噪声已经由角加速度块提供，因此这一部分给第三维填零。

**④ 半径的 $1/r^2$ 来自对数变量的误差传播。** 状态保存的是：

$$
\ell=\ln r
$$

对小扰动做一阶近似：

$$
\delta\ell\approx\frac{1}{r}\delta r
$$

因此：

$$
\operatorname{Var}(\delta\ell)
\approx\frac{\operatorname{Var}(\delta r)}{r^2}
$$

这就是下面这项的来源：

```cpp
config.radius / (r * r)
```

同样大小的物理半径误差，在大半径目标上对应更小的相对误差，所以对数空间中的方差更小。高度没有取对数，直接加入相应方差即可。普通目标分别处理两组半径和板间高度差；前哨站的两个复用状态槽存储两块板相对基准板的高度差。

**半径、高度部分的注释与时间尺度需要统一。** 当前实现没有乘 `dt`：

```cpp
q(idx::LOG_R1, idx::LOG_R1) = config.radius / (r1 * r1);
q(idx::HEIGHT, idx::HEIGHT) = config.height;
```

这相当于把配置解释为“每次预测增加的方差”。如果按注释理解为“单位时间的随机游走方差增长率”，则应当是：

$$
Q_{\ell\ell}=\frac{q_rdt}{r^2},\qquad
Q_{hh}=q_hdt
$$

`outpost_height` 同理。当前写法会使这些维度每秒注入的噪声量随预测频率变化。如果改为按时间累积，需要同时统一参数单位，并按原先的预测频率换算或重新标定参数。

**加速度噪声还需要区分离散模型和连续模型。** 前两部分的 $dt^4/4$ 公式，对应每步独立、步内近似恒定的随机加速度。如果采用连续时间白噪声加速度模型，其强度为 $q_a$，精确离散结果则是：

$$
Q_{pv}=q_a
\begin{bmatrix}
dt^3/3 & dt^2/2\\
dt^2/2 & dt
\end{bmatrix}
$$

前一种模型的 $\sigma_a^2$ 是单步加速度方差，单位为 $\mathrm{m^2/s^4}$；后一种模型的 $q_a$ 是连续白噪声强度，单位为 $\mathrm{m^2/s^3}$。两种参数不能直接使用相同数值替换，调参时需要与所用公式保持一致。
