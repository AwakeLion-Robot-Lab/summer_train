# GTSAM 因子图估计器的接入与移植

本文记录 `l3_estimation/gtsam_est/` 的边界、状态量约定和实际移植结果。当前已经
接通 L2 检测、逐板 PnP、因子、ISAM2 目标、关联、四态 Tracker 和 L4 公共快照；
`gtsam_est_smoke` 会校验解析 Jacobian 与合成轨迹。

移植来源是 `jlu_vision_26-master/src/auto_aim/armor_tracker/`（`factors.hpp`、
`factors.cpp`、`target.hpp`、`target.cpp`、`types.cpp`）。

## 1. L3 的两个后端与共享边界

```text
                 ┌─────────────────────────────────────────┐
 L2 检测 ───────>│ 共享：toArmorObservation + PnpSolver     │
                 └───────────────┬─────────────────────────┘
                                 │  L3Estimation::Armor（世界系位姿 + 板 yaw）
                 ┌───────────────┴───────────────┐
                 │                               │
        filter_est::Tracker            gtsam_est::Tracker
        （整车 EKF，逐帧递推）          （因子图，ISAM2 增量优化）
                 │                               │
                 └───────────────┬───────────────┘
                                 │  L3Estimation::TrackedTarget（后端中立）
                                 v
                          L4Planning / L5Control
```

共享层（`include/l3_estimation/` 根目录）里的东西两个后端都用，**不允许**反向
依赖任何一个后端目录：

| 文件 | 内容 |
| --- | --- |
| `types.hpp` | `Armor` 观测、共享物理 `TargetConfig`、`ArmorConfig` / `TrackerConfig`、`EstimatorBackend` |
| `pnp_solver.hpp` | 单板 PnP 与 yaw 搜索，两个后端共用同一份实现 |
| `tracked_target.hpp` | 后端中立的 L3 → L4 契约，见第 2 节 |
| `tracker.hpp` | `ITracker` 抽象 + `makeTracker()` 工厂 + `toArmorObservation()` |

后端专属结构分开放置：`filter_est/config.hpp` 与 `filter_est/ekf.hpp` 只服务 EKF；
`gtsam_est/config.hpp`、`factors.hpp` 和 `target.hpp` 只服务因子图。`TrackedTarget`
只保存固定十一维物理状态、协方差和元数据，不含 `ExtendedKalmanFilter`、
`gtsam::Values` 或任何优化器生命周期。

## 2. 状态量映射

`TrackedTarget` 的十一维状态与吉大 `RobotTargetState` 是 1:1 的：

| jlu | newvision `x[i]` | 说明 |
| --- | --- | --- |
| `center_position.x()` | `x[0]` `xc` | 旋转中心，米 |
| `center_velocity.x()` | `x[1]` `vx` | |
| `center_position.y()` | `x[2]` `yc` | |
| `center_velocity.y()` | `x[3]` `vy` | |
| `center_position.z()` | `x[4]` `z` | **0/2 号板**的高度 |
| `center_velocity.z()` | `x[5]` `vz` | |
| `center_yaw` | `x[6]` `yaw` | 弧度，周期量 |
| `center_vyaw` | `x[7]` `v_yaw` | |
| `radius_a` | `x[8]` `r1` | 0/2 号板的半径 |
| `radius_b` | `x[9]` `r2 - r1` | 存的是**差值**，别直接写 `radius_b` |
| `dz` | `x[10]` `z2 - z1` | |

顺序是硬约定，代码通过 `TargetStateIndex` 命名索引读取。装甲板展开公式以
`target_state.cpp::armorPosition()` 为准：

```text
angle = limit_rad(x[6] + id * 2π / n)
xyz   = [x[0] - r*cos(angle), x[2] - r*sin(angle), z]
        其中 n==4 且 id 为奇数时 r = x[8]+x[9]，z = x[4]+x[10]；否则 r = x[8]，z = x[4]
```

吉大的 `ArmorIndex` 从 −x 轴起逆时针均分，效果一致，但**不要照搬那个枚举**：
本项目的下标就是物理板号。

## 3. 移植时必须改的四处

1. **时钟**。吉大用 `system_clock`，本项目全线 `steady_clock`（见 CLAUDE.md 的
   跨层契约）。照搬会在回放里出现负 `dt`。
2. **角点顺序**。统一为左上、右上、右下、左下。重投影因子吃像素角点时尤其要注意。
3. **世界系**。本项目的世界系是 MCU 的 `imu_abs`，不是吉大的 `odom`。
   `PnpSolver` 同时给出相机系和世界系位姿：关联使用 `xyz/ypr_in_world`；图中的单板
   `Pose3` 必须保持为相机系，供重投影因子直接投影，几何因子再用
   `T_world_camera` **恰好变换一次**。不要把世界系 Pose3 插进图后又重复乘外参。
4. **图管理**。见下一节。

## 4. 图管理暂与 JLU 保持一致

上游实现只在目标进入 `LOST` 时才清空整张图，因此变量（每帧的 `X(k)` `V(k)`
`R(k)` `W(k)` 和装甲板位姿）与因子随跟踪时长无界增长。上游 issue
[#6](https://github.com/Fskaaaaaaaa/jlu_vision_26/issues/6) 给出的实测：

| 跟踪时长 | `calculateEstimate()` | 后端总延迟 | 等效帧率 |
| --- | --- | --- | --- |
| 1 s | 0.406 ms | 1.232 ms | 164.8 FPS |
| 25 s（2622 帧） | 9.911 ms | 11.819 ms | 81.7 FPS |

瓶颈在 `getDelta()`——贝叶斯树越长，增量更新的传播范围越大。上游维护者的
回复是当前部署仅靠 LOST 清图，实战中长时间连续锁定较少，因此暂未加其他图管理。

本次的第一目标是先复现 JLU 行为，因此后端使用默认构造的 `gtsam::ISAM2`，
只在目标进入 LOST 后随 `Target` 一起清空。不额外暴露固定窗长、重线性化门限或
重线性化周期等参数。

这不代表 issue #6 的增长问题消失了。待状态与 `cmd_yaw` 逐帧对齐后，再把图管理当成
一个独立优化任务，并用长时间回放对比延迟和估计结果；不在当前基线里混入新参数。

## 5. 因子集合

```text
[运动段：恒速度 + 恒角速度，与 filter_est 的 F 矩阵是同一个模型]
  X(k-1), V(k-1) --TranslationFactor--> X(k)
  R(k-1), W(k-1) --YawFactor---------> R(k)
  V(k-1)         --VelocityFactor----> V(k)
  W(k-1)         --VyawFactor--------> W(k)

[观测段：把"中心 -> 装甲板"的位移拆成切向和径向]
  A, X(k), R(k)    --ArmorRadiusCenterZFactor--> 偶数号板观测
  B, Z, X(k), R(k) --ArmorRadiusDZFactor------> 奇数号板观测

[可选] 四角点 --ArmorReprojFactor--> 装甲板位姿
```

切向 / 径向分开是这套方案相对 EKF 的核心差别：半径**只**由径向残差约束，整车
yaw 的相位误差不会经切向残差把半径一路拉小。本项目的 EKF 在
`FilterEst::Target::updateObservation()` 里用 `[方位角, 俯仰角, 距离, 板 yaw]`
观测，半径和 yaw 在那里是耦合的——这正是回放对比时要重点看的那个量。

`GtsamEst::Config` 的噪声全部是**标准差 sigma**；`FilterEst::TargetConfig` 的
观测噪声是方差。它们位于不同后端目录，避免把两套记法混用。

## 6. 已实现链路与验证

1. `factors.hpp` / `factors.cpp`：运动、几何和四角点重投影因子及解析 Jacobian。
2. `target.hpp` / `target.cpp`：键分配、首批联合优化、每帧预测初值、ISAM2 更新、
   多板关联和后端中立快照。
3. `gtsam_est/tracker.cpp`：共享 PnP、四态生命周期、异常/时间连续性复位。
4. `xmake run gtsam_est_smoke`：解析/数值 Jacobian 对比，以及 120 帧合成轨迹上的
   有限状态、速度/半径收敛和完整图保留。
5. 回放验证：`xmake run auto_aim_test -- <record> --estimator=gtsam`，与
   `--estimator=filter` 的曲线逐帧对比。重点看半径 `x[8]`、`x[9]` 的收敛过程和
   整车 yaw 在换板瞬间的连续性。

当前公共 `TrackedTarget` 固定为十一维四板整车模型。三板目标可以跟踪，但按共享
半径、共享中心高度展开；JLU `OutpostTarget` 的 `Z(0)` / `Z(1)` / `Z(2)` 三块板
独立高度尚未塞进公共状态。若后续要精确复现前哨站倾斜安装，应单独扩展三板物理
状态或增加后端中立的逐板几何快照，不能把 GTSAM Key 暴露给 L4。

## 7. 安装 GTSAM

上游没有 4.3 正式版，只有 `4.3a0` / `4.3a1` / `4.3a2` 三个 alpha tag；吉大 README
里说的"4.3 版本"指这条线，用最新的 `4.3a2`。

Ubuntu 22.04 上除 Boost 外的依赖都已满足（cmake 3.22.1 ≥ 3.16、系统 Eigen 3.4.0、
g++-13）。Boost 需要 ≥ 1.70，jammy 的 1.74 够用。

Boost 组件按 4.3a2 的 `cmake/HandleBoost.cmake` 来，只要六个（别照搬网上常见的
`libboost-all-dev`，那会多装 1 GB）：

```bash
sudo apt install -y libboost-graph-dev libboost-serialization-dev \
                    libboost-program-options-dev libboost-random-dev \
                    libboost-timer-dev libboost-chrono-dev

git clone --depth 1 --branch 4.3a2 https://github.com/borglab/gtsam.git ~/src/gtsam
cmake -S ~/src/gtsam -B ~/src/gtsam/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/gtsam \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DCMAKE_INSTALL_RPATH=$HOME/.local/gtsam/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
cmake --build ~/src/gtsam/build -j4 && cmake --install ~/src/gtsam/build
```

三个 flag 是必须的，别用默认值：

| flag | 默认 | 为什么必须改 |
| --- | --- | --- |
| `GTSAM_USE_SYSTEM_EIGEN` | OFF | 默认会用 GTSAM 自带的 Eigen 3.3.x，而本项目用系统 Eigen 3.4.0。两份 Eigen 的类型跨库传递是 ODR 违规，症状是随机的对齐崩溃而不是编译错误。 |
| `GTSAM_WITH_TBB` | ON | 本项目是单线程流水线，实时循环要的是每帧确定的延迟而不是吞吐；顺便避开 Ubuntu 22.04 的 oneTBB 与 tbbmalloc 链接问题。 |
| `GTSAM_BUILD_WITH_MARCH_NATIVE` | OFF | 保持 OFF。打开后 GTSAM 与本项目的向量化宽度不一致，会引入 Eigen 对齐问题。 |

并行度要压：GTSAM 的翻译单元很吃内存（峰值 1~2 GB/TU），开发机 32 核但可用内存
只有 4 GB 左右，`-j32` 必然 OOM，`-j4` 实测可以跑完。

configure 成功的标志是这几行，对不上就别急着编：

```text
Use System Eigen                                 : ON
Use Intel TBB                                    : TBB not found
Build for native architecture                    : Disabled
Build shared GTSAM libraries                     : Enabled
```

装到 `$HOME/.local/gtsam` 是为了不需要 sudo；xmake 侧用 `--gtsam_root=` 指过去。
想跟吉大一样装到 `/usr/local` 就把两处路径都改掉、`cmake --install` 前面加 `sudo`，
这时 `--gtsam_root` 可以不传（默认就是 `/usr/local`）。

### 装到非标准前缀时必须设 CMAKE_INSTALL_RPATH

上面那两个 `CMAKE_INSTALL_RPATH*` 参数不是可选的，漏了会编译链接都成功、**一运行
就挂**：

```text
error while loading shared libraries: libmetis-gtsam.so: cannot open shared object file
```

原因是现代 binutils 生成的是 `DT_RUNPATH` 而不是 `DT_RPATH`，而 **RUNPATH 不参与
传递依赖的解析**：加载器解析 `libgtsam.so` 的 `NEEDED`（`libmetis-gtsam.so`、
`libcephes-gtsam.so.1`）时只看 *libgtsam.so 自己的* RUNPATH，不看可执行文件的。
GTSAM 默认不设 install rpath，于是它找不到自己捆绑的 metis / cephes。xmake 这边
把前缀加进 `rpathdirs` 也救不了——那只影响可执行文件自己的 RUNPATH。

验证方法：

```bash
readelf -d ~/.local/gtsam/lib/libgtsam.so | grep RUNPATH
# 应当输出 Library runpath: [/home/<user>/.local/gtsam/lib]
```

装在 `/usr/local/lib` 的话碰不到这个问题，因为它本来就在 ld.so 的默认搜索路径里。
吉大的 `rule("gtsam_deps")` 里那句 `runenvs LD_LIBRARY_PATH=/usr/local/lib` 是同一个
问题的另一种绕法。

## 8. 构建与切换

```bash
export GTSAM_ROOT=$HOME/.local/gtsam   # 装在 /usr/local 时不需要这行
xmake f --use_gtsam=y                  # 编入因子图后端
xmake run estimator_backend_smoke
xmake run auto_aim_test -- records/3m_high --estimator=gtsam
```

运行期后端由 `config/auto_aim.yaml` 的 `estimator.backend` 选（`filter` / `gtsam`）。
**选了没编进来的后端会在构造 `Tracker` 时抛 `std::runtime_error`，不会静默回退
到 EKF**：回退会让回放曲线看起来是因子图跑出来的，实际读的是卡尔曼的结果，这种
误导比起不来更贵。
