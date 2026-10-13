# pose_offset：图像与云台姿态的时间偏移标定

图像时间戳是相机取图返回的时刻，姿态时间戳是串口包解析的时刻，两者都是"到手时刻"，
各自比真实的曝光、IMU 采样晚一段。两段延迟之差就是 `pose_offset_ms`：取曝光那一刻的
云台姿态时，查的是 `图像时间戳 + pose_offset_ms`。

不补偿时，相机朝向误差约为 云台角速度 × 偏移。云台 2 rad/s、偏移 5 ms 时是 0.01 rad，
4 m 外约 4 cm，而且跟着云台运动走，滤波器会把它当成目标运动。

## 原理

静止目标的世界系视线方向应当恒定。扫一组候选偏移 τ，每个 τ 下用 `gimbalPoseAt(图像时刻 + τ)`
把枪管系的板位置转到世界系，看视线方向的离散度；离散度最小的 τ 就是偏移。推导见
`pose_offset.hpp` 开头的注释，背景见 `docs/ieskf_notes_and_stochastic_cloning.md` 5.4 节。

云台**必须来回甩**：匀速朝一个方向转时，偏移只造成一个恒定的视线偏差，离散度不随 τ 变，
偏移不可观。

## 步骤

1. 在 2~4 m 处放**一块静止**的装甲板（颜色不限），视野里别有别的板。
2. 电控**不要进自瞄**。程序不发任何控制命令，云台由操作手用遥控器左右、上下来回甩，
   幅度能让板一直留在视野里，频率 0.5~2 Hz 即可。
3. 在 newvision 目录下运行（本机 xmake 会把 `--` 原样传给程序，参数前不要写 `--`）：

   ```bash
   xmake build pose_offset_calib
   xmake run pose_offset_calib                 # 默认采 20 s，扫 ±30 ms，步长 0.5 ms
   xmake run pose_offset_calib -d=30 -r=50     # 加长采集、放宽范围
   xmake run pose_offset_calib --csv=offset.csv
   ```

4. 程序结束时打印代价曲线和结论。可信时最后一行形如 `pose_offset_ms: 4.5`，把它写进
   `config/serial_config.yaml`。结果是相对原始时间戳的绝对值，与配置里现有的值无关，
   直接覆盖即可。

配置读的是 `config/auto_aim.yaml`（模型、后端、板尺寸）、`config/camera_config.yaml`（相机与
标定）、`config/serial_config.yaml`（串口），和实机主程序同一份。缺 `T_barrel_camera` 时直接退出。

## 判读

| 输出 | 含义 |
| --- | --- |
| 视线角速度标准差 | 激励强度，低于 0.5 rad/s 判不可信，多甩几下 |
| 最优处剩余离散 | 检测与 PnP 的噪声底，通常零点几度 |
| 前一半 / 后一半 | 两半样本各自拟合，差超过 2 ms 判不可信（目标动过或检出不稳） |
| 最小值在边界 | 真值可能在扫描范围之外，加大 `-r` |
| 姿态不全 | 扫描窗口没完全落在串口姿态历史里（历史只存 64 包），这些帧被跳过而不是取边界值 |

退出码：0 可信，3 算出来了但不可信，1 硬件或样本不足，2 参数错误。

换相机、曝光时间、下位机固件或串口波特率后要重标。

## 文件

- `pose_offset.hpp/.cpp`：拟合核心，只依赖 Eigen；`tests/pose_offset_smoke.cpp` 用合成数据检查它
  （`xmake run pose_offset_smoke`，不需要硬件）。
- `pose_offset_calib.cpp`：实机采集，相机 → 检测 → PnP → 按候选偏移查姿态 → 拟合。
