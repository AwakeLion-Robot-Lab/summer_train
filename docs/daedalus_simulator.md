# Daedalus（bevy_robomaster_simulator）接入说明

`newvision` 已提供一个 C++20 客户端，直接连接模拟器默认启用的 Talos IPC v2
共享内存。它可以读取同步的相机图像、四组位姿、相机内参和底盘观测，也可以向
模拟器发送云台与开火建议。

## 快速运行

先从模拟器仓库启动 Daedalus。通过 `cargo run` 启动可以自动设置 Bevy 动态库
路径，并且让程序从正确的工作目录加载 `assets/`：

```bash
cd /home/rm/bevy_robomaster_simulator-master
cargo run --release
```

再开一个终端构建并启动客户端：

```bash
cd /home/rm/super/newvision
xmake f -m release
xmake build daedalus_client
xmake run daedalus_client --frames=100 --show
```

`--show` 会显示 OpenCV 的 BGR 图像；不需要窗口时去掉它。`--frames=0` 表示
持续运行，按 `Ctrl+C` 退出。

发送云台命令的例子：

```bash
xmake run daedalus_client \
  --yaw-deg=15 --pitch-deg=-8 --distance-m=3.5
```

需要在模拟器窗口中按一次 `F5` 打开 AutoAim，外部云台命令才会作用到机器人。
`--fire` 会把开火建议设为真，并可能让模拟器实际发射弹丸，请按需使用。角度参数
是模拟器 IPC 的原始绝对角度（度）；当前模拟器会以
`yaw = yaw_deg`、`pitch = -pitch_deg - 90` 的内部角度关系应用它们。

运行 `xmake run daedalus_client --help` 可查看全部参数，包括连接超时和自定义
共享内存文件路径。

## 整车预测可视化

`daedalus_client` 只验证原始图像和通信。要运行装甲识别、IESKF 整车跟踪并把
预测结果画回模拟器画面，使用下面的目标：

```bash
cd /home/rm/super/newvision
xmake f -m release --use_openvino=y
xmake build daedalus_vehicle_prediction
xmake run daedalus_vehicle_prediction --predict-ms=100 --enemy=any
```

画面颜色含义：

- 青色框：神经网络在当前帧检测到的装甲板；
- 绿色框：IESKF 当前整车状态展开出的全部物理装甲板；
- 橙色框：整车中心和 yaw 同时外推指定时间后的全部物理装甲板；
- 绿点到橙点的箭头：整车旋转中心在预测窗口内的平移；
- 橙色 `+100ms #N`：预测时刻的第 N 块物理装甲板。

默认同时接受红蓝目标。只跟踪一方可指定 `--enemy=red` 或 `--enemy=blue`。
按 `q` 或 `Esc` 退出。这个可视化程序不会发送云台或开火命令，因此不需要按
模拟器的 `F5`。目标静止时绿色和橙色整车框会几乎重合，这是零速度预测的正常
结果；移动或旋转模拟器里的假人后，两组框会明显分开。

无窗口运行并保存最后一张叠加图：

```bash
xmake run daedalus_vehicle_prediction \
  --frames=100 --headless --save=/tmp/daedalus_prediction.png
```

运行 `xmake run daedalus_vehicle_prediction --help` 可查看预测时间、显示缩放、
模型配置和共享内存路径等参数。

## 在代码中使用

公开接口位于
`include/l1_sensor/simulator/daedalus_client.hpp`。最小读取循环如下：

```cpp
#include "l1_sensor/simulator/daedalus_client.hpp"

#include <chrono>
#include <iostream>

int main()
{
  L1Sensor::DaedalusClient simulator;
  if (!simulator.connect() || !simulator.isSimulatorAlive()) {
    std::cerr << simulator.lastError() << '\n';
    return 1;
  }

  L1Sensor::DaedalusFrame frame;
  while (simulator.readFrame(frame, std::chrono::milliseconds{500})) {
    // frame.image_bgr 是拥有自身内存的 CV_8UC3，可直接交给 OpenCV/newvision。
    const auto& gimbal = frame.pose(L1Sensor::DaedalusPoseKind::Gimbal);
    std::cout << frame.sequence << ' ' << frame.image_bgr.cols << 'x'
              << frame.image_bgr.rows << " qw=" << gimbal.quaternion[0] << '\n';

    // 参数依次为 yaw(deg)、pitch(deg)、目标距离(m)、开火建议。
    if (!simulator.sendGimbalCommand(15.0F, -8.0F, 3.5F, false)) {
      std::cerr << simulator.lastError() << '\n';
      return 2;
    }
  }
}
```

`DaedalusFrame` 中包含：

- `image_bgr`：从模拟器 RGB8 转换得到的独立 BGR8 图像，下一帧到来后仍然有效；
- `sequence`、`timestamp_ns`、`capture_time`：帧号、UNIX 纳秒时间戳和换算后的
  `steady_clock` 采集时刻；
- `Gimbal`、`Odom`、`Muzzle`、`Camera` 四组同步位姿；位置单位为米，四元数顺序
  为 `w, x, y, z`；
- `camera_info`：`fx/fy/cx/cy`、畸变参数和图像尺寸；
- `chassis`：车体速度、角速度、轮速和 IMU 风格观测；
- `auto_aim_enabled`：模拟器的 F5 AutoAim 状态。

## 协议与限制

默认共享内存文件为：

```text
/tmp/talos_ipc_meta
/tmp/talos_ipc_image_pool
```

Talos 图像和位姿通道采用单生产者、单消费者（SPSC）三缓冲协议。一个模拟器只能
同时由一个 `DaedalusClient` 消费帧；不要并行启动多个客户端。客户端每次读取时会
同时消费图像及四组位姿，保证模拟器能够继续发布下一帧。

客户端在连接时校验 Talos magic、协议版本、元数据大小和 1440×1080 图像布局，
不会把不兼容的共享内存误当成图像读取。

## 验证与排错

不启动模拟器也可以运行离线协议测试：

```bash
xmake build daedalus_client_smoke
xmake run daedalus_client_smoke
```

常见问题：

- `cannot open /tmp/talos_ipc_meta`：模拟器尚未启动，或启动时禁用了默认 `talos`
  feature；
- `shared-memory heartbeat is stale`：共享内存来自已经退出的旧进程，重新启动模拟器；
- `incomplete Daedalus frame bundle`：通常是另一个客户端也在消费同一通道，关闭它；
- 能收到图像但云台不动：先让模拟器窗口获得焦点，再按 `F5` 打开 AutoAim；
- `distance_m == -1` 是协议约定的无效命令，模拟器会忽略它。
