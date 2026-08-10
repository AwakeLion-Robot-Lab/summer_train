# Daedalus 仿真器（Talos SHM）接入说明

把 `bevy_robomaster_simulator`（Daedalus）的仿真相机、云台姿态与真值接入
`newvision`（纯 C++、无 ROS2）。通信走 Talos 共享内存 IPC：

- `/tmp/talos_ipc_meta`（3712 B）：元数据、位姿、云台命令、内参、真值；
- `/tmp/talos_ipc_image_pool`（约 13.3 MB）：1440×1080 RGB8 图像三缓冲。

## 运行步骤

```bash
# 1. 启动仿真器（默认启用 talos feature）
cd bevy_robomaster_simulator
cargo run

# 2. 仿真器窗口里按 F5 开启自瞄订阅（否则云台命令不会被消费）

# 3. 构建并运行闭环工具
cd summer_train
xmake f -m release --use_openvino=y   # 首次配置
xmake build talos_auto_aim
xmake run talos_auto_aim -- --enemy=red --shoot=0
```

也可只验证共享内存读取：

```bash
xmake build talos_shm_smoke
xmake run talos_shm_smoke
```

云台角度实时显示（在仿真器里手动控制枪口时观察 yaw/pitch/roll 与底盘 yaw；
窗口内按 `Z` 记录基准角，显示相对变化；`--no-gui` 仅控制台输出）：

```bash
xmake build talos_gimbal_debug
xmake run talos_gimbal_debug
```

## 命令行选项

```text
--camera-config config/talos_camera_config.yaml  相机与 Talos 配置
--l3-config      config/l3_config.yaml          L3 参数
--model-path     model/armor_model/armor.xml    OpenVINO 装甲模型
--device         CPU                            推理设备
--enemy          red|blue                       敌方阵营（默认 red）
--shoot          0|1                            允许开火（默认 0）
--no-gui                                        关闭 OpenCV 界面
--timeout-ms     50                             单帧读取超时
--shm-dir        /tmp                           共享内存目录
--gt-gate-m      0.6                            真值匹配门限（米）
--prediction-ms  100                            整车未来轮廓时长，0 关闭
--show-armor-text                               显示装甲类别/置信度/XYZ/RPY
```

窗口按键：

```text
Space     暂停 / 继续
N / Right 暂停时单帧步进
L         切换装甲板文字（类别、置信度、XYZ、RPY、物理面 P0..）
Q / Esc   退出
```

画面叠加（与 `l3_video_replay` 一致）：

- 绿色：本帧有观测更新的 EKF 整车模型；
- 橙色：仅预测的整车模型；
- 紫色：按 `--prediction-ms` 外推的未来整车轮廓；
- 青色十字：敌方 GroundTruth 车身位置标记。

## 数据链路与关键约定

1. **图像**：共享内存为 RGB8；`TalosCamera` 转成 OpenCV BGR 后交给检测管线。
2. **时间戳**：SHM 时间为 UNIX 纳秒，读取器在启动时标定 `steady_clock` 偏移后
   映射为 `steady_clock::time_point`，与 L3 的单调时钟语义一致。
3. **同步握手**：每帧必须同时消费 image 与 Gimbal/Odom/Muzzle/Camera 四路
   pose 槽，否则仿真器 `synchronized_frame_consumed` 不通过、不再发布下一帧。
   读取器在 `readFrame()` 内部完成，使用者无需关心。
4. **单消费者**：Talos 三缓冲协议只支持一个消费者；不要同时运行
   `talos_auto_aim`、`talos_shm_smoke` 或多个读取进程。
5. **云台命令**：仿真器约定 `local_yaw = yaw_deg.to_radians()`、
   `pitch = (-pitch_deg - 90°).to_radians()`（-90° 为水平）。
   工具在 `computeAimSetpoint()` 中按世界系 Z-up、yaw=0 指向 +X 计算
   `bearing - chassis_yaw`，再经 `TalosSerial::updateCommand()` 转换。
6. **坐标系**：仿真器发布的云台四元数是 ROS 相机系（X 前、Y 左、Z 上），
   L3 PnP 输出在 OpenCV 相机系（X 右、Y 下、Z 前）。
   `TalosSerial::gimbalPoseAt()` 已做固定轴转换后作为 `R_world_barrel`。
   若实测方位错 90°/镜像，只需修改该转换矩阵或 `computeAimSetpoint()` 的符号。
   **L3 世界系 = 云台系（相机为原点）**：`TargetState.center` 是相机到目标的
   位移，不是绝对世界坐标。因此整车叠加和瞄准都按纯旋转使用（与
   `l3_video_replay` 一致），只有 GroundTruth 投影才叠加相机位置 `p_gimbal`。
7. **真值**：`GroundTruthBatch` 的 `team`（0=红/1=蓝）与 `armor_label`
   （与 `ArmorClass` 整数一致）。真值位置是车辆根部，因此叠加标记画在车身下方，
   匹配指标使用 3D 距离（默认门限 0.6 m）而非像素距离。
8. **开火**：`--shoot=1` 且目标处于 `Tracking`、距离在 0.3~25 m 时，
   `fire_advice=1` 才会写回；仿真器还需按 F5 开启订阅。

## 新增文件

- `include/l1_sensor/talos/`、`src/l1_sensor/talos/`：共享内存布局与读取器；
- `include/l1_sensor/camera/talos_camera.hpp`、`src/l1_sensor/camera/talos_camera.cpp`：
  相机后端（`camera_name: talos` 可被 `L1Sensor::Camera` 与现有工具复用）；
- `include/l1_sensor/serial/talos_serial.hpp`、`src/l1_sensor/serial/talos_serial.cpp`：
  仿真器姿态源 + 云台命令写回；
- `tools/talos_auto_aim.cpp`：闭环工具；
- `config/talos_camera_config.yaml`、`tests/talos_shm_smoke.cpp`。

对旧文件的改动仅两处：`src/l1_sensor/camera/camera.cpp`（新增 talos 分支）、
`xmake.lua`（新增两个 target）。
