# 录制 auto_aim_test 回放数据

在 `newvision/` 目录运行。需要接好 `config/camera_config.yaml` 指定的相机和
`config/serial_config.yaml` 指定的下位机串口；下位机需持续发送 IMU 姿态。

```bash
xmake build record
xmake run record -o=records/repro_01 -d=20 -f=30
```

`-o` 是无扩展名的输出路径；不写时按本地时间生成 `records/YYYY-MM-DD_HH-MM-SS`。
默认打开实时相机窗口，显示未经绘制的原始画面。在窗口中按 `q` 或 Esc 结束录制，
也可按 Ctrl-C。无显示器时加 `--preview=false`。
`-d=0`（默认）会一直录到手动结束。`--max-frames=300` 可按配对帧数停止。
已有同名文件时程序报错，不覆盖旧录像。录制时串口只接收状态，不发送控制包。

每个已保存视频帧在同名 TXT 里恰有一行：

```text
图像相对首帧的秒数 quaternion_w quaternion_x quaternion_y quaternion_z
```

图像时间戳来自相机接口；姿态按 `pose_offset_ms` 对齐，再用前后串口包插值。
如果找不到包围图像时刻的两包姿态，该图像不写入任一文件。保存的是下位机的
IMU 四元数，所以回放时用默认 `--convention=imu`，并沿用**录制时相同的**
`serial_config.yaml` 中 `R_imu_barrel` 和相机标定。TXT 是真实图像时间戳；
AVI 按指定 FPS 编码，回放算法使用 TXT 时间戳推进。

```bash
xmake run auto_aim_test -- records/repro_01 -c=config/camera_config.yaml --convention=imu
```

录像格式可以无硬件检查：

```bash
xmake run record_writer_smoke
xmake run serial_worker_smoke
```

第一项生成三帧临时 AVI/TXT 并读回验证；第二项用伪终端检查串口只读模式。
