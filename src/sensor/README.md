# 公共相机与通信模块

源码位于 `src/sensor/`，头文件位于 `include/sensor/`，供装甲板、能量机关和运行时共用。命名空间为 `L1Sensor`，头文件统一使用 `#include "sensor/..."`，项目的 `include/` 已作为公开搜索路径。

- `camera/`：相机帧采集、相机标定和相机到枪管的坐标变换配置。
- `serial/`：串口配置、底层收发、协议解析、机器人状态和按图像时间查询云台姿态。
- `serial/serial_command.hpp`：公共 `L1Sensor::SerialCommand`，包含 yaw、pitch、shoot 与可选前馈字段。装甲板中的 `L5Control::SerialCommand` 是该类型的兼容别名。

`RobotState::mode` 包含自瞄、大小符与前哨站模式。运行时负责把公共状态转换成各算法的输入，并把算法输出转换成公共串口命令。算法模块不应各自创建相机或串口收发线程。

本模块由现有 `newvision` 库统一构建。相机 SDK 与串口库仍位于 `tools/`；日志和姿态数学函数当前复用 `L6Telemetry`，实现位于 `src/armor/l5_telemetry/`。目录迁移不改变相机配置、协议字段、单位、线程或控制行为。
