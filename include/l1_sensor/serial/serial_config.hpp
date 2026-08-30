#pragma once

#include <Eigen/Geometry>

#include <cstddef>
#include <string>

namespace L1Sensor {

// 串口运行参数；默认值用于快速跑通，也可由 YAML 覆盖。
struct SerialConfig {
  bool enable = true;
  std::string device = "/dev/ttyACM0";
  int baud_rate = 1000000;
  int read_timeout_ms = 20;
  int tx_rate_hz = 200;
  int command_timeout_ms = 100;
  int reconnect_interval_ms = 500;
  std::size_t rx_buffer_size = 256;
  bool packet_loss_check_enable = true;

  // 外参命名沿用 T_A_B：R_imu_barrel 把 barrel 系中的向量转到下位机 IMU 系。
  // world 取 imu_abs（IMU 轴向），barrel 是独立定义的右手系，两者链式复合：
  //   R_world_barrel = R_world_imu * R_imu_barrel
  //
  // barrel 系约定：x 轴指向枪口（瞄准方向），z 轴朝上，y 轴朝左。
  // x 轴必须是瞄准方向，PnpSolver::optimize_yaw 用 R_world_barrel 的 yaw
  // 作为 140 度搜索窗口的中心，轴向选错会让真值落到窗口之外。
  //
  // 加载时会校验正交且行列式为 +1，因此这里必然是右手系，不会混入镜像矩阵。
  Eigen::Matrix3d R_imu_barrel = Eigen::Matrix3d::Identity();

  // 该外参是否为单位阵之外的值；用于跳过恒等变换并输出启动日志。
  bool imuBarrelRotationNeeded() const noexcept
  {
    return !R_imu_barrel.isApprox(Eigen::Matrix3d::Identity());
  }
};

// 从 YAML 文件读取串口配置，缺失字段使用 SerialConfig 默认值。
SerialConfig loadSerialConfig(const std::string& config_path);

}  // namespace L1Sensor
