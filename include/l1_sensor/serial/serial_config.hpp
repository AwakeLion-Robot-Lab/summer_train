#pragma once

#include <Eigen/Geometry>

#include <chrono>
#include <cstddef>
#include <string>

namespace L1Sensor {

/**
 * @brief 串口运行参数
 *
 * 默认值用于快速跑通，也可由 YAML 覆盖。
 */
struct SerialConfig {
  /// 总开关；关闭时 SerialWorker::start() 直接返回 false
  bool enable = true;

  /// 串口设备路径
  std::string device = "/dev/ttyACM0";

  /// 波特率
  int baud_rate = 1000000;

  /// 单次读的超时，ms
  int read_timeout_ms = 20;

  /// 发送线程的发包频率，Hz
  int tx_rate_hz = 200;

  /// 控制命令的有效期，ms；超时后保持最后角度并强制关火
  int command_timeout_ms = 100;

  /// 串口断开后的重连间隔，ms
  int reconnect_interval_ms = 500;

  /// 接收线程单次读的缓冲区大小，字节
  std::size_t rx_buffer_size = 256;

  /// 是否按 seq 检测丢包
  bool packet_loss_check_enable = true;

  /// waitPose 最多等多久（ms）：图像到手后，等它之后的那包姿态到齐再插值。
  /// 正常只等一个下位机发包周期；等满说明串口断流，退回取最新一包。
  int pose_wait_ms = 20;

  /// 图像与姿态时间戳之间的固定偏移（ms）：曝光那一刻的云台姿态，查的是
  /// 图像时间戳 + pose_offset_ms。两边打的都是到手时刻（取图返回、串口包解析），
  /// 差值是"曝光读出传输"与"下位机发包串口"两段延迟之差，正值表示姿态晚到。
  /// 只能实机标定，见 tools/pose_offset。
  double pose_offset_ms = 0.0;

  /**
   * @brief 图像时间戳对应的姿态查询时刻
   * @param image 图像时间戳
   * @return image + pose_offset_ms
   */
  std::chrono::steady_clock::time_point poseTime(
    std::chrono::steady_clock::time_point image) const
  {
    return image + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(pose_offset_ms));
  }

  /// 外参命名沿用 T_A_B：R_imu_barrel 把 barrel 系中的向量转到下位机 IMU 系。
  /// world 取 imu_abs（IMU 轴向），barrel 是独立定义的右手系，两者链式复合：
  ///   R_world_barrel = R_world_imu * R_imu_barrel
  ///
  /// barrel 系约定：x 轴指向枪口（瞄准方向），z 轴朝上，y 轴朝左。
  /// x 轴必须是瞄准方向，PnpSolver::optimize_yaw 用 R_world_barrel 的 yaw
  /// 作为 140 度搜索窗口的中心，轴向选错会让真值落到窗口之外。
  ///
  /// 加载时会校验正交且行列式为 +1，因此这里必然是右手系，不会混入镜像矩阵。
  Eigen::Matrix3d R_imu_barrel = Eigen::Matrix3d::Identity();

  /**
   * @brief 该外参是否为单位阵之外的值
   *
   * 用于跳过恒等变换并输出启动日志。
   *
   * @return R_imu_barrel 不是单位阵时返回 true
   */
  bool imuBarrelRotationNeeded() const noexcept
  {
    return !R_imu_barrel.isApprox(Eigen::Matrix3d::Identity());
  }
};

/**
 * @brief 从 YAML 文件读取串口配置
 *
 * 缺失字段使用 SerialConfig 默认值。
 *
 * @param config_path 串口配置 YAML 路径（如 config/serial_config.yaml）
 * @return 串口配置
 */
SerialConfig loadSerialConfig(const std::string& config_path);

}  // namespace L1Sensor
