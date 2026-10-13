#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

#include "l1_sensor/serial/robot_state.hpp"
#include "l5_control/serial_command.hpp"

namespace L1Sensor {

/**
 * @brief 视觉与下位机之间的串口帧协议：打包控制命令、从字节流中解析状态帧
 */
class SerialProtocol {
public:
  /**
   * @brief 协议帧头：用于定位帧起点、描述数据长度、记录序号和命令类型
   */
  struct __attribute__((packed)) HeaderFrame {
    /// 帧起始字节
    std::uint8_t sof = 0xA0;

    /// 数据段长度，字节
    std::uint16_t data_length = 0;

    /// 循环递增的帧序号，用于丢包检测
    std::uint8_t seq = 0;

    /// 帧头 CRC8
    std::uint8_t crc8 = 0;

    /// 命令类型
    std::uint16_t cmd_id = 0;
  };

  /**
   * @brief 下位机发给视觉的状态数据
   */
  struct __attribute__((packed)) RxPayload {
    /// 云台横滚角，rad
    float roll = 0.0F;

    /// 云台偏航角，rad
    float yaw = 0.0F;

    /// 云台俯仰角，rad
    float pitch = 0.0F;

    /// 弹速，m/s
    float bullet_speed = 0.0F;

    /// 枪口热量，目前不读
    float heat = 0.0F;

    /// 敌方颜色：0 红、1 蓝，其余为未知
    std::uint8_t enemy_color = 2;

    /// 工作模式：1 自瞄、2 小符、3 大符、4 前哨，其余为空闲
    std::uint8_t mode = 0;
  };

  /**
   * @brief 视觉发给下位机的控制数据
   */
  struct __attribute__((packed)) TxPayload {
    /// 目标偏航角，世界系绝对角，rad
    float yaw = 0.0F;

    /// 目标俯仰角，世界系绝对角，rad
    float pitch = 0.0F;

    /// 是否开火
    std::uint8_t shoot = 0;
  };

  /**
   * @brief 完整接收帧：帧头 + 状态数据 + CRC16
   */
  struct __attribute__((packed)) RxPacket {
    /// 帧头
    HeaderFrame frame_header;

    /// 状态数据
    RxPayload data;

    /// 整帧 CRC16
    std::uint16_t crc16 = 0;
  };

  /**
   * @brief 完整发送帧：帧头 + 控制数据 + CRC16
   */
  struct __attribute__((packed)) TxPacket {
    /// 帧头
    HeaderFrame frame_header;

    /// 控制数据
    TxPayload data;

    /// 整帧 CRC16
    std::uint16_t crc16 = 0;
  };

  /**
   * @brief 把控制命令打包成可以直接写入串口的字节流，并分配循环递增的发送 seq
   * @param command 控制命令
   * @return 完整的发送帧字节
   */
  std::vector<std::uint8_t> encodeCommand(const L5Control::SerialCommand& command);

  /**
   * @brief 输入串口原始字节流，解析其中全部完整合法的状态帧
   *
   * 不完整的尾部会保留到下一次调用继续解析。
   *
   * @param bytes 本次读到的原始字节
   * @return 本次解析出的全部状态
   */
  std::vector<RobotState> feed(std::span<const std::uint8_t> bytes);

  /**
   * @brief 清空当前接收缓存和协议状态
   */
  void reset();

  /**
   * @brief 开关 seq 丢包检测
   *
   * 切换和解析会串行化，避免重置 seq 基准时发生数据竞争。
   *
   * @param enable true 为开启
   */
  void setPacketLossCheckEnable(bool enable);

  /**
   * @brief 查询当前是否启用 seq 丢包检测
   * @return 启用返回 true
   */
  bool packetLossCheckEnable() const;

  /**
   * @brief 累计检测到的丢包数量
   * @return 丢包数
   */
  std::uint64_t droppedPacketCount() const;

private:
  /// 帧起始字节
  static constexpr std::uint8_t kSof = 0xA0;

  /// 视觉 -> 下位机的命令类型
  static constexpr std::uint16_t kTxCmdId = 0x0001;

  /// 下位机 -> 视觉的命令类型
  static constexpr std::uint16_t kRxCmdId = 0x0002;

  /// 数据段长度上限，超过视为帧头损坏
  static constexpr std::uint16_t kMaxPayloadLength = 256;

  /**
   * @brief 计算帧头 CRC8
   * @param bytes 参与校验的字节
   * @return CRC8 值
   */
  static std::uint8_t crc8(std::span<const std::uint8_t> bytes);

  /**
   * @brief 校验帧头 CRC8 是否正确
   * @param header 含 CRC8 字段的帧头字节
   * @return 校验通过返回 true
   */
  static bool checkCrc8(std::span<const std::uint8_t> header);

  /**
   * @brief 计算整帧 CRC16
   * @param bytes 参与校验的字节
   * @return CRC16 值
   */
  static std::uint16_t crc16(std::span<const std::uint8_t> bytes);

  /**
   * @brief 校验整帧 CRC16 是否正确
   * @param packet 含末尾 CRC16 的整帧字节
   * @return 校验通过返回 true
   */
  static bool checkCrc16(std::span<const std::uint8_t> packet);

  /**
   * @brief 把协议数据帧转换成视觉内部使用的 RobotState
   * @param packet 已校验的接收帧
   * @return 机器人状态，时间戳为解析时刻
   */
  static RobotState toRobotState(const RxPacket& packet);

  /**
   * @brief 根据连续 seq 判断是否丢包，并更新累计丢包计数
   * @param seq 本帧序号
   * @return 检测到丢包返回 true
   */
  bool checkPacketLoss(std::uint8_t seq);

  /// 串行化解析与配置切换
  mutable std::mutex mutex_;

  /// 跨调用保留的未解析字节
  std::vector<std::uint8_t> rx_buffer_;

  /// 是否启用 seq 丢包检测
  bool packet_loss_check_enable_ = true;

  /// 是否已有 seq 基准
  bool has_rx_seq_ = false;

  /// 上一帧接收的 seq
  std::uint8_t last_rx_seq_ = 0;

  /// 已提示过的其他合法命令类型，避免高频辅助帧刷满日志。
  std::unordered_set<std::uint16_t> ignored_command_ids_;

  /// 下一帧发送要用的 seq
  std::uint8_t next_tx_seq_ = 0;

  /// 累计丢包数
  std::uint64_t dropped_packet_count_ = 0;
};

}  // namespace L1Sensor
