#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "l1_sensor/serial/robot_state.hpp"
#include "l5_control/serial_command.hpp"

namespace L1Sensor {

class SerialProtocol {
public:
  // 协议帧头：用于定位帧起点、描述数据长度、记录序号和命令类型。
  struct __attribute__((packed)) HeaderFrame {
    std::uint8_t sof = 0xA0;
    std::uint16_t data_length = 0;
    std::uint8_t seq = 0;
    std::uint8_t crc8 = 0;
    std::uint16_t cmd_id = 0;
  };

  // 下位机发给视觉的状态数据。
  struct __attribute__((packed)) RxPayload {
    float roll = 0.0F;
    float yaw = 0.0F;
    float pitch = 0.0F;
    float bullet_speed = 0.0F;
    float heat = 0.0F;
    std::uint8_t enemy_color = 2;
    std::uint8_t mode = 0;
  };

  // 视觉发给下位机的控制数据。
  struct __attribute__((packed)) TxPayload {
    float yaw = 0.0F;
    float pitch = 0.0F;
    std::uint8_t shoot = 0;
  };

  // 完整接收帧：帧头 + 状态数据 + CRC16。
  struct __attribute__((packed)) RxPacket {
    HeaderFrame frame_header;
    RxPayload data;
    std::uint16_t crc16 = 0;
  };

  // 完整发送帧：帧头 + 控制数据 + CRC16。
  struct __attribute__((packed)) TxPacket {
    HeaderFrame frame_header;
    TxPayload data;
    std::uint16_t crc16 = 0;
  };

  // 把控制命令打包成可以直接写入串口的字节流，并分配循环递增的发送 seq。
  std::vector<std::uint8_t> encodeCommand(const L5Control::SerialCommand& command);

  // 输入串口原始字节流，解析并返回其中全部完整合法状态帧。
  // 不完整的尾部会保留到下一次调用继续解析。
  std::vector<RobotState> feed(std::span<const std::uint8_t> bytes);

  // 清空当前接收缓存和协议状态。
  void reset();

  // 开关 seq 丢包检测；切换和解析会串行化，避免重置 seq 基准时发生数据竞争。
  void setPacketLossCheckEnable(bool enable);

  // 查询当前是否启用 seq 丢包检测。
  bool packetLossCheckEnable() const;

  // 返回累计检测到的丢包数量。
  std::uint64_t droppedPacketCount() const;

private:
  static constexpr std::uint8_t kSof = 0xA0;
  static constexpr std::uint16_t kTxCmdId = 0x0001;
  static constexpr std::uint16_t kRxCmdId = 0x0002;
  static constexpr std::uint16_t kMaxPayloadLength = 256;

  // 计算帧头 CRC8。
  static std::uint8_t crc8(std::span<const std::uint8_t> bytes);

  // 校验帧头 CRC8 是否正确。
  static bool checkCrc8(std::span<const std::uint8_t> header);

  // 计算整帧 CRC16。
  static std::uint16_t crc16(std::span<const std::uint8_t> bytes);

  // 校验整帧 CRC16 是否正确。
  static bool checkCrc16(std::span<const std::uint8_t> packet);

  // 把协议数据帧转换成视觉内部使用的 RobotState。
  static RobotState toRobotState(const RxPacket& packet);

  // 根据连续 seq 判断是否丢包，并更新累计丢包计数。
  bool checkPacketLoss(std::uint8_t seq);

  mutable std::mutex mutex_;
  std::vector<std::uint8_t> rx_buffer_;
  bool packet_loss_check_enable_ = true;
  bool has_rx_seq_ = false;
  std::uint8_t last_rx_seq_ = 0;
  std::uint8_t next_tx_seq_ = 0;
  std::uint64_t dropped_packet_count_ = 0;
};

}  // namespace L1Sensor
