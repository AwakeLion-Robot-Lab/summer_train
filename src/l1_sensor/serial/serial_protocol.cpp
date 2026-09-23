#include "l1_sensor/serial/serial_protocol.hpp"
#include "l6_telemetry/logger.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>

namespace L1Sensor {

// 将上层控制命令封装为完整发送帧，并补齐 CRC 校验。
std::vector<std::uint8_t>
SerialProtocol::encodeCommand(const L5Control::SerialCommand &command) {
  std::lock_guard<std::mutex> lock(mutex_);

  TxPacket packet;
  packet.frame_header.sof = kSof;
  packet.frame_header.data_length = sizeof(TxPayload);
  packet.frame_header.seq = next_tx_seq_;
  next_tx_seq_ = static_cast<std::uint8_t>(next_tx_seq_ + 1U);
  packet.frame_header.cmd_id = kTxCmdId;
  packet.data.yaw = static_cast<float>(command.yaw);
  packet.data.pitch = static_cast<float>(command.pitch);
  packet.data.shoot = command.shoot ? 1 : 0;

  auto header_body =
      std::span{reinterpret_cast<const std::uint8_t *>(&packet.frame_header),
                offsetof(HeaderFrame, crc8)};
  packet.frame_header.crc8 = crc8(header_body);

  auto packet_bytes = std::span{reinterpret_cast<const std::uint8_t *>(&packet),
                                sizeof(packet) - sizeof(packet.crc16)};
  packet.crc16 = crc16(packet_bytes);

  std::vector<std::uint8_t> bytes(sizeof(packet));
  std::memcpy(bytes.data(), &packet, sizeof(packet));
  return bytes;
}

// 向协议解析器追加原始字节，并一次取出缓存中的全部完整合法状态帧。
std::vector<RobotState>
SerialProtocol::feed(std::span<const std::uint8_t> bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<RobotState> states;
  rx_buffer_.insert(rx_buffer_.end(), bytes.begin(), bytes.end());

  while (rx_buffer_.size() >= sizeof(HeaderFrame)) {
    const auto head_pos = std::find(rx_buffer_.begin(), rx_buffer_.end(), kSof);

    if (head_pos == rx_buffer_.end()) {
      L6Telemetry::logWarn("serial protocol can't match sof");
      rx_buffer_.clear();
      break;
    }
    // 删除 sof 前面的干扰字节。计数是为了让"被吃掉的字节"可见——不计的话，
    // 下位机若发了别的 SOF 的帧，这里会一声不响地删掉整帧。
    skipped_byte_count_ +=
        static_cast<std::uint64_t>(std::distance(rx_buffer_.begin(), head_pos));
    rx_buffer_.erase(rx_buffer_.begin(), head_pos);

    if (rx_buffer_.size() < sizeof(HeaderFrame)) {
      break;
    }

    // 取出 HeaderFrame 中 cmd_id 之前的字节，用于校验帧头 CRC8。
    const auto header_bytes = std::span<const std::uint8_t>{
        rx_buffer_.data(), offsetof(HeaderFrame, cmd_id)};
    if (!checkCrc8(header_bytes)) {
      L6Telemetry::logWarn("serial protocol crc8 check failed");
      rx_buffer_.erase(rx_buffer_.begin());
      continue;
    }

    HeaderFrame header;
    std::memcpy(&header, rx_buffer_.data(), sizeof(header));
    if (header.data_length > kMaxPayloadLength) {
      L6Telemetry::logWarn("serial protocol payload length too large",
                           static_cast<int>(header.data_length));
      rx_buffer_.erase(rx_buffer_.begin());
      continue;
    }

    const auto frame_size =
        sizeof(HeaderFrame) + header.data_length + sizeof(std::uint16_t);
    if (rx_buffer_.size() < frame_size) {
      break;
    }

    const auto packet_bytes =
        std::span<const std::uint8_t>{rx_buffer_.data(), frame_size};
    if (!checkCrc16(packet_bytes)) {
      L6Telemetry::logWarn("serial protocol crc16 check failed");
      rx_buffer_.erase(rx_buffer_.begin());
      continue;
    }

    // seq 由下位机**逐帧**自增，与帧类型无关。所以丢包判断必须对所有通过
    // CRC 的帧做，不能只对本实现认识的那一种——否则下位机每插一帧别的类型
    // （现场是 0x21），序号就跳一格，会被当成丢包。实测这样会让 rx_dropped
    // 反过来比 rx 还大，明显不合理。
    if (packet_loss_check_enable_) {
      checkPacketLoss(header.seq);
    }

    if (header.cmd_id == kRxCmdId && header.data_length == sizeof(RxPayload) &&
        frame_size == sizeof(RxPacket)) {
      RxPacket packet;
      std::memcpy(&packet, packet_bytes.data(), sizeof(packet));
      rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + frame_size);
      states.push_back(toRobotState(packet));
      continue;
    }

    // 静默丢弃本实现不认识的帧。现场下位机稳定在发一种 0x21 帧（28 字节，
    // 约 90 Hz），这是协议尚未对齐、不是运行故障，每帧打一条只会把真正的
    // 告警冲掉——上一版在这里打 cmd_id/data_length/frame_size 与期望值的对照，
    // 正是靠它认出 0x02 的 payload 是 22 字节而不是 18；原因查明之后它就只剩
    // 噪声了。要再排一次结构体失配，从 b438f6e 把那段捡回来即可。
    // 帧数本身不会丢：seq 检查在上面对所有帧做，落在 rx_dropped 里。
    rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + frame_size);
  }

  return states;
}

// 清空接收缓存、序号状态和累计丢包计数。
void SerialProtocol::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  rx_buffer_.clear();
  has_rx_seq_ = false;
  last_rx_seq_ = 0;
  next_tx_seq_ = 0;
  dropped_packet_count_ = 0;
  skipped_byte_count_ = 0;
}

// 开关 seq 丢包检测；切换后重新建立 seq 基准，避免误报。
void SerialProtocol::setPacketLossCheckEnable(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);

  packet_loss_check_enable_ = enable;
  has_rx_seq_ = false;
  last_rx_seq_ = 0;
}

// 返回当前是否启用 seq 丢包检测。
bool SerialProtocol::packetLossCheckEnable() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return packet_loss_check_enable_;
}

// 返回累计检测到的丢包数量。
std::uint64_t SerialProtocol::droppedPacketCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return dropped_packet_count_;
}

// 返回累计被 SOF 搜索丢弃的字节数。
std::uint64_t SerialProtocol::skippedByteCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return skipped_byte_count_;
}

// 按当前协议的 CRC8 参数计算校验值。
std::uint8_t SerialProtocol::crc8(std::span<const std::uint8_t> bytes) {
  std::uint8_t crc = 0xFF;

  for (const auto byte : bytes) {
    crc ^= byte;
    for (int i = 0; i < 8; ++i) {
      if ((crc & 0x01) != 0) {
        crc = static_cast<std::uint8_t>((crc >> 1) ^ 0x8C);
      } else {
        crc = static_cast<std::uint8_t>(crc >> 1);
      }
    }
  }

  return crc;
}

// 校验帧头 CRC8，失败说明帧头字段不可信。
bool SerialProtocol::checkCrc8(std::span<const std::uint8_t> header) {
  if (header.size() < offsetof(HeaderFrame, cmd_id)) {
    return false;
  }

  const auto body = header.first(offsetof(HeaderFrame, crc8));
  return crc8(body) == header[offsetof(HeaderFrame, crc8)];
}

// 按当前协议的 CRC16 参数计算整帧校验值。
std::uint16_t SerialProtocol::crc16(std::span<const std::uint8_t> bytes) {
  std::uint16_t crc = 0xFFFF;

  for (const auto byte : bytes) {
    crc ^= byte;
    for (int i = 0; i < 8; ++i) {
      if ((crc & 0x0001) != 0) {
        crc = static_cast<std::uint16_t>((crc >> 1) ^ 0x8408);
      } else {
        crc = static_cast<std::uint16_t>(crc >> 1);
      }
    }
  }

  return crc;
}

// 校验整帧 CRC16，失败说明 payload 或 cmd_id 可能已损坏。
bool SerialProtocol::checkCrc16(std::span<const std::uint8_t> packet) {
  if (packet.size() < sizeof(std::uint16_t)) {
    return false;
  }

  const auto expected = static_cast<std::uint16_t>(
      packet[packet.size() - 2] |
      (static_cast<std::uint16_t>(packet[packet.size() - 1]) << 8));
  const auto body = packet.first(packet.size() - sizeof(expected));
  return crc16(body) == expected;
}

// 根据连续递增的 seq 判断合法接收帧之间是否发生丢包。
bool SerialProtocol::checkPacketLoss(std::uint8_t seq) {
  if (!has_rx_seq_) {
    has_rx_seq_ = true;
    last_rx_seq_ = seq;
    return false;
  }

  if (seq == last_rx_seq_) {
    L6Telemetry::logDebug("serial rx seq repeated", static_cast<int>(seq));
    return false;
  }

  const auto expected = static_cast<std::uint8_t>(last_rx_seq_ + 1);
  if (seq == expected) {
    last_rx_seq_ = seq;
    return false;
  }

  const auto dropped = static_cast<std::uint8_t>(seq - expected);
  dropped_packet_count_ += dropped;
  // 只在跳号超过 1 时告警。现场实测下位机每 21 个 seq 会稳定空转一个号，
  // 逐条打印就是每秒十几行、几分钟上万条，会把真正的告警冲掉；单号跳空只
  // 进计数器，从遥测的 serial/rx_dropped 曲线看。连跳多号才是异常。
  if (dropped > 1) {
    L6Telemetry::logWarn("serial rx packet lost", static_cast<int>(dropped),
                         "last", static_cast<int>(last_rx_seq_), "current",
                         static_cast<int>(seq));
  }
  last_rx_seq_ = seq;
  return true;
}

// 将接收帧中的 payload 转换成视觉内部统一使用的 RobotState。
RobotState SerialProtocol::toRobotState(const RxPacket &packet) {
  RobotState state;
  state.rpy.roll = packet.data.roll;
  state.rpy.yaw = packet.data.yaw;
  state.rpy.pitch = packet.data.pitch;
  state.bullet_speed = packet.data.bullet_speed;
  state.heat = packet.data.heat;
  state.timestamp = std::chrono::steady_clock::now();

  switch (packet.data.enemy_color) {
  case 0:
    state.enemy_color = EnemyColor::Red;
    break;
  case 1:
    state.enemy_color = EnemyColor::Blue;
    break;
  default:
    state.enemy_color = EnemyColor::Unknown;
    break;
  }

  switch (packet.data.mode) {
  case 1:
    state.mode = WorkMode::AutoAim;
    break;
  case 2:
    state.mode = WorkMode::SmallBuff;
    break;
  case 3:
    state.mode = WorkMode::BigBuff;
    break;
  case 4:
    state.mode = WorkMode::Outpost;
    break;
  default:
    state.mode = WorkMode::Idle;
    break;
  }

  return state;
}

} // namespace L1Sensor
