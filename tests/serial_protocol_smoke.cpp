#include "l1_sensor/serial/serial_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace {

using Protocol = L1Sensor::SerialProtocol;

std::uint8_t crc8(std::span<const std::uint8_t> bytes)
{
  std::uint8_t crc = 0xFF;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int i = 0; i < 8; ++i) {
      crc = (crc & 0x01) != 0
              ? static_cast<std::uint8_t>((crc >> 1) ^ 0x8C)
              : static_cast<std::uint8_t>(crc >> 1);
    }
  }
  return crc;
}

std::uint16_t crc16(std::span<const std::uint8_t> bytes)
{
  std::uint16_t crc = 0xFFFF;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int i = 0; i < 8; ++i) {
      crc = (crc & 0x0001) != 0
              ? static_cast<std::uint16_t>((crc >> 1) ^ 0x8408)
              : static_cast<std::uint16_t>(crc >> 1);
    }
  }
  return crc;
}

std::vector<std::uint8_t> makeStatePacket(
  std::uint8_t seq, float yaw, float pitch)
{
  Protocol::RxPacket packet{};
  packet.frame_header.sof = 0xA0;
  packet.frame_header.data_length = sizeof(Protocol::RxPayload);
  packet.frame_header.seq = seq;
  packet.frame_header.cmd_id = 0x0002;
  packet.data.yaw = yaw;
  packet.data.pitch = pitch;
  packet.data.roll = 0.1F;
  packet.data.bullet_speed = 23.0F;
  packet.data.enemy_color = 1;
  packet.data.mode = 1;

  const auto header = std::span{
    reinterpret_cast<const std::uint8_t*>(&packet.frame_header),
    offsetof(Protocol::HeaderFrame, crc8)};
  packet.frame_header.crc8 = crc8(header);

  const auto body = std::span{
    reinterpret_cast<const std::uint8_t*>(&packet),
    sizeof(packet) - sizeof(packet.crc16)};
  packet.crc16 = crc16(body);

  std::vector<std::uint8_t> bytes(sizeof(packet));
  std::memcpy(bytes.data(), &packet, sizeof(packet));
  return bytes;
}

std::uint8_t txSequence(std::span<const std::uint8_t> bytes)
{
  if (bytes.size() != sizeof(Protocol::TxPacket)) {
    return 0xFF;
  }

  Protocol::TxPacket packet{};
  std::memcpy(&packet, bytes.data(), sizeof(packet));
  return packet.frame_header.seq;
}

}  // namespace

int main()
{
  // 线格式守卫：改这个数之前先确认下位机的结构体真的变了。现场实测下位机
  // 的 0x02 payload 是 22 字节（四个姿态/弹速 float + heat float + 两个 uint8）。
  static_assert(sizeof(Protocol::RxPayload) == 22);

  const auto first = makeStatePacket(10, 1.0F, 2.0F);
  const auto second = makeStatePacket(11, 3.0F, 4.0F);

  std::vector<std::uint8_t> merged = first;
  merged.insert(merged.end(), second.begin(), second.end());

  Protocol protocol;
  const auto states = protocol.feed(merged);
  if (states.size() != 2 || states[0].rpy.yaw != 1.0 ||
      states[1].rpy.yaw != 3.0 || states[1].rpy.pitch != 4.0 ||
      states[1].bullet_speed != 23.0) {
    std::cerr << "SerialProtocol did not drain concatenated packets\n";
    return 1;
  }

  Protocol fragmented_protocol;
  const auto split = first.size() / 2;
  if (!fragmented_protocol.feed(
        std::span<const std::uint8_t>{first.data(), split}).empty()) {
    std::cerr << "SerialProtocol accepted an incomplete packet\n";
    return 2;
  }

  const auto fragmented_states = fragmented_protocol.feed(
    std::span<const std::uint8_t>{first.data() + split, first.size() - split});
  if (fragmented_states.size() != 1 || fragmented_states.front().rpy.yaw != 1.0) {
    std::cerr << "SerialProtocol failed to reassemble a fragmented packet\n";
    return 3;
  }

  if (protocol.droppedPacketCount() != 0) {
    std::cerr << "SerialProtocol reported a false packet loss\n";
    return 4;
  }

  Protocol tx_protocol;
  L5Control::SerialCommand command{};
  for (int expected = 0; expected < 256; ++expected) {
    const auto bytes = tx_protocol.encodeCommand(command);
    if (txSequence(bytes) != static_cast<std::uint8_t>(expected)) {
      std::cerr << "SerialProtocol tx seq is not monotonic\n";
      return 5;
    }
  }

  if (txSequence(tx_protocol.encodeCommand(command)) != 0) {
    std::cerr << "SerialProtocol tx seq did not wrap after 255\n";
    return 6;
  }

  // ---- 下行帧两种格式 ----
  // 线格式守卫：改这两个数之前先确认电控那边的结构体真的跟着变了。
  static_assert(sizeof(Protocol::TxPayload) == 9);
  static_assert(sizeof(Protocol::TxFeedforwardPayload) == 25);

  // 默认必须仍是旧格式。这条断言是现场安全前提：电控固件没更新之前，谁都不该
  // 因为改了别处的默认值就把车上的下行帧换掉。
  {
    Protocol format_protocol;
    if (format_protocol.commandFormat() != Protocol::CommandFormat::Angle) {
      std::cerr << "SerialProtocol must default to the angle-only format\n";
      return 7;
    }
    const auto bytes = format_protocol.encodeCommand(command);
    if (bytes.size() != sizeof(Protocol::TxPacket)) {
      std::cerr << "default tx frame size changed\n";
      return 8;
    }
  }

  // 切到前馈格式：长度、cmd_id、六个浮点字段都要对得上，CRC 要自洽。
  {
    Protocol ff_protocol;
    ff_protocol.setCommandFormat(Protocol::CommandFormat::Feedforward);

    L5Control::SerialCommand ff{};
    ff.yaw = 0.25;
    ff.pitch = -0.125;
    ff.shoot = true;
    ff.yaw_velocity = 1.5;
    ff.yaw_acceleration = -32.0;
    ff.pitch_velocity = 0.75;
    ff.pitch_acceleration = 8.0;

    const auto bytes = ff_protocol.encodeCommand(ff);
    if (bytes.size() != sizeof(Protocol::TxFeedforwardPacket)) {
      std::cerr << "feedforward tx frame has the wrong size\n";
      return 9;
    }

    Protocol::TxFeedforwardPacket packet{};
    std::memcpy(&packet, bytes.data(), bytes.size());
    // 字面值而不是引用内部常量：这是给电控看的线格式契约，改常量不该悄悄
    // 让守卫跟着变。0x0003 = 带前馈的下行帧，0x0001 = 只有角度的旧格式。
    if (packet.frame_header.cmd_id != 0x0003 ||
        packet.frame_header.data_length !=
          sizeof(Protocol::TxFeedforwardPayload)) {
      std::cerr << "feedforward tx frame header is wrong\n";
      return 10;
    }
    if (packet.data.yaw != 0.25F || packet.data.pitch != -0.125F ||
        packet.data.yaw_velocity != 1.5F ||
        packet.data.yaw_acceleration != -32.0F ||
        packet.data.pitch_velocity != 0.75F ||
        packet.data.pitch_acceleration != 8.0F || packet.data.shoot != 1) {
      std::cerr << "feedforward tx payload does not carry the command\n";
      return 11;
    }

    // CRC 必须覆盖新增字段：改一个加速度字节，整帧 CRC16 就必须对不上。
    const auto body = std::span<const std::uint8_t>{
      bytes.data(), bytes.size() - sizeof(std::uint16_t)};
    if (crc16(body) != packet.crc16) {
      std::cerr << "feedforward tx frame crc16 is inconsistent\n";
      return 12;
    }
    auto tampered = bytes;
    tampered[offsetof(Protocol::TxFeedforwardPacket, data) +
             offsetof(Protocol::TxFeedforwardPayload, yaw_acceleration)] ^= 0x01;
    const auto tampered_body = std::span<const std::uint8_t>{
      tampered.data(), tampered.size() - sizeof(std::uint16_t)};
    if (crc16(tampered_body) == packet.crc16) {
      std::cerr << "feedforward tx crc16 does not cover the new fields\n";
      return 13;
    }
  }

  std::cout << "SerialProtocol smoke test passed\n";
  return 0;
}
