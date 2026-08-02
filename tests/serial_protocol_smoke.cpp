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
  packet.data.heat = 42.0F;
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
  const auto first = makeStatePacket(10, 1.0F, 2.0F);
  const auto second = makeStatePacket(11, 3.0F, 4.0F);

  std::vector<std::uint8_t> merged = first;
  merged.insert(merged.end(), second.begin(), second.end());

  Protocol protocol;
  const auto states = protocol.feed(merged);
  if (states.size() != 2 || states[0].rpy.yaw != 1.0 ||
      states[1].rpy.yaw != 3.0 || states[1].rpy.pitch != 4.0 ||
      states[1].heat != 42.0) {
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

  std::cout << "SerialProtocol smoke test passed\n";
  return 0;
}
