#include "sensor/serial/serial_worker.hpp"
#include <pty.h>
#include <poll.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Pty {
  int master = -1, slave = -1;
  char name[128]{};
  Pty() { check(openpty(&master, &slave, name, nullptr, nullptr) == 0, "openpty failed"); }
  ~Pty() { if (master >= 0) close(master); if (slave >= 0) close(slave); }
};
std::vector<L1Sensor::SerialProtocol::TxPayload> capture(int fd, int milliseconds) {
  std::vector<std::uint8_t> bytes;
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
  while (std::chrono::steady_clock::now() < end) {
    pollfd descriptor{fd, POLLIN, 0};
    if (poll(&descriptor, 1, 5) > 0) {
      std::array<std::uint8_t, 512> buffer{};
      const auto count = read(fd, buffer.data(), buffer.size());
      if (count > 0) bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + count);
    }
  }
  std::vector<L1Sensor::SerialProtocol::TxPayload> result;
  using Protocol = L1Sensor::SerialProtocol;
  for (std::size_t pos = 0; pos + sizeof(Protocol::TxPacket) <= bytes.size();) {
    Protocol::TxPacket packet;
    std::memcpy(&packet, bytes.data() + pos, sizeof(packet));
    if (packet.frame_header.sof == 0xA0 && packet.frame_header.cmd_id == 1 &&
        packet.frame_header.data_length == sizeof(Protocol::TxPayload)) {
      result.push_back(packet.data);
      pos += sizeof(packet);
    } else {
      ++pos;
    }
  }
  return result;
}
}  // namespace

int main() {
  Pty pty;
  L1Sensor::SerialConfig config;
  config.device = pty.name;
  config.baud_rate = 115200;
  config.read_timeout_ms = 2;
  config.tx_rate_hz = 100;
  config.command_timeout_ms = 150;
  L1Sensor::SerialWorker worker(config);
  check(worker.start(), "worker failed to start");
  check(capture(pty.master, 50).size() >= 2, "no initial serial frames");
  L1Sensor::SerialCommand command;
  command.yaw = .25;
  command.pitch = -.12;
  command.shoot = true;
  worker.updateCommand(command, true);
  const auto single = capture(pty.master, 90);
  int pulses = 0;
  for (const auto& packet : single) {
    pulses += packet.shoot != 0;
    check(std::abs(packet.yaw - .25) < 1e-5 && std::abs(packet.pitch + .12) < 1e-5,
          "single shot stopped angle updates");
  }
  check(single.size() >= 4 && pulses == 1, "single shot was lost or repeated");
  worker.updateCommand(command);  // 装甲板原有的持续开火语义。
  const auto continuous = capture(pty.master, 70);
  pulses = 0;
  for (const auto& packet : continuous) pulses += packet.shoot != 0;
  check(pulses >= 3, "continuous fire behavior changed");
  // 丢帧超时后两种模式都应保持角度、关火。
  (void)capture(pty.master, 180);
  for (const auto& packet : capture(pty.master, 50)) check(!packet.shoot, "expired command fired");
  worker.updateCommand(command, true);
  pulses = 0;
  for (const auto& packet : capture(pty.master, 70)) pulses += packet.shoot != 0;
  check(pulses == 1, "next single shot could not rearm");
  command.shoot = false;
  worker.updateCommand(command);
  for (const auto& packet : capture(pty.master, 40)) check(!packet.shoot, "safe hold fired");
  worker.stop();
  std::cout << "serial single shot checks passed\n";
}
