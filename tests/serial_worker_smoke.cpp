#include "l1_sensor/serial/serial_worker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <optional>
#include <poll.h>
#include <pty.h>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Protocol = L1Sensor::SerialProtocol;

class PtyPair {
public:
  bool open()
  {
    std::array<char, 128> slave_name{};
    if (::openpty(&master_fd_, &slave_fd_, slave_name.data(), nullptr, nullptr) != 0) {
      return false;
    }

    device_ = slave_name.data();
    const int flags = ::fcntl(master_fd_, F_GETFL, 0);
    return flags >= 0 && ::fcntl(master_fd_, F_SETFL, flags | O_NONBLOCK) == 0;
  }

  ~PtyPair()
  {
    if (slave_fd_ >= 0) {
      ::close(slave_fd_);
    }
    if (master_fd_ >= 0) {
      ::close(master_fd_);
    }
  }

  int masterFd() const { return master_fd_; }
  const std::string& device() const { return device_; }

private:
  int master_fd_ = -1;
  int slave_fd_ = -1;
  std::string device_;
};

std::optional<Protocol::TxPacket> popPacket(std::vector<std::uint8_t>& pending)
{
  while (pending.size() >= sizeof(Protocol::TxPacket)) {
    if (pending.front() != 0xA0) {
      pending.erase(pending.begin());
      continue;
    }

    Protocol::TxPacket packet{};
    std::memcpy(&packet, pending.data(), sizeof(packet));
    if (packet.frame_header.data_length != sizeof(Protocol::TxPayload) ||
        packet.frame_header.cmd_id != 0x0001) {
      pending.erase(pending.begin());
      continue;
    }

    pending.erase(pending.begin(), pending.begin() + sizeof(packet));
    return packet;
  }

  return std::nullopt;
}

std::optional<Protocol::TxPacket> readPacket(
  int fd, std::vector<std::uint8_t>& pending, Clock::time_point deadline)
{
  while (Clock::now() < deadline) {
    if (const auto packet = popPacket(pending)) {
      return packet;
    }

    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - Clock::now());
    const int timeout_ms = static_cast<int>(std::max<std::int64_t>(remaining.count(), 1));
    pollfd poll_fd{fd, POLLIN, 0};
    if (::poll(&poll_fd, 1, timeout_ms) <= 0) {
      continue;
    }

    std::array<std::uint8_t, 256> bytes{};
    const auto bytes_read = ::read(fd, bytes.data(), bytes.size());
    if (bytes_read > 0) {
      pending.insert(pending.end(), bytes.begin(), bytes.begin() + bytes_read);
    }
  }

  return popPacket(pending);
}

bool isNextSequence(std::uint8_t previous, std::uint8_t current)
{
  return current == static_cast<std::uint8_t>(previous + 1U);
}

}  // namespace

int main()
{
  PtyPair pty;
  if (!pty.open()) {
    std::cerr << "Failed to create a pseudo terminal\n";
    return 1;
  }

  L1Sensor::SerialConfig config;
  config.device = pty.device();
  config.baud_rate = 115200;
  config.read_timeout_ms = 5;
  config.tx_rate_hz = 200;
  config.command_timeout_ms = 80;
  config.reconnect_interval_ms = 5;
  config.packet_loss_check_enable = false;

  int result = 0;
  {
    L1Sensor::SerialWorker worker(config);
    if (!worker.start()) {
      std::cerr << "SerialWorker failed to start on pseudo terminal\n";
      result = 2;
    } else {
      worker.updateCommand({1.25, -0.5, true});

      std::vector<std::uint8_t> pending;
      std::optional<std::uint8_t> previous_sequence;
      int shoot_packets = 0;
      bool saw_safe_command = false;
      const auto deadline = Clock::now() + std::chrono::seconds(1);

      while (Clock::now() < deadline && !saw_safe_command) {
        const auto packet = readPacket(
          pty.masterFd(), pending, Clock::now() + std::chrono::milliseconds(100));
        if (!packet) {
          continue;
        }

        if (previous_sequence && !isNextSequence(*previous_sequence, packet->frame_header.seq)) {
          std::cerr << "SerialWorker sent a non-incrementing seq\n";
          result = 3;
          break;
        }
        previous_sequence = packet->frame_header.seq;

        if (packet->data.shoot != 0) {
          ++shoot_packets;
          if (std::abs(packet->data.yaw - 1.25F) > 1e-6F ||
              std::abs(packet->data.pitch + 0.5F) > 1e-6F) {
            std::cerr << "SerialWorker altered a fresh command\n";
            result = 4;
            break;
          }
          continue;
        }

        if (shoot_packets >= 2 && std::abs(packet->data.yaw - 1.25F) <= 1e-6F &&
            std::abs(packet->data.pitch + 0.5F) <= 1e-6F) {
          saw_safe_command = true;
        }
      }

      if (result == 0 && shoot_packets < 2) {
        std::cerr << "SerialWorker did not transmit fresh commands before timeout\n";
        result = 5;
      }
      if (result == 0 && !saw_safe_command) {
        std::cerr << "SerialWorker did not transmit a safe command after timeout\n";
        result = 6;
      }
      if (result == 0) {
        const auto sent_command = worker.latestSentCommand();
        if (!sent_command || sent_command->shoot ||
            std::abs(sent_command->yaw - 1.25) > 1e-6 ||
            std::abs(sent_command->pitch + 0.5) > 1e-6) {
          std::cerr << "SerialWorker did not retain the last written command\n";
          result = 8;
        }
      }
    }

    const auto stop_started = Clock::now();
    worker.stop();
    if (result == 0 && Clock::now() - stop_started > std::chrono::milliseconds(250)) {
      std::cerr << "SerialWorker stop took too long\n";
      result = 7;
    }
  }

  if (result == 0) {
    std::cout << "SerialWorker smoke test passed\n";
  }
  return result;
}
