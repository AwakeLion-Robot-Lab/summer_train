#include "l1_sensor/serial/serial_worker.hpp"

#include "l6_telemetry/math.hpp"

#include <Eigen/Geometry>

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

std::uint8_t crc8(std::span<const std::uint8_t> bytes)
{
  std::uint8_t crc = 0xFF;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
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
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x0001) != 0
              ? static_cast<std::uint16_t>((crc >> 1) ^ 0x8408)
              : static_cast<std::uint16_t>(crc >> 1);
    }
  }
  return crc;
}

// 构造一帧下位机状态，用于让 gimbal 历史里出现可查询的姿态。
std::vector<std::uint8_t> makeStatePacket(
  std::uint8_t seq, float roll, float pitch, float yaw)
{
  Protocol::RxPacket packet{};
  packet.frame_header.sof = 0xA0;
  packet.frame_header.data_length = sizeof(Protocol::RxPayload);
  packet.frame_header.seq = seq;
  packet.frame_header.cmd_id = 0x0002;
  packet.data.roll = roll;
  packet.data.pitch = pitch;
  packet.data.yaw = yaw;
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
  // 取一个非单位阵的合法右手旋转（绕 z 转 180 度），让轴向转换真正生效。
  config.R_imu_barrel = Eigen::Vector3d{-1.0, -1.0, 1.0}.asDiagonal();

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

      // gimbalPoseAt 必须返回 barrel -> world，即在下位机上报的姿态右乘一次
      // R_imu_barrel。roll 和 pitch 都取非零值，这样单边右乘、单边左乘和
      // 双边相似变换三种写法互不相同，写反了这里就会失败。
      if (result == 0) {
        const auto state_packet = makeStatePacket(0, 0.1F, 0.2F, 0.3F);
        if (::write(pty.masterFd(), state_packet.data(), state_packet.size()) !=
            static_cast<ssize_t>(state_packet.size())) {
          std::cerr << "Failed to inject a state packet\n";
          result = 9;
        }

        std::optional<L1Sensor::RobotState> state;
        const auto state_deadline = Clock::now() + std::chrono::milliseconds(500);
        while (result == 0 && Clock::now() < state_deadline) {
          state = worker.latestState();
          if (state) {
            break;
          }
          pollfd idle{pty.masterFd(), 0, 0};
          ::poll(&idle, 1, 5);
        }

        if (result == 0 && !state) {
          std::cerr << "SerialWorker did not parse the injected state\n";
          result = 10;
        }

        if (result == 0) {
          const auto pose = worker.gimbalPoseAt(Clock::now());
          if (!pose) {
            std::cerr << "SerialWorker returned no gimbal pose\n";
            result = 11;
          } else {
            // 用解析出来的 rpy 反推期望值，避免 float 精度参与比较。
            const Eigen::Matrix3d R_world_imu =
              L6Telemetry::rpyToQuaternion(
                state->rpy.roll, state->rpy.pitch, state->rpy.yaw)
                .toRotationMatrix();
            const Eigen::Matrix3d expected = R_world_imu * config.R_imu_barrel;
            if (!pose->toRotationMatrix().isApprox(expected, 1e-9)) {
              std::cerr
                << "SerialWorker composed R_imu_barrel the wrong way round\n";
              result = 12;
            }
          }
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
