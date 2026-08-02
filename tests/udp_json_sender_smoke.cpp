#include "l6_telemetry/udp_json_sender.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>

namespace {

class SocketHandle {
public:
  explicit SocketHandle(int fd) : fd_(fd) {}
  ~SocketHandle()
  {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  SocketHandle(const SocketHandle&) = delete;
  SocketHandle& operator=(const SocketHandle&) = delete;

  [[nodiscard]] int get() const { return fd_; }

private:
  int fd_;
};

}  // namespace

int main()
{
  const SocketHandle receiver(::socket(AF_INET, SOCK_DGRAM, 0));
  if (receiver.get() < 0) {
    std::cerr << "Failed to create UDP receiver\n";
    return 1;
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(receiver.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) {
    std::cerr << "Failed to bind UDP receiver\n";
    return 2;
  }

  socklen_t address_length = sizeof(address);
  if (::getsockname(receiver.get(), reinterpret_cast<sockaddr*>(&address), &address_length) < 0) {
    std::cerr << "Failed to query UDP receiver port\n";
    return 3;
  }

  const timeval timeout{1, 0};
  if (::setsockopt(receiver.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    std::cerr << "Failed to configure UDP receive timeout\n";
    return 4;
  }

  const auto port = ntohs(address.sin_port);
  L6Telemetry::UdpJsonSender sender("127.0.0.1", port);
  if (sender.host() != "127.0.0.1" || sender.port() != port) {
    std::cerr << "UDP sender endpoint mismatch\n";
    return 5;
  }

  const nlohmann::json expected = {
    {"t", 1.25},
    {"gimbal_yaw", -3.5},
    {"tracking", true},
    {"target", {{"x", 1.0}, {"y", 2.0}, {"z", 3.0}}},
  };
  if (!sender.send(expected)) {
    std::cerr << "Failed to send UDP JSON payload\n";
    return 6;
  }

  std::array<char, 4096> buffer{};
  const auto received = ::recv(receiver.get(), buffer.data(), buffer.size(), 0);
  if (received <= 0) {
    std::cerr << "Failed to receive UDP JSON payload\n";
    return 7;
  }

  try {
    const auto actual = nlohmann::json::parse(buffer.data(), buffer.data() + received);
    if (actual != expected) {
      std::cerr << "Received JSON payload mismatch\n";
      return 8;
    }
  } catch (const std::exception& error) {
    std::cerr << "Failed to parse received JSON payload: " << error.what() << '\n';
    return 9;
  }

  std::cout << "UDP JSON sender smoke test passed\n";
  return 0;
}
