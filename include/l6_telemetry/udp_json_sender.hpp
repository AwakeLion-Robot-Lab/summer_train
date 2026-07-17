#pragma once

#include <netinet/in.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

namespace L6Telemetry {

class UdpJsonSender {
public:
  static constexpr std::size_t kMaxPayloadSize = 65507;

  explicit UdpJsonSender(std::string host = "127.0.0.1", uint16_t port = 9870);
  ~UdpJsonSender();

  UdpJsonSender(const UdpJsonSender&) = delete;
  UdpJsonSender& operator=(const UdpJsonSender&) = delete;
  UdpJsonSender(UdpJsonSender&&) = delete;
  UdpJsonSender& operator=(UdpJsonSender&&) = delete;

  [[nodiscard]] bool send(const nlohmann::json& payload) noexcept;

  [[nodiscard]] const std::string& host() const noexcept;
  [[nodiscard]] uint16_t port() const noexcept;

private:
  int socket_fd_ = -1;
  sockaddr_in destination_{};
  std::string host_;
  uint16_t port_ = 0;
  std::mutex mutex_;
};

}  // namespace L6Telemetry
