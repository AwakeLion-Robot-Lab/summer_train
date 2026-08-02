#include "l6_telemetry/udp_json_sender.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace L6Telemetry {

UdpJsonSender::UdpJsonSender(std::string host, uint16_t port)
  : host_(std::move(host)), port_(port)
{
  if (host_.empty()) {
    throw std::invalid_argument("UDP JSON destination host must not be empty");
  }
  if (port_ == 0) {
    throw std::invalid_argument("UDP JSON destination port must not be zero");
  }

  destination_.sin_family = AF_INET;
  destination_.sin_port = htons(port_);
  if (::inet_pton(AF_INET, host_.c_str(), &destination_.sin_addr) != 1) {
    throw std::invalid_argument("UDP JSON destination must be a valid IPv4 address: " + host_);
  }

  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0) {
    throw std::system_error(errno, std::generic_category(), "failed to create UDP JSON socket");
  }
}

UdpJsonSender::~UdpJsonSender()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
  }
}

bool UdpJsonSender::send(const nlohmann::json& payload) noexcept
{
  std::string serialized;
  try {
    serialized = payload.dump();
  } catch (...) {
    return false;
  }

  if (serialized.size() > kMaxPayloadSize) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const auto sent = ::sendto(socket_fd_, serialized.data(), serialized.size(), 0,
                             reinterpret_cast<const sockaddr*>(&destination_),
                             sizeof(destination_));
  return sent >= 0 && static_cast<std::size_t>(sent) == serialized.size();
}

const std::string& UdpJsonSender::host() const noexcept
{
  return host_;
}

uint16_t UdpJsonSender::port() const noexcept
{
  return port_;
}

}  // namespace L6Telemetry
