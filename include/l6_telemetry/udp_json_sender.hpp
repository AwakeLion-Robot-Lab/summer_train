#pragma once

#include <netinet/in.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

namespace L6Telemetry {

/**
 * @brief 把 JSON 以 UDP 报文发给 PlotJuggler
 *
 * UDP 无连接，没人接收也不阻塞。
 */
class UdpJsonSender {
public:
  /// 单个 UDP 报文的最大载荷，字节
  static constexpr std::size_t kMaxPayloadSize = 65507;

  /**
   * @brief 创建 socket 并解析目标地址
   * @param host 目标 IP
   * @param port 目标端口
   */
  explicit UdpJsonSender(std::string host = "127.0.0.1", uint16_t port = 9870);

  /**
   * @brief 关闭 socket
   */
  ~UdpJsonSender();

  UdpJsonSender(const UdpJsonSender&) = delete;
  UdpJsonSender& operator=(const UdpJsonSender&) = delete;
  UdpJsonSender(UdpJsonSender&&) = delete;
  UdpJsonSender& operator=(UdpJsonSender&&) = delete;

  /**
   * @brief 序列化并发送一个 JSON
   * @param payload 要发送的 JSON
   * @return 序列化失败、超过 kMaxPayloadSize 或没发完整时返回 false
   */
  [[nodiscard]] bool send(const nlohmann::json& payload) noexcept;

  /**
   * @brief 目标 IP
   * @return IP 字符串
   */
  const std::string& host() const noexcept;

  /**
   * @brief 目标端口
   * @return 端口号
   */
  uint16_t port() const noexcept;

private:
  /// UDP socket，创建失败时为 -1
  int socket_fd_ = -1;

  /// 目标地址
  sockaddr_in destination_{};

  /// 目标 IP
  std::string host_;

  /// 目标端口
  uint16_t port_ = 0;

  /// 串行化 send()
  std::mutex mutex_;
};

}  // namespace L6Telemetry
