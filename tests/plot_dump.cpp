// 遥测落盘：监听 runtime 发给 PlotJuggler 的 UDP JSON，每个数据报原样写一行到
// <out-dir>/<月日_时分秒>.jsonl，同时转发到 forward 端口，PlotJuggler 改听那个
// 端口即可照常看曲线。PlotJuggler 的"保存布局"不含数据，离线分析靠这份文件。
//
// 刻意不挂进 AutoAimRuntime：落盘只是调试手段，主链路只管发 UDP。
// 不依赖相机和串口。用法（先开它，再开 auto_aim）：
//   xmake run plot_dump                        # 听 9870，转发 9871，写到 records/plot
//   xmake run plot_dump -- --forward=0         # 不转发
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/core/utility.hpp>

#include <array>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{port | 9870 | 监听端口，与 auto_aim.yaml 的 debug.plot_port 一致}"
  "{forward | 9871 | 原样转发到 127.0.0.1 的这个端口，0 表示不转发}"
  "{out-dir o | records/plot | 输出目录}";

std::atomic<bool> g_running{true};

void onSignal(int)
{
  g_running = false;
}

std::string timeStamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  std::ostringstream stream;
  stream << std::put_time(&local, "%m%d_%H%M%S");
  return stream.str();
}

sockaddr_in loopback(std::uint16_t port)
{
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  return address;
}

}  // namespace

int main(int argc, char** argv)
{
  cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
  if (cli.get<bool>("help")) {
    cli.printMessage();
    return 0;
  }
  const int port = cli.get<int>("port");
  const int forward = cli.get<int>("forward");
  const std::filesystem::path out_dir = cli.get<std::string>("out-dir");

  const int listen_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (listen_fd < 0) {
    std::cerr << "socket 失败\n";
    return 1;
  }
  // 听所有网卡：plot_host 指向别的机器（笔记本连 NUC）时也能收到。
  sockaddr_in listen_address = loopback(static_cast<std::uint16_t>(port));
  listen_address.sin_addr.s_addr = htonl(INADDR_ANY);
  if (::bind(listen_fd, reinterpret_cast<const sockaddr*>(&listen_address),
             sizeof(listen_address)) != 0) {
    std::cerr << "bind " << port << " 失败：PlotJuggler 是否还在听这个端口？"
              << "把它改听 " << forward << "\n";
    ::close(listen_fd);
    return 1;
  }

  const int forward_fd = forward > 0 ? ::socket(AF_INET, SOCK_DGRAM, 0) : -1;
  const sockaddr_in forward_address = loopback(static_cast<std::uint16_t>(forward));

  std::filesystem::create_directories(out_dir);
  const std::filesystem::path path = out_dir / (timeStamp() + ".jsonl");
  std::ofstream out(path);
  if (!out) {
    std::cerr << "无法写入 " << path << "\n";
    return 1;
  }
  std::cout << "听 " << port << "，写 " << path
            << (forward > 0 ? "，转发 " + std::to_string(forward) : std::string{})
            << "；Ctrl+C 结束\n";

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  // 与 UdpJsonSender::kMaxPayloadSize 一致，一个数据报就是一帧完整的 JSON。
  std::array<char, 65507> buffer{};
  std::uint64_t count = 0;
  pollfd poll_fd{listen_fd, POLLIN, 0};
  while (g_running) {
    // 带超时的 poll，Ctrl+C 后最多 200 ms 退出，不会卡在 recv 上。
    if (::poll(&poll_fd, 1, 200) <= 0) {
      continue;
    }
    const ssize_t size = ::recv(listen_fd, buffer.data(), buffer.size(), 0);
    if (size <= 0) {
      continue;
    }
    out.write(buffer.data(), size);
    out.put('\n');
    if (forward_fd >= 0) {
      (void)::sendto(forward_fd, buffer.data(), static_cast<std::size_t>(size), 0,
                     reinterpret_cast<const sockaddr*>(&forward_address),
                     sizeof(forward_address));
    }
    if (++count % 500 == 0) {
      out.flush();
      std::cout << "已收 " << count << " 帧\n";
    }
  }

  out.flush();
  std::cout << "共 " << count << " 帧，写入 " << path << "\n";
  if (forward_fd >= 0) {
    ::close(forward_fd);
  }
  ::close(listen_fd);
  return 0;
}
