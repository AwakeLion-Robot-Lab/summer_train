#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l6_telemetry/logger.hpp"

#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

// 真实下位机串口烟雾测试。
//
// 执行顺序：
// 1. 从 YAML 读取串口参数并启动 SerialWorker；
// 2. 从零开始递增 yaw/pitch，并始终禁止开火，直到按 Ctrl+C 停止；
// 3. 观察下位机是否持续回传合法 RobotState；
// 4. 根据收发计数判断链路是否基本正常。
//
// 注意：当前协议没有下位机 ACK。本测试中的 tx 只表示数据已完整写入视觉端串口，
// rx 表示已收到并解析下位机状态帧；若要证明下位机执行了某条命令，需要固件回传 ACK 或 seq。
namespace {

// 每 100 ms 改变一次待发送角度；发送线程仍按 config.tx_rate_hz 发送最新值。
constexpr auto kCommandStepInterval = std::chrono::milliseconds{100};

// 这是通信可读性测试值，不用于控制电机：下位机日志会依次看到
// yaw/pitch 为 0、1、2 ... 10。达到 10 后回到 0，循环继续，既方便观察，
// 也避免长时间运行时数值无限增大。
constexpr double kYawStepInProtocolUnit = 1.0;
constexpr double kPitchStepInProtocolUnit = 1.0;
constexpr double kMaxCommandAngleInProtocolUnit = 10.0;

// 第二个可选参数可设置自动结束秒数；0（默认）表示一直运行到 Ctrl+C。
bool parseDuration(const char* value, int& duration_seconds)
{
  try {
    duration_seconds = std::stoi(value);
  } catch (const std::exception&) {
    return false;
  }

  return duration_seconds >= 0;
}

// 信号处理函数只设置标志；主线程负责停止 worker、输出统计和释放串口。
volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int)
{
  stop_requested = 1;
}

}  // namespace

int main(int argc, char* argv[])
{
  // 可选参数：配置文件路径、自动停止秒数（0 表示不限制时间）。
  const std::string config_path =
    argc >= 2 ? argv[1] : "config/serial_config.yaml";
  int duration_seconds = 0;
  if (argc >= 3 && !parseDuration(argv[2], duration_seconds)) {
    std::cerr << "Usage: serial_hardware_smoke [config_path] [duration_seconds: 0=until Ctrl-C]\n";
    return 1;
  }
  if (argc > 3) {
    std::cerr << "Usage: serial_hardware_smoke [config_path] [duration_seconds: 0=until Ctrl-C]\n";
    return 1;
  }

  // 初始化日志后读取配置；配置中的 device、baud_rate 等会直接用于真实串口。
  L6Telemetry::initLogger();
  const auto config = L1Sensor::loadSerialConfig(config_path);
  if (!config.enable) {
    std::cerr << "Serial is disabled in " << config_path << "\n";
    return 2;
  }

  std::cout << "Starting hardware serial smoke test on " << config.device;
  if (duration_seconds == 0) {
    std::cout << " until Ctrl+C.\n";
  } else {
    std::cout << " for " << duration_seconds << " seconds.\n";
  }
  std::cout << "A repeating no-fire communication pattern is sent: yaw/pitch 0 -> "
            << kMaxCommandAngleInProtocolUnit << " -> 0, shoot=false.\n";

  // Ctrl+C / SIGTERM 会令循环自然退出，以便 stop() 正常回收收发线程。
  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);

  // SerialWorker 内部会启动一个接收线程和一个发送线程，并在断开时尝试重连。
  L1Sensor::SerialWorker serial(config);
  if (!serial.start()) {
    std::cerr << "SerialWorker failed to start\n";
    return 3;
  }

  // 第一个命令为 {yaw=0, pitch=0, shoot=false}；随后 yaw/pitch 小步递增。
  // 无论数值如何变化，shoot 始终为 false，测试不会请求发射。
  L5Control::SerialCommand test_command{};
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);
  auto next_step = std::chrono::steady_clock::now();

  while (stop_requested == 0 &&
         (duration_seconds == 0 || std::chrono::steady_clock::now() < deadline)) {
    // 每 10 ms 续期一次当前命令。真正写串口由 SerialWorker 的发送线程负责。
    serial.updateCommand(test_command);

    const auto now = std::chrono::steady_clock::now();
    if (now >= next_step) {
      // [TX queued] 表示刚交给 SerialWorker 的命令；[TX written] 表示最近一次
      // 已完整写入视觉端串口的命令。二者不同可帮助定位发送线程或串口的问题。
      std::cout << "[TX queued] yaw=" << test_command.yaw
                << " pitch=" << test_command.pitch
                << " shoot=" << std::boolalpha << test_command.shoot << std::noboolalpha;
      if (const auto sent_command = serial.latestSentCommand()) {
        std::cout << " | [TX written] yaw=" << sent_command->yaw
                  << " pitch=" << sent_command->pitch
                  << " shoot=" << std::boolalpha << sent_command->shoot << std::noboolalpha;
      } else {
        std::cout << " | [TX written] none";
      }
      std::cout << " | worker: tx=" << serial.sentCommandCount()
                << " rx=" << serial.receivedStateCount()
                << " tx_failed=" << serial.failedCommandCount()
                << " rx_dropped=" << serial.droppedPacketCount();

      // tx：视觉端完整写入的控制帧数；rx：成功解析的下位机状态帧数。
      // tx_failed：写串口失败/不完整次数；rx_dropped：按接收 seq 推断的丢包数。
      if (const auto state = serial.latestState()) {
        // latestState() 是 optional：没有收到状态时不能解引用。
        std::cout << " | [RX] mode=" << L1Sensor::toString(state->mode)
                  << " roll=" << state->rpy.roll
                  << " yaw=" << state->rpy.yaw
                  << " pitch=" << state->rpy.pitch
                  << " bullet_speed=" << state->bullet_speed
                  << " heat=" << state->heat
                  << " enemy_color=" << L1Sensor::toString(state->enemy_color);
      } else {
        std::cout << " | [RX] state=none";
      }
      std::cout << '\n';

      // 本次先发当前值（首帧为 0），再计算下一台阶；达到 10 后从 0 重新开始。
      test_command.yaw += kYawStepInProtocolUnit;
      if (test_command.yaw > kMaxCommandAngleInProtocolUnit) {
        test_command.yaw = 0.0;
      }
      test_command.pitch += kPitchStepInProtocolUnit;
      if (test_command.pitch > kMaxCommandAngleInProtocolUnit) {
        test_command.pitch = 0.0;
      }
      next_step = now + kCommandStepInterval;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (stop_requested != 0) {
    std::cout << "Stop signal received; shutting down serial worker.\n";
  }

  // 先停止并 join 收发线程，再读取最终统计，避免线程仍在更新计数。
  serial.stop();

  const auto sent = serial.sentCommandCount();
  const auto received = serial.receivedStateCount();
  const auto failed = serial.failedCommandCount();
  const auto dropped = serial.droppedPacketCount();
  std::cout << "Serial hardware smoke summary: tx=" << sent
            << " rx=" << received
            << " tx_failed=" << failed
            << " rx_dropped=" << dropped << '\n';
  L6Telemetry::flushLogger();

  // 三个失败条件分别对应：视觉端没写出、没收到下位机回传、写入中途失败。
  if (sent == 0) {
    std::cerr << "No complete control frame was written; check device path and permissions.\n";
    return 4;
  }
  if (received == 0) {
    std::cerr << "No valid state frame was received; check lower-controller TX and protocol settings.\n";
    return 5;
  }
  if (failed != 0) {
    std::cerr << "At least one control frame write failed.\n";
    return 6;
  }

  std::cout << "Serial hardware smoke test passed.\n";
  return 0;
}
