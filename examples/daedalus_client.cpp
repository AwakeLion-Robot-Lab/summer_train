#include "l1_sensor/simulator/daedalus_client.hpp"

#include <opencv2/highgui.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

std::atomic<bool> running{true};

void stopOnSignal(int)
{
  running = false;
}

struct Options {
  L1Sensor::DaedalusPaths paths;
  std::size_t frame_limit = 0;
  int connect_timeout_ms = 5000;
  bool show = false;
  bool send_command = false;
  float yaw_deg = 0.0F;
  float pitch_deg = 0.0F;
  float distance_m = 3.0F;
  bool fire = false;
};

std::string valueAfter(std::string_view argument, std::string_view prefix)
{
  if (!argument.starts_with(prefix)) {
    return {};
  }
  return std::string{argument.substr(prefix.size())};
}

Options parseOptions(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      std::cout
        << "Usage: daedalus_client [options]\n"
        << "  --show                    display the BGR camera stream\n"
        << "  --frames=N                stop after N frames (0 means forever)\n"
        << "  --yaw-deg=DEG             send absolute yaw command\n"
        << "  --pitch-deg=DEG           send absolute pitch command\n"
        << "  --distance-m=M            command target distance (-1 is invalid)\n"
        << "  --fire                    set fire_advice=1\n"
        << "  --connect-timeout-ms=MS   wait for simulator startup\n"
        << "  --metadata=PATH           override metadata mmap file\n"
        << "  --image-pool=PATH         override image-pool mmap file\n\n"
        << "Press F5 in Daedalus to enable remote auto aim; q/Esc exits --show.\n";
      std::exit(0);
    }

    if (argument == "--show") {
      options.show = true;
    } else if (argument == "--fire") {
      options.fire = true;
      options.send_command = true;
    } else if (const auto value = valueAfter(argument, "--frames="); !value.empty()) {
      options.frame_limit = std::stoull(value);
    } else if (const auto value = valueAfter(argument, "--yaw-deg="); !value.empty()) {
      options.yaw_deg = std::stof(value);
      options.send_command = true;
    } else if (const auto value = valueAfter(argument, "--pitch-deg="); !value.empty()) {
      options.pitch_deg = std::stof(value);
      options.send_command = true;
    } else if (const auto value = valueAfter(argument, "--distance-m="); !value.empty()) {
      options.distance_m = std::stof(value);
      options.send_command = true;
    } else if (const auto value = valueAfter(argument, "--connect-timeout-ms=");
               !value.empty()) {
      options.connect_timeout_ms = std::stoi(value);
    } else if (const auto value = valueAfter(argument, "--metadata="); !value.empty()) {
      options.paths.metadata = value;
    } else if (const auto value = valueAfter(argument, "--image-pool="); !value.empty()) {
      options.paths.image_pool = value;
    } else {
      throw std::invalid_argument("unknown or incomplete option: " + std::string{argument});
    }
  }

  if (options.connect_timeout_ms < 0) {
    throw std::invalid_argument("--connect-timeout-ms must not be negative");
  }
  return options;
}

bool connectWithRetry(
  L1Sensor::DaedalusClient& client,
  std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (client.connect() && client.isSimulatorAlive()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Options options = parseOptions(argc, argv);
    std::signal(SIGINT, stopOnSignal);
    std::signal(SIGTERM, stopOnSignal);

    L1Sensor::DaedalusClient client(options.paths);
    if (!connectWithRetry(
          client, std::chrono::milliseconds{options.connect_timeout_ms})) {
      const std::string error = client.lastError();
      std::cerr << "Cannot connect to a live Daedalus simulator: "
                << (error.empty() ? "shared-memory heartbeat is stale" : error)
                << '\n';
      return 1;
    }

    const auto camera = client.cameraInfo();
    std::cout << "Connected to Daedalus Talos IPC v2";
    if (camera.valid()) {
      std::cout << " | camera=" << camera.width << 'x' << camera.height
                << " fx=" << camera.fx << " fy=" << camera.fy;
    }
    std::cout << '\n';

    if (options.send_command) {
      std::cout << "Command: yaw=" << options.yaw_deg
                << " deg pitch=" << options.pitch_deg
                << " deg distance=" << options.distance_m
                << " m fire=" << std::boolalpha << options.fire
                << " (press F5 in the simulator to apply it)\n";
    }

    std::size_t received = 0;
    while (running && (options.frame_limit == 0 || received < options.frame_limit)) {
      L1Sensor::DaedalusFrame frame;
      if (!client.readFrame(frame, std::chrono::milliseconds{500})) {
        const std::string error = client.lastError();
        if (!error.empty()) {
          std::cerr << "Frame read failed: " << error << '\n';
          return 2;
        }
        if (!client.isSimulatorAlive()) {
          std::cerr << "Daedalus heartbeat stopped\n";
          return 3;
        }
        continue;
      }

      ++received;
      if (options.send_command && !client.sendGimbalCommand(
            options.yaw_deg,
            options.pitch_deg,
            options.distance_m,
            options.fire)) {
        std::cerr << "Command send failed: " << client.lastError() << '\n';
        return 4;
      }

      if (received == 1 || received % 30 == 0) {
        const auto& gimbal = frame.pose(L1Sensor::DaedalusPoseKind::Gimbal);
        std::cout << "frame=" << frame.sequence << " image="
                  << frame.image_bgr.cols << 'x' << frame.image_bgr.rows
                  << " gimbal_q=[" << std::fixed << std::setprecision(3)
                  << gimbal.quaternion[0] << ',' << gimbal.quaternion[1] << ','
                  << gimbal.quaternion[2] << ',' << gimbal.quaternion[3]
                  << "] auto_aim=" << std::boolalpha << frame.auto_aim_enabled
                  << '\n';
      }

      if (options.show) {
        cv::imshow("Daedalus camera", frame.image_bgr);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
          break;
        }
      }
    }

    if (options.show) {
      cv::destroyWindow("Daedalus camera");
    }
    std::cout << "Disconnected after " << received << " frame(s)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 64;
  }
}
