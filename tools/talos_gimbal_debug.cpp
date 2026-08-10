// 云台角度实时显示工具。
//
// 读取 Daedalus 仿真器 Talos 共享内存中的云台姿态（PoseIndex::Gimbal），
// 在 OpenCV 窗口叠加显示相对角度与底盘 yaw，并在控制台按固定频率输出。
// 用途：在仿真器里手动控制枪口（方向键），观察云台角度读数，辅助瞄准坐标校准。
//
// 角度约定（以原点/中位为基准，从原点开始）：
//  - roll 正：沿前进方向右手定则，前轮侧（左侧）抬起；
//  - yaw 正：向左转（俯视逆时针）；
//  - pitch 正：低头。
// 启动时自动把当前姿态记录为原点（Z 键可重新记录），显示值 = 相对原点的
// 角度（ZYX 分解：R = Rz(yaw)Ry(pitch)Rx(roll)），原点处三者均为 0。
//
// 注意：
//  - 单消费者：不要与 talos_auto_aim / talos_shm_smoke 同时运行；
//  - 发布姿态是 ROS 系四元数，含模型自带的固定俯仰偏置（约 -25°）；
//    相对原点显示会自动消掉该偏置，原始四元数仍单独展示。

#include "l1_sensor/talos/talos_reader.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

const char* kCommandLineKeys =
  "{help h usage ? | | 显示命令行帮助}"
  "{shm-dir | /tmp | Talos 共享内存目录}"
  "{no-gui | | 关闭 OpenCV 界面}"
  "{print-hz | 5.0 | 控制台输出频率（Hz）}";

// 标准 ZYX 欧拉分解：R = Rz(yaw)Ry(pitch)Rx(roll)，yaw 绕 Z、pitch 绕 Y。
// 对纯 Ry 旋转返回干净分支（如 pitch=-25°），避免 Eigen eulerAngles
// 的等价表示（180°/-155°/180°）造成误解。
struct Rpy {
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

Rpy quatToRpyZyx(const float q[4])  // wxyz
{
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  Rpy out;
  out.yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  const double sp = 2.0 * (w * y - z * x);
  out.pitch = std::abs(sp) >= 1.0
                ? std::copysign(3.14159265358979323846 / 2.0, sp)
                : std::asin(sp);
  out.roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  return out;
}

void quatConjugate(const float q[4], float out[4])  // wxyz
{
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
}

void quatMultiply(const float a[4], const float b[4], float out[4])  // wxyz
{
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

struct Angles {
  double yaw = 0.0;     // rad
  double pitch = 0.0;   // rad
  double roll = 0.0;    // rad
  float quat[4]{1.0F, 0.0F, 0.0F, 0.0F};
};

std::string formatRelAngles(const Angles& value)
{
  std::ostringstream text;
  text << std::fixed << std::setprecision(1)
       << "rel(yaw=" << value.yaw * kRadToDeg
       << " pitch=" << value.pitch * kRadToDeg
       << " roll=" << value.roll * kRadToDeg << ")deg";
  return text.str();
}

int run(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, kCommandLineKeys);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  const std::string shm_dir = parser.get<std::string>("shm-dir");
  const bool gui_enabled = !parser.has("no-gui");
  const double print_hz = parser.get<double>("print-hz");
  if (!parser.check() || print_hz <= 0.0 || print_hz > 100.0) {
    parser.printErrors();
    return 2;
  }

  L1Sensor::talos::TalosReader reader{shm_dir};
  if (!reader.open()) {
    std::cerr << "talos shared memory not found in " << shm_dir
              << "; is the Daedalus simulator running?\n";
    return 2;
  }

  if (gui_enabled) {
    cv::namedWindow("talos gimbal debug", cv::WINDOW_NORMAL);
    cv::resizeWindow("talos gimbal debug", 960, 720);
  }

  float origin_quat[4]{1.0F, 0.0F, 0.0F, 0.0F};
  bool origin_set = false;
  bool running = true;
  auto last_print = Clock::now();
  const auto print_interval =
    std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>{1.0 / print_hz});

  while (running) {
    L1Sensor::talos::TalosFrame frame;
    if (!reader.readFrame(frame, std::chrono::milliseconds{100})) {
      if (gui_enabled) {
        const int key = cv::waitKey(10);
        if (key == 27 || key == 'q' || key == 'Q') {
          running = false;
        }
      }
      continue;
    }

    const auto gimbal = reader.pose(L1Sensor::talos::PoseIndex::Gimbal);
    const L1Sensor::talos::ChassisObservation chassis =
      reader.chassisObservation();
    if (!gimbal) {
      continue;
    }

    if (!origin_set) {
      for (int i = 0; i < 4; ++i) {
        origin_quat[i] = gimbal->quaternion[i];
      }
      origin_set = true;
      std::cout << "[gimbal] origin captured (neutral pose); "
                << "press Z to re-capture\n";
    }

    // 相对原点角度：origin⁻¹ * current，再按 ZYX 分解。
    float inv_origin[4];
    float rel_quat[4];
    quatConjugate(origin_quat, inv_origin);
    quatMultiply(inv_origin, gimbal->quaternion, rel_quat);
    const Rpy rel_rpy = quatToRpyZyx(rel_quat);
    Angles rel_angles;
    rel_angles.yaw = rel_rpy.yaw;
    rel_angles.pitch = rel_rpy.pitch;
    rel_angles.roll = rel_rpy.roll;
    for (int i = 0; i < 4; ++i) {
      rel_angles.quat[i] = rel_quat[i];
    }
    const double chassis_yaw = static_cast<double>(chassis.rpy_rad[2]);

    if (Clock::now() - last_print >= print_interval) {
      std::cout << std::fixed << std::setprecision(1)
                << "[gimbal] seq=" << frame.seq
                << " " << formatRelAngles(rel_angles)
                << " | chassis_yaw=" << chassis_yaw * kRadToDeg << "deg"
                << " | q=(" << std::setprecision(4)
                << gimbal->quaternion[0] << ','
                << gimbal->quaternion[1] << ','
                << gimbal->quaternion[2] << ','
                << gimbal->quaternion[3] << ")" << std::endl;
      last_print = Clock::now();
    }

    if (gui_enabled) {
      cv::Mat display;
      if (frame.rgb && frame.width == L1Sensor::talos::kImageWidth
          && frame.height == L1Sensor::talos::kImageHeight) {
        const cv::Mat rgb_view(
          static_cast<int>(frame.height), static_cast<int>(frame.width),
          CV_8UC3, const_cast<std::uint8_t*>(frame.rgb));
        cv::cvtColor(rgb_view, display, cv::COLOR_RGB2BGR);
      } else {
        display = cv::Mat::zeros(720, 1280, CV_8UC3);
      }

      std::ostringstream line1;
      line1 << "gimbal " << formatRelAngles(rel_angles)
            << (origin_set ? " (origin)" : "");
      std::ostringstream line2;
      line2 << std::fixed << std::setprecision(1)
            << "chassis_yaw=" << chassis_yaw * kRadToDeg << "deg"
            << "  q=(" << std::setprecision(4)
            << gimbal->quaternion[0] << ", "
            << gimbal->quaternion[1] << ", "
            << gimbal->quaternion[2] << ", "
            << gimbal->quaternion[3] << ")";
      const std::string hint = "Z: set origin  Q/Esc: quit";
      cv::putText(
        display, line1.str(), cv::Point(16, 42),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2, cv::LINE_AA);
      cv::putText(
        display, line2.str(), cv::Point(16, 78),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 1, cv::LINE_AA);
      cv::putText(
        display, hint, cv::Point(16, display.rows - 16),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
      cv::imshow("talos gimbal debug", display);

      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        running = false;
      } else if (key == 'z' || key == 'Z') {
        for (int i = 0; i < 4; ++i) {
          origin_quat[i] = gimbal->quaternion[i];
        }
        origin_set = true;
        std::cout << "[gimbal] origin re-captured\n";
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  }

  if (gui_enabled) {
    cv::destroyWindow("talos gimbal debug");
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "talos gimbal debug failed: " << error.what() << '\n';
    return 1;
  }
}
