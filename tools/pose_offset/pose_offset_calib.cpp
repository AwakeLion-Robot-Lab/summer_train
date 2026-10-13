// 标定图像时间戳与云台姿态时间戳之间的固定偏移，结果填进
// config/serial_config.yaml 的 pose_offset_ms。原理见 pose_offset.hpp，用法见 README.md。
//
// 要相机、下位机串口和一块静止装甲板。程序只读姿态、不发任何控制命令，云台由
// 操作手用遥控器来回甩；采完只打印结果，不改配置文件。没有窗口，ssh 上能跑。
//
// 扫的是相对原始时间戳的绝对偏移，直接查 gimbalPoseAt(图像时刻 + τ)，不经过
// 配置里现有的 pose_offset_ms，所以结果与当前配置无关，可以直接覆盖。

#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"
#include "tools/pose_offset/pose_offset.hpp"

#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

const char* kKeys =
  "{help h usage ? |     | Show command line help}"
  "{duration d     | 20  | Capture seconds}"
  "{range r        | 30  | Scan offsets in [-range, range] ms}"
  "{step s         | 0.5 | Offset step in ms}"
  "{csv            |     | Also write the cost curve to this CSV}";

// 判可信的经验门限。
// 视线角速度标准差低于它，说明云台没有来回甩，偏移基本不可观。
constexpr double kMinRateStd = 0.5;
// 前后两半各自拟合的最优值差超过它，说明目标动过或检测不稳。
constexpr double kMaxHalfGapMs = 2.0;
constexpr int kMinFrames = 200;

Clock::duration ms(double value)
{
  return std::chrono::duration_cast<Clock::duration>(
    std::chrono::duration<double, std::milli>(value));
}

// 等到时间戳晚于 time 的那包姿态，最多等 budget。waitPose 一次只等 pose_wait_ms，
// 扫描范围比它宽时要多等几轮。
bool waitUntil(const L1Sensor::SerialWorker& serial, Clock::time_point time, Clock::duration budget)
{
  const auto deadline = Clock::now() + budget;
  while (Clock::now() < deadline) {
    if (serial.waitPose(time)) {
      return true;
    }
    // pose_wait_ms 配成 0 时 waitPose 立即返回，别空转。
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

struct Counts {
  int frames{0};
  int no_armor{0};
  int no_pnp{0};
  int no_pose{0};
};

void report(const std::vector<double>& offsets, const tools::OffsetFit& fit, double step)
{
  const auto [low, high] = std::minmax_element(fit.cost.begin(), fit.cost.end());
  const auto deg = [](double cost) { return std::sqrt(cost) * 180.0 / std::numbers::pi; };

  std::printf("\n  offset_ms  spread_deg\n");
  const auto stride = std::max<std::size_t>(1, static_cast<std::size_t>(std::lround(2.0 / step)));
  for (std::size_t k = 0; k < offsets.size(); k += stride) {
    const double ratio = *high > *low ? (fit.cost[k] - *low) / (*high - *low) : 0.0;
    std::printf(
      "  %9.1f  %10.4f  %s\n", offsets[k], deg(fit.cost[k]),
      std::string(static_cast<std::size_t>(std::lround(ratio * 40.0)), '#').c_str());
  }

  std::printf("\n样本 %d 帧，视线角速度标准差 %.2f rad/s，最优处剩余离散 %.3f°\n",
    fit.frames, fit.rate_std, fit.floor_deg);
  std::printf("最优偏移 %.1f ms（前一半 %.1f ms，后一半 %.1f ms）\n",
    fit.best_ms, fit.first_ms, fit.second_ms);
}

int run(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, kKeys);
  parser.about("Calibrate the image-to-gimbal-pose timestamp offset (pose_offset_ms).");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  const double duration = parser.get<double>("duration");
  const double range = parser.get<double>("range");
  const double step = parser.get<double>("step");
  const std::string csv = parser.get<std::string>("csv");
  if (!parser.check()) {
    parser.printErrors();
    return 2;
  }
  if (!(duration > 0.0 && range > 0.0 && step > 0.0 && step <= range)) {
    LOG_ERROR("need duration > 0 and 0 < step <= range");
    return 2;
  }

  std::vector<double> offsets;
  for (double tau = -range; tau <= range + 1e-9; tau += step) {
    offsets.push_back(tau);
  }

  const auto config = runtime::loadConfig("config/auto_aim.yaml");
  L1Sensor::Camera camera{"config/camera_config.yaml"};
  const auto& calibration = camera.calibration();
  // 没有外参就没有枪管系位置，不拿单位阵顶替。
  if (!calibration || !calibration->T_barrel_camera) {
    LOG_ERROR("camera calibration or T_barrel_camera missing");
    return 1;
  }
  const auto detector = runtime::makeDetector(config);
  L3Estimation::PnpSolver pnp{*calibration, config.armor};
  if (!pnp.ready()) {
    LOG_ERROR("pnp solver not ready, check calibration and armor config");
    return 1;
  }
  // single_pnp 要求先给枪管姿态才肯解。这里只取它的相机系结果，转到枪管系由
  // 下面显式乘外参完成，所以给单位阵不影响结果。
  pnp.set_R_world_barrel(Eigen::Quaterniond::Identity());

  const auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial{serial_config};
  if (!serial.start()) {
    LOG_ERROR("serial worker failed to start");
    return 1;
  }

  LOG_INFO(
    "pose offset capture started", "seconds", duration, "range_ms", range, "step_ms", step,
    "configured_offset_ms", serial_config.pose_offset_ms);

  std::vector<tools::OffsetFrame> frames;
  Counts counts;
  cv::Mat image;
  Clock::time_point timestamp;
  const auto start = Clock::now();
  auto last_log = start;
  while (Clock::now() - start < ms(duration * 1000.0)) {
    if (!camera.read(image, timestamp)) {
      continue;
    }
    ++counts.frames;

    // 视野里只该有一块板；万一检出多块，取面积最大的那块。
    const auto armors = detector.detect(image);
    if (armors.empty()) {
      ++counts.no_armor;
      continue;
    }
    const auto& largest = *std::max_element(
      armors.begin(), armors.end(), [](const auto& a, const auto& b) {
        return L6Telemetry::polygonArea(a.corners) < L6Telemetry::polygonArea(b.corners);
      });
    auto armor = L3Estimation::toObservation(largest);
    pnp.single_pnp(armor);
    if (armor.name == L3Estimation::ArmorName::Unknown) {
      ++counts.no_pnp;
      continue;
    }
    const Eigen::Vector3d in_barrel = *calibration->T_barrel_camera * armor.xyz_in_camera;

    // 整个扫描窗口都要落在姿态历史里：右端等它到齐，左端确认没滚出去，否则
    // gimbalPoseAt 会静默返回边界值。
    const auto first = timestamp + ms(-range);
    const auto last = timestamp + ms(range);
    const bool arrived = waitUntil(serial, last, ms(range + 100.0));
    const auto span = serial.poseSpan();
    if (!arrived || !span || span->first > first || span->second < last) {
      ++counts.no_pose;
      continue;
    }

    tools::OffsetFrame frame;
    frame.reserve(offsets.size());
    for (const double tau : offsets) {
      const auto pose = serial.gimbalPoseAt(timestamp + ms(tau));
      if (!pose) {
        break;
      }
      frame.push_back(pose->toRotationMatrix() * in_barrel);
    }
    if (frame.size() != offsets.size()) {
      ++counts.no_pose;
      continue;
    }
    frames.push_back(std::move(frame));

    if (Clock::now() - last_log > std::chrono::seconds{2}) {
      last_log = Clock::now();
      LOG_INFO(
        "pose offset capture", "used", frames.size(), "frames", counts.frames, "no_armor",
        counts.no_armor, "no_pnp", counts.no_pnp, "no_pose", counts.no_pose);
    }
  }
  serial.stop();
  camera.stop();

  std::printf(
    "\n读到 %d 帧：可用 %zu，没检出板 %d，PnP 失败 %d，姿态不全 %d\n", counts.frames,
    frames.size(), counts.no_armor, counts.no_pnp, counts.no_pose);

  const auto fit = tools::fitOffset(offsets, frames);
  if (!fit) {
    std::printf("可用样本太少，无法拟合。确认板在视野里、能被检出。\n");
    return 1;
  }
  report(offsets, *fit, step);

  if (!csv.empty()) {
    std::ofstream out{csv};
    out << "offset_ms,cost_rad2\n";
    for (std::size_t k = 0; k < offsets.size(); ++k) {
      out << offsets[k] << ',' << fit->cost[k] << '\n';
    }
  }

  std::vector<std::string> problems;
  if (fit->frames < kMinFrames) {
    problems.push_back("样本不足 " + std::to_string(kMinFrames) + " 帧，加长 -d");
  }
  if (fit->rate_std < kMinRateStd) {
    problems.push_back("视线角速度起伏太小：云台要来回甩，不能不动或匀速单向转");
  }
  if (fit->edge) {
    problems.push_back("最小值落在扫描边界，加大 -r");
  }
  if (std::abs(fit->first_ms - fit->second_ms) > kMaxHalfGapMs) {
    problems.push_back("前后两半不一致：目标可能动过，或检出不稳定");
  }

  if (!problems.empty()) {
    std::printf("\n结果不可信，不要写进配置：\n");
    for (const auto& problem : problems) {
      std::printf("  - %s\n", problem.c_str());
    }
    return 3;
  }
  std::printf("\n写进 config/serial_config.yaml：\npose_offset_ms: %.1f\n", fit->best_ms);
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  L6Telemetry::initLogger();
  try {
    const int result = run(argc, argv);
    L6Telemetry::flushLogger();
    return result;
  } catch (const std::exception& error) {
    LOG_ERROR("pose offset calibration failed", error.what());
    L6Telemetry::flushLogger();
    return 1;
  }
}
