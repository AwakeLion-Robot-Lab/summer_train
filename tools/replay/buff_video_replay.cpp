// Buff 离线视频回放工具。
//
// 该入口只读取录像，在内存中运行 L2～L5 链路；不会初始化相机、串口，也不会
// 下发控制命令。即使 buff.yaml 打开了 shoot_enable，这里仍会强制关闭实际开火，
// 画面上的 SHOOT READY 仅表示源算法在该帧满足开火时序。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/robot_state.hpp"
#include "l6_telemetry/buff/overlay.hpp"
#include "runtime/buff/buff_config.hpp"
#include "runtime/buff/buff_runtime.hpp"
#include "vision_tools/replay/frame_timeline.hpp"

#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 YAML，支持本工程和 ROS camera_info 格式}"
  "{buff-config b | config/buff.yaml | Buff 参数 YAML}"
  "{enemy | blue | 敌方颜色：red / blue}"
  "{wait w | 1 | all 模式每帧最少等待毫秒数，0 表示从首帧单步}"
  "{start-index s | 0 | 起始帧下标}"
  "{end-index e | 0 | 结束帧下标，0 表示到视频末尾}"
  "{output o | | 导出标注视频；留空表示不导出}"
  "{overwrite | false | 允许覆盖已存在的导出视频}"
  "{show-original | false | 同步显示未绘制标注的原始视频窗口}"
  "{playback-mode | realtime | all 逐帧处理；realtime 落后时跳帧追赶源视频时间}"
  "{timing-source | auto | auto 优先读取同名 TXT 采集时间；video 使用 AVI FPS；pose 强制 TXT}"
  "{sync-playback | true | all 模式检测足够快时按源视频 FPS 控制播放节奏}"
  "{headless | false | 不创建窗口，只统计或导出视频}"
  "{@input | | 输入视频路径}";
//xmake run buff_video_replay --calibration=/home/rm/FosuVision-main/src/ros2_hik_camera-main/config/camera_info.yaml --enemy=red /home/rm/FosuVision-main/src/rm_auto_aim/armor_detector/test/buffVideo.avi

void require(bool condition, const std::string& message)
{
  if (!condition) throw std::runtime_error(message);
}

L1Sensor::EnemyColor parseEnemyColor(const std::string& text)
{
  if (text == "red") return L1Sensor::EnemyColor::Red;
  if (text == "blue") return L1Sensor::EnemyColor::Blue;
  throw std::runtime_error("enemy 必须是 red 或 blue");
}

enum class PlaybackMode { All, Realtime };

PlaybackMode parsePlaybackMode(const std::string& text)
{
  if (text == "all") return PlaybackMode::All;
  if (text == "realtime") return PlaybackMode::Realtime;
  throw std::runtime_error("playback-mode 必须是 all 或 realtime");
}

// ROS camera_info.yaml 把矩阵写成 {rows, cols, data}，本工程则使用二维数组。
// 回放工具同时支持两者，方便直接检查其他工程留下的录像和标定。
YAML::Node rosMatrixToSequence(const YAML::Node& matrix,
                               int expected_rows,
                               int expected_cols,
                               const std::string& name)
{
  require(matrix && matrix.IsMap(), name + " 必须是 YAML map");
  const int rows = matrix["rows"] ? matrix["rows"].as<int>() : expected_rows;
  const int cols = matrix["cols"] ? matrix["cols"].as<int>() : expected_cols;
  const YAML::Node data = matrix["data"];
  require(rows == expected_rows && cols == expected_cols,
          name + " 尺寸不符合预期");
  require(data && data.IsSequence() &&
            data.size() == static_cast<std::size_t>(rows * cols),
          name + ".data 数量不正确");

  YAML::Node result(YAML::NodeType::Sequence);
  for (int row = 0; row < rows; ++row) {
    YAML::Node values(YAML::NodeType::Sequence);
    for (int col = 0; col < cols; ++col) {
      values.push_back(data[static_cast<std::size_t>(row * cols + col)].as<double>());
    }
    result.push_back(values);
  }
  return result;
}

L1Sensor::CameraCalibration loadReplayCalibration(const std::string& path)
{
  const YAML::Node root = YAML::LoadFile(path);
  if (root["calibration"]) {
    return L1Sensor::loadCameraCalibration(root["calibration"], path);
  }

  require(root["image_width"] && root["image_height"],
          path + " 既没有 calibration 节点，也不是 ROS camera_info 格式");
  YAML::Node calibration(YAML::NodeType::Map);
  calibration["image_width"] = root["image_width"].as<int>();
  calibration["image_height"] = root["image_height"].as<int>();
  calibration["camera_matrix"] =
    rosMatrixToSequence(root["camera_matrix"], 3, 3, "camera_matrix");

  const YAML::Node distortion = root["distortion_coefficients"];
  require(distortion && distortion.IsMap() && distortion["data"] &&
            distortion["data"].IsSequence(),
          "distortion_coefficients.data 缺失");
  YAML::Node coefficients(YAML::NodeType::Sequence);
  for (const auto& value : distortion["data"]) {
    coefficients.push_back(value.as<double>());
  }
  calibration["distortion_coefficients"] = coefficients;
  return L1Sensor::loadCameraCalibration(calibration, path);
}

Eigen::Matrix3d rosToOpenCv()
{
  Eigen::Matrix3d matrix;
  matrix << 0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
            1.0, 0.0, 0.0;
  return matrix;
}

std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_camera,
  const Eigen::Vector3d& t_world_camera)
{
  const Eigen::Vector3d camera_ros =
    q_world_camera.conjugate() * (world - t_world_camera);
  const Eigen::Vector3d camera_cv = rosToOpenCv() * camera_ros;
  if (!camera_cv.allFinite() || camera_cv.z() <= 1e-6) return std::nullopt;

  const std::vector<cv::Point3d> points{{camera_cv.x(), camera_cv.y(), camera_cv.z()}};
  std::vector<cv::Point2d> projected;
  cv::projectPoints(points, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0),
                    calibration.camera_matrix,
                    calibration.distortion_coefficients, projected);
  if (projected.empty() || !std::isfinite(projected.front().x) ||
      !std::isfinite(projected.front().y)) {
    return std::nullopt;
  }
  return cv::Point2f{static_cast<float>(projected.front().x),
                     static_cast<float>(projected.front().y)};
}

void putTextLine(cv::Mat& image, const std::string& text, int row,
                 const cv::Scalar& color = {245, 245, 245})
{
  const cv::Point origin{14, 26 + row * 22};
  // 黑色描边保证文字在红灯和白色顶灯上仍可读。
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar{0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              color, 1, cv::LINE_AA);
}

void drawReplayOverlay(cv::Mat& image,
                       const runtime::buff::Result& result,
                       const L1Sensor::CameraCalibration& calibration,
                       std::uint64_t frame_index,
                       double video_seconds,
                       double detect_ms,
                       double update_ms)
{
  L6Telemetry::Buff::drawOverlay(
    image, result.detections, result.addition, result.plan);

  const Eigen::Quaterniond q_world_camera = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d t_world_camera = Eigen::Vector3d::Zero();
  if (result.plan.valid) {
    const auto center = projectWorldPoint(
      result.plan.center, calibration, q_world_camera, t_world_camera);
    const auto attack = projectWorldPoint(
      result.plan.attack, calibration, q_world_camera, t_world_camera);
    if (center) {
      cv::drawMarker(image, *center, cv::Scalar{255, 0, 255},
                     cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    }
    if (attack) {
      cv::drawMarker(image, *attack, cv::Scalar{0, 255, 0},
                     cv::MARKER_TILTED_CROSS, 18, 2, cv::LINE_AA);
    }
    if (center && attack) {
      cv::line(image, *center, *attack, cv::Scalar{180, 180, 180}, 1,
               cv::LINE_AA);
    }

    if (result.plan.attack_box_valid) {
      std::vector<cv::Point> box;
      box.reserve(result.plan.attack_box.size());
      for (const auto& corner : result.plan.attack_box) {
        const auto projected = projectWorldPoint(
          corner, calibration, q_world_camera, t_world_camera);
        if (!projected) {
          box.clear();
          break;
        }
        box.emplace_back(cvRound(projected->x), cvRound(projected->y));
      }
      if (box.size() == result.plan.attack_box.size()) {
        const cv::Scalar box_color = result.plan.algorithm_shoot
          ? cv::Scalar{0, 255, 0} : cv::Scalar{0, 165, 255};
        cv::polylines(image, box, true, box_color, 3, cv::LINE_AA);
        const auto label = *std::min_element(
          box.begin(), box.end(), [](const cv::Point& lhs, const cv::Point& rhs) {
            return lhs.y < rhs.y;
          });
        cv::putText(image, "L4 PLAN", label + cv::Point{4, -6},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar{0, 0, 0},
                    3, cv::LINE_AA);
        cv::putText(image, "L4 PLAN", label + cv::Point{4, -6},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1,
                    cv::LINE_AA);
      }
    }
  }

  // 原 overlay 的状态占用了左上角第一行，回放信息从第二行开始。
  putTextLine(image,
              cv::format("frame=%llu  t=%.3fs  icon=%zu  target=%zu",
                         static_cast<unsigned long long>(frame_index), video_seconds,
                         result.detections.icons.size(),
                         result.detections.bullseyes.size()),
              1);
  putTextLine(image,
              cv::format("tracked=%zu  predicted=%zu  detect=%.2fms  update=%.2fms",
                         result.addition.tracked.size(),
                         result.addition.predicted.size(), detect_ms, update_ms),
              2);
  if (result.plan.valid) {
    putTextLine(image,
                cv::format("yaw=%+.2fdeg pitch=%+.2fdeg fly=%.1fms blade=%d speed=%.2f(%s)",
                           result.plan.yaw * kRadToDeg,
                           result.plan.pitch * kRadToDeg,
                           result.plan.fly_time * 1000.0,
                           result.plan.blade_index,
                           result.plan.bullet_speed.value,
                           result.plan.bullet_speed.measured ? "input" : "fallback"),
                3);
  } else {
    putTextLine(image, "yaw=--  pitch=--  fly=--  blade=--", 3,
                cv::Scalar{120, 120, 255});
  }
  putTextLine(image,
              result.plan.algorithm_shoot
                ? "SHOOT READY (diagnostic only; serial disabled)"
                : "NO SHOOT (offline replay; serial disabled)",
              4, result.plan.algorithm_shoot ? cv::Scalar{80, 255, 80}
                                             : cv::Scalar{120, 220, 255});
  putTextLine(image,
              "yellow=icon  red/green=target  cyan=pred  white=track  orange/green=L4 plan",
              5, cv::Scalar{210, 210, 210});
  putTextLine(image, "keys: SPACE pause  N step  A/D +/-1s  J/L +/-5s  R restart  Q quit",
              6, cv::Scalar{210, 210, 210});
  putTextLine(image, "FIXED CAMERA POSE - no camera, no serial, no command output",
              7, cv::Scalar{80, 180, 255});
}

int outputFourcc(const std::filesystem::path& path)
{
  const std::string extension = path.extension().string();
  if (extension == ".mp4" || extension == ".m4v" || extension == ".mov") {
    return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  }
  return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
}

struct Statistics {
  std::uint64_t processed{0};
  std::uint64_t skipped{0};
  std::uint64_t observation_frames{0};
  std::uint64_t model_frames{0};
  std::uint64_t model_detections{0};
  std::uint64_t paired_frames{0};
  std::uint64_t plan_frames{0};
  std::uint64_t plan_box_frames{0};
  std::uint64_t degraded_frames{0};
  std::uint64_t shoot_ready_frames{0};
  int first_plan_box_frame{-1};
  int last_plan_box_frame{-1};
  double detect_ms{0.0};
  double update_ms{0.0};
  double decode_ms{0.0};
  double overlay_ms{0.0};
  double write_ms{0.0};
};

double percent(std::uint64_t value, std::uint64_t total)
{
  return total == 0 ? 0.0 : 100.0 * static_cast<double>(value) /
                                static_cast<double>(total);
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
    if (cli.get<bool>("help")) {
      cli.printMessage();
      return 0;
    }
    if (!cli.check()) {
      cli.printErrors();
      return 1;
    }

    const std::filesystem::path input{cli.get<std::string>("@input")};
    const std::filesystem::path output{cli.get<std::string>("output")};
    const std::string calibration_path = cli.get<std::string>("calibration");
    const std::string buff_config_path = cli.get<std::string>("buff-config");
    const L1Sensor::EnemyColor enemy_color =
      parseEnemyColor(cli.get<std::string>("enemy"));
    const int wait_ms = cli.get<int>("wait");
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const bool headless = cli.get<bool>("headless");
    const bool overwrite = cli.get<bool>("overwrite");
    const bool show_original = cli.get<bool>("show-original");
    const bool sync_playback = cli.get<bool>("sync-playback");
    const PlaybackMode playback_mode =
      parsePlaybackMode(cli.get<std::string>("playback-mode"));
    const vision_tools::TimingSource timing_source =
      vision_tools::parseTimingSource(cli.get<std::string>("timing-source"));

    require(!input.empty(), "必须提供输入视频路径");
    require(wait_ms >= 0, "wait 不能为负数");
    require(start_index >= 0, "start-index 不能为负数");
    require(end_index == 0 || end_index >= start_index,
            "end-index 必须为 0 或不小于 start-index");
    require(std::filesystem::exists(input), "输入视频不存在：" + input.string());

    if (!output.empty()) {
      const auto input_absolute = std::filesystem::absolute(input).lexically_normal();
      const auto output_absolute = std::filesystem::absolute(output).lexically_normal();
      require(input_absolute != output_absolute, "导出路径不能覆盖输入视频");
      require(overwrite || !std::filesystem::exists(output),
              "导出文件已存在；换一个路径或显式添加 --overwrite=true");
      if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
      }
    }

    const L1Sensor::CameraCalibration calibration =
      loadReplayCalibration(calibration_path);
    runtime::buff::Config config = runtime::buff::loadConfig(buff_config_path);
    // 离线工具永远不继承实机的开火许可。仍保留 plan.algorithm_shoot，便于
    // 检查原算法何时认为可以开火，但 Controller 输出会被强制压成 false。
    config.control.shoot_enable = false;
    runtime::buff::Runtime runtime(calibration, config);

    cv::VideoCapture video(input.string());
    require(video.isOpened(), "无法打开视频：" + input.string());
    double video_fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(video_fps) || video_fps <= 0.0) video_fps = 30.0;
    const int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
    const vision_tools::FrameTimeline timeline =
      vision_tools::loadFrameTimeline(input, video_fps, total_frames, timing_source);
    require(timeline.capture_seconds.empty() ||
              start_index < static_cast<int>(timeline.capture_seconds.size()),
            "start-index 超出录制时间戳范围");
    const cv::Size video_size{
      static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
      static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))};
    require(calibration.matchesImageSize(video_size),
            cv::format("标定分辨率 %dx%d 与视频 %dx%d 不一致",
                       calibration.image_size.width, calibration.image_size.height,
                       video_size.width, video_size.height));

    video.set(cv::CAP_PROP_POS_FRAMES, start_index);
    int frame_index = start_index;
    bool paused = !headless && wait_ms == 0;
    bool quit = false;
    cv::VideoWriter writer;
    cv::Mat last_canvas;
    Statistics statistics;
    const auto epoch = std::chrono::steady_clock::time_point{};
    auto playback_origin = std::chrono::steady_clock::now();
    int playback_origin_frame = start_index;
    const auto playbackDeadline = [&](int index) {
      return playback_origin +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>{
            timeline.secondsAt(index) - timeline.secondsAt(playback_origin_frame)});
    };

    std::cout << "input=" << input << "  resolution=" << video_size.width << 'x'
              << video_size.height << "  fps=" << std::fixed << std::setprecision(3)
              << video_fps << "  frames=" << total_frames
              << "  timing=" << (timeline.capture_seconds.empty() ? "video" : "pose")
              << "  capture_fps=" << timeline.fps() << '\n';
    std::cout << "enemy=" << cli.get<std::string>("enemy")
              << "  calibration=" << calibration_path
              << "  buff_config=" << buff_config_path
              << "  playback_mode="
              << (playback_mode == PlaybackMode::Realtime ? "realtime" : "all") << '\n';
    std::cout << "offline safety: camera disabled, serial disabled, shoot output forced false\n";
    if (!headless) {
      std::cout << "keys: SPACE pause/resume, N step, A/D +/-1s, J/L +/-5s, "
                   "R restart, Q/ESC quit\n";
      cv::namedWindow("buff video replay", cv::WINDOW_NORMAL);
      if (show_original) {
        cv::namedWindow("buff original", cv::WINDOW_NORMAL);
      }
    }

    const auto replay_begin = std::chrono::steady_clock::now();
    playback_origin = replay_begin;
    // GUI 暂停期间不重复解码同一帧；只有明确单步或恢复播放才进入下一轮。
    while (!quit) {
      if (end_index > 0 && frame_index > end_index) break;
      if (playback_mode == PlaybackMode::Realtime && !paused) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_seconds =
          std::chrono::duration<double>(now - playback_origin).count();
        int target = timeline.frameAt(
          timeline.secondsAt(playback_origin_frame) +
          std::max(0.0, elapsed_seconds));
        if (total_frames > 0) target = std::min(target, total_frames - 1);
        if (end_index > 0) target = std::min(target, end_index);
        while (frame_index < target) {
          if (!video.grab()) {
            quit = true;
            break;
          }
          ++frame_index;
          ++statistics.skipped;
          // 被跳过的帧沿用上一张标注画面，导出视频仍保持源时间轴长度。
          if (writer.isOpened() && !last_canvas.empty()) {
            const auto write_begin = std::chrono::steady_clock::now();
            writer.write(last_canvas);
            statistics.write_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - write_begin).count();
          }
        }
        if (quit) break;
        if (headless && frame_index > target) {
          std::this_thread::sleep_until(playbackDeadline(frame_index));
        }
      }
      cv::Mat frame;
      const auto decode_begin = std::chrono::steady_clock::now();
      if (!video.read(frame) || frame.empty()) break;
      statistics.decode_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decode_begin).count();

      const double seconds = timeline.secondsAt(frame_index);
      const auto timestamp = epoch +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>{seconds});

      const auto detect_begin = std::chrono::steady_clock::now();
      auto detections = runtime.detect(frame, enemy_color);
      statistics.model_frames += !detections.model_detections.empty();
      statistics.model_detections += detections.model_detections.size();
      const double detect_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - detect_begin).count();
      const bool has_observation =
        !detections.icons.empty() || !detections.bullseyes.empty();
      const bool has_pair =
        !detections.icons.empty() && !detections.bullseyes.empty();

      const auto update_begin = std::chrono::steady_clock::now();
      const auto result = runtime.update(
        std::move(detections), Eigen::Quaterniond::Identity(),
        Eigen::Vector3d::Zero(), timestamp, timestamp,
        config.planner.fallback_bullet_speed);
      const double update_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - update_begin).count();

      ++statistics.processed;
      statistics.observation_frames += has_observation;
      statistics.paired_frames += has_pair;
      statistics.plan_frames += result.plan.valid;
      statistics.plan_box_frames += result.plan.attack_box_valid;
      if (result.plan.attack_box_valid) {
        if (statistics.first_plan_box_frame < 0) {
          statistics.first_plan_box_frame = frame_index;
        }
        statistics.last_plan_box_frame = frame_index;
      }
      statistics.degraded_frames += result.plan.valid && result.plan.degraded;
      statistics.shoot_ready_frames += result.plan.algorithm_shoot;
      statistics.detect_ms += detect_ms;
      statistics.update_ms += update_ms;

      cv::Mat canvas;
      if (!headless || !output.empty()) {
        const auto overlay_begin = std::chrono::steady_clock::now();
        canvas = frame.clone();
        drawReplayOverlay(canvas, result, calibration,
                          static_cast<std::uint64_t>(frame_index), seconds,
                          detect_ms, update_ms);
        statistics.overlay_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - overlay_begin).count();
      }

      if (!output.empty()) {
        if (!writer.isOpened()) {
          require(writer.open(output.string(), outputFourcc(output), timeline.fps(),
                              canvas.size(), true),
                  "无法创建导出视频：" + output.string());
        }
        const auto write_begin = std::chrono::steady_clock::now();
        writer.write(canvas);
        statistics.write_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - write_begin).count();
        if (playback_mode == PlaybackMode::Realtime) last_canvas = canvas;
      }

      if (headless) {
        if (statistics.processed % 500 == 0) {
          std::cout << "processed=" << statistics.processed
                    << " frame=" << frame_index << '\n';
        }
        ++frame_index;
        continue;
      }

      // 两个窗口共享同一次解码得到的 frame，只在同一个事件循环中刷新，因此
      // 暂停、单步和跳转后始终是同一帧，不会出现原片与叠加层错一帧。
      if (show_original) cv::imshow("buff original", frame);
      cv::imshow("buff video replay", canvas);
      bool advance = false;
      bool seeked = false;
      while (!advance && !seeked && !quit) {
        int delay_ms = std::max(wait_ms, 1);
        if (!paused && playback_mode == PlaybackMode::Realtime) {
          const auto deadline = playbackDeadline(frame_index + 1);
          const double remaining_ms = std::chrono::duration<double, std::milli>(
            deadline - std::chrono::steady_clock::now()).count();
          delay_ms = remaining_ms > 0.0
            ? std::max(1, static_cast<int>(std::ceil(remaining_ms))) : 1;
        } else if (!paused && sync_playback) {
          const double remaining_ms = std::chrono::duration<double, std::milli>(
            playbackDeadline(frame_index + 1) -
            std::chrono::steady_clock::now()).count();
          if (remaining_ms > 0.0) {
            delay_ms = std::max(delay_ms,
                                static_cast<int>(std::ceil(remaining_ms)));
          }
        }
        const int key = cv::waitKeyEx(paused ? 0 : delay_ms);
        if (key < 0) {
          advance = true;
          continue;
        }
        const int ascii = key & 0xff;
        if (ascii == 'q' || key == 27) {
          quit = true;
        } else if (ascii == ' ') {
          paused = !paused;
          if (!paused) {
            // 暂停时间不能算进视频时钟，否则恢复时会误判落后数秒。
            playback_origin = std::chrono::steady_clock::now();
            playback_origin_frame = frame_index + 1;
            advance = true;
          }
        } else if (ascii == 'n' || ascii == '.') {
          advance = true;
        } else {
          int target = frame_index;
          if (ascii == 'a') target = timeline.frameAt(seconds - 1.0);
          else if (ascii == 'd') target = timeline.frameAt(seconds + 1.0);
          else if (ascii == 'j') target = timeline.frameAt(seconds - 5.0);
          else if (ascii == 'l') target = timeline.frameAt(seconds + 5.0);
          else if (ascii == 'r') target = start_index;
          else continue;

          const int upper = end_index > 0
            ? end_index
            : (total_frames > 0 ? total_frames - 1 : target);
          target = std::clamp(target, start_index, std::max(start_index, upper));
          video.set(cv::CAP_PROP_POS_FRAMES, target);
          runtime.reset();
          frame_index = target;
          paused = true;
          last_canvas.release();
          playback_origin = std::chrono::steady_clock::now();
          playback_origin_frame = target;
          seeked = true;
        }
      }
      if (!seeked) ++frame_index;
    }

    const double replay_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - replay_begin).count();

    writer.release();
    if (!headless) {
      cv::destroyWindow("buff video replay");
      if (show_original) cv::destroyWindow("buff original");
    }

    std::cout << std::fixed << std::setprecision(3)
              << "processed=" << statistics.processed
              << " skipped=" << statistics.skipped
              << " wall_fps="
              << (replay_seconds > 0.0 ? statistics.processed / replay_seconds : 0.0)
              << " model_frames=" << statistics.model_frames
              << " model_detections=" << statistics.model_detections
              << " observation=" << statistics.observation_frames
              << " (" << percent(statistics.observation_frames, statistics.processed) << "%)"
              << " paired=" << statistics.paired_frames
              << " (" << percent(statistics.paired_frames, statistics.processed) << "%)"
              << " plan=" << statistics.plan_frames
              << " (" << percent(statistics.plan_frames, statistics.processed) << "%)"
              << " plan_box=" << statistics.plan_box_frames
              << " plan_box_range=[" << statistics.first_plan_box_frame
              << ',' << statistics.last_plan_box_frame << ']'
              << " degraded=" << statistics.degraded_frames
              << " shoot_ready=" << statistics.shoot_ready_frames
              << " avg_detect_ms="
              << (statistics.processed ? statistics.detect_ms / statistics.processed : 0.0)
              << " avg_update_ms="
              << (statistics.processed ? statistics.update_ms / statistics.processed : 0.0)
              << " avg_decode_ms="
              << (statistics.processed ? statistics.decode_ms / statistics.processed : 0.0)
              << " avg_overlay_ms="
              << (statistics.processed ? statistics.overlay_ms / statistics.processed : 0.0)
              << " total_write_ms=" << statistics.write_ms
              << '\n';
    if (!output.empty()) std::cout << "annotated_video=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "buff_video_replay: " << error.what() << '\n';
    return 1;
  }
}
