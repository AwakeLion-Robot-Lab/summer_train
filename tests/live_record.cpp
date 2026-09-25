// 实机录像：只开相机和串口，把原始帧和曝光时刻的 IMU 姿态写成
// <out-dir>/<月日_时分秒>.avi + .txt，格式与 records/ 下的录像一致（txt 每行
// `t w x y z`，t 为相对首帧的秒数），track_diag / auto_aim_test 用默认的
// --convention=imu 直接回放。
//
// 刻意不挂进 AutoAimRuntime：录像只是离线复现现场的手段，不该出现在主链路里。
// 这里不跑检测，帧率会比实跑高；回放按 txt 的时间戳走，不受影响。
//
// 需要相机和串口硬件。用法：
//   xmake run live_record                          # 录到 records/live，q / Ctrl+C 结束
//   xmake run live_record -- --preview=false       # 无显示器（ssh）时关掉预览
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l6_telemetry/logger.hpp"

#include <Eigen/Geometry>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace {

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{camera-config | config/camera_config.yaml | 相机配置}"
  "{serial-config | config/serial_config.yaml | 串口配置，同时提供 R_imu_barrel}"
  "{out-dir o | records/live | 输出目录}"
  "{preview | true | 显示预览窗口；无显示器时设 false}"
  "{duration | 0 | 录多少秒后自动停，0 表示一直录}";

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
  std::ostringstream out;
  out << std::put_time(&local, "%m%d_%H%M%S");
  return out.str();
}

// 编码放在后台线程：1440x1080 的 MJPG 编码要几毫秒，放取图循环里会拖慢帧率、
// 让相机缓冲积压。队列满了就丢帧并计数；丢帧时 txt 同样不写那一行，avi 与
// txt 始终逐帧对齐。
class FrameRecorder
{
public:
  FrameRecorder(std::filesystem::path stem, std::size_t max_queue)
  : stem_(std::move(stem)), max_queue_(max_queue), text_(stem_.string() + ".txt")
  {
    text_ << std::fixed << std::setprecision(9);
    worker_ = std::thread([this] { loop(); });
  }

  ~FrameRecorder() { finish(); }

  // 排空队列、关闭 avi。之后 written() 才是最终帧数。
  void finish()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    video_.release();
  }

  bool ok() const { return text_.is_open() && !failed_; }

  void push(
    const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp,
    const Eigen::Quaterniond & q_world_imu)
  {
    if (!first_) {
      first_ = timestamp;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.size() >= max_queue_) {
        ++dropped_;
        return;
      }
    }
    Item item{
      frame.clone(), std::chrono::duration<double>(timestamp - *first_).count(),
      q_world_imu.normalized()};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  std::size_t written() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return written_;
  }

  std::size_t dropped() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
  }

private:
  struct Item
  {
    cv::Mat frame;
    double seconds{0.0};
    Eigen::Quaterniond q;
  };

  void loop()
  {
    while (true) {
      Item item;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) {
          return;  // stop_ 且已排空
        }
        item = std::move(queue_.front());
        queue_.pop_front();
      }

      // 帧尺寸要等第一帧才知道，所以 VideoWriter 在这里懒打开。
      if (!video_.isOpened()) {
        video_.open(
          stem_.string() + ".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 100.0,
          item.frame.size(), item.frame.channels() == 3);
        if (!video_.isOpened()) {
          L6Telemetry::logError("live_record: 无法创建", stem_.string() + ".avi");
          failed_ = true;
          std::lock_guard<std::mutex> lock(mutex_);
          queue_.clear();
          return;
        }
        video_.set(cv::VIDEOWRITER_PROP_QUALITY, 95);
      }

      video_.write(item.frame);
      text_ << item.seconds << ' ' << item.q.w() << ' ' << item.q.x() << ' ' << item.q.y()
            << ' ' << item.q.z() << '\n';
      // 逐行刷盘：进程被强杀时缓冲里的姿态会丢，回放就只剩 avi 没有 txt。
      text_.flush();
      std::lock_guard<std::mutex> lock(mutex_);
      ++written_;
    }
  }

  std::filesystem::path stem_;
  std::size_t max_queue_;
  std::optional<std::chrono::steady_clock::time_point> first_;
  cv::VideoWriter video_;
  std::ofstream text_;
  std::atomic<bool> failed_{false};

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  bool stop_{false};
  std::size_t written_{0};
  std::size_t dropped_{0};
  std::thread worker_;
};

}  // namespace

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
  if (cli.get<bool>("help")) {
    cli.printMessage();
    return 0;
  }
  const bool preview = cli.get<bool>("preview");
  const double duration = cli.get<double>("duration");

  L6Telemetry::initLogger();
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  L1Sensor::Camera camera(cli.get<std::string>("camera-config"));
  const auto serial_config = L1Sensor::loadSerialConfig(cli.get<std::string>("serial-config"));
  L1Sensor::SerialWorker serial(serial_config);
  if (!serial.start()) {
    std::cerr << "串口启动失败：没有云台姿态的帧回放时无法使用\n";
    L6Telemetry::flushLogger();
    return 1;
  }

  const std::filesystem::path dir = cli.get<std::string>("out-dir");
  std::filesystem::create_directories(dir);
  const std::filesystem::path stem = dir / timeStamp();
  std::size_t no_pose = 0;
  {
    FrameRecorder recorder(stem, 64);
    if (!recorder.ok()) {
      std::cerr << "无法创建 " << stem.string() << ".txt\n";
      return 1;
    }
    std::cout << "录到 " << stem.string() << ".avi/.txt，q 或 Ctrl+C 结束\n";
    if (preview) {
      cv::namedWindow("live_record", cv::WINDOW_NORMAL);
    }

    const auto start = std::chrono::steady_clock::now();
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
    struct Pending {
      cv::Mat frame;
      std::chrono::steady_clock::time_point timestamp;
    };
    std::deque<Pending> pending;
    while (g_running && recorder.ok()) {
      if (duration > 0.0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >
            duration) {
        break;
      }
      if (!camera.read(frame, timestamp)) {
        continue;
      }
      // 曝光时刻的姿态要晚 pose_delay_ms 才到。这里不能像 runtime 那样原地等，
      // 否则取图被卡住、帧率掉一半；先排队，姿态到齐的帧再写出。排队太久
      // （串口断流）的帧按边界值写出，不无限堆积。
      pending.push_back({frame.clone(), timestamp});
      while (!pending.empty()) {
        const auto& head = pending.front();
        const bool ready = serial.poseReady(head.timestamp);
        const bool stale = std::chrono::steady_clock::now() - head.timestamp >
                           std::chrono::milliseconds(200);
        if (!ready && !stale) {
          break;
        }
        // 录的是 IMU 约定下的姿态：gimbalPoseAt 已乘过 R_imu_barrel，这里乘回去，
        // 回放按 --convention=imu 再乘同一份，两边用的是 serial_config.yaml 里同一个数。
        const auto q_world_barrel = serial.gimbalPoseAt(head.timestamp);
        if (!q_world_barrel) {
          ++no_pose;
        } else {
          recorder.push(
            head.frame, head.timestamp,
            Eigen::Quaterniond(
              q_world_barrel->toRotationMatrix() * serial_config.R_imu_barrel.transpose()));
        }
        pending.pop_front();
      }

      if (preview) {
        cv::Mat shown = frame.clone();
        const std::string status = "rec " + std::to_string(recorder.written()) + "  drop " +
                                   std::to_string(recorder.dropped()) + "  no-pose " +
                                   std::to_string(no_pose);
        cv::putText(shown, status, {12, 36}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
        cv::imshow("live_record", shown);
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
          break;
        }
      }
    }
    recorder.finish();
    std::cout << "写入 " << recorder.written() << " 帧，丢帧 " << recorder.dropped()
              << "，缺姿态跳过 " << no_pose << '\n';
  }

  serial.stop();
  camera.stop();
  std::cout << stem.string() << '\n';
  L6Telemetry::flushLogger();
  return 0;
}
