#include "mindvision.hpp"

#if defined(NEWVISION_ENABLE_LIBUSB) && __has_include(<libusb-1.0/libusb.h>)
#include <libusb-1.0/libusb.h>
#define NEWVISION_HAS_LIBUSB 1
#else
#define NEWVISION_HAS_LIBUSB 0
#endif

#include <stdexcept>
#include "l6_telemetry/logger.hpp"



using namespace std::chrono_literals;

namespace io
{
MindVision::MindVision(double exposure_ms, double gamma, const std::string & vid_pid)
: exposure_ms_(exposure_ms),
  gamma_(gamma),
  vid_(-1),
  pid_(-1)
{
  set_vid_pid(vid_pid);
#if NEWVISION_HAS_LIBUSB
  if (libusb_init(NULL)) L6Telemetry::logWarn("Unable to init libusb!");
#else
  L6Telemetry::logWarn("libusb header not found; USB reset is disabled.");
#endif

  try_open();

  // 守护线程独占重连、关闭和重开；关闭 SDK 前始终先等待采集线程退出。
  daemon_thread_ = std::thread{[this] {
    while (!stop_requested_.load()) {
      std::this_thread::sleep_for(100ms);

      if (stop_requested_.load() || healthy_.load()) {
        continue;
      }

      if (capture_thread_.joinable()) {
        capture_thread_.join();
      }

      close();
      if (stop_requested_.load()) {
        break;
      }
      reset_usb();
      if (stop_requested_.load()) {
        break;
      }
      try_open();
    }
  }};
}

MindVision::~MindVision()
{
  stop();
  if (daemon_thread_.joinable()) {
    daemon_thread_.join();
  }
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  close();
  L6Telemetry::logInfo("Mindvision destructed.");
}

bool MindVision::read(
  cv::Mat & img,
  std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
  CameraData data;
  const auto result = buffer_.readFor(data, timeout);
  if (result != tools::LatestBuffer<CameraData>::ReadStatus::Value) {
    img.release();
    timestamp = {};
    return false;
  }

  img = data.img;
  timestamp = data.timestamp;
  return true;
}

void MindVision::stop()
{
  stop_requested_.store(true);
  healthy_.store(false);
  buffer_.close();
}

void MindVision::open()
{
  if (stop_requested_.load()) {
    return;
  }

  int camera_num = 1;
  tSdkCameraDevInfo camera_info_list;
  tSdkCameraCapbility camera_capbility;
  CameraSdkInit(1);
  CameraEnumerateDevice(&camera_info_list, &camera_num);

  if (camera_num == 0) throw std::runtime_error("Not found camera!");

  if (CameraInit(&camera_info_list, -1, -1, &handle_) != CAMERA_STATUS_SUCCESS)
    throw std::runtime_error("Failed to init camera!");

  CameraGetCapability(handle_, &camera_capbility);
  width_ = camera_capbility.sResolutionRange.iWidthMax;
  height_ = camera_capbility.sResolutionRange.iHeightMax;

  CameraSetAeState(handle_, FALSE);                        // 关闭自动曝光
  CameraSetExposureTime(handle_, exposure_ms_ * 1e3);      // 设置曝光
  CameraSetGamma(handle_, gamma_ * 1e2);                   // 设置伽马
  CameraSetIspOutFormat(handle_, CAMERA_MEDIA_TYPE_BGR8);  // 设置输出格式为BGR
  CameraSetTriggerMode(handle_, 0);                        // 设置为连续采集模式
  CameraSetFrameSpeed(handle_, 1);                         // 设置为低帧率模式

  CameraPlay(handle_);

  // 取图线程
  capture_thread_ = std::thread{[this] {
    tSdkFrameHead head;
    BYTE * raw = nullptr;

    healthy_.store(true);
    while (!stop_requested_.load()) {
      std::this_thread::sleep_for(1ms);

      auto img = cv::Mat(height_, width_, CV_8UC3);

      auto status = CameraGetImageBuffer(handle_, &head, &raw, 100);
      auto timestamp = std::chrono::steady_clock::now();

      if (status != CAMERA_STATUS_SUCCESS) {
        if (!stop_requested_.load()) {
          L6Telemetry::logWarn("Camera dropped!");
        }
        healthy_.store(false);
        break;
      }

      const auto process_status = CameraImageProcess(handle_, raw, img.data, &head);
      const auto release_status = CameraReleaseImageBuffer(handle_, raw);
      raw = nullptr;
      if (process_status != CAMERA_STATUS_SUCCESS || release_status != CAMERA_STATUS_SUCCESS) {
        L6Telemetry::logWarn("MindVision image process or release failed.");
        healthy_.store(false);
        break;
      }

      if (!buffer_.write({img, timestamp})) {
        break;
      }
    }

    healthy_.store(false);
  }};

  L6Telemetry::logInfo("Mindvision opened.");
}

void MindVision::try_open()
{
  if (stop_requested_.load()) {
    return;
  }

  try {
    open();
  } catch (const std::exception & e) {
    healthy_.store(false);
    close();
    L6Telemetry::logWarn("{}", e.what());
  }
}

void MindVision::close()
{
  if (handle_ == -1) return;
  CameraUnInit(handle_);
  handle_ = -1;
  width_ = 0;
  height_ = 0;
}

void MindVision::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    L6Telemetry::logWarn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    L6Telemetry::logWarn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void MindVision::reset_usb() const
{
#if NEWVISION_HAS_LIBUSB
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    L6Telemetry::logWarn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    L6Telemetry::logWarn("Unable to reset usb!");
  else
    L6Telemetry::logInfo("Reset usb successfully :)");

  libusb_close(handle);
#else
  L6Telemetry::logWarn("USB reset skipped because libusb support is not available.");
#endif
}

}  // namespace io
