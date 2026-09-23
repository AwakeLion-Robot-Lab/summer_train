#include "hikrobot.hpp"

#if __has_include(<libusb-1.0/libusb.h>)
#include <libusb-1.0/libusb.h>
#define NEWVISION_HAS_LIBUSB 1
#else
#define NEWVISION_HAS_LIBUSB 0
#endif

#include <unordered_map>

#include "l6_telemetry/logger.hpp"

using namespace std::chrono_literals;

namespace io {
HikRobot::HikRobot(double exposure_ms, double gain, const std::string &vid_pid)
    : exposure_us_(exposure_ms * 1e3), gain_(gain), daemon_quit_(false),
      capturing_(false), capture_quit_(false), handle_(nullptr), vid_(-1),
      pid_(-1) {
  set_vid_pid(vid_pid);
#if NEWVISION_HAS_LIBUSB
  if (libusb_init(NULL))
    L6Telemetry::logWarn("Unable to init libusb!");
#else
  L6Telemetry::logWarn("libusb header not found; USB reset is disabled.");
#endif

  daemon_thread_ = std::thread{[this] {
    L6Telemetry::logInfo("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_)
        continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    L6Telemetry::logInfo("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot() {
  stop();
  if (daemon_thread_.joinable()) {
    daemon_thread_.join();
  }
  L6Telemetry::logInfo("HikRobot destructed.");
}

bool HikRobot::read(cv::Mat &img,
                    std::chrono::steady_clock::time_point &timestamp,
                    std::chrono::milliseconds timeout) {
  CameraData data;
  const auto result = buffer.readFor(data, timeout);
  if (result != tools::LatestBuffer<CameraData>::ReadStatus::Value) {
    img.release();
    timestamp = {};
    return false;
  }

  img = data.img;
  timestamp = data.timestamp;
  return true;
}

void HikRobot::stop() {
  daemon_quit_.store(true);
  capture_quit_.store(true);
  buffer.close();
}

void HikRobot::capture_start() {
  if (daemon_quit_.load()) {
    return;
  }

  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_EnumDevices failed: {:#x}", ret);
    return;
  }

  if (device_list.nDeviceNum == 0) {
    L6Telemetry::logWarn("Not found camera!");
    return;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_CreateHandle failed: {:#x}", ret);
    handle_ = nullptr;
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_OpenDevice failed: {:#x}", ret);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return;
  }

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 165);

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  capture_thread_ = std::thread{[this] {
    L6Telemetry::logInfo("HikRobot's capture thread started.");

    capturing_ = true;

    MV_FRAME_OUT raw{};

    while (!capture_quit_.load()) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 10;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        if (!capture_quit_.load()) {
          L6Telemetry::logWarn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        }
        break;
      }

      // 曝光中点而不是到达时刻。CLAUDE.md 的跨层契约写的是"A frame's
      // timestamp is its exposure instant"，而 MV_CC_GetImageBuffer 返回时
      // 整帧已经传完了，直接 now() 会把整个曝光段算进时间戳，下游
      // SerialWorker::gimbalPoseAt() 就会在 IMU 历史里查到偏晚的姿态。
      //
      // 调研过的 12 份开源里只有 awakening 做了这件事
      // （src/utils/drivers/{hik,mv,daheng}_camera 三个驱动都是
      // `frame.timestamp = current_time - half_exposure`）；sp_vision、
      // Climber、jlu、rmcs 全部直接取到达时刻。这里照 awakening 的做法。
      //
      // 仍未补偿的是读出和 USB 传输耗时，它们同样让时间戳偏晚，但既不是常量
      // 也无法从 SDK 问出来。海康的 stFrameInfo.nHostTimeStamp 本来是更好的
      // 来源，jlu 试过又退回去了（hikrobot.cpp:194 "硬件时间戳似乎两帧才更新
      // 一次，有点怪"），所以这里不用它。
      const auto half_exposure = std::chrono::microseconds(
          static_cast<long long>(exposure_us_ / 2.0));
      auto timestamp = std::chrono::steady_clock::now() - half_exposure;
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight),
                  CV_8U, raw.pBufAddr);

      const auto &frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      // L1 对上层统一输出 OpenCV 的 BGR。MindVision 同样配置为 BGR，L2 因此不需要
      // 根据相机品牌猜测通道顺序，也不会把红蓝装甲板识别反。
      //
      // 这张表**不是**同名对应，改成同名会让红蓝整个对调：
      // GenICam/海康的 BayerRG8 按传感器左上角 2x2 命名，即 RGGB；而 OpenCV 的
      // COLOR_BayerXY2BGR 按**第二行的第二、三列**命名，两套命名整整错开一位。
      // RGGB 的第二行是 G B G B，取下标 1、2 得 "BG"，所以 BayerRG8 必须配
      // COLOR_BayerBG2BGR。合成图实测：RGGB 传感器拍纯红时，同名的
      // COLOR_BayerRG2BGR 输出 BGR=(255,0,0) 也就是蓝色，四种格式全部如此。
      // 红蓝对调会直接反转敌我过滤，属于静默的致命错误，改这里前先跑合成图验证。
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes>
          type_map = {{PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGB2BGR},
                      {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerBG2BGR},
                      {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGR2BGR},
                      {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerRG2BGR}};
      const auto conversion = type_map.find(pixel_type);
      if (conversion == type_map.end()) {
        L6Telemetry::logWarn("Unsupported HikRobot pixel type",
                             static_cast<int>(pixel_type));
        MV_CC_FreeImageBuffer(handle_, &raw);
        break;
      }

      try {
        cv::cvtColor(img, dst_image, conversion->second);
      } catch (const cv::Exception &e) {
        L6Telemetry::logWarn("HikRobot color conversion failed", e.what());
        MV_CC_FreeImageBuffer(handle_, &raw);
        break;
      }

      img = dst_image;

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        L6Telemetry::logWarn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }

      if (!buffer.write({img, timestamp})) {
        break;
      }
    }

    capturing_ = false;
    L6Telemetry::logInfo("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop() {
  capture_quit_ = true;
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  capturing_ = false;

  if (handle_ == nullptr) {
    return;
  }

  auto *handle = handle_;
  handle_ = nullptr;

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_StopGrabbing failed: {:#x}", ret);
  }

  ret = MV_CC_CloseDevice(handle);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_CloseDevice failed: {:#x}", ret);
  }

  ret = MV_CC_DestroyHandle(handle);
  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_DestroyHandle failed: {:#x}", ret);
  }
}

void HikRobot::set_float_value(const std::string &name, double value) {
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name,
                         value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string &name, unsigned int value) {
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name,
                         value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string &vid_pid) {
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

void HikRobot::reset_usb() const {
#if NEWVISION_HAS_LIBUSB
  if (vid_ == -1 || pid_ == -1)
    return;

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
  L6Telemetry::logWarn(
      "USB reset skipped because libusb support is not available.");
#endif
}

} // namespace io
