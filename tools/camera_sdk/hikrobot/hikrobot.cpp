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

namespace io
{
HikRobot::HikRobot(double exposure_ms, double gain, const std::string & vid_pid)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  daemon_quit_(false),
  capturing_(false),
  capture_quit_(false),
  handle_(nullptr),
  vid_(-1),
  pid_(-1)
{
  set_vid_pid(vid_pid);
#if NEWVISION_HAS_LIBUSB
  if (libusb_init(NULL)) L6Telemetry::logWarn("Unable to init libusb!");
#else
  L6Telemetry::logWarn("libusb header not found; USB reset is disabled.");
#endif

  daemon_thread_ = std::thread{[this] {
    
    L6Telemetry::logInfo("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    L6Telemetry::logInfo("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot()
{
  stop();
  if (daemon_thread_.joinable()) {
    daemon_thread_.join();
  }
  L6Telemetry::logInfo("HikRobot destructed.");
}

bool HikRobot::read(
  cv::Mat & img,
  std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
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

void HikRobot::stop()
{
  daemon_quit_.store(true);
  capture_quit_.store(true);
  buffer.close();
}

void HikRobot::capture_start()
{
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

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      const auto conversion = type_map.find(pixel_type);
      if (conversion == type_map.end()) {
        L6Telemetry::logWarn("Unsupported HikRobot pixel type", static_cast<int>(pixel_type));
        MV_CC_FreeImageBuffer(handle_, &raw);
        break;
      }

      try {
        cv::cvtColor(img, dst_image, conversion->second);
      } catch (const cv::Exception& e) {
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

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  capturing_ = false;

  if (handle_ == nullptr) {
    return;
  }

  auto* handle = handle_;
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

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    L6Telemetry::logWarn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
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

void HikRobot::reset_usb() const
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
