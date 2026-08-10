#include "l1_sensor/camera/talos_camera.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace L1Sensor {
namespace {

CameraCalibration calibrationFromShm(const talos::CameraInfo& info)
{
  CameraCalibration calibration;
  calibration.image_size =
    cv::Size(static_cast<int>(info.width), static_cast<int>(info.height));
  calibration.camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
  calibration.camera_matrix.at<double>(0, 0) = info.fx;
  calibration.camera_matrix.at<double>(1, 1) = info.fy;
  calibration.camera_matrix.at<double>(0, 2) = info.cx;
  calibration.camera_matrix.at<double>(1, 2) = info.cy;
  calibration.camera_matrix.at<double>(2, 2) = 1.0;
  calibration.distortion_coefficients = cv::Mat::zeros(5, 1, CV_64F);
  if (info.distortion[0] != 0.0 || info.distortion[1] != 0.0
      || info.distortion[2] != 0.0 || info.distortion[3] != 0.0
      || info.distortion[4] != 0.0) {
    for (int i = 0; i < 5; ++i) {
      calibration.distortion_coefficients.at<double>(i) = info.distortion[i];
    }
  }
  // 仿真器相机挂在云台上，默认相机→枪管为单位外参；配置可覆盖。
  calibration.T_barrel_camera = Eigen::Isometry3d::Identity();
  return calibration;
}

}  // namespace

TalosCamera::TalosCamera(const std::string& config_path)
  : config_(talos::loadTalosSimConfig(config_path))
{
  reader_ = std::make_shared<talos::TalosReader>(config_.shm_dir);
  if (!reader_->open()) {
    throw std::runtime_error(
      "TalosCamera: cannot open shared memory in " + config_.shm_dir
      + " (is the Daedalus simulator running?)");
  }
  buildCalibration();
}

TalosCamera::TalosCamera(
  std::shared_ptr<talos::TalosReader> reader,
  talos::TalosSimConfig config)
  : reader_(std::move(reader)), config_(std::move(config))
{
  if (!reader_ || !reader_->isOpen()) {
    throw std::runtime_error("TalosCamera: reader is not open");
  }
  buildCalibration();
}

void TalosCamera::buildCalibration()
{
  if (config_.calibration) {
    calibration_ = config_.calibration;
    return;
  }
  calibration_ = calibrationFromShm(reader_->cameraInfo());
}

bool TalosCamera::read(
  cv::Mat& img,
  std::chrono::steady_clock::time_point& timestamp,
  std::chrono::milliseconds timeout)
{
  talos::TalosFrame frame;
  if (!reader_->readFrame(frame, timeout)) {
    return false;
  }
  if (!frame.rgb
      || frame.width != talos::kImageWidth
      || frame.height != talos::kImageHeight) {
    return false;
  }
  // 共享内存是 RGB8；检测管线按 OpenCV BGR 约定，读帧时转换并拷贝。
  const cv::Mat rgb_view(
    static_cast<int>(frame.height), static_cast<int>(frame.width), CV_8UC3,
    const_cast<std::uint8_t*>(frame.rgb));
  cv::cvtColor(rgb_view, img, cv::COLOR_RGB2BGR);
  timestamp = reader_->toSteadyTime(frame.timestamp_ns);
  return true;
}

void TalosCamera::stop()
{
  // mmap 生命周期由 TalosReader 管理；stop() 为空实现以匹配 CameraBase。
}

}  // namespace L1Sensor
