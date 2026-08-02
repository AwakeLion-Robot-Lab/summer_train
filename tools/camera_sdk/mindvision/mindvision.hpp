#ifndef IO__MINDVISION_HPP
#define IO__MINDVISION_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "CameraApi.h"
#include "l1_sensor/camera/camera.hpp"
#include "latesbuffer.hpp"

namespace io
{
class MindVision : public L1Sensor::CameraBase
{
public:
  MindVision(double exposure_ms, double gamma, const std::string & vid_pid);
  ~MindVision() override;
  bool read(
    cv::Mat & img,
    std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout) override;
  void stop() override;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_ms_, gamma_;
  CameraHandle handle_ = -1;
  int height_ = 0;
  int width_ = 0;
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> healthy_{false};
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::LatestBuffer<CameraData> buffer_;
  int vid_, pid_;

  void open();
  void try_open();
  void close();
  void set_vid_pid(const std::string & vid_pid);
  void reset_usb() const;
};

}  // namespace io

#endif  // IO__MINDVISION_HPP
