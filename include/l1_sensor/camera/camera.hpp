#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

namespace L1Sensor {

/**
 * @brief 相机 SDK 封装的公共接口
 *
 * hikrobot / mindvision 各自实现它，Camera 按配置里的 camera_name 选一个。
 */
class CameraBase {
public:
  virtual ~CameraBase() = default;

  /**
   * @brief 取一帧图像
   * @param img 输出的 BGR 图像
   * @param timestamp 输出的图像时间戳（steady_clock）
   * @param timeout 最长等待时间
   * @return 超时或取图失败时返回 false
   */
  virtual bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout) = 0;

  /**
   * @brief 停止取流并释放设备
   */
  virtual void stop() = 0;
};

/**
 * @brief 相机：按配置选 SDK 实现，并持有可选的相机标定
 */
class Camera {
public:
  /**
   * @brief 读取相机配置并打开对应的 SDK 实现
   *
   * 配置里有 calibration: 节点时一并加载标定；camera_name 不受支持时抛异常。
   *
   * @param config_path 相机配置 YAML 路径（如 config/camera_config.yaml）
   */
  explicit Camera(const std::string& config_path);

  /**
   * @brief 取一帧图像
   * @param img 输出的 BGR 图像
   * @param timestamp 输出的图像时间戳（steady_clock）
   * @param timeout 最长等待时间
   * @return 超时或取图失败时返回 false
   */
  bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{50});

  /**
   * @brief 停止取流
   */
  void stop();

  /**
   * @brief 相机标定参数
   *
   * L2 应以图像曝光时刻对应的内参和畸变参数进行 PnP。
   *
   * @return 标定参数；未配置时为空
   */
  const std::optional<CameraCalibration>& calibration() const noexcept;

private:
  /// 具体的相机 SDK 实现
  std::unique_ptr<CameraBase> camera_;

  /// 相机标定，未配置时为空
  std::optional<CameraCalibration> calibration_;
};

}  // namespace L1Sensor
