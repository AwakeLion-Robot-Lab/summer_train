#pragma once

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <string>

namespace L1Sensor {

// 一台相机在固定分辨率、ROI 和 binning 设置下的内参与畸变参数。
// camera_matrix 与 distortion_coefficients 均在加载时规范为 CV_64FC1。
struct CameraCalibration {
  cv::Size image_size{};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;

  // 标定仅对生成它时的图像尺寸有效；ROI、binning 或分辨率变化后应重新标定。
  bool matchesImageSize(const cv::Size& size) const noexcept;
};

// 从相机配置中的 calibration: 节点读取标定数据。camera_matrix 使用 3 行数组，
// distortion_coefficients 使用一维数组；source_name 仅用于生成可定位的错误信息。
CameraCalibration loadCameraCalibration(const YAML::Node& node, const std::string& source_name);

}  // namespace L1Sensor
