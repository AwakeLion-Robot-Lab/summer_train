#pragma once

#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <optional>
#include <string>

namespace L1Sensor {

// 一台相机在固定分辨率、ROI 和 binning 设置下的标定参数。
// camera_matrix 与 distortion_coefficients 均在加载时规范为 CV_64FC1。
//
// 外参命名统一采用 T_A_B：把 B 坐标系中的点转换到 A 坐标系。
// T_barrel_camera 是相机与枪管之间不随帧变化的静态机械外参。
// 未标定时保持 std::nullopt，禁止用单位阵冒充有效标定。
struct CameraCalibration {
  cv::Size image_size{};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;

  // camera optical frame -> barrel frame
  std::optional<Eigen::Isometry3d> T_barrel_camera;

  // 标定仅对生成它时的图像尺寸有效；ROI、binning 或分辨率变化后
  // 应重新标定。
  bool matchesImageSize(const cv::Size& size) const noexcept;

  bool barrelExtrinsicsReady() const noexcept;
};

// 从相机配置中的 calibration: 节点读取标定数据。camera_matrix 使用 3 行数组，
// distortion_coefficients 使用一维数组。可选的 T_barrel_camera 使用
// {rotation: 3x3, translation: [x,y,z]}，平移单位为米。
// source_name 仅用于生成可定位的错误信息。
CameraCalibration loadCameraCalibration(const YAML::Node& node, const std::string& source_name);

}  // namespace L1Sensor
