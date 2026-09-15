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
//
// barrel 是右手系：x 指向枪口（瞄准方向），z 朝上，y 朝左，允许与 IMU 轴向
// 不同，差异写在 config/serial_config.yaml 的 R_imu_barrel 里。
// T_barrel_camera 必须标定到这个 barrel 系；相机光学系为 z 前 / x 右 / y 下，
// 纯轴向部分是 [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]，机械安装角再叠加上去。
// 轴向不一致时程序不会报错，只会让世界系姿态整体错掉。
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
