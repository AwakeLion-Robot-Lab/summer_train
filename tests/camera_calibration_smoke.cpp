#include "l1_sensor/camera/camera_calibration.hpp"

#include <cmath>
#include <iostream>

#include <yaml-cpp/yaml.h>

int main()
{
  const auto camera_config = YAML::LoadFile("tests/data/camera_calibration_inline.yaml");
  const auto calibration =
    L1Sensor::loadCameraCalibration(camera_config["calibration"], "inline smoke config");

  if (!calibration.matchesImageSize({1440, 1080}) ||
      calibration.matchesImageSize({1280, 1024})) {
    std::cerr << "CameraCalibration image-size validation failed\n";
    return 1;
  }

  if (calibration.camera_matrix.type() != CV_64FC1 ||
      calibration.camera_matrix.rows != 3 || calibration.camera_matrix.cols != 3 ||
      calibration.distortion_coefficients.type() != CV_64FC1 ||
      calibration.distortion_coefficients.rows != 1 ||
      calibration.distortion_coefficients.cols != 5) {
    std::cerr << "CameraCalibration did not normalize matrices\n";
    return 2;
  }

  if (std::abs(calibration.camera_matrix.at<double>(0, 0) - 1210.0) > 1e-9 ||
      std::abs(calibration.distortion_coefficients.at<double>(0, 0) + 0.12) > 1e-9) {
    std::cerr << "CameraCalibration values were loaded incorrectly\n";
    return 3;
  }

  std::cout << "CameraCalibration smoke test passed\n";
  return 0;
}
