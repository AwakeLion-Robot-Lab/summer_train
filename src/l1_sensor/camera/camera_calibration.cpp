#include "l1_sensor/camera/camera_calibration.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "l6_telemetry/logger.hpp"

namespace L1Sensor {
namespace {

[[noreturn]] void invalidCalibration(const std::string &path,
                                     const std::string &reason) {
  L6Telemetry::logError("camera calibration invalid", path, reason);
  throw std::runtime_error("Invalid camera calibration '" + path +
                           "': " + reason);
}

int readRequiredInt(const YAML::Node &node, const std::string &key,
                    const std::string &source_name) {
  const auto value_node = node[key];
  if (!value_node) {
    invalidCalibration(source_name, "missing " + key);
  }

  try {
    return value_node.as<int>();
  } catch (const YAML::Exception &e) {
    invalidCalibration(source_name, "invalid " + key + ": " + e.what());
  }
}

cv::Mat readRequiredMatrix(const YAML::Node &node, const std::string &key,
                           const std::string &source_name) {
  const auto matrix_node = node[key];
  if (!matrix_node || !matrix_node.IsSequence() || matrix_node.size() == 0) {
    invalidCalibration(source_name, key + " must be a non-empty YAML sequence");
  }

  const bool nested_rows = matrix_node[0].IsSequence();
  const std::size_t rows = nested_rows ? matrix_node.size() : 1;
  const std::size_t cols =
      nested_rows ? matrix_node[0].size() : matrix_node.size();
  if (cols == 0) {
    invalidCalibration(source_name, key + " contains an empty row");
  }

  cv::Mat matrix(static_cast<int>(rows), static_cast<int>(cols), CV_64FC1);
  try {
    for (std::size_t row = 0; row < rows; ++row) {
      const auto values = nested_rows ? matrix_node[row] : matrix_node;
      if (!values.IsSequence() || values.size() != cols) {
        invalidCalibration(source_name,
                           key + " rows must have the same length");
      }
      for (std::size_t col = 0; col < cols; ++col) {
        matrix.at<double>(static_cast<int>(row), static_cast<int>(col)) =
            values[col].as<double>();
      }
    }
  } catch (const YAML::Exception &e) {
    invalidCalibration(source_name, "invalid " + key + ": " + e.what());
  }

  if (!cv::checkRange(matrix)) {
    invalidCalibration(source_name, key + " contains non-finite values");
  }
  return matrix;
}

void validatePinholeCoefficients(const cv::Mat &coefficients,
                                 const std::string &path) {
  const auto count = coefficients.total();
  if (count != 4 && count != 5 && count != 8 && count != 12 && count != 14) {
    std::ostringstream reason;
    reason << "pinhole distortion_coefficients must contain 4, 5, 8, 12, or 14 "
              "values, got "
           << count;
    invalidCalibration(path, reason.str());
  }
}

CameraCalibration makeCalibration(cv::Size image_size, cv::Mat camera_matrix,
                                  cv::Mat distortion_coefficients,
                                  const std::string &source_name) {
  CameraCalibration calibration;
  calibration.image_size = image_size;
  if (calibration.image_size.width <= 0 || calibration.image_size.height <= 0) {
    invalidCalibration(source_name, "image size must be positive");
  }

  calibration.camera_matrix = std::move(camera_matrix);
  calibration.distortion_coefficients =
      std::move(distortion_coefficients).reshape(1, 1).clone();

  if (calibration.camera_matrix.rows != 3 ||
      calibration.camera_matrix.cols != 3) {
    invalidCalibration(source_name, "camera_matrix must be 3x3");
  }
  if (calibration.camera_matrix.at<double>(0, 0) <= 0.0 ||
      calibration.camera_matrix.at<double>(1, 1) <= 0.0) {
    invalidCalibration(source_name,
                       "camera_matrix focal lengths must be positive");
  }

  validatePinholeCoefficients(calibration.distortion_coefficients, source_name);

  return calibration;
}

} // namespace

bool CameraCalibration::matchesImageSize(const cv::Size &size) const noexcept {
  return size == image_size;
}

CameraCalibration loadCameraCalibration(const YAML::Node &node,
                                        const std::string &source_name) {
  if (!node || !node.IsMap()) {
    invalidCalibration(source_name, "calibration must be a YAML map");
  }

  return makeCalibration(
      {readRequiredInt(node, "image_width", source_name),
       readRequiredInt(node, "image_height", source_name)},
      readRequiredMatrix(node, "camera_matrix", source_name),
      readRequiredMatrix(node, "distortion_coefficients", source_name),
      source_name);
}

} // namespace L1Sensor
