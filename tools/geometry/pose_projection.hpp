#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <vector>

namespace tools::geometry {

// Rodrigues rotation vector followed by camera-frame translation.
using PoseParameters = Eigen::Matrix<double, 6, 1>;

bool finite(const cv::Point3d& point);
bool rigid(const Eigen::Isometry3d& transform);
cv::Matx33d rotation(const PoseParameters& parameters);
bool inFront(const std::vector<cv::Point3d>& model, const PoseParameters& parameters);
std::vector<cv::Point2d> project(const std::vector<cv::Point3d>& model,
    const PoseParameters& parameters, const cv::Mat& camera, const cv::Mat& distortion);

} // namespace tools::geometry
