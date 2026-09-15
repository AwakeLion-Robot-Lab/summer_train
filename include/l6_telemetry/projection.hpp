#pragma once

#include <ceres/jet.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <stdexcept>
#include <vector>

// 针孔 + Brown-Conrady 畸变投影，模板化在标量类型上。照搬 awakening 的
// utils::project_points_jets。
//
// 为什么不用 cv::projectPoints：它只认 double，自动微分穿不过去。整条链
// 状态 → 板位姿 → 相机系 → 像素 必须是纯代数、处处可微的，Jet 才能一路走到
// 底——这是 UVL 观测模型可行的技术前提。眼下 ObsBase 类型擦除后走的是中心
// 差分，但观测模型本身保持模板化，将来去掉类型擦除即可直接换成 Jet 求 H。
namespace L6Telemetry {

// 把物点从自身坐标系经 pose_in_camera 变换到相机光学系，再投影成像素。
//
// pose_in_camera 是"物点所在坐标系 → 相机光学系（x 右 / y 下 / z 前）"的变换。
//
// 注意没有 Zc <= 0 的保护，与上游一致：板转到相机背面时 Xc/Zc 会翻符号甚至
// 除零。调用方靠可见性筛选（按板法向排序只取最正对的几块）挡住，那是启发式
// 而非保证。
template <typename T>
void projectPoints(
  const std::vector<cv::Point3f> & object_points,
  const Eigen::Transform<T, 3, Eigen::Isometry> & pose_in_camera, const cv::Mat & camera_matrix,
  const cv::Mat & distortion_coefficients, std::vector<Eigen::Matrix<T, 2, 1>> & image_points)
{
  if (object_points.empty()) {
    return;
  }
  if (camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3) {
    throw std::runtime_error("projectPoints: 相机内参矩阵无效");
  }
  if (distortion_coefficients.empty()) {
    throw std::runtime_error("projectPoints: 畸变系数为空");
  }

  const Eigen::Matrix<T, 3, 3> & rotation = pose_in_camera.linear();
  const Eigen::Matrix<T, 3, 1> & translation = pose_in_camera.translation();

  const T fx = T(camera_matrix.at<double>(0, 0));
  const T fy = T(camera_matrix.at<double>(1, 1));
  const T cx = T(camera_matrix.at<double>(0, 2));
  const T cy = T(camera_matrix.at<double>(1, 2));

  const auto coefficient = [&](int i) -> double {
    return distortion_coefficients.rows == 1 ? distortion_coefficients.at<double>(0, i)
                                             : distortion_coefficients.at<double>(i, 0);
  };
  const int count = distortion_coefficients.rows * distortion_coefficients.cols;
  const T k1 = count > 0 ? T(coefficient(0)) : T(0.0);
  const T k2 = count > 1 ? T(coefficient(1)) : T(0.0);
  const T p1 = count > 2 ? T(coefficient(2)) : T(0.0);
  const T p2 = count > 3 ? T(coefficient(3)) : T(0.0);
  const T k3 = count > 4 ? T(coefficient(4)) : T(0.0);

  image_points.clear();
  image_points.reserve(object_points.size());

  for (const auto & point : object_points) {
    const Eigen::Matrix<T, 3, 1> in_object(T(point.x), T(point.y), T(point.z));
    const Eigen::Matrix<T, 3, 1> in_camera = rotation * in_object + translation;

    // 透视除法
    const T normalized_x = in_camera(0) / in_camera(2);
    const T normalized_y = in_camera(1) / in_camera(2);

    const T r2 = normalized_x * normalized_x + normalized_y * normalized_y;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;

    const T radial = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;
    const T distorted_x = normalized_x * radial + T(2.0) * p1 * normalized_x * normalized_y +
                          p2 * (r2 + T(2.0) * normalized_x * normalized_x);
    const T distorted_y = normalized_y * radial +
                          p1 * (r2 + T(2.0) * normalized_y * normalized_y) +
                          T(2.0) * p2 * normalized_x * normalized_y;

    image_points.emplace_back(fx * distorted_x + cx, fy * distorted_y + cy);
  }
}

}  // namespace L6Telemetry
