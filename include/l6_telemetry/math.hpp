#ifndef TOOLS__MATH_HPP
#define TOOLS__MATH_HPP

#include <Eigen/Geometry>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <chrono>

namespace L6Telemetry {

/**
 * @brief 计算时间差 a - b
 * @param a 被减时刻
 * @param b 减去的时刻
 * @return a - b，单位：s，可为负
 */
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

/**
 * @brief 把角度折回 (-π, π]
 * @param angle 角度，rad
 * @return 折回后的角度；非有限值原样返回
 */
double limit_rad(double angle);

/**
 * @brief OpenCV 旋转矩阵转 Eigen
 * @param rotation 3x3 旋转矩阵
 * @return 同一个矩阵
 */
Eigen::Matrix3d toEigen(
  const cv::Matx33d& rotation) noexcept;

/**
 * @brief Eigen 旋转矩阵转 OpenCV
 * @param rotation 3x3 旋转矩阵
 * @return 同一个矩阵
 */
cv::Matx33d toCv(
  const Eigen::Matrix3d& rotation) noexcept;

/**
 * @brief 使用鞋带公式计算四边形的像素面积
 * @param corners 四个角点，按顺序相连
 * @return 面积，pixel²，非负
 */
double polygonArea(
  const std::array<cv::Point2f, 4>& corners) noexcept;

/**
 * @brief 两点间距离的平方
 * @param lhs 第一个点
 * @param rhs 第二个点
 * @return 距离平方，pixel²
 */
double squaredDistance(
  const cv::Point2f& lhs,
  const cv::Point2f& rhs) noexcept;

/**
 * @brief 四元数分解为欧拉角
 *
 * 欧拉角固定使用 [yaw, pitch, roll]；axis0/1/2 指定旋转轴顺序。
 *
 * @param quaternion 旋转
 * @param axis0 第一个旋转轴，0/1/2 对应 x/y/z
 * @param axis1 第二个旋转轴
 * @param axis2 第三个旋转轴
 * @param extrinsic true 为外旋，false 为内旋
 * @return 三个欧拉角，rad，顺序与 axis0/1/2 对应
 */
Eigen::Vector3d eulers(
  Eigen::Quaterniond quaternion,
  int axis0,
  int axis1,
  int axis2,
  bool extrinsic = false);

/**
 * @brief 旋转矩阵分解为欧拉角，参数含义同四元数版本
 * @param rotation 旋转矩阵
 * @param axis0 第一个旋转轴，0/1/2 对应 x/y/z
 * @param axis1 第二个旋转轴
 * @param axis2 第三个旋转轴
 * @param extrinsic true 为外旋，false 为内旋
 * @return 三个欧拉角，rad
 */
Eigen::Vector3d eulers(
  const Eigen::Matrix3d& rotation,
  int axis0,
  int axis1,
  int axis2,
  bool extrinsic = false);

/**
 * @brief YPR 转旋转矩阵
 *
 * YPR 固定为 [yaw, pitch, roll]，旋转顺序为 Rz(yaw)Ry(pitch)Rx(roll)。
 *
 * @param ypr [yaw, pitch, roll]，rad
 * @return 旋转矩阵
 */
Eigen::Matrix3d yprToRotation(const Eigen::Vector3d& ypr);

/**
 * @brief 旋转矩阵转 YPR，yprToRotation 的逆
 * @param rotation 旋转矩阵
 * @return [yaw, pitch, roll]，rad
 */
Eigen::Vector3d rotationToYpr(const Eigen::Matrix3d& rotation);

/**
 * @brief 直角坐标转 [方位角, 俯仰角, 距离]
 * @param xyz [x, y, z]，meter
 * @return [atan2(y, x), atan2(z, 水平距离), 直线距离]，角度 rad，距离 meter
 */
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);

/**
 * @brief 欧拉角转四元数，按 Rz(yaw)Ry(pitch)Rx(roll)
 * @param roll 横滚角，rad
 * @param pitch 俯仰角，rad
 * @param yaw 偏航角，rad
 * @return 单位四元数
 */
Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw);

/**
 * @brief 两个姿态之间球面线性插值
 * @param a 起点姿态
 * @param b 终点姿态
 * @param k 插值系数，钳到 [0, 1]；0 得 a，1 得 b
 * @return 单位四元数
 */
Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k);

}  // namespace L6Telemetry

#endif  // TOOLS__MATH_HPP
