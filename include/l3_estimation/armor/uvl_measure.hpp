#pragma once

#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/projection.hpp"

#include <ceres/jet.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <array>
#include <vector>

// UVL 观测：图像平面上一条灯条的四个几何量。照搬 awakening 的 UVLMeasure。
//
// 与"PnP 解位姿再滤波"的根本区别在于把观测退回到更接近传感器原始输出的地方。
// PnP 的误差高度各向异性且相关（深度与 yaw 强相关，量级差一两个数量级），
// 给它一个对角 R 就是撒谎；而像素误差近似各向同性，对角 R 诚实得多。
//
// 观测取 [角度, 中心 x, 中心 y, 长度] 而不是两个端点的四个坐标，是因为这四个
// 量物理意义正交：长度吸收纵向误差、中心把纵向误差平均掉一半、角度只受横向
// 误差影响且被长度归一化。于是三类量可以各给各的 sigma，对角 R 才成立。
// 推导见 docs/esekf_uvl_port.md 第 5.4 节。
namespace L3Estimation {

namespace uvl
{
enum { ANGLE, CENTER_X, CENTER_Y, LENGTH, kMeasureSize };
}

constexpr int kUvlMeasureSize = uvl::kMeasureSize;
using UvlVector = Eigen::Matrix<double, kUvlMeasureSize, 1>;

// 一条灯条上下两端点在装甲板系的三维坐标。
//
// 板系约定 x = 板面法向、y = 左、z = 上，与 PnpSolver::armorPoints 同源；
// 左灯条在 +y 侧，右灯条在 -y 侧。
inline std::vector<cv::Point3f> armorLightPoints3D(
  ArmorName name, bool is_left, const ArmorConfig & config)
{
  const auto type = armorTypeOf(name);
  const double width =
    (type == ArmorType::Big) ? config.big_width : config.small_width;
  const float half_width = static_cast<float>(width / 2.0);
  const float half_height = static_cast<float>(config.height / 2.0);
  const float y = is_left ? half_width : -half_width;
  return {{0.0F, y, half_height}, {0.0F, y, -half_height}};
}

// 一个 UVL 观测的上下文：哪块板的哪条灯条，以及那一帧相机在哪。
//
// camera_in_world 必须是**按图像曝光时刻**取到的相机位姿，不是"当前"位姿。
// 图像与 IMU 的时间对齐就发生在取这个量的时候。
struct UvlContext
{
  int armor_num{4};
  int id{0};
  bool is_left{true};
  ArmorName name{ArmorName::Unknown};
  ArmorConfig armor_config{};
  // 相机光学系（x 右 / y 下 / z 前）在世界系下的位姿。
  Eigen::Isometry3d camera_in_world{Eigen::Isometry3d::Identity()};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
};

struct UvlMeasure
{
  template <typename T>
  using ImagePoint = Eigen::Matrix<T, 2, 1>;

  UvlContext ctx;

  // 状态 → 该灯条两端点的像素坐标。把前面所有层串起来。
  //
  // 状态只通过 armorPose 进入，其余全是常量（内参、外参、板尺寸、板编号）。
  // 观测与状态之间只有这一条通路，滤波器才能用它反解状态。
  template <typename T>
  std::array<ImagePoint<T>, 2> projectPointsOf(const T * x) const
  {
    const auto pose_in_world =
      VehicleModel::armorPose<T>(x, ctx.id, ctx.armor_num, ctx.name);

    Eigen::Transform<T, 3, Eigen::Isometry> camera_in_world_jet;
    camera_in_world_jet.matrix() = ctx.camera_in_world.matrix().template cast<T>();

    const auto pose_in_camera = camera_in_world_jet.inverse() * pose_in_world;

    const std::vector<cv::Point3f> object_points =
      armorLightPoints3D(ctx.name, ctx.is_left, ctx.armor_config);

    std::vector<ImagePoint<T>> image_points;
    projectPoints<T>(
      object_points, pose_in_camera, ctx.camera_matrix, ctx.distortion_coefficients,
      image_points);

    return {image_points[0], image_points[1]};
  }

  // 两个端点 → 四维观测。
  //
  // 预测值和观测值走的是**同一个函数**：前者喂投影出来的点，后者喂检测出来的
  // 点。两边定义一旦漂移就是灾难性的隐蔽 bug，所以不拆成两份。
  //
  // atan2 的参数是 (Δx, Δy) 而不是通常的 (Δy, Δx)：量的是偏离**竖直**方向的
  // 角。灯条基本竖直，这样 α 在 0 附近工作，残差归一化和线性化都干净；用通常
  // 写法 α 会跑到 ±π/2 附近。
  template <typename T>
  static void pointsToObservation(
    const ImagePoint<T> & top, const ImagePoint<T> & bottom, T * z)
  {
    const ImagePoint<T> delta = top - bottom;
    const ImagePoint<T> center = (top + bottom) / T(2.0);
    z[uvl::ANGLE] = ceres::atan2(delta.x(), delta.y());
    z[uvl::CENTER_X] = center.x();
    z[uvl::CENTER_Y] = center.y();
    z[uvl::LENGTH] = ceres::sqrt(delta.squaredNorm());
  }

  template <typename T>
  void operator()(const T * x, T * z) const
  {
    const auto points = projectPointsOf<T>(x);
    pointsToObservation<T>(points[0], points[1], z);
  }

  // 调试与关联用：直接拿到预测的两个像素点。
  std::pair<cv::Point2f, cv::Point2f> projectedPoints(const Eigen::VectorXd & x) const
  {
    const auto points = projectPointsOf<double>(x.data());
    return {
      cv::Point2f(static_cast<float>(points[0].x()), static_cast<float>(points[0].y())),
      cv::Point2f(static_cast<float>(points[1].x()), static_cast<float>(points[1].y()))};
  }

  // 只有角度活在 S¹ 上，其余三维是普通欧氏量。
  //
  // 求 H 的中心差分差的正是这个函数而不是 z_pred，就为了让缠绕归一化进到导数
  // 里——否则预测值分居 ±π 两侧时会差出一个 2π 的假梯度。
  template <typename T>
  static Eigen::Matrix<T, kUvlMeasureSize, 1> residual(
    const Eigen::Matrix<T, kUvlMeasureSize, 1> & z_pred,
    const Eigen::Matrix<T, kUvlMeasureSize, 1> & z)
  {
    Eigen::Matrix<T, kUvlMeasureSize, 1> v = z - z_pred;
    v[uvl::ANGLE] = VehicleModel::normalizeAngle(v[uvl::ANGLE]);
    return v;
  }
};

// 从检测到的两个像素端点构造观测量。与预测共用 pointsToObservation。
inline UvlVector uvlMeasurementFrom(const cv::Point2f & top, const cv::Point2f & bottom)
{
  const Eigen::Vector2d top_point(top.x, top.y);
  const Eigen::Vector2d bottom_point(bottom.x, bottom.y);
  UvlVector observation;
  UvlMeasure::pointsToObservation<double>(top_point, bottom_point, observation.data());
  return observation;
}

}  // namespace L3Estimation
