#pragma once

#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l6_telemetry/projection.hpp"

#include <ceres/jet.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <array>
#include <vector>

// UVL 观测：一条灯条在图像上的四个几何量 [角度, 中心 x, 中心 y, 长度]。
// 观测函数是「整车状态 → 灯条两端点的三维坐标 → 投影到像素 → 折成这四个量」，
// 滤波器拿它和检测到的端点算残差。
//
// 观测停在像素而不是先做 PnP 再滤波：PnP 的误差各向异性且强相关（深度与 yaw
// 量级差一两个数量级），配对角 R 不成立；像素误差近似各向同性。四个量的物理
// 意义也相互正交——长度吸收纵向误差、中心把纵向误差平均掉一半、角度只受横向
// 误差影响，所以三类量各给各的 sigma。推导见 docs/esekf_uvl_port.md 5.4 节。
namespace L3Estimation {

namespace uvl
{
enum { ANGLE, CENTER_X, CENTER_Y, LENGTH, kMeasureSize };
}

constexpr int kUvlMeasureSize = uvl::kMeasureSize;
using UvlVector = Eigen::Matrix<double, kUvlMeasureSize, 1>;

// 一条灯条上下两端点在装甲板系里的坐标，顺序是先上后下。板宽按 name 取大板
// 或小板，左灯条在 +y 侧、右灯条在 -y 侧。
//
// 板系约定 x = 板面法向、y = 左、z = 上，与 PnpSolver 的物点同源。
inline std::vector<cv::Point3f> lightPoints3D(
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

// 一次 UVL 观测的上下文：哪辆车的第几块板、左灯还是右灯、板的几何尺寸，以及
// 那一帧的相机位姿和内参。观测函数里除状态外的量全在这里。
//
// camera_in_world 必须按图像曝光时刻取，不是"当前"位姿：图像与 IMU 的时间
// 对齐就发生在取这个量的时候。
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

  // 状态 → 该灯条两端点的像素坐标：整车状态经 armorPose 得到板在世界系的
  // 位姿 → 用 camera_in_world 的逆转到相机系 → 带畸变投影。
  //
  // 状态只从 armorPose 这一条路进来，其余（内参、外参、板尺寸、板编号）都是
  // 常量，所以这个函数能对状态求导，滤波器才能反解状态。模板参数 T 同时吃
  // double 和 ceres::Jet。
  template <typename T>
  std::array<ImagePoint<T>, 2> projectPointsOf(const T * x) const
  {
    const auto pose_in_world =
      VehicleModel::armorPose<T>(x, ctx.id, ctx.armor_num, ctx.name);

    Eigen::Transform<T, 3, Eigen::Isometry> camera_in_world_jet;
    camera_in_world_jet.matrix() = ctx.camera_in_world.matrix().template cast<T>();

    const auto pose_in_camera = camera_in_world_jet.inverse() * pose_in_world;

    const std::vector<cv::Point3f> object_points =
      lightPoints3D(ctx.name, ctx.is_left, ctx.armor_config);

    std::vector<ImagePoint<T>> image_points;
    L6Telemetry::projectPoints<T>(
      object_points, pose_in_camera, ctx.camera_matrix, ctx.distortion_coefficients,
      image_points);

    return {image_points[0], image_points[1]};
  }

  // 两个端点 → 四维观测：角度取两端点差向量偏离竖直方向的夹角，中心取两点
  // 中点，长度取两点距离。
  //
  // 预测值和观测值走同一个函数，前者喂投影出来的点、后者喂检测出来的点：拆成
  // 两份一旦定义漂移就是很难发现的 bug。
  //
  // atan2 的参数是 (Δx, Δy) 而不是通常的 (Δy, Δx)，量的是偏离竖直方向的角。
  // 灯条基本竖直，这样角度在 0 附近工作，缠绕归一化和线性化都干净。
  template <typename T>
  static void pointsToUvl(
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
    pointsToUvl<T>(points[0], points[1], z);
  }

  // 同一条投影链路，但直接返回两个像素点，给关联和叠加显示用。
  std::pair<cv::Point2f, cv::Point2f> projectedPoints(const Eigen::VectorXd & x) const
  {
    const auto points = projectPointsOf<double>(x.data());
    return {
      cv::Point2f(static_cast<float>(points[0].x()), static_cast<float>(points[0].y())),
      cv::Point2f(static_cast<float>(points[1].x()), static_cast<float>(points[1].y()))};
  }

  // 观测残差 z - z_pred，其中角度一维要按 S¹ 归一化到 (-π, π]，另外三维是
  // 普通减法。
  //
  // 求 H 的中心差分差的是这个函数而不是 z_pred，为的就是让缠绕归一化进到导数
  // 里：否则两个预测值分居 ±π 两侧时会差出一个 2π 的假梯度。
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

// 把检测到的两个像素端点折成观测向量，与预测值共用 pointsToUvl。
inline UvlVector toUvl(const cv::Point2f & top, const cv::Point2f & bottom)
{
  const Eigen::Vector2d top_point(top.x, top.y);
  const Eigen::Vector2d bottom_point(bottom.x, bottom.y);
  UvlVector observation;
  UvlMeasure::pointsToUvl<double>(top_point, bottom_point, observation.data());
  return observation;
}

// 一维观测：左右灯条中心在相机 z 轴上的深度差，配出完整一块板时才有。
// 观测函数同样从整车状态出发，只是投影换成取两个灯条中心的相机系 z 之差。
//
// 只取 PnP 里对斜视姿态最有辨识度的这一维，抖动大的绝对深度和完整姿态不进
// 滤波器。
constexpr int kDepthDiffMeasureSize = 1;
using DepthDiffVector = Eigen::Matrix<double, kDepthDiffMeasureSize, 1>;

struct DepthDiffMeasure
{
  UvlContext ctx;

  template <typename T>
  void operator()(const T * x, T * z) const
  {
    const auto armor_in_world =
      VehicleModel::armorPose<T>(x, ctx.id, ctx.armor_num, ctx.name);

    Eigen::Transform<T, 3, Eigen::Isometry> camera_in_world_jet;
    camera_in_world_jet.matrix() = ctx.camera_in_world.matrix().template cast<T>();
    const auto armor_in_camera = camera_in_world_jet.inverse() * armor_in_world;

    const auto centerInCamera = [&](bool is_left) {
      const std::vector<cv::Point3f> points =
        lightPoints3D(ctx.name, is_left, ctx.armor_config);
      Eigen::Matrix<T, 3, 1> center = Eigen::Matrix<T, 3, 1>::Zero();
      for (const cv::Point3f& point : points) {
        center += armor_in_camera * Eigen::Matrix<T, 3, 1>(
          T(point.x), T(point.y), T(point.z));
      }
      return (center / T(static_cast<double>(points.size()))).eval();
    };
    // 左右灯条中心在相机系下的深度差。
    z[0] = centerInCamera(true).z() - centerInCamera(false).z();
  }

  template <typename T>
  static Eigen::Matrix<T, kDepthDiffMeasureSize, 1> residual(
    const Eigen::Matrix<T, kDepthDiffMeasureSize, 1>& z_pred,
    const Eigen::Matrix<T, kDepthDiffMeasureSize, 1>& z)
  {
    return z - z_pred;
  }
};

}  // namespace L3Estimation
