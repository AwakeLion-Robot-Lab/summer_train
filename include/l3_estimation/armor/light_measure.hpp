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

// 灯条端点观测：一根灯条上下两个端点的像素坐标
// [u_top, v_top, u_bottom, v_bottom]，与 rmcs_auto_aim_v2 的做法相同。
// 观测函数是「整车状态 → 灯条两端点的三维坐标 → 带畸变投影到像素」，滤波器
// 拿它和检测到的端点直接相减。
//
// 观测停在像素而不是先做 PnP 再滤波：PnP 的误差各向异性且强相关（深度与 yaw
// 量级差一两个数量级），配不出对角 R。端点是检测器的直接输出，误差结构在
// 灯条自身的坐标系里写得干净，见 lightCov。
//
// 不再先折成 [角度, 中心, 长度] 再观测：UVL 是同两个端点的可逆映射，信息一
// 样多，差别只在同一块对角 R 在两种参数化下代表的端点噪声不同——UVL 的对角
// R 等价于端点之间带相关。换句话说换参数化换不来信息，只换来角度的 ±π 缠绕，
// 以及中心用图像系、角度与长度用灯条系的混基。直接观测端点，残差就是普通减法。
namespace L3Estimation {

namespace endpoint
{
enum { TOP_U, TOP_V, BOTTOM_U, BOTTOM_V, kMeasureSize };
}

constexpr int kLightMeasureSize = endpoint::kMeasureSize;
using LightVector = Eigen::Matrix<double, kLightMeasureSize, 1>;
using LightCov = Eigen::Matrix<double, kLightMeasureSize, kLightMeasureSize>;

// 一根灯条上下两端点在装甲板系里的坐标，顺序是先上后下。板宽按 name 取大板
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

// 一次端点观测的上下文：哪辆车的第几块板、左灯还是右灯、板的几何尺寸，以及
// 那一帧的相机位姿和内参。观测函数里除状态外的量全在这里。
//
// camera_in_world 必须按图像曝光时刻取，不是"当前"位姿：图像与 IMU 的时间
// 对齐就发生在取这个量的时候。
struct LightContext
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

struct LightMeasure
{
  template <typename T>
  using ImagePoint = Eigen::Matrix<T, 2, 1>;

  LightContext ctx;

  // 状态 → 该灯条两端点的像素坐标：整车状态经 armorPose 得到板在世界系的
  // 位姿 → 用 camera_in_world 的逆转到相机系 → 带畸变投影。
  //
  // 状态只从 armorPose 这一条路进来，其余（内参、外参、板尺寸、板编号）都是
  // 常量，所以这个函数能对状态求导，滤波器才能反解状态。模板参数 T 同时吃
  // double 和 ceres::Jet。
  //
  // rmcs 的解析 Jacobian 只用了针孔模型；这里 H 由滤波器对整条链做中心差分，
  // 畸变自然算在里面。
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

  template <typename T>
  void operator()(const T * x, T * z) const
  {
    const auto points = projectPointsOf<T>(x);
    z[endpoint::TOP_U] = points[0].x();
    z[endpoint::TOP_V] = points[0].y();
    z[endpoint::BOTTOM_U] = points[1].x();
    z[endpoint::BOTTOM_V] = points[1].y();
  }

  // 同一条投影链路，但直接返回两个像素点，给关联和叠加显示用。
  std::pair<cv::Point2f, cv::Point2f> projectedPoints(const Eigen::VectorXd & x) const
  {
    const auto points = projectPointsOf<double>(x.data());
    return {
      cv::Point2f(static_cast<float>(points[0].x()), static_cast<float>(points[0].y())),
      cv::Point2f(static_cast<float>(points[1].x()), static_cast<float>(points[1].y()))};
  }

  // 观测残差 z - z_pred。四维都是像素坐标，没有缠绕，普通减法即可。
  template <typename T>
  static Eigen::Matrix<T, kLightMeasureSize, 1> residual(
    const Eigen::Matrix<T, kLightMeasureSize, 1> & z_pred,
    const Eigen::Matrix<T, kLightMeasureSize, 1> & z)
  {
    return z - z_pred;
  }
};

// 把检测到的两个像素端点排成观测向量，排布与 LightMeasure 的输出一致。
inline LightVector toLight(const cv::Point2f & top, const cv::Point2f & bottom)
{
  LightVector z;
  z << top.x, top.y, bottom.x, bottom.y;
  return z;
}

// 由上指向下的灯条单位方向。两端点重合时方向无定义，返回 (0, 1)，调用方
// 此时应当把两个方向的 sigma 取成一样，方向就不再起作用。
inline Eigen::Vector2d lightDirection(const cv::Point2f & top, const cv::Point2f & bottom)
{
  const Eigen::Vector2d delta(bottom.x - top.x, bottom.y - top.y);
  const double length = delta.norm();
  return length > 1e-6 ? Eigen::Vector2d(delta / length) : Eigen::Vector2d(0.0, 1.0);
}

// 一根灯条的 4×4 观测协方差。
//
// 两个端点的定位误差按互相独立处理，所以是两个 2×2 块、块间为零。每块在灯条
// 自身坐标系里是对角的——沿灯条 sigma_along、垂直灯条 sigma_perp——再旋到
// 图像系：
//   Σ = σ∥²·e·eᵀ + σ⊥²·(I − e·eᵀ)，e 为灯条方向。
// 两个 sigma 相等时退化为各向同性，e 不再起作用。
//
// 块间为零是实测之后的选择，不是偷懒——上下端点的误差物理上确实相关（细亮斑
// 横向整体平移），但把这个相关写进 R 在 pred_px 上不划算，数据见 auto_aim.yaml
// 的 sigma_min_px 注释。别再重新推导一遍，那条链子很容易长出来。
inline LightCov lightCov(
  const cv::Point2f & top, const cv::Point2f & bottom, double sigma_along,
  double sigma_perp)
{
  const Eigen::Vector2d e = lightDirection(top, bottom);
  const Eigen::Matrix2d along = e * e.transpose();
  const Eigen::Matrix2d block =
    sigma_along * sigma_along * along +
    sigma_perp * sigma_perp * (Eigen::Matrix2d::Identity() - along);

  LightCov cov = LightCov::Zero();
  cov.block<2, 2>(endpoint::TOP_U, endpoint::TOP_U) = block;
  cov.block<2, 2>(endpoint::BOTTOM_U, endpoint::BOTTOM_U) = block;
  return cov;
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
  LightContext ctx;

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
