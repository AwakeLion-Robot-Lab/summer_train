#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/armor/light_measure.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/filter/error_state_ekf.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/types.hpp>

#include <memory>
#include <utility>

/**
 * @file
 * @brief 整车滤波器的观测侧
 *
 * 一帧里构造任何一个观测块都要用的那组常量，以及由它造出灯条端点观测和深度差
 * 观测。
 *
 * 这里只管几何与投影，不管噪声大小——R 是滤波器的事，留在 EskfTarget 的配置
 * 里（见 EskfTarget::lightObs）。分开的好处是关联和更新能共用同一条投影链路，
 * 而门限的松紧只在一处可调。
 */
namespace L3Estimation {

/// 整车滤波器的具体类型。状态在流形上，⊞/⊟ 由 VehicleModel 提供。
using VehicleFilter = ErrorStateEkf<VehicleModel::kStateSize, VehicleModel::Motion>;

/// 类型擦除的一个观测块
using VehicleObs = std::shared_ptr<VehicleFilter::ObsBase>;

/**
 * @brief 一帧的观测上下文
 *
 * 这辆车的几何（类别、板数、板尺寸）加上这一帧的相机位姿和内参。整帧不变，
 * 所以拼一次传下去，不要每根灯条重新组装一遍。
 *
 * camera_in_world 必须按图像曝光时刻取，不是"当前"位姿：图像与 IMU 的时间对齐
 * 就发生在取这个量的时候。
 */
struct ObsContext
{
  /// 车辆类别
  ArmorName name{ArmorName::Unknown};

  /// 物理装甲板数量
  int armor_num{4};

  /// 装甲板几何尺寸
  ArmorConfig armor{};

  /// 相机光学系（x 右 / y 下 / z 前）在世界系下的位姿。
  Eigen::Isometry3d camera_in_world{Eigen::Isometry3d::Identity()};

  /// 相机内参
  cv::Mat camera_matrix;

  /// 畸变系数
  cv::Mat distortion_coefficients;

  /**
   * @brief 补上板编号和左右，就是一次端点观测的完整上下文
   * @param id 物理板编号
   * @param is_left true 为左灯条
   * @return 端点观测上下文
   */
  LightContext light(int id, bool is_left) const
  {
    LightContext ctx;
    ctx.armor_num = armor_num;
    ctx.id = id;
    ctx.is_left = is_left;
    ctx.name = name;
    ctx.armor_config = armor;
    ctx.camera_in_world = camera_in_world;
    ctx.camera_matrix = camera_matrix;
    ctx.distortion_coefficients = distortion_coefficients;
    return ctx;
  }

  /**
   * @brief 由状态投影出某块板某条灯条的上下端点像素坐标
   *
   * 关联、ROI 和叠加层都走它，与更新时滤波器内部用的是同一条链路。
   *
   * @param id 物理板编号
   * @param is_left true 为左灯条
   * @param state 整车状态
   * @return (上端点, 下端点)，像素坐标
   */
  std::pair<cv::Point2f, cv::Point2f> project(
    int id, bool is_left, const Eigen::VectorXd & state) const
  {
    const LightMeasure measure{light(id, is_left)};
    return measure.projectedPoints(state);
  }
};

/**
 * @brief 合成相机光学系在世界系的位姿
 *
 * 由相机标定的 camera -> barrel 外参和当帧枪管姿态合成。世界系原点取枪管原点，
 * 与 PnpSolver 的约定一致。
 *
 * @param calibration 相机标定；没有 T_barrel_camera 时只用枪管姿态
 * @param q_world_barrel 曝光时刻的枪管姿态
 * @return 相机光学系在世界系下的位姿
 */
inline Eigen::Isometry3d cameraInWorld(
  const L1Sensor::CameraCalibration & calibration,
  const Eigen::Quaterniond & q_world_barrel)
{
  Eigen::Isometry3d barrel_in_world = Eigen::Isometry3d::Identity();
  barrel_in_world.linear() = q_world_barrel.toRotationMatrix();
  if (!calibration.T_barrel_camera) {
    return barrel_in_world;
  }
  return barrel_in_world * (*calibration.T_barrel_camera);
}

/**
 * @brief 一根灯条的上下端点做成一个四维观测
 *
 * R 由 lightCov 按灯条方向写成各向异性的块。
 *
 * @param ctx 本帧的观测上下文
 * @param top 检测到的上端点
 * @param bottom 检测到的下端点
 * @param id 物理板编号
 * @param is_left true 为左灯条
 * @param sigma_along 沿灯条方向的端点噪声，px，已由调用方按灯条长度和独立与否算好
 * @param sigma_perp 垂直灯条方向的端点噪声，px，同上
 * @return 观测块
 */
inline VehicleObs makeLightObs(
  const ObsContext & ctx, const cv::Point2f & top, const cv::Point2f & bottom, int id,
  bool is_left, double sigma_along, double sigma_perp)
{
  const LightMeasure measure{ctx.light(id, is_left)};
  const LightCov r_cov = lightCov(top, bottom, sigma_along, sigma_perp);

  return VehicleFilter::makeObs<kLightMeasureSize>(
    toLight(top, bottom), measure, [r_cov](const LightVector &) { return r_cov; },
    [](const LightVector & z_pred, const LightVector & z_obs) {
      return LightMeasure::residual<double>(z_pred, z_obs);
    });
}

/**
 * @brief 左右灯条中心的相机系深度差，一维观测
 *
 * 只有配出完整一块板时才有这个观测，用来补住纯重投影在斜视方向的退化。
 *
 * @param ctx 本帧的观测上下文
 * @param id 物理板编号
 * @param depth_diff 观测到的深度差（左 − 右），meter，见 PnpSolver::lights_depth_diff
 * @param sigma 深度差噪声，meter；R 取 sigma² / 2
 * @return 观测块
 */
inline VehicleObs makeDepthObs(
  const ObsContext & ctx, int id, double depth_diff, double sigma)
{
  const DepthDiffMeasure measure{ctx.light(id, true)};
  DepthDiffVector z;
  z[0] = depth_diff;

  Eigen::Matrix<double, kDepthDiffMeasureSize, kDepthDiffMeasureSize> r_cov;
  r_cov.setZero();
  r_cov(0, 0) = sigma * sigma / 2.0;

  return VehicleFilter::makeObs<kDepthDiffMeasureSize>(
    z, measure, [r_cov](const DepthDiffVector &) { return r_cov; },
    [](const DepthDiffVector & z_pred, const DepthDiffVector & z_obs) {
      return DepthDiffMeasure::residual<double>(z_pred, z_obs);
    });
}

}  // namespace L3Estimation
