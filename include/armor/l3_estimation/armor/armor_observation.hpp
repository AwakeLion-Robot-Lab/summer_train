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

// 整车滤波器的观测侧：一帧里构造任何一个观测块都要用的那组常量，以及由它造出
// 灯条端点观测和深度差观测。
//
// 这里只管几何与投影，不管噪声大小——R 是滤波器的事，留在 EskfTarget 的配置
// 里（见 EskfTarget::lightObs）。分开的好处是关联和更新能共用同一条投影链路，
// 而门限的松紧只在一处可调。
namespace L3Estimation {

// 整车滤波器的具体类型。状态在流形上，⊞/⊟ 由 VehicleModel 提供。
using VehicleFilter = ErrorStateEkf<VehicleModel::kStateSize, VehicleModel::Motion>;
using VehicleObs = std::shared_ptr<VehicleFilter::ObsBase>;

// 一帧的观测上下文：这辆车的几何（类别、板数、板尺寸）加上这一帧的相机位姿
// 和内参。整帧不变，所以拼一次传下去，不要每根灯条重新组装一遍。
//
// camera_in_world 必须按图像曝光时刻取，不是"当前"位姿：图像与 IMU 的时间对齐
// 就发生在取这个量的时候。
struct ObsContext
{
  ArmorName name{ArmorName::Unknown};
  int armor_num{4};
  ArmorConfig armor{};
  // 相机光学系（x 右 / y 下 / z 前）在世界系下的位姿。
  Eigen::Isometry3d camera_in_world{Eigen::Isometry3d::Identity()};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;

  // 补上板编号和左右，就是一次端点观测的完整上下文。
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

  // 由状态投影出某块板某条灯条的上下端点像素坐标。关联、ROI 和叠加层都走它，
  // 与更新时滤波器内部用的是同一条链路。
  std::pair<cv::Point2f, cv::Point2f> project(
    int id, bool is_left, const Eigen::VectorXd & state) const
  {
    const LightMeasure measure{light(id, is_left)};
    return measure.projectedPoints(state);
  }
};

// 由相机标定的 camera -> barrel 外参和当帧枪管姿态合成相机光学系在世界系的
// 位姿。世界系原点取枪管原点，与 PnpSolver 的约定一致。
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

// 一根灯条的上下端点做成一个四维观测。sigma 已由调用方按灯条长度和独立与否
// 算好，单位 px；R 由 lightCov 按灯条方向写成各向异性的块。
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

// 左右灯条中心的相机系深度差，一维。只有配出完整一块板时才有这个观测，用来
// 补住纯重投影在斜视方向的退化。
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
