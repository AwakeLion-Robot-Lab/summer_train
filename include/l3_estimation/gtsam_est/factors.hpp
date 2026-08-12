#pragma once

// 整车运动与装甲板观测的 GTSAM 因子。移植自 jlu_vision_26-master 的
// src/auto_aim/armor_tracker/include/factors.hpp。
//
// 整个文件在没有 GTSAM 时编译成空翻译单元。
#ifdef NEWVISION_USE_GTSAM

#include "l3_estimation/types.hpp"

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

namespace L3Estimation::GtsamEst {

// 因子图连接关系（k 是帧号，A/B/Z 是整段共用的常量）：
//
//   [运动段：恒速度 + 恒角速度，与 filter_est 的 F 矩阵是同一个模型]
//     X(k-1), V(k-1) --TranslationFactor--> X(k)
//     R(k-1), W(k-1) --YawFactor---------> R(k)
//     V(k-1)         --VelocityFactor----> V(k)
//     W(k-1)         --VyawFactor--------> W(k)
//
//   [观测段：把"中心 -> 装甲板"的位移拆成切向和径向]
//     A, X(k), R(k) --ArmorRadiusCenterZFactor--> 偶数号板观测
//     B, Z, X(k), R(k) --ArmorRadiusDZFactor----> 奇数号板观测
//
// 切向/径向分开是这套方案相对 EKF 的关键改进：半径**只**由径向残差约束，
// 整车 yaw 的相位误差不会经切向残差把半径一路拉小。EKF 后端使用
// [方位角, 俯仰角, 距离, 板 yaw] 观测，半径和 yaw 在那里是耦合的。
//
// ArmorReprojFactor 直接吃四个像素角点，优化单板 Pose3。

// X(k) = X(k-1) + V(k-1) * dt
class TranslationFactor
    : public gtsam::NoiseModelFactorN<gtsam::Point3, gtsam::Vector3,
                                      gtsam::Point3> {
  using Base =
      gtsam::NoiseModelFactorN<gtsam::Point3, gtsam::Vector3, gtsam::Point3>;

public:
  TranslationFactor(const gtsam::SharedNoiseModel &model, gtsam::Key x_pre,
                    gtsam::Key v_pre, gtsam::Key x_cur, double dt);
  gtsam::Vector evaluateError(const gtsam::Point3 &x_pre,
                              const gtsam::Vector3 &v_pre,
                              const gtsam::Point3 &x_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2,
                              gtsam::OptionalMatrixType H3) const override;

private:
  double dt_;
};

// R(k) = R(k-1) + W(k-1) * dt，注意 yaw 是周期量，残差要过 limit_rad。
class YawFactor
    : public gtsam::NoiseModelFactorN<gtsam::Rot2, double, gtsam::Rot2> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Rot2, double, gtsam::Rot2>;

public:
  YawFactor(const gtsam::SharedNoiseModel &model, gtsam::Key r_pre,
            gtsam::Key w_pre, gtsam::Key r_cur, double dt);
  gtsam::Vector evaluateError(const gtsam::Rot2 &r_pre, const double &w_pre,
                              const gtsam::Rot2 &r_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2,
                              gtsam::OptionalMatrixType H3) const override;

private:
  double dt_;
};

// V(k) = V(k-1)，过程噪声由 GtsamEst::Config::velocity_factor_sigma 给。
class VelocityFactor
    : public gtsam::NoiseModelFactorN<gtsam::Vector3, gtsam::Vector3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Vector3, gtsam::Vector3>;

public:
  VelocityFactor(const gtsam::SharedNoiseModel &model, gtsam::Key v_pre,
                 gtsam::Key v_cur);

  gtsam::Vector evaluateError(const gtsam::Vector3 &r_pre,
                              const gtsam::Vector3 &r_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2) const override;
};


// W(k) = W(k-1)。
class VyawFactor : public gtsam::NoiseModelFactorN<double, double> {
  using Base = gtsam::NoiseModelFactorN<double, double>;

public:
  VyawFactor(const gtsam::SharedNoiseModel &model, gtsam::Key w_pre,
             gtsam::Key w_cur);

  gtsam::Vector evaluateError(const double &w_pre, const double &w_cur,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2) const override;
};

// 偶数号板：位置 = 中心 - radiusA * [cos(angle), sin(angle), 0]，高度取中心 z。
class ArmorRadiusCenterZFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, double, gtsam::Rot2,
                                      gtsam::Point3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, double, gtsam::Rot2,
                                        gtsam::Point3>;

public:
  ArmorRadiusCenterZFactor(const gtsam::SharedNoiseModel &model,
                           gtsam::Key armor_pose_key, gtsam::Key radius_key,
                           gtsam::Key center_yaw_key,
                           gtsam::Key center_point_key,
                           const Eigen::Isometry3d &T_world_camera,
                           int armor_index, double radius_min,
                           double radius_max, int armor_count = 4);

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &armor_pose_camera, const double &radius,
                const gtsam::Rot2 &center_yaw,
                const gtsam::Point3 &center_point, gtsam::OptionalMatrixType H1,
                gtsam::OptionalMatrixType H2, gtsam::OptionalMatrixType H3,
                gtsam::OptionalMatrixType H4) const override;

private:
  Eigen::Isometry3d T_world_camera_;
  int armor_index_;
  double radius_min_, radius_max_;
  int armor_count_;
};

// 奇数号板：半径换成 radiusB，高度加 deltaZ。
class ArmorRadiusDZFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, double, double, gtsam::Rot2,
                                      gtsam::Point3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, double, double,
                                        gtsam::Rot2, gtsam::Point3>;

public:
  ArmorRadiusDZFactor(const gtsam::SharedNoiseModel &model,
                      gtsam::Key armor_pose_key, gtsam::Key radius_key,
                      gtsam::Key dz_key, gtsam::Key center_yaw_key,
                      gtsam::Key center_point_key,
                      const Eigen::Isometry3d &T_world_camera,
                      int armor_index, double radius_min,
                      double radius_max, int armor_count = 4);

  gtsam::Vector
  evaluateError(const gtsam::Pose3 &armor_pose_camera, const double &radius,
                const double &dz, const gtsam::Rot2 &center_yaw,
                const gtsam::Point3 &center_point, gtsam::OptionalMatrixType H1,
                gtsam::OptionalMatrixType H2, gtsam::OptionalMatrixType H3,
                gtsam::OptionalMatrixType H4,
                gtsam::OptionalMatrixType H5) const override;

private:
  Eigen::Isometry3d T_world_camera_;
  int armor_index_;
  double radius_min_, radius_max_;
  int armor_count_;
};

// 四角点重投影。相机内参走 gtsam::Cal3DS2，畸变系数从 CameraCalibration 转过来。
class ArmorReprojFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3>;

public:
  ArmorReprojFactor(const gtsam::SharedNoiseModel &model,
                    gtsam::Key armor_pose_key, const cv::Mat &camera_matrix,
                    const cv::Mat &distortion_coefficients,
                    ArmorType type, const ArmorConfig &armor_config,
                    int point_index,
                    Eigen::Vector2d px_point);
  gtsam::Vector evaluateError(const gtsam::Pose3 &armor_pose_camera,
                              gtsam::OptionalMatrixType H) const override;

private:
  gtsam::Point2 px_point_;
  gtsam::Point3 armor_point_;
  gtsam::Cal3DS2 calib_;
};

}  // namespace L3Estimation::GtsamEst

#endif  // NEWVISION_USE_GTSAM
