#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

#include <Eigen/Geometry>

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// 自瞄调试叠加层。实机 runtime 和离线回放共用同一份绘制，否则两边会漂——
// 回放里看着对的东西实机上可能画错，那时候你会先怀疑算法。
//
// L6 依赖 L1~L5 是允许的方向（数据只向下游流，遥测在最下游）。
namespace L6Telemetry {

// 一帧要画的全部东西，全部是引用：叠加层关闭时调用方什么都不用构造。
struct AimOverlayInput
{
  const std::vector<L2Perception::Armor>& detections;
  const std::optional<L3Estimation::EskfTarget>& target;
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  const L4Planning::Plan& plan;
  const L5Control::FireDecision& fire;
  // 曝光时刻的枪管姿态，用于把世界系点投回图像。
  const std::optional<Eigen::Quaterniond>& q_world_barrel;
};

// 在原图上画：检测角点、滤波器展开的整车、Plan 的命中板，以及一行状态文字。
// 就地修改 image。
void drawAimOverlay(
  cv::Mat& image, const AimOverlayInput& input,
  const L3Estimation::PnpSolver& solver,
  const L1Sensor::CameraCalibration& calibration);

// 以下是叠加层的组成部件，离线回放另外还要画代价曲线和外推框，所以单独导出。
cv::Point toPixel(const cv::Point2f& point);

void drawOutlinedText(
  cv::Mat& image, const std::string& text, cv::Point origin,
  const cv::Scalar& color, double scale = 0.6);

// 把一个世界系点投到图像上。整车旋转中心不是装甲板，用不了 reproject_armor。
std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& point_in_world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel);

// 把一组整车装甲板位姿画成闭合四边形。
void drawVehicle(
  cv::Mat& image, const std::vector<Eigen::Vector4d>& armor_poses,
  L3Estimation::ArmorType type, L3Estimation::ArmorName name,
  const L3Estimation::PnpSolver& solver, const cv::Scalar& color, int thickness,
  cv::Point image_offset = {});

}  // namespace L6Telemetry
