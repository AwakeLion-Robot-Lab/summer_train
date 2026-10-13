#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

#include <Eigen/Geometry>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

/**
 * @file
 * @brief 自瞄调试叠加层
 *
 * 实机 runtime 和离线回放共用同一份绘制，否则两边会漂——回放里看着对的东西
 * 实机上可能画错，那时候你会先怀疑算法。
 *
 * L6 依赖 L1~L5 是允许的方向（数据只向下游流，遥测在最下游）。
 */
namespace L6Telemetry {

/**
 * @brief 一帧要画的全部东西
 *
 * 全部是引用：叠加层关闭时调用方什么都不用构造。
 */
struct AimOverlayInput
{
  /// L2 检出的装甲板
  const std::vector<L2Perception::Armor>& detections;

  /// L3 的目标快照
  const std::optional<L3Estimation::EskfTarget>& target;

  /// 跟踪状态
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};

  /// 本帧规划
  const L4Planning::Plan& plan;

  /// 本帧开火判定
  const L5Control::FireDecision& fire;

  /// 曝光时刻的枪管姿态，用于把世界系点投回图像。
  const std::optional<Eigen::Quaterniond>& q_world_barrel;
};

/**
 * @brief 在原图上画叠加层
 *
 * 画检测角点、滤波器展开的整车、Plan 的命中板，以及一行状态文字。
 *
 * @param image 原图，就地修改
 * @param input 要画的内容
 * @param solver 用于重投影装甲板，需已设置当帧枪管姿态
 * @param calibration 相机标定
 */
void drawAimOverlay(
  cv::Mat& image, const AimOverlayInput& input,
  const L3Estimation::PnpSolver& solver,
  const L1Sensor::CameraCalibration& calibration);

// 以下是叠加层的组成部件，离线回放另外还要画代价曲线和外推框，所以单独导出。

/**
 * @brief 亚像素点四舍五入成整数像素
 * @param point 亚像素点
 * @return 整数像素点
 */
cv::Point toPixel(const cv::Point2f& point);

/**
 * @brief 画带黑色描边的文字，在任何背景上都看得清
 * @param image 目标图像，就地修改
 * @param text 文字
 * @param origin 文字左下角
 * @param color 文字颜色
 * @param scale 字号
 */
void drawOutlinedText(
  cv::Mat& image, const std::string& text, cv::Point origin,
  const cv::Scalar& color, double scale = 0.6);

/**
 * @brief 把一个世界系点投到图像上
 *
 * 整车旋转中心不是装甲板，用不了 reproject_armor。
 *
 * @param point_in_world 世界系点，meter
 * @param calibration 相机标定
 * @param q_world_barrel 曝光时刻的枪管姿态
 * @return 像素坐标；缺 T_barrel_camera、点在相机背后或投影失败时为空
 */
std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& point_in_world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel);

using ArmorCorners = std::array<cv::Point2f, 4>;

/**
 * @brief 按滤波器的完整姿态投影整车，角点顺序为左上、右上、右下、左下
 *
 * 与端点观测共用 ObsContext::project，保留车体倾斜及其对各板高度的影响。
 * 缺少外参、未初始化或投影无效时返回空。
 */
std::vector<ArmorCorners> projectVehicleArmors(
  const L3Estimation::EskfTarget& target,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel);

/** @brief 绘制保留完整姿态的整车，参数与 projectVehicleArmors 相同 */
void drawVehicle(
  cv::Mat& image, const L3Estimation::EskfTarget& target,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel, const cv::Scalar& color, int thickness,
  cv::Point image_offset = {});

/**
 * @brief 把一组整车装甲板位姿画成闭合四边形
 * @param image 目标图像，就地修改
 * @param armor_poses 每块板的 [x, y, z, yaw]，世界系
 * @param type 板型
 * @param name 车辆类别，决定后仰角
 * @param solver 用于重投影装甲板，需已设置当帧枪管姿态
 * @param color 线条颜色
 * @param thickness 线宽
 * @param image_offset 画到 image 上时整体平移的像素量（image 是裁剪图时用）
 */
void drawVehicle(
  cv::Mat& image, const std::vector<Eigen::Vector4d>& armor_poses,
  L3Estimation::ArmorType type, L3Estimation::ArmorName name,
  const L3Estimation::PnpSolver& solver, const cv::Scalar& color, int thickness,
  cv::Point image_offset = {});

}  // namespace L6Telemetry
