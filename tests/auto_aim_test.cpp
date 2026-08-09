// 离线回放测试，结构对照 sp_vision_25 的 tests/auto_aim_test.cpp：
// 读取 records/ 下的 avi + txt（每行 "t w x y z"），逐帧跑
// 识别 -> PnP -> 整车跟踪，并把当前观测的 PnP yaw 搜索代价曲线画出来。
//
// 代价曲线是这个测试存在的主要理由：PnpSolver::optimize_yaw 搜索使四角点
// 重投影平方和最小的世界系 yaw，曲线能直接看出该代价有几个坑、求解器落点
// 是不是全局最小点。代价在整周有两个极小值是常态而不是异常，所以横轴画
// 整周而不是开窗——开窗会把另一个坑藏起来，正好藏住最需要看见的东西。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/tracker.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/predictor.hpp"
#include "l5_control/controller.hpp"
#include "l5_control/fire_decision.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "l6_telemetry/udp_json_sender.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;
constexpr double kDegToRad = std::numbers::pi / 180.0;

// 曲线覆盖整周，与 optimize_yaw 的粗扫范围一致。
constexpr double kSearchRangeDegrees = 360.0;
// 曲线采样步长比求解器 10 度的粗扫细得多，用来核对高斯牛顿细化后的落点
// 是不是真的落在坑底，而不只是落在正确的坑里。
constexpr double kCostStepDegrees = 0.5;

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{model m | model/armor_model/armor.xml | OpenVINO 装甲板模型}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方颜色：red / blue / any}"
  "{convention | imu | 录像四元数约定：imu / sp}"
  "{serial-config | config/serial_config.yaml | convention=imu 时读 R_imu_barrel}"
  "{predict-time p | 0.1 | 整车预测外推时长（秒），<=0 表示不画预测}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{wait w | 30 | 每帧 waitKey 毫秒，0 表示逐帧手动推进}"
  "{view | sp | 叠加层：sp（只画当前 EKF 整车和瞄准板）/ full（全部调试层）}"
  "{overlay-offset | 0 | full 视图下整车叠加层上移的像素数；sp 视图恒为 0}"
  "{plot | auto | PnP 代价曲线窗口：auto（只在 view=full 时开）/ true / false}"
  "{require-quality | false | 观测是否必须通过全部 ArmorQuality 门限}"
  "{bullet-speed | 23.0 | 回放没有裁判系统数据，用这个弹速喂 L4（m/s）}"
  "{command-jump | 10.0 | 相邻帧命令 yaw 跳变超过该角度即判为 command_jump（度）}"
  "{fire-limits | false | 填一组占位火控参数，让 fire_feasible 时序可观测}"
  "{@input-path | records/3m_high | avi 和 txt 文件的路径（不含后缀）}";

struct PoseSample {
  double seconds{0.0};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

// 一条 yaw 搜索代价曲线：横轴是相对枪管 yaw 的偏角，纵轴是四角点重投影
// 像素距离之和，与 PnpSolver::armor_reprojection_error 的定义一致。
struct YawCostCurve {
  double barrel_yaw{std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> offsets_degrees;
  std::vector<double> costs;
  double best_offset_degrees{std::numeric_limits<double>::quiet_NaN()};
  double best_yaw{std::numeric_limits<double>::quiet_NaN()};
  double best_cost{std::numeric_limits<double>::infinity()};
  // 局部极小值个数。大于 1 说明代价不是单峰的，三分搜索这类假设单峰的
  // 算法会随初值落进不同的坑里。
  std::size_t local_minima{0};
};

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] std::string_view stateName(L3Estimation::TrackState state) noexcept
{
  switch (state) {
  case L3Estimation::TrackState::Lost:
    return "lost";
  case L3Estimation::TrackState::Detecting:
    return "detecting";
  case L3Estimation::TrackState::Tracking:
    return "tracking";
  case L3Estimation::TrackState::TempLost:
    return "temp_lost";
  }
  return "unknown";
}

[[nodiscard]] const char* phaseName(L4Planning::AimPhase phase) noexcept
{
  switch (phase) {
  case L4Planning::AimPhase::SingleArmor:
    return "single";
  case L4Planning::AimPhase::WholeCarArmor:
    return "car-armor";
  case L4Planning::AimPhase::WholeCarCenter:
    return "car-center";
  }
  return "unknown";
}

[[nodiscard]] const char* planErrorName(L4Planning::PlanError error) noexcept
{
  switch (error) {
  case L4Planning::PlanError::None:            return "none";
  case L4Planning::PlanError::NoTarget:        return "no-target";
  case L4Planning::PlanError::NotTracking:     return "not-tracking";
  case L4Planning::PlanError::NoArmor:         return "no-armor";
  case L4Planning::PlanError::NoCalibration:   return "no-calib";
  case L4Planning::PlanError::BadBulletSpeed:  return "bad-speed";
  case L4Planning::PlanError::BallisticFailed: return "ballistic";
  case L4Planning::PlanError::NotConverged:    return "not-converged";
  case L4Planning::PlanError::OutOfRange:      return "out-of-range";
  case L4Planning::PlanError::Switching:       return "switching";
  case L4Planning::PlanError::OutOfWindow:     return "out-of-window";
  }
  return "unknown";
}

// 拒绝原因拼成一行，画在图上。数值曲线看得出"没开火"，看不出"为什么"。
[[nodiscard]] std::string rejectReasons(const L5Control::FireDecision& decision)
{
  std::string text;
  for (const auto reason : decision.reasons) {
    if (!text.empty()) {
      text += ',';
    }
    text += L5Control::toString(reason);
  }
  return text.empty() ? std::string{"-"} : text;
}

[[nodiscard]] L2Perception::ArmorColor parseEnemyColor(const std::string& value)
{
  if (value == "red") {
    return L2Perception::ArmorColor::Red;
  }
  if (value == "blue") {
    return L2Perception::ArmorColor::Blue;
  }
  if (value == "any") {
    return L2Perception::ArmorColor::Unknown;
  }
  throw std::invalid_argument("enemy 必须是 red、blue 或 any");
}

[[nodiscard]] bool readPose(std::istream& input, PoseSample& sample)
{
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (!(input >> sample.seconds >> w >> x >> y >> z)) {
    return false;
  }
  sample.q = Eigen::Quaterniond{w, x, y, z};
  require(
    std::isfinite(sample.seconds) && sample.q.coeffs().allFinite() &&
      sample.q.squaredNorm() > 1e-12,
    "四元数文本包含非有限值");
  sample.q.normalize();
  return true;
}

// 录像里的四元数是 MCU 给出的 imu_abs 姿态。两种约定的差别只在于 barrel
// 轴向怎么定义：
//   sp  —— 录像和它配套的 T_barrel_camera 都按 sp_vision 标定，barrel 的
//          x、y 轴与 IMU 相反，且 sp 把 world 跟着 barrel 一起重标记了，
//          所以是双边相似变换 R^T * R_world_imu * R。
//   imu —— 本项目实机约定，world 固定为 imu_abs，只做单边复合
//          R_world_imu * R_imu_barrel（见 SerialWorker::gimbalPoseAt）。
//          R_imu_barrel 从 serial_config.yaml 读，与实机同一份数值，
//          不在这里另写一份。
// 两者自洽。用哪个取决于录像配套的 T_barrel_camera 是按哪套约定标定的：
// sp 的 barrel 与 IMU 差 180 度绕 z，本项目的 barrel 由 R_imu_barrel 描述。
[[nodiscard]] Eigen::Quaterniond toWorldBarrelPose(
  const PoseSample& sample,
  bool sp_convention,
  const Eigen::Matrix3d& R_imu_barrel)
{
  Eigen::Matrix3d R_sp_flip = Eigen::Matrix3d::Identity();
  R_sp_flip(0, 0) = -1.0;
  R_sp_flip(1, 1) = -1.0;
  const Eigen::Matrix3d R_world_imu = sample.q.toRotationMatrix();
  const Eigen::Matrix3d R_world_barrel = sp_convention
    ? Eigen::Matrix3d{R_sp_flip.transpose() * R_world_imu * R_sp_flip}
    : Eigen::Matrix3d{R_world_imu * R_imu_barrel};
  return Eigen::Quaterniond(R_world_barrel).normalized();
}

[[nodiscard]] cv::Point toPixel(const cv::Point2f& point)
{
  return {
    static_cast<int>(std::lround(point.x)),
    static_cast<int>(std::lround(point.y))};
}

void drawOutlinedText(
  cv::Mat& image,
  const std::string& text,
  cv::Point origin,
  const cv::Scalar& color,
  double scale = 0.6)
{
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 4,
    cv::LINE_AA);
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

// 把一个世界系点投到图像上。整车的旋转中心不是装甲板，用不了
// reproject_armor，所以这里单独走一次 world -> camera -> pixel。
[[nodiscard]] std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& point_in_world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  if (!calibration.T_barrel_camera || !point_in_world.allFinite()) {
    return std::nullopt;
  }
  Eigen::Isometry3d T_world_barrel = Eigen::Isometry3d::Identity();
  T_world_barrel.linear() = q_world_barrel.toRotationMatrix();
  const Eigen::Vector3d point_in_camera =
    (T_world_barrel * *calibration.T_barrel_camera).inverse() * point_in_world;
  // 相机后方的点投影出来是镜像的假点，直接丢掉。
  if (!point_in_camera.allFinite() || point_in_camera.z() <= 1e-6) {
    return std::nullopt;
  }

  std::vector<cv::Point2d> projected;
  try {
    cv::projectPoints(
      std::vector<cv::Point3d>{
        {point_in_camera.x(), point_in_camera.y(), point_in_camera.z()}},
      cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), calibration.camera_matrix,
      calibration.distortion_coefficients, projected);
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
  if (projected.size() != 1 || !std::isfinite(projected[0].x) ||
      !std::isfinite(projected[0].y)) {
    return std::nullopt;
  }
  return cv::Point2f(projected[0]);
}

// 把一组整车装甲板位姿画成闭合四边形。
void drawVehicle(
  cv::Mat& image,
  const std::vector<Eigen::Vector4d>& armor_poses,
  L3Estimation::ArmorType type,
  L3Estimation::ArmorName name,
  const L3Estimation::PnpSolver& solver,
  const cv::Scalar& color,
  int thickness,
  cv::Point image_offset = {})
{
  for (const Eigen::Vector4d& xyza : armor_poses) {
    const auto image_points =
      solver.reproject_armor(xyza.head<3>(), xyza[3], type, name);
    for (std::size_t index = 0; index < image_points.size(); ++index) {
      cv::line(
        image, toPixel(image_points[index]) + image_offset,
        toPixel(image_points[(index + 1) % image_points.size()]) + image_offset,
        color,
        thickness, cv::LINE_AA);
    }
  }
}

[[nodiscard]] bool isFilterInputArmor(
  const L3Estimation::Armor& armor,
  const L3Estimation::ArmorConfig& armor_config,
  bool require_quality)
{
  // 与 Tracker::observationUsable 保持一致，避免把被滤掉的坏解画出来。
  return armor.name != L3Estimation::ArmorName::Unknown &&
    armor.xyz_in_world.allFinite() &&
    std::isfinite(armor.ypr_in_world[0]) &&
    std::isfinite(armor.area) && armor.area >= armor_config.min_area &&
    (!require_quality || armor.quality.valid());
}

// 将当前帧实际送入目标滤波器的单板 PnP 位姿重投影为红框。
void drawFilterInputArmors(
  cv::Mat& image,
  const std::vector<L3Estimation::Armor>& observations,
  const L3Estimation::ArmorConfig& armor_config,
  bool require_quality,
  const L3Estimation::PnpSolver& solver,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  for (const auto& armor : observations) {
    if (!isFilterInputArmor(armor, armor_config, require_quality)) {
      continue;
    }

    const auto image_points = solver.reproject_armor(
      armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);
    if (image_points.size() != armor.points.size()) {
      continue;
    }
    for (std::size_t index = 0; index < image_points.size(); ++index) {
      cv::line(
        image, toPixel(image_points[index]),
        toPixel(image_points[(index + 1) % image_points.size()]),
        {0, 255, 0}, 2, cv::LINE_AA);
    }

    // armorPoints 使用局部 x=0 的 y-z 平面，因此局部 +x 是装甲板法向；
    // 它在世界系中的方向正是 yaw 所表示的朝向。
    const double pitch = armor.name == L3Estimation::ArmorName::Outpost
      ? -15.0 * std::numbers::pi / 180.0
      : 15.0 * std::numbers::pi / 180.0;
    const double yaw = armor.ypr_in_world[0];
    const Eigen::Vector3d normal_in_world{
      std::cos(yaw) * std::cos(pitch),
      std::sin(yaw) * std::cos(pitch),
      -std::sin(pitch)};
    constexpr double kArrowLengthMeters = 0.16;
    const auto arrow_start = projectWorldPoint(
      armor.xyz_in_world, calibration, q_world_barrel);
    const auto arrow_end = projectWorldPoint(
      armor.xyz_in_world + kArrowLengthMeters * normal_in_world,
      calibration, q_world_barrel);
    if (arrow_start && arrow_end && image_points.size() == 4) {
      // 透视投影不保持垂直关系。为了让图像上的箭头直观看起来垂直于
      // 装甲板，使用投影后长边的二维垂线；三维法向投影只用于决定正负方向。
      const cv::Point2f long_edge =
        (image_points[1] - image_points[0]) +
        (image_points[2] - image_points[3]);
      const double long_edge_length =
        std::hypot(long_edge.x, long_edge.y);
      if (long_edge_length > 1e-3) {
        cv::Point2f perpendicular{
          static_cast<float>(-long_edge.y / long_edge_length),
          static_cast<float>(long_edge.x / long_edge_length)};
        const cv::Point2f physical_direction =
          *arrow_end - *arrow_start;
        if (perpendicular.x * physical_direction.x +
              perpendicular.y * physical_direction.y < 0.0F) {
          perpendicular *= -1.0F;
        }

        const double short_edge_length =
          0.5 * (cv::norm(image_points[3] - image_points[0]) +
                 cv::norm(image_points[2] - image_points[1]));
        const double arrow_length =
          std::clamp(0.8 * short_edge_length, 12.0, 64.0);
        const cv::Point2f arrow_tip =
          *arrow_start + perpendicular * static_cast<float>(arrow_length);
        cv::arrowedLine(
          image, toPixel(*arrow_start), toPixel(arrow_tip),
          {0, 255, 0}, 2, cv::LINE_AA, 0, 0.25);
      }
    }
  }
}

[[nodiscard]] const char* armorClassName(L3Estimation::ArmorName name) noexcept
{
  switch (name) {
  case L3Estimation::ArmorName::Guard:
    return "G";
  case L3Estimation::ArmorName::Hero:
    return "1";
  case L3Estimation::ArmorName::Engineer:
    return "2";
  case L3Estimation::ArmorName::Infantry3:
    return "3";
  case L3Estimation::ArmorName::Infantry4:
    return "4";
  case L3Estimation::ArmorName::Infantry5:
    return "5";
  case L3Estimation::ArmorName::Outpost:
    return "O";
  case L3Estimation::ArmorName::BaseSmall:
    return "Bs";
  case L3Estimation::ArmorName::BaseLarge:
    return "Bb";
  case L3Estimation::ArmorName::Unknown:
    break;
  }
  return "?";
}

// 与 PnpSolver::armor_reprojection_error 定义一致：把装甲板按给定世界系
// yaw 重投影，取四个对应角点的像素距离之和。
[[nodiscard]] double yawCost(
  const L3Estimation::PnpSolver& solver,
  const L3Estimation::Armor& armor,
  double yaw)
{
  // 必须和 PnpSolver::yaw_squared_cost 用同一个定义（像素残差平方和），
  // 否则画出来的最小点不是求解器真正在找的那个，对账就失去意义。
  const std::vector<cv::Point2f> projected =
    solver.reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (projected.size() != armor.points.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double cost = 0.0;
  for (std::size_t index = 0; index < armor.points.size(); ++index) {
    const cv::Point2f difference = armor.points[index] - projected[index];
    cost += static_cast<double>(difference.x) * difference.x +
      static_cast<double>(difference.y) * difference.y;
  }
  return cost;
}

[[nodiscard]] YawCostCurve sampleYawCost(
  const L3Estimation::PnpSolver& solver,
  const L3Estimation::Armor& armor,
  const Eigen::Quaterniond& q_world_barrel)
{
  YawCostCurve curve;
  curve.barrel_yaw =
    L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0)[0];

  const auto sample_count =
    static_cast<int>(kSearchRangeDegrees / kCostStepDegrees) + 1;
  curve.offsets_degrees.reserve(sample_count);
  curve.costs.reserve(sample_count);
  for (int index = 0; index < sample_count; ++index) {
    const double offset =
      -kSearchRangeDegrees / 2.0 + index * kCostStepDegrees;
    const double yaw =
      L6Telemetry::limit_rad(curve.barrel_yaw + offset * kDegToRad);
    const double cost = yawCost(solver, armor, yaw);
    curve.offsets_degrees.push_back(offset);
    curve.costs.push_back(cost);
    if (cost < curve.best_cost) {
      curve.best_cost = cost;
      curve.best_offset_degrees = offset;
      curve.best_yaw = yaw;
    }
  }

  // 只统计内部的严格局部极小值，端点不算，避免把截断处误判成极小值。
  for (std::size_t index = 1; index + 1 < curve.costs.size(); ++index) {
    const double cost = curve.costs[index];
    if (!std::isfinite(cost)) {
      continue;
    }
    if (cost < curve.costs[index - 1] && cost <= curve.costs[index + 1]) {
      ++curve.local_minima;
    }
  }
  return curve;
}

// 选一块装甲板画代价曲线：优先跟踪器当前关联的那块，其次取图像中心附近的。
[[nodiscard]] std::optional<std::size_t> selectArmor(
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<L3Estimation::TargetState>& target,
  const std::vector<Eigen::Vector4d>& target_armor_poses,
  const cv::Size& image_size)
{
  std::optional<std::size_t> selected;
  double best_score = std::numeric_limits<double>::infinity();

  if (target && target->armor_id >= 0 &&
      static_cast<std::size_t>(target->armor_id) < target_armor_poses.size()) {
    const Eigen::Vector3d tracked =
      target_armor_poses[static_cast<std::size_t>(target->armor_id)].head<3>();
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (observations[index].name != target->name) {
        continue;
      }
      const double score =
        (observations[index].xyz_in_world - tracked).squaredNorm();
      if (score < best_score) {
        best_score = score;
        selected = index;
      }
    }
    if (selected) {
      return selected;
    }
  }

  const cv::Point2f image_center{
    static_cast<float>(image_size.width) * 0.5F,
    static_cast<float>(image_size.height) * 0.5F};
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const cv::Point2f difference = observations[index].center - image_center;
    const double score = static_cast<double>(difference.x) * difference.x +
      static_cast<double>(difference.y) * difference.y;
    if (score < best_score) {
      best_score = score;
      selected = index;
    }
  }
  return selected;
}

// 代价曲线窗口。横轴是相对枪管 yaw 的偏角，竖线标出几个关键 yaw 的位置。
[[nodiscard]] cv::Mat drawCostPlot(
  const YawCostCurve* curve,
  const L3Estimation::Armor* armor,
  const L3Estimation::PnpSolver& solver,
  const std::optional<double>& ekf_armor_yaw,
  const std::optional<double>& predicted_armor_yaw,
  int frame_index)
{
  constexpr int kWidth = 960;
  constexpr int kHeight = 540;
  const cv::Rect graph{74, 132, kWidth - 74 - 24, kHeight - 132 - 56};
  cv::Mat plot(kHeight, kWidth, CV_8UC3, cv::Scalar{24, 24, 24});

  drawOutlinedText(
    plot, cv::format("frame=%d  PnP yaw-search cost", frame_index), {12, 28},
    {255, 255, 255}, 0.62);
  if (curve == nullptr || armor == nullptr) {
    drawOutlinedText(plot, "no PnP observation", {12, 62}, {0, 165, 255});
    return plot;
  }

  drawOutlinedText(
    plot,
    cv::format(
      "armor=%s %s  xyz=(%.2f,%.2f,%.2f)m  dist=%.2fm",
      armorClassName(armor->name),
      armor->type == L3Estimation::ArmorType::Big ? "big" : "small",
      armor->xyz_in_world.x(), armor->xyz_in_world.y(),
      armor->xyz_in_world.z(), armor->xyz_in_world.norm()),
    {12, 58}, {200, 200, 200}, 0.52);
  drawOutlinedText(
    plot,
    cv::format(
      "raw=%.1fdeg  filter input=%.1fdeg  curve min=%.1fdeg (cost=%.1fpx)",
      armor->yaw_raw * kRadToDeg, armor->ypr_in_world[0] * kRadToDeg,
      curve->best_yaw * kRadToDeg, curve->best_cost),
    {12, 84}, {0, 255, 255}, 0.52);
  drawOutlinedText(
    plot,
    cv::format(
      "local minima=%zu  step=%.1fdeg  range=+-%.0fdeg  gimbal yaw=%.1fdeg",
      curve->local_minima, kCostStepDegrees, kSearchRangeDegrees / 2.0,
      curve->barrel_yaw * kRadToDeg),
    {12, 110}, curve->local_minima > 1 ? cv::Scalar{0, 165, 255}
                                       : cv::Scalar{200, 200, 200},
    0.52);

  std::vector<double> finite_costs;
  finite_costs.reserve(curve->costs.size());
  std::copy_if(
    curve->costs.begin(), curve->costs.end(),
    std::back_inserter(finite_costs),
    [](double cost) { return std::isfinite(cost); });
  if (finite_costs.empty()) {
    drawOutlinedText(
      plot, "cost curve unavailable", {graph.x + 16, graph.y + 32},
      {0, 165, 255});
    return plot;
  }

  const auto [min_it, max_it] =
    std::minmax_element(finite_costs.begin(), finite_costs.end());
  double min_cost = *min_it;
  double max_cost = *max_it;
  if (max_cost - min_cost < 1e-9) {
    max_cost = min_cost + 1.0;
  }
  const double padding = 0.05 * (max_cost - min_cost);
  min_cost = std::max(0.0, min_cost - padding);
  max_cost += padding;

  const double min_offset = curve->offsets_degrees.front();
  const double max_offset = curve->offsets_degrees.back();
  const auto x_of = [&](double offset) {
    return graph.x + static_cast<int>(std::lround(
      (offset - min_offset) / (max_offset - min_offset) * graph.width));
  };
  const auto y_of = [&](double cost) {
    const double clamped = std::clamp(cost, min_cost, max_cost);
    return graph.y + graph.height - static_cast<int>(std::lround(
      (clamped - min_cost) / (max_cost - min_cost) * graph.height));
  };

  cv::rectangle(plot, graph, {100, 100, 100}, 1, cv::LINE_AA);
  for (int index = 0; index <= 4; ++index) {
    const double ratio = index / 4.0;
    const int x = graph.x + static_cast<int>(std::lround(ratio * graph.width));
    const int y = graph.y + static_cast<int>(std::lround(ratio * graph.height));
    cv::line(plot, {x, graph.y}, {x, graph.y + graph.height}, {55, 55, 55}, 1);
    cv::line(plot, {graph.x, y}, {graph.x + graph.width, y}, {55, 55, 55}, 1);
    cv::putText(
      plot, cv::format("%.0f", min_offset + ratio * (max_offset - min_offset)),
      {x - 14, graph.y + graph.height + 22}, cv::FONT_HERSHEY_SIMPLEX, 0.44,
      {190, 190, 190}, 1, cv::LINE_AA);
    cv::putText(
      plot, cv::format("%.1f", max_cost - ratio * (max_cost - min_cost)),
      {6, y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {190, 190, 190}, 1,
      cv::LINE_AA);
  }
  cv::putText(
    plot, "yaw offset from gimbal [deg] / cost = sum of 4 corner distances [px]",
    {graph.x, kHeight - 14}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {220, 220, 220}, 1,
    cv::LINE_AA);

  std::optional<cv::Point> previous;
  for (std::size_t index = 0; index < curve->costs.size(); ++index) {
    if (!std::isfinite(curve->costs[index])) {
      previous.reset();
      continue;
    }
    const cv::Point point{
      x_of(curve->offsets_degrees[index]), y_of(curve->costs[index])};
    if (previous) {
      cv::line(plot, *previous, point, {230, 230, 230}, 2, cv::LINE_AA);
    }
    previous = point;
  }
  cv::circle(
    plot, {x_of(curve->best_offset_degrees), y_of(curve->best_cost)}, 6,
    {0, 0, 255}, cv::FILLED, cv::LINE_AA);

  // 各个 yaw 换算成相对枪管的偏角后画竖线，落在搜索窗口外的不画。
  const auto draw_marker = [&](double yaw, const cv::Scalar& color,
                               const std::string& label, int row) {
    if (!std::isfinite(yaw)) {
      return;
    }
    const double offset = L6Telemetry::limit_rad(yaw - curve->barrel_yaw) *
      kRadToDeg;
    if (offset < min_offset || offset > max_offset) {
      return;
    }
    const int x = x_of(offset);
    cv::line(plot, {x, graph.y}, {x, graph.y + graph.height}, color, 1,
             cv::LINE_AA);
    cv::putText(
      plot, label, {x + 3, graph.y + 16 + row * 16}, cv::FONT_HERSHEY_SIMPLEX,
      0.42, color, 1, cv::LINE_AA);
  };
  const auto draw_cost_marker = [&](double yaw, double cost,
                                    const cv::Scalar& color) {
    if (!std::isfinite(yaw) || !std::isfinite(cost)) {
      return;
    }
    const double offset = L6Telemetry::limit_rad(yaw - curve->barrel_yaw) *
      kRadToDeg;
    if (offset < min_offset || offset > max_offset) {
      return;
    }
    cv::circle(
      plot, {x_of(offset), y_of(cost)}, 6, color, cv::FILLED, cv::LINE_AA);
  };
  draw_marker(armor->yaw_raw, {255, 255, 0}, "raw", 0);
  draw_marker(armor->ypr_in_world[0], {0, 255, 0}, "input", 1);
  draw_cost_marker(
    armor->ypr_in_world[0],
    yawCost(solver, *armor, armor->ypr_in_world[0]),
    {0, 255, 0});
  if (ekf_armor_yaw) {
    draw_marker(*ekf_armor_yaw, {0, 255, 0}, "ekf", 2);
  }
  if (predicted_armor_yaw) {
    draw_marker(*predicted_armor_yaw, {0, 165, 255}, "pred", 3);
  }

  return plot;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
    if (cli.get<bool>("help")) {
      cli.printMessage();
      return 0;
    }
    if (!cli.check()) {
      cli.printErrors();
      return 1;
    }

    const std::filesystem::path input_path{cli.get<std::string>(0)};
    const std::string video_path = input_path.string() + ".avi";
    const std::string text_path = input_path.string() + ".txt";
    const auto enemy_color = parseEnemyColor(cli.get<std::string>("enemy"));
    const std::string convention = cli.get<std::string>("convention");
    require(
      convention == "sp" || convention == "imu",
      "convention 必须是 sp 或 imu");
    const bool sp_convention = convention == "sp";
    // 单边复合分支用实机那份 R_imu_barrel，缺省即单位阵。
    const Eigen::Matrix3d R_imu_barrel = sp_convention
      ? Eigen::Matrix3d::Identity()
      : L1Sensor::loadSerialConfig(cli.get<std::string>("serial-config"))
          .R_imu_barrel;
    const double predict_time = cli.get<double>("predict-time");
    require(std::isfinite(predict_time), "predict-time 必须是有限值");
    const double bullet_speed = cli.get<double>("bullet-speed");
    require(
      std::isfinite(bullet_speed) && bullet_speed > 0.0,
      "bullet-speed 必须是正的有限值");
    const double command_jump_rad = cli.get<double>("command-jump") * kDegToRad;
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const int wait_ms = cli.get<int>("wait");

    // 叠加层口径。sp 视图刻意只保留 sp_vision auto_aim_test 画的那两样东西：
    // 当前 EKF 展开的全部装甲板（绿），和命中时刻瞄准的那块板（红）。这样
    // "框贴不贴板"才是可以直接目视判断的——多画一层前瞻框或者整体偏移，
    // 看到的就不再是姿态估计的对错，而是显示口径的差异。
    const std::string view = cli.get<std::string>("view");
    require(view == "sp" || view == "full", "view 必须是 sp 或 full");
    const bool full_view = view == "full";
    const int overlay_offset = full_view ? cli.get<int>("overlay-offset") : 0;
    require(overlay_offset >= 0, "overlay-offset 不能为负");
    // 代价曲线是 sp 没有的第二个窗口，sp 视图下除非显式要求否则不开。
    // 用三态字符串而不是 cli.has("plot")：CommandLineParser 对带默认值的键
    // 恒返回 true，has() 区分不出"用户写了"和"用了默认值"。
    const std::string plot_option = cli.get<std::string>("plot");
    require(
      plot_option == "auto" || plot_option == "true" || plot_option == "false",
      "plot 必须是 auto、true 或 false");
    const bool show_plot =
      plot_option == "auto" ? full_view : plot_option == "true";
    require(start_index >= 0, "start-index 不能为负");
    require(
      end_index == 0 || end_index >= start_index,
      "end-index 必须为 0 或不小于 start-index");

    L6Telemetry::initLogger();

    const std::string calibration_path = cli.get<std::string>("calibration");
    const YAML::Node calibration_yaml = YAML::LoadFile(calibration_path);
    require(
      static_cast<bool>(calibration_yaml["calibration"]),
      calibration_path + " 里没有 calibration: 节点");
    const auto calibration = L1Sensor::loadCameraCalibration(
      calibration_yaml["calibration"], calibration_path);
    // 没有 T_barrel_camera 就没有世界系位姿，代价曲线和整车都无从谈起。
    // 按"缺失标定保持缺失"的约定，这里直接失败，不拿单位阵顶替。
    require(
      calibration.barrelExtrinsicsReady(),
      calibration_path +
        " 缺少 T_barrel_camera，PnP 无法给出世界系位姿；"
        "先补标外参，或用 -c=tests/data/sp_auto_aim/camera_calibration.yaml");

    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = cli.get<std::string>("model");
    model_config.device = cli.get<std::string>("device");
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);
    require(backend->ready(), "OpenVINO 后端未就绪");
    L2Perception::ArmorDetector detector(std::move(backend));
    require(detector.ready(), "ArmorDetector 未就绪");

    const L3Estimation::ArmorConfig armor_config;
    L3Estimation::TrackerConfig tracker_config;
    // 回放默认只以 PnP 成功为门限，否则质量位没置起来时跟踪器一帧都不会起步，
    // 整车和预测叠加也就无从显示。要复现实机行为加 --require-quality=true。
    tracker_config.require_quality = cli.get<bool>("require-quality");
    L3Estimation::Tracker tracker(calibration, armor_config, tracker_config);
    require(tracker.ready(), "Tracker 拒绝了该标定");
    // 与 Tracker 内部同参数的求解器，只用来做重投影和代价曲线，不参与滤波。
    L3Estimation::PnpSolver solver(calibration, armor_config);
    require(solver.ready(), "诊断用 PnpSolver 拒绝了该标定");
    // L4 的整车外推：恒速度 + 恒角速度，中心和整车 yaw 一起推进。
    const L4Planning::Predictor predictor;
    // 完整的 L4 -> L5 链路。回放里这条链路第一次真正跑起来：runtime 目前在
    // tracker 之后就 break 了，planner_smoke 也只喂合成数据。
    L4Planning::Planner planner;
    // 默认 FireConfig 全是 nullopt，parametersReady() 恒假，fire_feasible 会
    // 一直是 0——这正是实机未标定时该有的行为。想在回放里看清"火控窗口什么时候
    // 打开"，用 --fire-limits=true 填一组占位值。它们不是标定结果，绝不能搬到
    // 实机配置里；shoot_enable 无论如何保持 false。
    L5Control::FireConfig fire_config;
    if (cli.get<bool>("fire-limits")) {
      fire_config.bullet_diameter = 0.017;
      fire_config.min_bullet_speed = 10.0;
      fire_config.max_bullet_speed = 32.0;
      fire_config.heat_limit = 1e9;
      fire_config.min_yaw = -std::numbers::pi;
      fire_config.max_yaw = std::numbers::pi;
      fire_config.min_pitch = -std::numbers::pi / 4.0;
      fire_config.max_pitch = std::numbers::pi / 4.0;
      std::cout << "已启用占位火控参数：fire_feasible 可观测，shoot 仍被 "
                   "shoot_enable=false 拦住\n";
    }
    const L5Control::FireDecider fire_decider{fire_config};
    const L5Control::Controller controller;

    cv::VideoCapture video(video_path);
    require(video.isOpened(), "无法打开录像：" + video_path);
    std::ifstream text(text_path);
    require(text.is_open(), "无法打开四元数文本：" + text_path);

    L6Telemetry::UdpJsonSender plotter;

    // 跳过 start-index 之前的帧，视频和文本必须同步前进。
    video.set(cv::CAP_PROP_POS_FRAMES, start_index);
    PoseSample skipped;
    for (int index = 0; index < start_index; ++index) {
      require(readPose(text, skipped), "四元数文本在 start-index 之前结束");
    }

    cv::Mat img;
    PoseSample pose;
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t frames = 0;
    std::size_t observation_frames = 0;
    std::size_t valid_pnp_observations = 0;
    std::size_t tracking_frames = 0;
    std::size_t multi_minimum_frames = 0;
    std::size_t plan_valid_frames = 0;
    std::size_t command_frames = 0;
    std::size_t fire_feasible_frames = 0;
    std::size_t plan_switch_frames = 0;
    std::size_t command_jump_frames = 0;
    // 拒绝原因直方图。fire_feasible 是 0 时，唯一有用的信息是"被哪一条拦住的"。
    std::map<L5Control::RejectReason, std::size_t> reject_histogram;
    // 回放里云台姿态来自录像，不是本规划器闭环出来的，所以 aim_error 基本必然
    // 触发。把误差量级和容差一起打出来，才能判断是"云台没跟"还是"规划跑偏"。
    std::vector<double> aim_yaw_errors;
    // 上一帧真正下发的命令，用来判 armor_switching 和 command_jump。
    int last_plan_armor_id = -1;
    std::optional<double> last_command_yaw;
    bool paused = false;

    for (int frame_index = start_index;; ++frame_index) {
      if (paused) {
        const int key = cv::waitKey(0);
        if (key == 'q' || key == 27) {
          break;
        }
        if (key == ' ') {
          paused = false;
        }
        continue;
      }
      if (end_index > 0 && frame_index > end_index) {
        break;
      }
      video.read(img);
      if (img.empty()) {
        break;
      }
      if (!readPose(text, pose)) {
        std::cout << "四元数文本已结束，回放停在最后一组配对帧\n";
        break;
      }
      if (frames == 0) {
        require(
          calibration.matchesImageSize(img.size()),
          "录像分辨率与标定不一致");
      }
      ++frames;

      const auto timestamp =
        t0 + std::chrono::microseconds(static_cast<long long>(pose.seconds * 1e6));
      const Eigen::Quaterniond q_world_barrel =
        toWorldBarrelPose(pose, sp_convention, R_imu_barrel);

      /// 自瞄核心逻辑

      const auto detect_start = std::chrono::steady_clock::now();
      auto armors = detector.detect(img);
      std::erase_if(armors, [enemy_color](const L2Perception::Armor& armor) {
        return enemy_color != L2Perception::ArmorColor::Unknown &&
          armor.color != enemy_color;
      });

      const auto track_start = std::chrono::steady_clock::now();
      solver.set_R_world_barrel(q_world_barrel);
      const auto target = tracker.track(armors, q_world_barrel, timestamp);
      const auto target_armor_poses = tracker.targetArmorPoses();
      const auto track_end = std::chrono::steady_clock::now();

      /// PnP 代价曲线

      const auto& observations = tracker.observations();
      const auto selected = selectArmor(
        observations, target, target_armor_poses, calibration.image_size);
      std::optional<YawCostCurve> curve;
      if (selected) {
        curve = sampleYawCost(solver, observations[*selected], q_world_barrel);
        if (curve->local_minima > 1) {
          ++multi_minimum_frames;
        }
      }
      std::optional<double> ekf_armor_yaw;
      if (target && target->armor_id >= 0 &&
          static_cast<std::size_t>(target->armor_id) <
            target_armor_poses.size()) {
        ekf_armor_yaw =
          target_armor_poses[static_cast<std::size_t>(target->armor_id)].w();
      }

      /// 整车预测：把当前 EKF 状态外推 predict_time 秒后重新展开所有装甲板
      std::optional<L3Estimation::TargetState> predicted;
      std::vector<Eigen::Vector4d> predicted_armor_poses;
      std::optional<double> predicted_armor_yaw;
      if (target && predict_time > 0.0) {
        predicted = predictor.predict(*target, predict_time);
        predicted_armor_poses = predictor.armorPoses(*predicted);
        if (target->armor_id >= 0 &&
            static_cast<std::size_t>(target->armor_id) <
              predicted_armor_poses.size()) {
          predicted_armor_yaw =
            predicted_armor_poses[static_cast<std::size_t>(target->armor_id)]
              .w();
        }
      }

      /// L4 规划 -> L5 火控 -> 串口命令

      // 回放没有裁判系统数据，弹速由命令行给定；模式和敌色按当前回放设定填，
      // 其余字段保持默认。这份 RobotState 是合成的，真实性仅限于弹速和姿态。
      L1Sensor::RobotState robot_state;
      robot_state.bullet_speed = bullet_speed;
      robot_state.enemy_color = enemy_color == L2Perception::ArmorColor::Red
        ? L1Sensor::EnemyColor::Red
        : enemy_color == L2Perception::ArmorColor::Blue
        ? L1Sensor::EnemyColor::Blue
        : L1Sensor::EnemyColor::Unknown;
      robot_state.mode = L1Sensor::WorkMode::AutoAim;
      const Eigen::Vector3d gimbal_ypr =
        L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0);
      robot_state.rpy.yaw = gimbal_ypr[0];
      robot_state.rpy.pitch = gimbal_ypr[1];
      robot_state.rpy.roll = gimbal_ypr[2];
      robot_state.timestamp = timestamp;

      // plan_time 用本帧实测的检测 + 跟踪耗时，而不是曝光时刻本身：
      // Planner 拿它算 image_to_plan，填曝光时刻等于白送一段延迟补偿。
      const auto plan_time = timestamp + (track_end - detect_start);
      const auto plan = planner.plan(target, robot_state, plan_time);

      L5Control::FireInput fire_input;
      fire_input.target = target;
      fire_input.plan = plan;
      fire_input.robot_state = robot_state;
      fire_input.now = plan_time;
      // 命中判据必须拿云台**实际**指向来比，这里就是录像里那份四元数。
      fire_input.actual_yaw = gimbal_ypr[0];
      fire_input.actual_pitch = gimbal_ypr[1];
      fire_input.calibration_ready = true;
      // 回放里 txt 与视频逐帧对齐，不存在陈旧数据；实机这两位由 SerialWorker
      // 的时间戳年龄决定，不能照抄这里的 true。
      fire_input.serial_fresh = true;
      fire_input.gimbal_pose_fresh = true;
      fire_input.armor_switching =
        plan.valid && last_plan_armor_id >= 0 && plan.armor_id != last_plan_armor_id;
      fire_input.command_jump = plan.valid && last_command_yaw &&
        std::abs(L6Telemetry::limit_rad(plan.yaw - *last_command_yaw)) >
          command_jump_rad;

      const auto fire_decision = fire_decider.decide(fire_input);
      const auto command = controller.makeCommand(plan, fire_decision);

      if (plan.valid) {
        ++plan_valid_frames;
        last_plan_armor_id = plan.armor_id;
        last_command_yaw = plan.yaw;
      } else {
        last_plan_armor_id = -1;
        last_command_yaw.reset();
      }
      if (command) {
        ++command_frames;
      }
      if (fire_decision.fire_feasible) {
        ++fire_feasible_frames;
      }
      if (fire_input.armor_switching) {
        ++plan_switch_frames;
      }
      if (fire_input.command_jump) {
        ++command_jump_frames;
      }
      for (const auto reason : fire_decision.reasons) {
        ++reject_histogram[reason];
      }
      if (plan.valid && fire_decision.tolerance.valid) {
        aim_yaw_errors.push_back(fire_decision.yaw_error * kRadToDeg);
      }

      if (!observations.empty()) {
        ++observation_frames;
      }
      for (const auto& observation : observations) {
        if (observation.quality.valid()) {
          ++valid_pnp_observations;
        }
      }
      if (tracker.state() == L3Estimation::TrackState::Tracking) {
        ++tracking_frames;
      }

      /// 调试输出

      L6Telemetry::logDebug(
        "[", frame_index, "] detect:",
        L6Telemetry::delta_time(track_start, detect_start) * 1e3,
        "ms tracker:",
        L6Telemetry::delta_time(track_end, track_start) * 1e3, "ms");

      if (full_view) {
        // L2 原始识别框：按识别颜色绘制，便于和红色滤波器输入位姿框对比。
        for (const auto& armor : armors) {
          const cv::Scalar color = armor.color == L2Perception::ArmorColor::Blue
            ? cv::Scalar{0, 0, 255}
            : armor.color == L2Perception::ArmorColor::Red
            ? cv::Scalar{255, 0, 0}
            : cv::Scalar{0, 255, 255};
          for (std::size_t index = 0; index < armor.corners.size(); ++index) {
            cv::line(
              img, toPixel(armor.corners[index]),
              toPixel(armor.corners[(index + 1) % armor.corners.size()]),
              color, 2, cv::LINE_AA);
          }
        }

        // 当前帧真正送入滤波器的位姿：绿色重投影框和绿色朝向箭头。
        drawFilterInputArmors(
          img, observations, armor_config, tracker_config.require_quality, solver,
          calibration, q_world_barrel);
      }

      // 绿色是当前 EKF 展开的全部物理装甲板，和 sp_vision 画的是同一个量：
      // 直接压在图像上，不偏移、不前瞻，所以"贴不贴板"可以目视判断。
      // full 视图额外画橙色的 predict_time 外推框，那是延迟补偿的目标位置，
      // 本来就该领先绿框（100 ms 实测约 40 px），不要当成估计误差。
      if (target) {
        const auto armor_type =
          L3Estimation::armorTypeOf(target->name).value_or(
            L3Estimation::ArmorType::Small);
        const cv::Point overlay_shift{0, -overlay_offset};
        if (full_view) {
          drawVehicle(
            img, predicted_armor_poses, armor_type, target->name, solver,
            {0, 165, 255}, 2, overlay_shift);
        }
        drawVehicle(
          img, target_armor_poses, armor_type, target->name, solver,
          {0, 255, 0}, 2, overlay_shift);

        // 红色是命中时刻真正瞄的那块板，对应 sp_vision 的 debug_aim_point。
        // 用规划自己的时域（延迟 + 飞行时间）重新展开整车再取 armor_id，
        // 而不是复用上面那个按 --predict-time 外推的快照——两者时域不同，
        // 混用会画出一块规划从没瞄过的板。
        if (plan.valid && plan.armor_id >= 0) {
          const double aim_time = plan.delay.beforeFire() + plan.fly_time;
          const auto aim_poses = predictor.armorPosesAt(*target, aim_time);
          if (static_cast<std::size_t>(plan.armor_id) < aim_poses.size()) {
            drawVehicle(
              img, {aim_poses[static_cast<std::size_t>(plan.armor_id)]},
              armor_type, target->name, solver, {0, 0, 255}, 2, overlay_shift);
          }
        }
      }

      // 瞄准点和火控判据用的那块实体板。两者在 WholeCarCenter 档会明显分开
      // ——瞄的是旋转圆上的代理点，判的是板。sp 没有这一层。
      if (full_view && plan.valid) {
        const auto aim_pixel =
          projectWorldPoint(plan.aim_point, calibration, q_world_barrel);
        if (aim_pixel) {
          const cv::Point center = toPixel(*aim_pixel);
          const cv::Scalar color = fire_decision.fire_feasible
            ? cv::Scalar{0, 255, 255}
            : cv::Scalar{160, 160, 160};
          cv::line(img, center + cv::Point{-14, 0}, center + cv::Point{14, 0}, color, 2,
                   cv::LINE_AA);
          cv::line(img, center + cv::Point{0, -14}, center + cv::Point{0, 14}, color, 2,
                   cv::LINE_AA);
          cv::circle(img, center, 18, color, 2, cv::LINE_AA);
        }
        if (plan.fire_armor_id >= 0) {
          const auto fire_pixel =
            projectWorldPoint(plan.fire_armor_point, calibration, q_world_barrel);
          if (fire_pixel) {
            cv::circle(img, toPixel(*fire_pixel), 9, {255, 0, 255}, 2, cv::LINE_AA);
          }
        }
      }

      drawOutlinedText(
        img,
        cv::format(
          "frame=%d state=%s det=%zu obs=%zu", frame_index,
          std::string(stateName(tracker.state())).c_str(), armors.size(),
          observations.size()),
        {10, 32}, {255, 255, 255});
      drawOutlinedText(
        img,
        cv::format(
          "gimbal yaw=%.2fdeg",
          L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0)[0] *
            kRadToDeg),
        {10, 62}, {255, 255, 255});
      if (full_view && selected && isFilterInputArmor(
            observations[*selected], armor_config, tracker_config.require_quality)) {
        const auto& armor = observations[*selected];
        drawOutlinedText(
          img,
          cv::format("filter input armor yaw=%.1fdeg",
                     armor.ypr_in_world[0] * kRadToDeg),
          {10, 92}, {0, 255, 0});
      }
      if (full_view && target) {
        drawOutlinedText(
          img,
          cv::format(
            "EKF center=(%.2f,%.2f,%.2f)m v=(%.2f,%.2f,%.2f)m/s yaw=%.1fdeg "
            "v_yaw=%.2frad/s r=%.3fm id=%d",
            target->position.x(), target->position.y(), target->position.z(),
            target->velocity.x(), target->velocity.y(), target->velocity.z(),
            target->yaw * kRadToDeg, target->v_yaw, target->radius,
            target->armor_id),
          {10, 122}, {0, 255, 0}, 0.55);
      }
      if (full_view && predicted) {
        drawOutlinedText(
          img,
          cv::format(
            "pred +%.0fms center=(%.2f,%.2f,%.2f)m yaw=%.1fdeg (delta=%.1fdeg)",
            predict_time * 1e3, predicted->position.x(),
            predicted->position.y(), predicted->position.z(),
            predicted->yaw * kRadToDeg,
            L6Telemetry::limit_rad(predicted->yaw - target->yaw) * kRadToDeg),
          {10, 152}, {0, 165, 255}, 0.55);
      }
      drawOutlinedText(
        img,
        plan.valid
          ? cv::format(
              "CMD yaw=%.2f pitch=%.2f deg | err yaw=%.2f pitch=%.2f | "
              "phase=%s armor=%d fire_armor=%d",
              plan.yaw * kRadToDeg, plan.pitch * kRadToDeg,
              L6Telemetry::limit_rad(plan.yaw - gimbal_ypr[0]) * kRadToDeg,
              L6Telemetry::limit_rad(plan.pitch - gimbal_ypr[1]) * kRadToDeg,
              phaseName(plan.aim_phase), plan.armor_id, plan.fire_armor_id)
          : cv::format("CMD not sent (plan %s)", planErrorName(plan.error)),
        {10, full_view ? 182 : 92},
        plan.valid ? cv::Scalar{0, 255, 255} : cv::Scalar{160, 160, 160}, 0.55);
      if (full_view) {
        drawOutlinedText(
          img,
          cv::format(
            "FIRE feasible=%d shoot=%d | %s", fire_decision.fire_feasible ? 1 : 0,
            command && command->shoot ? 1 : 0, rejectReasons(fire_decision).c_str()),
          {10, 212},
          fire_decision.fire_feasible ? cv::Scalar{0, 255, 0} : cv::Scalar{160, 160, 160},
          0.5);
      }
      nlohmann::json data;
      data["gimbal_yaw"] =
        L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0)[0] *
        kRadToDeg;
      data["armor_num"] = armors.size();

      // 装甲板原始观测数据
      if (selected) {
        const auto& armor = observations[*selected];
        data["armor_x"] = armor.xyz_in_world[0];
        data["armor_y"] = armor.xyz_in_world[1];
        data["armor_z"] = armor.xyz_in_world[2];
        data["armor_yaw"] = armor.ypr_in_world[0] * kRadToDeg;
        data["armor_yaw_raw"] = armor.yaw_raw * kRadToDeg;
        data["armor_distance"] = armor.xyz_in_world.norm();
        data["armor_quality_valid"] = armor.quality.valid();
      }

      // PnP yaw 搜索代价
      if (curve) {
        data["cost_min"] = curve->best_cost;
        data["cost_min_yaw"] = curve->best_yaw * kRadToDeg;
        data["cost_min_offset"] = curve->best_offset_degrees;
        data["cost_local_minima"] = curve->local_minima;
        if (selected) {
          data["cost_at_solver_yaw"] =
            yawCost(solver, observations[*selected],
                    observations[*selected].ypr_in_world[0]);
          data["cost_at_raw_yaw"] =
            yawCost(solver, observations[*selected],
                    observations[*selected].yaw_raw);
        }
      }

      // 观测器内部数据
      if (target) {
        data["x"] = target->position.x();
        data["vx"] = target->velocity.x();
        data["y"] = target->position.y();
        data["vy"] = target->velocity.y();
        data["z"] = target->position.z();
        data["vz"] = target->velocity.z();
        data["a"] = target->yaw * kRadToDeg;
        data["w"] = target->v_yaw;
        data["r"] = target->radius;
        data["last_id"] = target->armor_id;
        data["nis"] = target->nis;
        if (ekf_armor_yaw) {
          data["ekf_armor_yaw"] = *ekf_armor_yaw * kRadToDeg;
        }
      }

      // L4 -> L5：这才是真正决定下位机动作的一组量。
      // cmd_yaw 是 world 系绝对方位角，和 gimbal_yaw 同一个基准，可以直接相减。
      data["plan_valid"] = plan.valid ? 1 : 0;
      data["plan_error"] = static_cast<int>(plan.error);
      data["plan_armor_id"] = plan.armor_id;
      data["plan_aim_phase"] = static_cast<int>(plan.aim_phase);
      data["plan_aim_on_armor"] = plan.aim_on_armor ? 1 : 0;
      data["fire_armor_id"] = plan.fire_armor_id;
      data["fire_admissible"] = plan.fire_admissible ? 1 : 0;
      if (plan.valid) {
        data["cmd_yaw"] = plan.yaw * kRadToDeg;
        data["cmd_pitch"] = plan.pitch * kRadToDeg;
        // 云台要闭合的跟随误差。单看 cmd_yaw 是条平滑斜坡，抖动只在差值里看得见。
        data["cmd_yaw_error"] =
          L6Telemetry::limit_rad(plan.yaw - gimbal_ypr[0]) * kRadToDeg;
        data["cmd_pitch_error"] =
          L6Telemetry::limit_rad(plan.pitch - gimbal_ypr[1]) * kRadToDeg;
        data["fire_delta_angle"] = plan.fire_delta_angle * kRadToDeg;
        data["fly_time"] = plan.fly_time;
        data["before_fire"] = plan.delay.beforeFire();
        data["image_to_plan"] = plan.delay.image_to_plan;
      }
      data["cmd_sent"] = command ? 1 : 0;
      data["cmd_shoot"] = command && command->shoot ? 1 : 0;
      data["fire_feasible"] = fire_decision.fire_feasible ? 1 : 0;
      data["aim_yaw_error"] = fire_decision.yaw_error * kRadToDeg;
      data["aim_pitch_error"] = fire_decision.pitch_error * kRadToDeg;
      if (fire_decision.tolerance.valid) {
        data["tol_yaw"] = fire_decision.tolerance.yaw * kRadToDeg;
        data["tol_pitch"] = fire_decision.tolerance.pitch * kRadToDeg;
      }
      data["armor_switching"] = fire_input.armor_switching ? 1 : 0;
      data["command_jump"] = fire_input.command_jump ? 1 : 0;
      data["gimbal_pitch"] = gimbal_ypr[1] * kRadToDeg;

      // 整车预测数据
      if (predicted) {
        data["predict_time"] = predict_time;
        data["pred_x"] = predicted->position.x();
        data["pred_y"] = predicted->position.y();
        data["pred_z"] = predicted->position.z();
        data["pred_a"] = predicted->yaw * kRadToDeg;
        if (predicted_armor_yaw) {
          data["pred_armor_yaw"] = *predicted_armor_yaw * kRadToDeg;
        }
      }
      (void)plotter.send(data);

      if (show_plot) {
        cv::imshow(
          "pnp cost",
          drawCostPlot(
            curve ? &*curve : nullptr,
            selected ? &observations[*selected] : nullptr, solver, ekf_armor_yaw,
            predicted_armor_yaw, frame_index));
      }
      cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
      cv::imshow("reprojection", img);
      const int key = cv::waitKey(wait_ms);
      if (key == 'q' || key == 27) {
        break;
      }
      if (key == ' ') {
        paused = true;
      }
    }

    cv::destroyAllWindows();
    std::cout << "\n回放结束\n"
              << "帧数: " << frames << '\n'
              << "有 PnP 观测的帧: " << observation_frames << '\n'
              << "通过 ArmorQuality 全部门限的观测: " << valid_pnp_observations
              << '\n'
              << "Tracking 帧: " << tracking_frames << '\n'
              << "代价曲线出现多个局部极小值的帧: " << multi_minimum_frames
              << '\n'
              << "L4 规划成功的帧: " << plan_valid_frames << '\n'
              << "实际下发命令的帧: " << command_frames << '\n'
              << "fire_feasible 的帧: " << fire_feasible_frames
              << "（shoot_enable=false，不会真的开火）\n"
              << "L4 选板切换的帧: " << plan_switch_frames << '\n'
              << "命令 yaw 跳变超门限的帧: " << command_jump_frames << '\n'
              << "观测门限: "
              << (tracker_config.require_quality ? "ArmorQuality 全部通过"
                                                 : "仅 PnP 成功")
              << '\n';
    if (!aim_yaw_errors.empty()) {
      std::sort(aim_yaw_errors.begin(), aim_yaw_errors.end());
      const double median = aim_yaw_errors[aim_yaw_errors.size() / 2];
      std::cout << "命令与录像云台的 yaw 偏差中位数: " << median
                << " deg（回放不是闭环，这里大属正常）\n";
    }

    if (!reject_histogram.empty()) {
      std::cout << "火控拒绝原因（按出现帧数）:\n";
      std::vector<std::pair<L5Control::RejectReason, std::size_t>> reasons(
        reject_histogram.begin(), reject_histogram.end());
      std::sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
      });
      for (const auto& [reason, count] : reasons) {
        std::cout << "  " << L5Control::toString(reason) << ": " << count << '\n';
      }
    }

    if (observation_frames > 0 && valid_pnp_observations == 0) {
      std::cout << (tracker_config.require_quality
                      ? "提示: 所有观测都被 ArmorQuality 门限拒绝，跟踪器不会起步。\n"
                      : "提示: 质量位一个都没置起来，当前跟踪结果来自未经质量校验的"
                        "PnP；实机 require_quality 为 true，不会是这个行为。\n");
    }
    L6Telemetry::flushLogger();
    return frames > 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "auto_aim_test 失败: " << error.what() << '\n';
    return 1;
  }
}
