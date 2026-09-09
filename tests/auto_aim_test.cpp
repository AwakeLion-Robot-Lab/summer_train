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
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/tracker.hpp"
#include "runtime/auto_aim_config.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l4_planning/armor/predictor.hpp"
#include "l5_control/controller.hpp"
#include "l5_control/fire_decision.hpp"
#include "l6_telemetry/aim_overlay.hpp"
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

// 叠加层的绘制统一在 L6，实机 runtime 和这里共用同一份，
// 否则两边会漂——回放里看着对的东西实机上可能画错。
using L6Telemetry::drawFilterInputArmors;
using L6Telemetry::isFilterInputArmor;
using L6Telemetry::drawOutlinedText;
using L6Telemetry::drawVehicle;
using L6Telemetry::projectWorldPoint;
using L6Telemetry::toPixel;

constexpr double kRadToDeg = 180.0 / std::numbers::pi;
constexpr double kDegToRad = std::numbers::pi / 180.0;
// 忽略浮点量化级小步进；相邻两个大于该值的反向步进才记为波形折返。
constexpr double kDirectionStepThreshold = 0.05 * kDegToRad;

// 诊断曲线仍覆盖整周，用来显示 SP 为什么只搜枪管 yaw 附近：
// 整周里存在不可见的背面局部极小值。
constexpr double kSearchRangeDegrees = 360.0;
// 画图采样比求解器的 1 度枚举更细，不参与求解。
constexpr double kCostStepDegrees = 0.5;

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{model m | model/armor_model/yolov5.xml | OpenVINO 装甲板模型}"
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
  "{bullet-speed | 27.0 | 回放没有裁判系统数据；默认与 SP auto_aim_test 一致（m/s）}"
  "{command-jump | 10.0 | 相邻帧命令 yaw 跳变超过该角度即判为 command_jump（度）}"
  "{conf | 0 | 覆盖检测分数门（confidence/minimum/nms_score 三者同时设为该值），<=0 保持 layout 预设}"
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

std::string_view stateName(L3Estimation::TrackState state) noexcept
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

const char* planErrorName(L4Planning::PlanError error) noexcept
{
  switch (error) {
  case L4Planning::PlanError::None:            return "none";
  case L4Planning::PlanError::NoTarget:        return "no-target";
  case L4Planning::PlanError::BadBulletSpeed:  return "bad-speed";
  case L4Planning::PlanError::DelayNotCalibrated: return "delay-uncal";
  case L4Planning::PlanError::BallisticFailed: return "ballistic";
  case L4Planning::PlanError::OutOfWindow:     return "out-of-window";
  }
  return "unknown";
}

// 拒绝原因拼成一行，画在图上。数值曲线看得出"没开火"，看不出"为什么"。
std::string rejectReasons(const L5Control::FireDecision& decision)
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

L2Perception::ArmorColor parseEnemyColor(const std::string& value)
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

bool readPose(std::istream& input, PoseSample& sample)
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
Eigen::Quaterniond toWorldBarrelPose(
  const PoseSample& sample,
  bool sp_convention,
  const Eigen::Matrix3d& R_imu_barrel)
{
  // newvision_record.yaml 的 SP 对照配置使用单位 R_gimbal2imubody，Solver
  // 直接消费录像四元数。这里也直接返回原值，不能额外 normalize 或先转矩阵
  // 再构造四元数，否则会在 1 度 yaw 网格的等价极小值附近改变胜负。
  if (!sp_convention && R_imu_barrel.isIdentity(0.0)) {
    return sample.q;
  }

  Eigen::Matrix3d R_sp_flip = Eigen::Matrix3d::Identity();
  R_sp_flip(0, 0) = -1.0;
  R_sp_flip(1, 1) = -1.0;
  const Eigen::Matrix3d R_world_imu = sample.q.toRotationMatrix();
  const Eigen::Matrix3d R_world_barrel = sp_convention
    ? Eigen::Matrix3d{R_sp_flip.transpose() * R_world_imu * R_sp_flip}
    : Eigen::Matrix3d{R_world_imu * R_imu_barrel};
  return Eigen::Quaterniond(R_world_barrel);
}

const char* armorClassName(L3Estimation::ArmorName name) noexcept
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
double yawCost(
  const L3Estimation::PnpSolver& solver,
  const L3Estimation::Armor& armor,
  double yaw)
{
  // 必须和 sp_vision 的 yaw 搜索用同一个定义（四角点二维距离之和），
  // 否则画出来的最小点不是求解器真正在找的那个，对账就失去意义。
  const std::vector<cv::Point2f> projected =
    solver.reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (projected.size() != armor.points.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double cost = 0.0;
  for (std::size_t index = 0; index < armor.points.size(); ++index) {
    const cv::Point2f difference = armor.points[index] - projected[index];
    cost += std::hypot(
      static_cast<double>(difference.x), static_cast<double>(difference.y));
  }
  return cost;
}

YawCostCurve sampleYawCost(
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

// 复刻 PnpSolver::refine_double_armor 的配对规则：同类别、同板型、世界系间距落
// 在相邻两板的物理范围内，取最近的一块。间距边界与 pnp_solver.cpp 的
// kMinimumPairGap / kMaximumPairGap 一致——这里是诊断视图，不参与求解。
//
// 返回 partner 下标，以及 partner 相对所选板的朝向差（含符号）。世界系 y 指左，
// 方位角大的是左板，左板 yaw 比右板小 2π/n。
std::optional<std::pair<std::size_t, double>> selectPairPartner(
  const std::vector<L3Estimation::Armor>& observations,
  std::size_t index)
{
  constexpr double kMinimumPairGap = 0.1;
  constexpr double kMaximumPairGap = 0.75;

  const L3Estimation::Armor& armor = observations[index];
  const auto armor_count = L3Estimation::armorCountOf(armor.name);
  if (armor.name == L3Estimation::ArmorName::Unknown || !armor_count) {
    return std::nullopt;
  }

  std::optional<std::size_t> partner;
  double partner_gap = kMaximumPairGap;
  for (std::size_t other = 0; other < observations.size(); ++other) {
    if (other == index ||
        observations[other].name != armor.name ||
        observations[other].type != armor.type) {
      continue;
    }
    const double gap =
      (armor.xyz_in_world - observations[other].xyz_in_world).norm();
    if (gap < kMinimumPairGap || gap >= partner_gap) {
      continue;
    }
    partner_gap = gap;
    partner = other;
  }
  if (!partner) {
    return std::nullopt;
  }

  const double step = 2.0 * std::numbers::pi / static_cast<double>(*armor_count);
  const bool partner_is_left =
    observations[*partner].ypd_in_world[0] > armor.ypd_in_world[0];
  return std::make_pair(*partner, partner_is_left ? -step : step);
}

// 双板联合代价曲线。横轴仍是所选装甲板自身相对枪管的 yaw 偏角，纵轴换成两块板
// 共八个角点的距离之和，与 PnpSolver::optimize_yaw_pair 的代价同一定义：
// partner 的朝向恒为所选板 + partner_offset。
YawCostCurve sampleJointYawCost(
  const L3Estimation::PnpSolver& solver,
  const L3Estimation::Armor& armor,
  const L3Estimation::Armor& partner,
  double partner_offset,
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
    const double cost = yawCost(solver, armor, yaw) +
      yawCost(solver, partner, L6Telemetry::limit_rad(yaw + partner_offset));
    curve.offsets_degrees.push_back(offset);
    curve.costs.push_back(cost);
    if (cost < curve.best_cost) {
      curve.best_cost = cost;
      curve.best_offset_degrees = offset;
      curve.best_yaw = yaw;
    }
  }

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
std::optional<std::size_t> selectArmor(
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<L3Estimation::TrackedTarget>& target,
  const std::vector<Eigen::Vector4d>& target_armor_poses,
  const cv::Size& image_size)
{
  std::optional<std::size_t> selected;
  double best_score = std::numeric_limits<double>::infinity();

  if (target && target->last_id >= 0 &&
      static_cast<std::size_t>(target->last_id) < target_armor_poses.size()) {
    const Eigen::Vector3d tracked =
      target_armor_poses[static_cast<std::size_t>(target->last_id)].head<3>();
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
cv::Mat drawCostPlot(
  const YawCostCurve* curve,
  const YawCostCurve* joint_curve,
  double pair_offset_degrees,
  const L3Estimation::Armor* armor,
  const L3Estimation::PnpSolver& solver,
  const std::optional<double>& ekf_armor_yaw,
  const std::optional<double>& predicted_armor_yaw,
  int frame_index)
{
  constexpr int kWidth = 960;
  constexpr int kHeight = 540;
  // 顶部留出五行文字：帧号、板信息、单板 yaw、单峰统计、双板配对。
  const cv::Rect graph{74, 156, kWidth - 74 - 24, kHeight - 156 - 56};
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

  // 双板：联合曲线的极小点才是 refine_double_armor 实际采用的 yaw，单板曲线
  // 在正对枪口时接近平坦，两条线放在同一纵轴上才能看出约束起了多大作用。
  drawOutlinedText(
    plot,
    joint_curve == nullptr
      ? std::string{"double armor: unpaired (single-armor yaw in use)"}
      : cv::format(
          "double armor: pair offset=%+.0fdeg  joint min=%.1fdeg (cost=%.1fpx)"
          "  local minima=%zu",
          pair_offset_degrees, joint_curve->best_yaw * kRadToDeg,
          joint_curve->best_cost, joint_curve->local_minima),
    {12, 136},
    joint_curve == nullptr ? cv::Scalar{140, 140, 140}
                           : cv::Scalar{255, 0, 255},
    0.52);

  std::vector<double> finite_costs;
  finite_costs.reserve(curve->costs.size());
  std::copy_if(
    curve->costs.begin(), curve->costs.end(),
    std::back_inserter(finite_costs),
    [](double cost) { return std::isfinite(cost); });
  if (joint_curve != nullptr) {
    std::copy_if(
      joint_curve->costs.begin(), joint_curve->costs.end(),
      std::back_inserter(finite_costs),
      [](double cost) { return std::isfinite(cost); });
  }
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

  if (joint_curve != nullptr) {
    std::optional<cv::Point> previous_joint;
    for (std::size_t index = 0; index < joint_curve->costs.size(); ++index) {
      if (!std::isfinite(joint_curve->costs[index])) {
        previous_joint.reset();
        continue;
      }
      const cv::Point point{
        x_of(joint_curve->offsets_degrees[index]),
        y_of(joint_curve->costs[index])};
      if (previous_joint) {
        cv::line(plot, *previous_joint, point, {255, 0, 255}, 2, cv::LINE_AA);
      }
      previous_joint = point;
    }
    cv::circle(
      plot,
      {x_of(joint_curve->best_offset_degrees), y_of(joint_curve->best_cost)}, 6,
      {255, 0, 255}, cv::FILLED, cv::LINE_AA);
  }

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

    // L2/L3/L4 参数一律从 auto_aim.yaml 读，回放和实机用同一份数值——否则在
    // YAML 里调噪声或精修阈值，这里根本看不出变化。
    const auto runtime_config = runtime::loadAutoAimConfig("config/auto_aim.yaml");

    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = cli.get<std::string>("model");
    model_config.device = cli.get<std::string>("device");
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);
    require(backend->ready(), "OpenVINO 后端未就绪");
    // 模型来自 --model，没有 auto_aim.yaml 的 layout 可依，按输出形状探契约。
    auto decoder_config =
      L2Perception::armorDecoderConfigFor(L2Perception::probeOutputSpecs(*backend));
    // --conf 只为扫阈值实验存在：<=0 时保持 layout 预设，行为与不加这个参数完全一致。
    // 三道分数门要一起动——minimum_confidence 是 NMS 之后的门，单独降前两个不起作用。
    // 注意本文件的 decoder 配置来自输出形状探测，不读 auto_aim.yaml 的 decoder 覆盖项，
    // 因此改 yaml 对回放无效，只能走这里。
    const float conf_override = cli.get<float>("conf");
    if (conf_override > 0.0F) {
      decoder_config.confidence_threshold = conf_override;
      decoder_config.minimum_confidence = conf_override;
      decoder_config.nms_score_threshold = conf_override;
      std::cout << "检测分数门被 --conf 覆盖为 " << conf_override << '\n';
    }
    L2Perception::ArmorDetector detector(
      std::move(backend), decoder_config, L2Perception::ImagePreprocessConfig{},
      runtime_config.refiner);
    require(detector.ready(), "ArmorDetector 未就绪");

    const L3Estimation::ArmorConfig & armor_config = runtime_config.armor;
    L3Estimation::Tracker tracker(
      calibration, armor_config, runtime_config.tracker, runtime_config.target);
    require(tracker.ready(), "Tracker 拒绝了该标定");
    // 与 Tracker 内部同参数的求解器，只用来做重投影和代价曲线，不参与滤波。
    L3Estimation::PnpSolver solver(calibration, armor_config);
    require(solver.ready(), "诊断用 PnpSolver 拒绝了该标定");
    // L4 的整车外推：恒速度 + 恒角速度，中心和整车 yaw 一起推进。
    const L4Planning::Predictor predictor;
    // 完整的 L4 -> L5 链路。回放与 runtime 现在共用同一组
    // Planner / FireDecider / Controller 语义，这里另外负责离线诊断。
    L4Planning::Planner planner(runtime_config.plan);
    // 回放固定关闭实际开火，但仍记录 fire_feasible 的时序。
    L5Control::FireConfig fire_config = runtime_config.fire;
    fire_config.shoot_enable = false;
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
    // 双板：成功配对的帧，以及其中联合代价曲线仍非单峰的帧。
    std::size_t paired_frames = 0;
    std::size_t joint_multi_minimum_frames = 0;
    // 联合极小与单板极小的 yaw 之差，即双板约束把这块板拉动了多少度。
    double paired_yaw_shift_sum = 0.0;
    double paired_yaw_shift_max = 0.0;
    std::size_t plan_valid_frames = 0;
    std::size_t command_frames = 0;
    std::size_t fire_feasible_frames = 0;
    std::size_t plan_switch_frames = 0;
    std::size_t command_jump_frames = 0;
    std::size_t same_armor_direction_reversal_frames = 0;
    double largest_reversal_step = 0.0;
    // 拒绝原因直方图。fire_feasible 是 0 时，唯一有用的信息是"被哪一条拦住的"。
    std::map<L5Control::RejectReason, std::size_t> reject_histogram;
    // 回放里云台姿态来自录像，不是本规划器闭环出来的，所以 aim_error 基本必然
    // 触发。把误差量级和容差一起打出来，才能判断是"云台没跟"还是"规划跑偏"。
    std::vector<double> aim_yaw_errors;
    // 上一帧规划命令用于命令跳变检查和 L4 选板连续性诊断。
    int last_plan_armor_id = -1;
    std::optional<double> last_command_yaw;
    // 射击轨迹原值。过渡段期间它与下发命令不同，选板连续性只能用它来判。
    std::optional<double> last_shoot_yaw;
    std::optional<double> last_same_armor_step;
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
      std::optional<YawCostCurve> joint_curve;
      double pair_offset_degrees = 0.0;
      if (selected) {
        curve = sampleYawCost(solver, observations[*selected], q_world_barrel);
        if (curve->local_minima > 1) {
          ++multi_minimum_frames;
        }

        // 同帧存在配对时再算一条联合曲线。这条曲线的极小点就是
        // PnpSolver::refine_double_armor 实际写回两块板的 yaw。
        if (const auto partner = selectPairPartner(observations, *selected)) {
          pair_offset_degrees = partner->second * kRadToDeg;
          joint_curve = sampleJointYawCost(
            solver, observations[*selected], observations[partner->first],
            partner->second, q_world_barrel);
          ++paired_frames;
          if (joint_curve->local_minima > 1) {
            ++joint_multi_minimum_frames;
          }
          const double yaw_shift = std::abs(L6Telemetry::limit_rad(
            joint_curve->best_yaw - curve->best_yaw)) * kRadToDeg;
          paired_yaw_shift_sum += yaw_shift;
          paired_yaw_shift_max = std::max(paired_yaw_shift_max, yaw_shift);
        }
      }
      std::optional<double> ekf_armor_yaw;
      if (target && target->last_id >= 0 &&
          static_cast<std::size_t>(target->last_id) <
            target_armor_poses.size()) {
        ekf_armor_yaw =
          target_armor_poses[static_cast<std::size_t>(target->last_id)].w();
      }

      /// 整车预测：把当前 EKF 状态外推 predict_time 秒后重新展开所有装甲板
      std::optional<L3Estimation::TrackedTarget> predicted;
      std::vector<Eigen::Vector4d> predicted_armor_poses;
      std::optional<double> predicted_armor_yaw;
      if (target && predict_time > 0.0) {
        predicted = predictor.predict(*target, predict_time);
        predicted_armor_poses = predictor.armorPoses(*predicted);
        if (target->last_id >= 0 &&
            static_cast<std::size_t>(target->last_id) <
              predicted_armor_poses.size()) {
          predicted_armor_yaw =
            predicted_armor_poses[static_cast<std::size_t>(target->last_id)]
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

      // SP 的离线 auto_aim_test 以 to_now=false 调 Aimer，固定使用
      // 0.005 s 检测耗时，再叠加 Aimer 的高/低速延迟。
      const auto plan_time = timestamp;
      const auto plan = planner.plan(target, robot_state, plan_time, false);
      const int plan_armor_id =
        plan.fire.has_value() ? plan.fire->armor_id : -1;

      L5Control::FireInput fire_input;
      fire_input.target = target;
      // 跟踪状态不再挂在目标上，火控要靠它区分 Tracking 和 TempLost。
      fire_input.track_state = tracker.state();
      fire_input.plan = plan;
      // 命中判据必须拿云台**实际**指向来比，这里就是录像里那份四元数。
      fire_input.actual_yaw = gimbal_ypr[0];
      fire_input.actual_pitch = gimbal_ypr[1];
      // 只作为 L4 选板连续性诊断，不再参与 L5 开火判定。
      const bool plan_armor_changed =
        plan.valid() && plan_armor_id >= 0 && last_plan_armor_id >= 0 &&
        plan_armor_id != last_plan_armor_id;
      fire_input.command_jump = plan.valid() && last_command_yaw &&
        std::abs(L6Telemetry::limit_rad(plan.aim.yaw - *last_command_yaw)) >
          command_jump_rad;

      // 三角/锯齿波验收：换板帧允许一次跳变，同一物理板内不允许
      // 出现“下降 -> 回升 -> 继续下降”。这里不预设旋转方向，正反转录像都适用。
      //
      // 比的必须是**射击轨迹**原值，不是下发命令。切板过渡段就是在同一块板
      // 内故意反向减速（armor_id 要到真正切板才变），拿命令角来比的话，一开
      // 过渡段这里就会满屏报折返——那是规划在按设计工作，不是估计在抖。
      if (plan.valid() && last_shoot_yaw &&
          plan_armor_id == last_plan_armor_id) {
        const double step =
          L6Telemetry::limit_rad(plan.aim.shootYaw() - *last_shoot_yaw);
        if (std::abs(step) >= kDirectionStepThreshold) {
          if (last_same_armor_step && step * *last_same_armor_step < 0.0) {
            ++same_armor_direction_reversal_frames;
            largest_reversal_step =
              std::max(largest_reversal_step, std::abs(step));
          }
          last_same_armor_step = step;
        }
      } else {
        last_same_armor_step.reset();
      }

      const auto fire_decision = fire_decider.decide(fire_input);
      const auto command = controller.makeCommand(plan, fire_decision);

      if (plan.valid()) {
        ++plan_valid_frames;
        last_plan_armor_id = plan_armor_id;
        last_command_yaw = plan.aim.yaw;
        last_shoot_yaw = plan.aim.shootYaw();
      } else {
        last_plan_armor_id = -1;
        last_command_yaw.reset();
        last_shoot_yaw.reset();
      }
      if (command) {
        ++command_frames;
      }
      if (fire_decision.fire_feasible) {
        ++fire_feasible_frames;
      }
      if (plan_armor_changed) {
        ++plan_switch_frames;
      }
      if (fire_input.command_jump) {
        ++command_jump_frames;
      }
      for (const auto reason : fire_decision.reasons) {
        ++reject_histogram[reason];
      }
      if (plan.valid() && fire_decision.tolerance.valid) {
        aim_yaw_errors.push_back(fire_decision.yaw_error * kRadToDeg);
      }

      if (!observations.empty()) {
        ++observation_frames;
      }
      for (const auto& observation : observations) {
        if (observation.name != L3Estimation::ArmorName::Unknown) {
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
          img, observations, solver,
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

        // 红色是 Plan 直接保存的命中时刻实体板，对应 sp_vision 的
        // debug_aim_point；不再靠 armor_id 和延迟在回放层重复重建。
        if (plan.valid() && plan.fire.has_value()) {
          drawVehicle(
            img, {plan.fire->armor_pose}, armor_type, target->name, solver,
            {0, 0, 255}, 2, overlay_shift);
        }
      }

      // 瞄准点和火控判据用的那块实体板。两者在 WholeCarCenter 档会明显分开
      // ——瞄的是旋转圆上的代理点，判的是板。sp 没有这一层。
      if (full_view && plan.valid()) {
        const auto aim_pixel =
          projectWorldPoint(plan.aim.point, calibration, q_world_barrel);
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
        if (plan.fire.has_value()) {
          const auto fire_pixel =
            projectWorldPoint(plan.fire->point(), calibration, q_world_barrel);
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
      if (full_view && selected && isFilterInputArmor(observations[*selected])) {
        const auto& armor = observations[*selected];
        drawOutlinedText(
          img,
          cv::format("filter input armor yaw=%.1fdeg",
                     armor.ypr_in_world[0] * kRadToDeg),
          {10, 92}, {0, 255, 0});
      }
      // 内部状态前十一维：[xc, vx, yc, vy, z, vz, yaw, v_yaw, r1, r2-r1, z2-z1]。
      if (full_view && target) {
        const Eigen::VectorXd tx = target->ekf_x();
        drawOutlinedText(
          img,
          cv::format(
            "EKF center=(%.2f,%.2f,%.2f)m v=(%.2f,%.2f,%.2f)m/s yaw=%.1fdeg "
            "v_yaw=%.2frad/s r=%.3fm id=%d",
            tx[0], tx[2], tx[4], tx[1], tx[3], tx[5],
            tx[6] * kRadToDeg, tx[7], tx[8], target->last_id),
          {10, 122}, {0, 255, 0}, 0.55);
      }
      if (full_view && predicted) {
        const Eigen::VectorXd px = predicted->ekf_x();
        drawOutlinedText(
          img,
          cv::format(
            "pred +%.0fms center=(%.2f,%.2f,%.2f)m yaw=%.1fdeg (delta=%.1fdeg)",
            predict_time * 1e3, px[0], px[2], px[4], px[6] * kRadToDeg,
            L6Telemetry::limit_rad(px[6] - target->ekf_x()[6]) * kRadToDeg),
          {10, 152}, {0, 165, 255}, 0.55);
      }
      drawOutlinedText(
        img,
        plan.valid()
          ? cv::format(
              "CMD yaw=%.2f pitch=%.2f deg | err yaw=%.2f pitch=%.2f | "
              "armor=%d fire_armor=%d",
              plan.aim.yaw * kRadToDeg, plan.aim.pitch * kRadToDeg,
              L6Telemetry::limit_rad(plan.aim.yaw - gimbal_ypr[0]) * kRadToDeg,
              L6Telemetry::limit_rad(plan.aim.pitch - gimbal_ypr[1]) * kRadToDeg,
              plan_armor_id, plan_armor_id) +
              (plan.aim.blending ? std::string(" | BLEND") : std::string())
          : cv::format("CMD not sent (plan %s)", planErrorName(plan.reason)),
        {10, full_view ? 182 : 92},
        plan.valid() ? cv::Scalar{0, 255, 255} : cv::Scalar{160, 160, 160}, 0.55);
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
        data["armor_pnp_committed"] =
          armor.name != L3Estimation::ArmorName::Unknown;
      }

      // PnP yaw 搜索代价
      if (curve) {
        data["cost_min"] = curve->best_cost;
        data["cost_min_yaw"] = curve->best_yaw * kRadToDeg;
        data["cost_min_offset"] = curve->best_offset_degrees;
        data["cost_local_minima"] = curve->local_minima;
        // 双板联合曲线。single 与 joint 的极小 yaw 之差就是双板约束把这块板的
        // yaw 拉动了多少度，是回放里最直接的收益量。
        if (joint_curve) {
          data["joint_cost_min"] = joint_curve->best_cost;
          data["joint_cost_min_yaw"] = joint_curve->best_yaw * kRadToDeg;
          data["joint_cost_min_offset"] = joint_curve->best_offset_degrees;
          data["joint_cost_local_minima"] = joint_curve->local_minima;
          data["joint_minus_single_yaw"] =
            L6Telemetry::limit_rad(joint_curve->best_yaw - curve->best_yaw) *
            kRadToDeg;
          data["pair_offset"] = pair_offset_degrees;
        }
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
        const Eigen::VectorXd tx = target->ekf_x();
        data["x"] = tx[0];
        data["vx"] = tx[1];
        data["y"] = tx[2];
        data["vy"] = tx[3];
        data["z"] = tx[4];
        data["vz"] = tx[5];
        data["a"] = tx[6] * kRadToDeg;
        data["w"] = tx[7];
        data["r"] = tx[8];
        data["last_id"] = target->last_id;
        data["nis"] = target->ekf().last_nis;
        if (ekf_armor_yaw) {
          data["ekf_armor_yaw"] = *ekf_armor_yaw * kRadToDeg;
        }
      }

      // L4 -> L5：这才是真正决定下位机动作的一组量。
      // cmd_yaw 是 world 系绝对方位角，和 gimbal_yaw 同一个基准，可以直接相减。
      data["plan_valid"] = plan.valid() ? 1 : 0;
      data["plan_error"] = static_cast<int>(plan.reason);
      data["plan_armor_id"] = plan_armor_id;
      data["plan_aim_on_armor"] = plan.fire.has_value() &&
          (plan.aim.point - plan.fire->point()).norm() < 1e-9
        ? 1
        : 0;
      data["fire_armor_id"] = plan_armor_id;
      data["fire_admissible"] = plan.fireAdmissible() ? 1 : 0;
      if (plan.valid()) {
        data["cmd_yaw"] = plan.aim.yaw * kRadToDeg;
        data["cmd_pitch"] = plan.aim.pitch * kRadToDeg;
        // 射击轨迹原值。**必须和 cmd_yaw 画在同一张图上**：切板过渡段调的
        // 就是这两条线怎么分开又怎么合上，只画一条完全看不出过渡做没做、
        // 有没有做过头。跟随段两者重合是正常的，不是数据重复。
        data["shoot_yaw"] = plan.aim.shootYaw() * kRadToDeg;
        data["shoot_pitch"] = plan.aim.shootPitch() * kRadToDeg;
        data["blending"] = plan.aim.blending ? 1 : 0;
        // 过渡段规划出的峰值角加速度，对着 planning.blend 里配的上限看。
        // acc_limited 置 1 说明拉到最长时长仍然超限——重合度上不去是云台
        // 能力的物理限制，不是参数没调好。
        data["blend_peak_acc"] = plan.blend.peak_yaw_acceleration;
        data["blend_acc_limited"] = plan.blend.acceleration_limited ? 1 : 0;
        // 过渡终点比切板时刻晚了多少毫秒。稳定在一个帧周期附近是正常量化
        // 误差；明显更大说明 blend.horizon_ms 不够长，切板发现得太晚。
        data["blend_late_ms"] = plan.blend.late * 1e3;
        // 云台要闭合的跟随误差。单看 cmd_yaw 是条平滑斜坡，抖动只在差值里看得见。
        data["cmd_yaw_error"] =
          L6Telemetry::limit_rad(plan.aim.yaw - gimbal_ypr[0]) * kRadToDeg;
        data["cmd_pitch_error"] =
          L6Telemetry::limit_rad(plan.aim.pitch - gimbal_ypr[1]) * kRadToDeg;
        data["fire_delta_angle"] = plan.fire.has_value()
          ? plan.fire->facingAngle() * kRadToDeg
          : 0.0;
        data["fly_time"] = plan.timing.fly_time;
        data["before_fire"] = plan.timing.delay.beforeFire();
        data["image_to_plan"] = plan.timing.delay.image_to_plan;
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
      data["plan_armor_changed"] = plan_armor_changed ? 1 : 0;
      data["command_jump"] = fire_input.command_jump ? 1 : 0;
      data["gimbal_pitch"] = gimbal_ypr[1] * kRadToDeg;

      // 整车预测数据
      if (predicted) {
        const Eigen::VectorXd px = predicted->ekf_x();
        data["predict_time"] = predict_time;
        data["pred_x"] = px[0];
        data["pred_y"] = px[2];
        data["pred_z"] = px[4];
        data["pred_a"] = px[6] * kRadToDeg;
        if (predicted_armor_yaw) {
          data["pred_armor_yaw"] = *predicted_armor_yaw * kRadToDeg;
        }
      }
      (void)plotter.send(data);

      if (show_plot) {
        cv::imshow(
          "pnp cost",
          drawCostPlot(
            curve ? &*curve : nullptr, joint_curve ? &*joint_curve : nullptr,
            pair_offset_degrees,
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
              << "双板配对成功的帧: " << paired_frames << '\n'
              << "配对帧里联合代价曲线非单峰的帧: "
              << joint_multi_minimum_frames << '\n'
              << "双板相对单板的 yaw 改动: 均值 "
              << (paired_frames == 0
                    ? 0.0
                    : paired_yaw_shift_sum /
                        static_cast<double>(paired_frames))
              << " deg，最大 " << paired_yaw_shift_max << " deg\n"
              << "L4 规划成功的帧: " << plan_valid_frames << '\n'
              << "实际下发命令的帧: " << command_frames << '\n'
              << "fire_feasible 的帧: " << fire_feasible_frames
              << "（shoot_enable=false，不会真的开火）\n"
              << "L4 选板切换的帧: " << plan_switch_frames << '\n'
              << "命令 yaw 跳变超门限的帧: " << command_jump_frames << '\n'
              << "同一装甲板内方向折返的帧(>0.05deg): "
              << same_armor_direction_reversal_frames << '\n'
              << "最大折返单步: " << largest_reversal_step * kRadToDeg
              << " deg\n"
              << "观测门限: 仅 PnP 成功\n";
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
      std::cout << "提示: 没有任何一帧的 single_pnp 提交出位姿，跟踪器不会起步。"
                   "先查标定和曝光时刻姿态。\n";
    }
    L6Telemetry::flushLogger();
    return frames > 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "auto_aim_test 失败: " << error.what() << '\n';
    return 1;
  }
}
