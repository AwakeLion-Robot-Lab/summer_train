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
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"
#include "l4_planning/armor/planner.hpp"
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
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <numbers>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
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

// 逐层耗时。量的是"这台机器跑完一帧算法要多久"，纯 CPU 墙钟，和实机的端到端
// 延迟不是一回事——后者的分解在 L4Planning::Delay 里（曝光、串口往返、飞行
// 时间都不在这儿），两者不要混着看。
//
// 同一段在一帧里可以 lap 多次、累加成一条样本：绘图散在检测后、跟踪后、显示
// 前三处，不累加就分不清"绘图总共花了多少"。
class StageClock
{
public:
  using Clock = std::chrono::steady_clock;

  struct Stage
  {
    std::string name;
    double frame{0.0};
    std::vector<double> ms;
    // 仅回放才有的开销（诊断 PnP、面板、imshow），不计入管线合计。
    bool replay_only{false};
    // L2 内部的分项：已经含在所在层里了，单独列出来但不重复计入合计。
    bool sub{false};
  };

  void tick() noexcept { mark_ = Clock::now(); }

  void lap(std::string_view name, bool replay_only = false)
  {
    const auto now = Clock::now();
    Stage& stage = slot(name, replay_only, false);
    stage.frame +=
      std::chrono::duration<double, std::milli>(now - mark_).count();
    mark_ = now;
  }

  // 记一段别处已经量好的耗时，不动 mark_。给 L2 内部的分项用。
  void add(std::string_view name, double ms)
  {
    slot(name, false, true).frame += ms;
  }

  // 一帧结束：把每段这一帧的累计值落成一条样本。
  void flush()
  {
    for (Stage& stage : stages_) {
      stage.ms.push_back(stage.frame);
      stage.frame = 0.0;
    }
  }

  const std::vector<Stage>& stages() const noexcept { return stages_; }

private:
  Stage& slot(std::string_view name, bool replay_only, bool sub)
  {
    for (Stage& stage : stages_) {
      if (stage.name == name) {
        return stage;
      }
    }
    stages_.push_back(Stage{std::string{name}, 0.0, {}, replay_only, sub});
    return stages_.back();
  }

  Clock::time_point mark_{Clock::now()};
  std::vector<Stage> stages_;
};

// 就地排序取分位，调用方给的是副本。
double msPercentile(std::vector<double> values, double ratio)
{
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t index = std::min(
    values.size() - 1,
    static_cast<std::size_t>(ratio * static_cast<double>(values.size())));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

// setw 数的是字节，中文一个字三字节两列宽，直接 setw 会把表格排歪。
std::string padCol(const std::string& text, std::size_t width)
{
  std::size_t shown = 0;
  for (const char character : text) {
    // UTF-8 续字节 10xxxxxx 不占列；其余非 ASCII 首字节按两列算。
    const auto byte = static_cast<unsigned char>(character);
    if ((byte & 0xC0) == 0x80) {
      continue;
    }
    shown += byte < 0x80 ? 1 : 2;
  }
  return text + std::string(width > shown ? width - shown : 1, ' ');
}

void printStageTiming(const StageClock& clock)
{
  double pipeline_mean = 0.0;
  for (const auto& stage : clock.stages()) {
    if (!stage.replay_only && !stage.sub && !stage.ms.empty()) {
      pipeline_mean +=
        std::accumulate(stage.ms.begin(), stage.ms.end(), 0.0) /
        static_cast<double>(stage.ms.size());
    }
  }

  const auto row = [&](const StageClock::Stage& stage) {
    if (stage.ms.empty()) {
      return;
    }
    const double mean =
      std::accumulate(stage.ms.begin(), stage.ms.end(), 0.0) /
      static_cast<double>(stage.ms.size());
    std::cout << "  " << padCol(stage.name, 18)
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(8) << mean
              << std::setw(8) << msPercentile(stage.ms, 0.5)
              << std::setw(8) << msPercentile(stage.ms, 0.9)
              << std::setw(9) << *std::max_element(stage.ms.begin(), stage.ms.end());
    if (!stage.replay_only && pipeline_mean > 0.0) {
      std::cout << std::setw(8) << std::setprecision(1)
                << 100.0 * mean / pipeline_mean << '%';
    }
    std::cout << '\n';
  };

  std::cout << "\n-- 逐层耗时 ms（CPU 墙钟，不是实机端到端延迟）--\n"
            << "  " << padCol("段", 18)
            << std::right << std::setw(8) << "mean" << std::setw(8) << "p50"
            << std::setw(8) << "p90" << std::setw(9) << "max"
            << std::setw(8) << "占比" << '\n';
  for (const auto& stage : clock.stages()) {
    if (!stage.replay_only) {
      row(stage);
    }
  }
  std::cout << "  " << padCol("管线合计", 18)
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(8) << pipeline_mean
            << "   （" << std::setprecision(1)
            << (pipeline_mean > 0.0 ? 1000.0 / pipeline_mean : 0.0)
            << " fps 上限）\n";

  bool has_replay_only = false;
  for (const auto& stage : clock.stages()) {
    has_replay_only = has_replay_only || (stage.replay_only && !stage.ms.empty());
  }
  if (has_replay_only) {
    std::cout << "  -- 以下仅回放，实机管线里没有，不计入合计 --\n";
    for (const auto& stage : clock.stages()) {
      if (stage.replay_only) {
        row(stage);
      }
    }
  }
  std::cout << std::defaultfloat;
}

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{model m |  | 整板模型，留空用 auto_aim.yaml 的；给了就按输出名认 layout}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方颜色：red / blue / any}"
  "{convention | imu | 录像四元数约定：imu / sp}"
  "{serial-config | config/serial_config.yaml | convention=imu 时读 R_imu_barrel}"
  "{predict-time p | 0.1 | 整车预测外推时长（秒），<=0 表示不画预测}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{wait w | 30 | 每帧 waitKey 毫秒，0 表示逐帧手动推进}"
  "{view | sp | 叠加层：sp（只画当前估计整车和瞄准板）/ full（全部调试层）}"
  "{overlay-offset | 0 | full 视图下整车叠加层上移的像素数；sp 视图恒为 0}"
  "{plot | auto | PnP 代价曲线窗口：auto（只在 view=full 时开）/ true / false}"
  "{bullet-speed | 27.0 | 回放没有裁判系统数据；默认与 SP auto_aim_test 一致（m/s）}"
  "{command-jump | 10.0 | 相邻帧命令 yaw 跳变超过该角度即判为 command_jump（度）}"
  "{csv | | 逐帧状态导出路径，留空则不导出。用于量化抖动而不是靠肉眼看}"
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

struct FilterEstimate
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double yaw_rate{0.0};
  double radius1{0.0};
  std::optional<double> radius2;
  std::optional<double> height_offset;
  std::optional<double> dz1;
  std::optional<double> dz2;
  std::optional<double> roll;
  std::optional<double> pitch;
};

// 回放侧对整车目标的只读包装。awakening 同样让 Tracker 返回不带滤波器的
// ArmorTarget 副本，下游只在副本上外推。
class ReplayTarget
{
public:
  explicit ReplayTarget(L3Estimation::EskfTarget target)
  : value_(std::move(target))
  {
    syncIdentity();
  }

  Eigen::VectorXd ekf_x() const { return value_.ekf_x(); }

  FilterEstimate estimate() const
  {
    const Eigen::VectorXd x = value_.ekf_x();

    FilterEstimate estimate;
    estimate.center = {x[0], x[2], x[4]};
    estimate.velocity = {x[1], x[3], x[5]};
    estimate.yaw_rate = x[7];
    estimate.radius1 = x[8];

    // 姿态是完整 SO(3)，所以 yaw 要从旋转矩阵分解，不能直接读 x[6]。
    const Eigen::Vector3d ypr = L6Telemetry::eulers(
      L3Estimation::VehicleModel::stateRotation(x), 2, 1, 0);
    estimate.yaw = ypr[0];
    estimate.pitch = ypr[1];
    estimate.roll = ypr[2];

    if (value_.name == L3Estimation::ArmorName::Outpost) {
      estimate.dz1 = x[9];
      estimate.dz2 = x[10];
    } else if (value_.armor_num() == 4) {
      // ekf_x() 已经把第 9 维的 log(r2) 转回线性半径。
      estimate.radius2 = x[9];
      estimate.height_offset = x[10];
    }
    return estimate;
  }

  std::vector<Eigen::Vector4d> armor_xyza_list() const
  {
    return value_.armor_xyza_list();
  }

  void predict(double dt) { value_.predict(dt); }

  L4Planning::Plan plan(
    L4Planning::Planner& planner,
    const L1Sensor::RobotState& robot_state,
    L3Estimation::TimePoint plan_time,
    bool to_now) const
  {
    return planner.plan(
      std::optional<L3Estimation::EskfTarget>{value_}, robot_state, plan_time, to_now);
  }

  double lastNis() const noexcept { return value_.lastNis(); }
  int lastNisDof() const noexcept { return value_.lastNisDof(); }

  // 是否关联到过 0 号以外的板。为 false 时整车 yaw 与第二组半径几乎不可观测。
  bool jumped() const noexcept { return value_.jumped; }

  L3Estimation::ArmorName name{L3Estimation::ArmorName::Unknown};
  int last_id{-1};

private:
  void syncIdentity()
  {
    name = value_.name;
    last_id = value_.last_id;
  }

  L3Estimation::EskfTarget value_;
};

std::string filterKinematicsText(const FilterEstimate& estimate)
{
  std::ostringstream text;
  text << std::fixed << std::setprecision(2)
       << "center=(" << estimate.center.x() << ',' << estimate.center.y() << ','
       << estimate.center.z() << ")m v=(" << estimate.velocity.x() << ','
       << estimate.velocity.y() << ',' << estimate.velocity.z() << ")m/s";
  return text.str();
}

std::string filterGeometryText(
  const FilterEstimate& estimate,
  int last_id,
  double nis,
  int nis_dof)
{
  std::ostringstream text;
  text << std::fixed << std::setprecision(2)
       << "yaw=" << estimate.yaw * kRadToDeg << "deg omega="
       << estimate.yaw_rate << "rad/s" << std::setprecision(3)
       << " r1=" << estimate.radius1 << 'm';
  if (estimate.radius2) {
    text << " r2=" << *estimate.radius2 << 'm';
  }
  if (estimate.height_offset) {
    text << " dz=" << *estimate.height_offset << 'm';
  }
  if (estimate.dz1 && estimate.dz2) {
    text << " dz1=" << *estimate.dz1 << "m dz2=" << *estimate.dz2 << 'm';
  }
  if (estimate.roll && estimate.pitch) {
    text << std::setprecision(1) << " roll=" << *estimate.roll * kRadToDeg
         << "deg pitch=" << *estimate.pitch * kRadToDeg << "deg";
  }
  text << std::setprecision(2) << " id=" << last_id << " NIS=" << nis << '/'
       << nis_dof;
  return text.str();
}

class ReplayTracker
{
public:
  ReplayTracker(
    const L1Sensor::CameraCalibration& calibration,
    const L3Estimation::ArmorConfig& armor_config,
    const L3Estimation::EskfTrackerConfig& tracker_config,
    const L3Estimation::EskfTargetConfig& target_config)
  : tracker_(calibration, armor_config, tracker_config, target_config)
  {
  }

  bool ready() const noexcept { return tracker_.ready(); }

  std::optional<ReplayTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::vector<L2Perception::Light>& lights,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    L3Estimation::TimePoint timestamp)
  {
    auto target = tracker_.track(detections, lights, q_world_barrel, timestamp);
    if (target) {
      return ReplayTarget{std::move(*target)};
    }
    return std::nullopt;
  }

  std::optional<cv::Rect> lightRoi(
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    L3Estimation::TimePoint timestamp, const cv::Size& image_size) const
  {
    return tracker_.lightRoi(q_world_barrel, timestamp, image_size);
  }

  // 本帧关联到的板数与编号。关联在编号间来回跳会让整车 yaw 每帧偏 2π/N。
  int lastMatchCount() const noexcept { return tracker_.lastMatchCount(); }
  std::string lastMatchedIdsString() const { return tracker_.lastMatchedIds(); }

  const std::vector<L3Estimation::UsedLight>& usedLights() const noexcept
  {
    return tracker_.usedLights();
  }

  // 送给网络的 ROI：整车先验驱动的显式空间注意力。
  cv::Rect netFocusRoi(
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    L3Estimation::TimePoint timestamp, const cv::Size& image_size,
    double target_wh_ratio) const
  {
    return tracker_.netFocusRoi(
      q_world_barrel, timestamp, image_size, target_wh_ratio);
  }

  const std::vector<L3Estimation::Armor>& observations() const noexcept
  {
    return tracker_.observations();
  }

  std::vector<Eigen::Vector4d> armorPoses() const
  {
    return tracker_.armorPoses();
  }

  L3Estimation::TrackState state() const noexcept { return tracker_.state(); }

  L4Planning::Plan plan(
    L4Planning::Planner& planner,
    const std::optional<ReplayTarget>& target,
    const L1Sensor::RobotState& robot_state,
    L3Estimation::TimePoint plan_time,
    bool to_now) const
  {
    if (target) {
      return target->plan(planner, robot_state, plan_time, to_now);
    }
    return planner.plan(
      std::optional<L3Estimation::EskfTarget>{}, robot_state, plan_time, to_now);
  }

private:
  L3Estimation::EskfTracker tracker_;
};

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

const char* armorColorName(L2Perception::ArmorColor color) noexcept
{
  switch (color) {
  case L2Perception::ArmorColor::Red:
    return "red";
  case L2Perception::ArmorColor::Blue:
    return "blue";
  case L2Perception::ArmorColor::Unknown:
    break;
  }
  return "unknown";
}

cv::Scalar armorDisplayColor(L2Perception::ArmorColor color) noexcept
{
  switch (color) {
  case L2Perception::ArmorColor::Red:
    return {0, 0, 255};
  case L2Perception::ArmorColor::Blue:
    return {255, 0, 0};
  case L2Perception::ArmorColor::Unknown:
    break;
  }
  return {0, 255, 255};
}

std::optional<cv::Rect> armorImageRoi(
  const L2Perception::Armor& armor, const cv::Size& image_size)
{
  std::vector<cv::Point2f> points;
  points.reserve(armor.corners.size());
  for (const cv::Point2f& point : armor.corners) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      return std::nullopt;
    }
    points.push_back(point);
  }

  cv::Rect roi = cv::boundingRect(points);
  roi &= cv::Rect(0, 0, image_size.width, image_size.height);
  return roi.empty() ? std::nullopt : std::optional<cv::Rect>{roi};
}

cv::Mat makeRecognitionPanel(
  const cv::Mat& source, const std::vector<L2Perception::Armor>& armors,
  L2Perception::ArmorColor enemy_color)
{
  constexpr int kTileWidth = 360;
  constexpr int kTileHeight = 260;
  constexpr int kColumns = 2;
  constexpr std::size_t kMaxShown = 4;

  if (armors.empty()) {
    cv::Mat panel(220, 480, CV_8UC3, cv::Scalar{24, 24, 24});
    drawOutlinedText(
      panel, "recognized: none", {28, 62}, {160, 160, 160}, 0.8);
    drawOutlinedText(
      panel, "no armor ROI in this frame", {28, 104},
      {120, 120, 120}, 0.55);
    return panel;
  }

  std::vector<std::size_t> order(armors.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
    order.begin(), order.end(), [&armors](std::size_t left, std::size_t right) {
      return armors[left].confidence > armors[right].confidence;
    });

  const std::size_t shown = std::min(order.size(), kMaxShown);
  const int columns = shown == 1 ? 1 : kColumns;
  const int rows = static_cast<int>((shown + columns - 1) / columns);
  cv::Mat panel(
    rows * kTileHeight, columns * kTileWidth, CV_8UC3,
    cv::Scalar{24, 24, 24});

  for (std::size_t tile_index = 0; tile_index < shown; ++tile_index) {
    const auto& armor = armors[order[tile_index]];
    const int column = static_cast<int>(tile_index) % columns;
    const int row = static_cast<int>(tile_index) / columns;
    const cv::Rect tile_rect(
      column * kTileWidth, row * kTileHeight, kTileWidth, kTileHeight);
    cv::rectangle(panel, tile_rect, {70, 70, 70}, 1, cv::LINE_AA);

    const bool filtered =
      enemy_color != L2Perception::ArmorColor::Unknown &&
      armor.color != enemy_color;
    const bool refined = armor.corner_source == L2Perception::CornerSource::Refined;
    const std::string label = cv::format(
      "[%zu] %s %s  conf=%.2f  %s%s",
      tile_index + 1, armorColorName(armor.color),
      armorClassName(L2Perception::armorClassFromId(armor.class_id)),
      armor.confidence, refined ? "refined" : "net", filtered ? "  ignored" : "");
    drawOutlinedText(
      panel, label, tile_rect.tl() + cv::Point{8, 28},
      armorDisplayColor(armor.color), 0.58);

    const auto roi = armorImageRoi(armor, source.size());
    if (!roi) {
      drawOutlinedText(
        panel, "invalid ROI", tile_rect.tl() + cv::Point{12, 84},
        {0, 255, 255}, 0.6);
      continue;
    }

    const int pad_x = std::max(12, roi->width / 2);
    const int pad_y = std::max(12, roi->height / 2);
    cv::Rect crop_roi(
      roi->x - pad_x, roi->y - pad_y,
      roi->width + 2 * pad_x, roi->height + 2 * pad_y);
    crop_roi &= cv::Rect(0, 0, source.cols, source.rows);
    if (crop_roi.empty()) {
      continue;
    }

    cv::Mat crop = source(crop_roi).clone();
    const cv::Scalar color = armorDisplayColor(armor.color);
    const cv::Point crop_offset{-crop_roi.x, -crop_roi.y};
    const cv::Rect local_roi(
      roi->x - crop_roi.x, roi->y - crop_roi.y,
      roi->width, roi->height);
    cv::rectangle(crop, local_roi, color, 2, cv::LINE_AA);
    for (std::size_t index = 0; index < armor.corners.size(); ++index) {
      cv::line(
        crop, toPixel(armor.corners[index]) + crop_offset,
        toPixel(armor.corners[(index + 1) % armor.corners.size()]) +
          crop_offset,
        color, 1, cv::LINE_AA);
    }

    const cv::Rect content(
      tile_rect.x + 6, tile_rect.y + 38,
      tile_rect.width - 12, tile_rect.height - 44);
    const double scale = std::min(
      static_cast<double>(content.width) / crop.cols,
      static_cast<double>(content.height) / crop.rows);
    const cv::Size resized_size{
      std::min(
        content.width,
        std::max(1, static_cast<int>(std::round(crop.cols * scale)))),
      std::min(
        content.height,
        std::max(1, static_cast<int>(std::round(crop.rows * scale))))};
    cv::Mat resized;
    cv::resize(crop, resized, resized_size, 0.0, 0.0, cv::INTER_LINEAR);
    const cv::Rect destination(
      content.x + (content.width - resized.cols) / 2,
      content.y + (content.height - resized.rows) / 2,
      resized.cols, resized.rows);
    resized.copyTo(panel(destination));
  }

  if (armors.size() > shown) {
    drawOutlinedText(
      panel, cv::format("+%zu more", armors.size() - shown),
      {panel.cols - 110, panel.rows - 12}, {180, 180, 180}, 0.5);
  }
  return panel;
}

cv::Mat makeUsedLightPanel(
  const cv::Mat& source,
  const std::vector<L3Estimation::UsedLight>& update_lights,
  std::size_t independent_candidate_count)
{
  constexpr int kPanelWidth = 720;
  constexpr int kPanelHeight = 430;
  constexpr int kHeaderHeight = 76;
  cv::Mat panel(
    kPanelHeight, kPanelWidth, CV_8UC3, cv::Scalar{24, 24, 24});

  const auto isolated_count = static_cast<std::size_t>(std::count_if(
    update_lights.begin(), update_lights.end(),
    [](const L3Estimation::UsedLight& light) {
      return light.isolated;
    }));
  drawOutlinedText(
    panel,
    cv::format(
      "lights used=%zu  armor=%zu  isolated=%zu  independent candidates=%zu",
      update_lights.size(), update_lights.size() - isolated_count,
      isolated_count, independent_candidate_count),
    {12, 28}, {255, 255, 255}, 0.58);
  drawOutlinedText(
    panel,
    "cyan=armor corners   magenta=matched isolated   filled dot=top",
    {12, 57}, {180, 180, 180}, 0.50);

  if (update_lights.empty()) {
    drawOutlinedText(
      panel, "no light entered the IESKF update in this frame",
      {28, 150}, {120, 120, 120}, 0.68);
    return panel;
  }

  std::vector<cv::Point2f> points;
  points.reserve(update_lights.size() * 2);
  double max_length = 0.0;
  for (const auto& light : update_lights) {
    if (!std::isfinite(light.top.x) || !std::isfinite(light.top.y) ||
        !std::isfinite(light.bottom.x) || !std::isfinite(light.bottom.y)) {
      continue;
    }
    points.push_back(light.top);
    points.push_back(light.bottom);
    max_length = std::max(max_length, cv::norm(light.top - light.bottom));
  }
  if (points.empty()) {
    drawOutlinedText(
      panel, "used light coordinates are invalid", {28, 150},
      {0, 255, 255}, 0.68);
    return panel;
  }

  const cv::Rect image_rect(0, 0, source.cols, source.rows);
  cv::Rect crop_roi = cv::boundingRect(points);
  const int padding = std::max(
    24, static_cast<int>(std::round(max_length * 0.8)));
  crop_roi = cv::Rect(
    crop_roi.x - padding, crop_roi.y - padding,
    crop_roi.width + padding * 2, crop_roi.height + padding * 2);
  crop_roi &= image_rect;
  if (crop_roi.empty()) {
    drawOutlinedText(
      panel, "used lights are outside the image", {28, 150},
      {0, 255, 255}, 0.68);
    return panel;
  }

  const cv::Rect content(
    12, kHeaderHeight, kPanelWidth - 24,
    kPanelHeight - kHeaderHeight - 12);
  const double scale = std::min(
    static_cast<double>(content.width) / crop_roi.width,
    static_cast<double>(content.height) / crop_roi.height);
  const cv::Size resized_size{
    std::max(1, static_cast<int>(std::round(crop_roi.width * scale))),
    std::max(1, static_cast<int>(std::round(crop_roi.height * scale)))};
  cv::Mat resized;
  cv::resize(
    source(crop_roi), resized, resized_size, 0.0, 0.0,
    cv::INTER_LINEAR);
  const cv::Rect destination(
    content.x + (content.width - resized.cols) / 2,
    content.y + (content.height - resized.rows) / 2,
    resized.cols, resized.rows);
  resized.copyTo(panel(destination));

  const auto panelPoint = [&](const cv::Point2f& point) {
    return cv::Point{
      destination.x + static_cast<int>(
        std::round((point.x - crop_roi.x) * scale)),
      destination.y + static_cast<int>(
        std::round((point.y - crop_roi.y) * scale))};
  };
  for (const auto& light : update_lights) {
    if (!std::isfinite(light.top.x) || !std::isfinite(light.top.y) ||
        !std::isfinite(light.bottom.x) || !std::isfinite(light.bottom.y)) {
      continue;
    }
    const cv::Scalar color = light.isolated
      ? cv::Scalar{255, 0, 255}
      : cv::Scalar{255, 255, 0};
    const cv::Point top = panelPoint(light.top);
    const cv::Point bottom = panelPoint(light.bottom);
    cv::line(panel, top, bottom, color, light.isolated ? 4 : 3, cv::LINE_AA);
    cv::circle(panel, top, 6, color, cv::FILLED, cv::LINE_AA);
    cv::circle(panel, bottom, 6, color, 2, cv::LINE_AA);
    const cv::Point center = (top + bottom) / 2;
    drawOutlinedText(
      panel,
      cv::format(
        "id=%d %c %s", light.armor_id, light.is_left ? 'L' : 'R',
        light.isolated ? "isolated" : "armor"),
      center + cv::Point{8, -8}, color, 0.48);
  }

  return panel;
}

// 一根被采纳的侧边灯条，配上它关联到的那块物理板在模型里对应的那根预测灯条。
// ROI 面板把两者画在一起，才能目视判断关联对不对：正常时两根基本重合，差出
// 一整根灯条的间距通常意味着关联到了隔壁那块板，或者左右配反了。
//
// 预测灯条取本帧更新之后的状态（后验），画出来的是残差而不是关联时用的先验
// 偏差。关联错了的话后验会被这根观测拖着走，残差照样看得出来。
struct SideLightMatch
{
  std::size_t light_id{0};
  cv::Point2f top{};
  cv::Point2f bottom{};
};

// 侧边灯条专用面板：把 light_roi 放大铺满，画出这一帧的全部灯条候选，按
// "滤波器采纳了" / "进了 L3 但被 matchLight 毙掉" / "被判成已检出装甲板自己
// 的灯条而丢掉" 三档分开。
//
// 之所以要单独一个窗口：灯条模型只在 light_roi 里跑，而 ROI 在整图上只占很
// 小一块，压在 reprojection 上根本看不清端点。另外 used lights 面板画的是
// 滤波器实际消费的量（含装甲板自己的两根），跟"侧边灯条检出了没有"不是同
// 一个问题——后者要连被丢掉的候选一起看才判得出来。
//
// 中间那一档是这里唯一的看点：交给 L3 的候选里大半会被 matchLight 的长度门
// 和角度门毙掉，只看"进了 L3"会把检出量当成采纳量。
cv::Mat makeSideLightPanel(
  const cv::Mat& source,
  const std::vector<L2Perception::Light>& candidates,
  const std::vector<L2Perception::Light>& kept,
  const std::vector<L3Estimation::UsedLight>& update_lights,
  const std::vector<SideLightMatch>& matches,
  const std::optional<cv::Rect>& light_roi)
{
  // light_roi 是整车框扩出来的，通常又宽又扁（3 m 处约 4:1），固定的方形面板
  // 会把放大倍率卡在宽度上、下半张全是黑边。所以面板宽度固定、高度跟着 ROI
  // 的宽高比走，倍率上限 6 是免得目标很近时糊成马赛克。
  constexpr int kPanelWidth = 1100;
  constexpr int kHeaderHeight = 76;
  constexpr double kMaxZoom = 6.0;

  // kept 和 UsedLight::light_id 都保留了 lastLights() 里的下标，靠 id 回查
  // 哪些候选活到了 L3、哪些又活过了 matchLight 的门限。
  std::set<std::size_t> kept_ids;
  for (const auto& light : kept) {
    kept_ids.insert(light.id);
  }
  std::set<std::size_t> used_ids;
  for (const auto& light : update_lights) {
    if (light.isolated) {
      used_ids.insert(light.light_id);
    }
  }

  const cv::Rect image_rect(0, 0, source.cols, source.rows);
  const cv::Rect roi = light_roi ? (*light_roi & image_rect) : cv::Rect{};

  const int content_width = kPanelWidth - 24;
  const double scale = roi.area() > 0
    ? std::min(kMaxZoom, static_cast<double>(content_width) / roi.width)
    : 1.0;
  const cv::Size resized_size{
    std::max(1, static_cast<int>(std::round(roi.width * scale))),
    std::max(1, static_cast<int>(std::round(roi.height * scale)))};
  const int panel_height = roi.area() > 0
    ? kHeaderHeight + std::clamp(resized_size.height, 180, 620) + 12
    : 200;
  cv::Mat panel(panel_height, kPanelWidth, CV_8UC3, cv::Scalar{24, 24, 24});

  drawOutlinedText(
    panel,
    cv::format(
      "side lights  candidates=%zu  to L3=%zu  used by filter=%zu  "
      "dropped(owned by armor)=%zu",
      candidates.size(), kept.size(), used_ids.size(),
      candidates.size() - kept.size()),
    {12, 28}, {255, 255, 255}, 0.58);
  drawOutlinedText(
    panel,
    "thick=used by IESKF   thin=rejected by matchLight   gray=owned by armor   "
    "green=model prediction, d=endpoint residual",
    {12, 57}, {180, 180, 180}, 0.46);

  if (roi.area() <= 0) {
    drawOutlinedText(
      panel, "no light roi this frame (target not tracked)", {28, 140},
      {120, 120, 120}, 0.68);
    return panel;
  }

  const cv::Rect content(
    12, kHeaderHeight, content_width, panel_height - kHeaderHeight - 12);
  cv::Mat resized;
  cv::resize(source(roi), resized, resized_size, 0.0, 0.0, cv::INTER_NEAREST);
  const cv::Rect destination(
    content.x + (content.width - resized.cols) / 2,
    content.y + (content.height - resized.rows) / 2,
    resized.cols, resized.rows);
  resized.copyTo(panel(destination));
  cv::rectangle(panel, destination, {70, 70, 70}, 1);
  drawOutlinedText(
    panel, cv::format("roi %dx%d  x%.1f", roi.width, roi.height, scale),
    {destination.x + 6, destination.y + 18}, {150, 150, 150}, 0.44);

  const auto panelPoint = [&](const cv::Point2f& point) {
    return cv::Point{
      destination.x + static_cast<int>(std::round((point.x - roi.x) * scale)),
      destination.y + static_cast<int>(std::round((point.y - roi.y) * scale))};
  };

  if (candidates.empty()) {
    drawOutlinedText(
      panel, "the light model returned nothing in this roi",
      {destination.x + 10, destination.y + destination.height - 12},
      {0, 255, 255}, 0.55);
    return panel;
  }

  for (const auto& light : candidates) {
    if (!std::isfinite(light.top.x) || !std::isfinite(light.top.y) ||
        !std::isfinite(light.bottom.x) || !std::isfinite(light.bottom.y)) {
      continue;
    }
    const bool to_l3 = kept_ids.count(light.id) != 0;
    const bool used = used_ids.count(light.id) != 0;
    // 被 matchLight 毙掉的那一档压暗而不是变灰：灰色留给"根本没进 L3"，两者
    // 的原因完全不同，颜色要分得开。
    const cv::Scalar bar = armorDisplayColor(light.color);
    const cv::Scalar color = used ? bar
      : to_l3 ? cv::Scalar{bar[0] * 0.45, bar[1] * 0.45, bar[2] * 0.45}
              : cv::Scalar{110, 110, 110};
    const int thickness = used ? 3 : 1;
    const int radius = used ? 6 : 3;
    const cv::Point top = panelPoint(light.top);
    const cv::Point bottom = panelPoint(light.bottom);
    cv::line(panel, top, bottom, color, thickness, cv::LINE_AA);
    cv::circle(panel, top, radius, color, cv::FILLED, cv::LINE_AA);
    cv::circle(panel, bottom, radius, color, 2, cv::LINE_AA);
    // 灯条在 ROI 里挨得近，标签压在一起就读不出来了：按 id 交替放在上下端点
    // 外侧，再按 id % 3 错开一行。
    const int stagger = static_cast<int>(light.id % 3) * 15;
    const cv::Point anchor = (light.id % 2) == 0
      ? top + cv::Point{10, -10 - stagger}
      : bottom + cv::Point{10, 22 + stagger};
    drawOutlinedText(
      panel,
      cv::format(
        "#%zu L=%.0f %.0fdeg%s", light.id, light.length,
        static_cast<double>(light.tilt_angle_deg), used ? " IN" : ""),
      anchor, color, 0.44);
  }

  // 模型预测的那根灯条：绿色，与整车叠加层同色。细灰线把观测端点连到对应的
  // 预测端点，两根重合时连线缩成一点；d 是两个端点距离的平均值。
  const cv::Scalar predicted_color{0, 255, 0};
  for (const auto& match : matches) {
    if (!std::isfinite(match.top.x) || !std::isfinite(match.top.y) ||
        !std::isfinite(match.bottom.x) || !std::isfinite(match.bottom.y)) {
      continue;
    }
    const cv::Point top = panelPoint(match.top);
    const cv::Point bottom = panelPoint(match.bottom);
    cv::line(panel, top, bottom, predicted_color, 2, cv::LINE_AA);
    cv::circle(panel, top, 5, predicted_color, cv::FILLED, cv::LINE_AA);
    cv::circle(panel, bottom, 5, predicted_color, 2, cv::LINE_AA);

    const auto observed = std::find_if(
      candidates.begin(), candidates.end(),
      [&](const L2Perception::Light& light) {
        return light.id == match.light_id;
      });
    if (observed == candidates.end()) {
      continue;
    }
    cv::line(panel, panelPoint(observed->top), top, {190, 190, 190}, 1, cv::LINE_AA);
    cv::line(
      panel, panelPoint(observed->bottom), bottom, {190, 190, 190}, 1, cv::LINE_AA);
    const double residual = 0.5 *
      (cv::norm(observed->top - match.top) +
       cv::norm(observed->bottom - match.bottom));
    drawOutlinedText(
      panel, cv::format("d=%.1f", residual), bottom + cv::Point{9, 17},
      predicted_color, 0.44);
  }

  return panel;
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

// 选一块装甲板画代价曲线：优先跟踪器当前关联的那块，其次取图像中心附近的。
std::optional<std::size_t> selectArmor(
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<ReplayTarget>& target,
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
  const L3Estimation::Armor* armor,
  const L3Estimation::PnpSolver& solver,
  const std::optional<double>& ekf_armor_yaw,
  const std::optional<double>& predicted_armor_yaw,
  int frame_index)
{
  constexpr int kWidth = 960;
  constexpr int kHeight = 540;
  // 顶部留出四行文字：帧号、板信息、单板 yaw、单峰统计。
  const cv::Rect graph{74, 130, kWidth - 74 - 24, kHeight - 130 - 56};
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
      "raw=%.1fdeg  pnp yaw=%.1fdeg  curve min=%.1fdeg (cost=%.1fpx)",
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
    // 当前估计器展开的全部装甲板（绿），和命中时刻瞄准的那块板（红）。这样
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
    // YAML 里调噪声或灯条门限，这里根本看不出变化。
    const auto runtime_config = runtime::loadConfig("config/auto_aim.yaml");

    // 检测器与实机同一个工厂组装，只有模型路径和设备允许命令行覆盖。命令行换了
    // 模型时 YAML 里的 layout 未必配得上，按模型输出名认。
    runtime::AutoAimConfig detector_config = runtime_config;
    const std::string model_override = cli.get<std::string>("model");
    if (!model_override.empty()) {
      detector_config.inference.model_path = model_override;
    }
    detector_config.inference.device = cli.get<std::string>("device");
    const L2Perception::ArmorDetector detector =
      runtime::makeDetector(detector_config, !model_override.empty());
    require(detector.ready(), "ArmorDetector 未就绪");

    const L3Estimation::ArmorConfig & armor_config = runtime_config.armor;
    ReplayTracker tracker(
      calibration, armor_config, runtime_config.ieskf_tracker,
      runtime_config.ieskf_target);
    require(tracker.ready(), "EskfTracker 拒绝了该标定");
    // 与 Tracker 内部同参数的求解器，只用来做重投影和代价曲线，不参与滤波。
    L3Estimation::PnpSolver solver(calibration, armor_config);
    require(solver.ready(), "诊断用 PnpSolver 拒绝了该标定");
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
    StageClock clock;
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
    std::size_t same_armor_direction_reversal_frames = 0;
    double largest_reversal_step = 0.0;
    // 拒绝原因直方图。fire_feasible 是 0 时，唯一有用的信息是"被哪一条拦住的"。
    std::map<L5Control::RejectReason, std::size_t> reject_histogram;
    // 回放里云台姿态来自录像，不是本规划器闭环出来的，所以 aim_error 基本必然
    // 触发。把误差量级和容差一起打出来，才能判断是"云台没跟"还是"规划跑偏"。
    std::vector<double> aim_yaw_errors;
    // 上一帧规划命令用于命令跳变检查和 L4 选板连续性诊断。
    // 逐帧状态导出。抖动是个时间序列问题，肉眼看单帧看不出来源——是观测噪声
    // 直接透传、还是关联在编号间跳、还是 clamp 在硬复位，只有把每帧的状态和
    // 关联结果落到文件里逐帧比才能分开。
    const std::string csv_path = cli.get<std::string>("csv");
    std::ofstream state_csv;
    if (!csv_path.empty()) {
      state_csv.open(csv_path);
      require(state_csv.is_open(), "无法打开 csv 输出路径: " + csv_path);
      state_csv << "frame,t,state,ndet,nlight,nmatch,armor_ids,jumped,"
                   "xc,yc,zc,vx,vy,vz,yaw_deg,vyaw,r1,r2,h,roll_deg,pitch_deg,"
                   "nis,nis_dof,roi_x,roi_y,roi_w,roi_h,refined,net_kept\n";
      state_csv << std::fixed << std::setprecision(6);
    }

    int last_plan_armor_id = -1;
    std::optional<double> last_command_yaw;
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
      clock.tick();
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
      clock.lap("L1 回放读帧");

      const auto timestamp =
        t0 + std::chrono::microseconds(static_cast<long long>(pose.seconds * 1e6));
      const Eigen::Quaterniond q_world_barrel =
        toWorldBarrelPose(pose, sp_convention, R_imu_barrel);

      /// 自瞄核心逻辑

      const std::optional<cv::Rect> light_roi = tracker.lightRoi(
        q_world_barrel, timestamp, img.size());
      // 整车先验驱动的显式空间注意力：ROI 已按网络输入宽高比修正并扩成方形，
      // 远距小目标裁剪后再 resize 相当于局部放大。目标丢失时它自动退化为整图。
      const cv::Rect net_roi = tracker.netFocusRoi(
        q_world_barrel, timestamp, img.size(), detector.net_aspect_ratio());
      clock.lap("L3 ROI 先验");
      L2Perception::ArmorFrame detection_frame =
        detector.detectFrame(img, light_roi, net_roi, enemy_color);
      clock.lap("L2 检测");
      {
        // L2 常年占掉管线九成，只报总数没法定位是网络、精修还是侧边灯条那一路
        // 贵。这些分项已经含在"L2 检测"里，不重复计入合计。
        const L2Perception::DetectTiming& t = detector.lastTiming();
        clock.add("  ├ 预处理", t.preprocess);
        clock.add("  ├ 整板推理", t.infer);
        clock.add("  ├ 解码", t.decode);
        clock.add("  ├ 角点精修", t.refine);
        clock.add("  ├ 数字分类", t.number);
        clock.add("  └ 侧边灯条", t.side_light);
      }
      const std::vector<L2Perception::Armor> recognized_armors =
        detection_frame.armors;
      const cv::Mat recognition_panel =
        makeRecognitionPanel(img, recognized_armors, enemy_color);
      clock.lap("绘图/显示", true);
      auto& armors = detection_frame.armors;
      std::erase_if(armors, [enemy_color](const L2Perception::Armor& armor) {
        return enemy_color != L2Perception::ArmorColor::Unknown &&
          armor.color != enemy_color;
      });

      solver.set_R_world_barrel(q_world_barrel);
      const auto target = tracker.track(
        armors, detection_frame.lights, q_world_barrel, timestamp);
      const auto& used_lights = tracker.usedLights();
      clock.lap("L3 跟踪");
      const auto target_armor_poses = tracker.armorPoses();
      const cv::Mat used_light_panel = makeUsedLightPanel(
        img, used_lights, detection_frame.lights.size());

      // 每根采纳的侧边灯条关联到哪块板的哪一侧，都记在 UsedLight 里；按这个
      // 编号把模型展开的那块板重投影回来，取同一侧的两个角点，就是它本该长的
      // 位置。角点顺序左上、右上、右下、左下，左灯条是 0/3，右灯条是 1/2。
      std::vector<SideLightMatch> side_light_matches;
      if (target) {
        const auto side_armor_type =
          L3Estimation::armorTypeOf(target->name).value_or(
            L3Estimation::ArmorType::Small);
        for (const auto& used : used_lights) {
          if (!used.isolated || used.armor_id < 0) {
            continue;
          }
          const auto armor_index = static_cast<std::size_t>(used.armor_id);
          if (armor_index >= target_armor_poses.size()) {
            continue;
          }
          const Eigen::Vector4d& xyza = target_armor_poses[armor_index];
          const auto corners = solver.reproject_armor(
            xyza.head<3>(), xyza[3], side_armor_type, target->name);
          if (corners.size() != 4) {
            continue;
          }
          side_light_matches.push_back(
            {used.light_id, used.is_left ? corners[0] : corners[1],
             used.is_left ? corners[3] : corners[2]});
        }
      }

      // lastLights() 是颜色过滤后、剔除已检出装甲板自己的灯条之前的全部候选，
      // detection_frame.lights 是真正交给 L3 的那批，两者一起画才看得出侧边
      // 灯条是"没检出"还是"检出了但被判给了某块板"。
      const cv::Mat side_light_panel = makeSideLightPanel(
        img, detector.lastLights(), detection_frame.lights, used_lights,
        side_light_matches, light_roi);
      clock.lap("绘图/显示", true);
      const std::optional<FilterEstimate> filter_estimate = target
        ? std::optional<FilterEstimate>{target->estimate()}
        : std::nullopt;

      /// PnP 代价曲线

      // IESKF 的正常观测入口刻意不跑 PnP；这份副本仅供 full 视图下既有的
      // PnP 代价曲线诊断，不进入滤波器。
      std::vector<L3Estimation::Armor> diagnostic_pnp_observations;
      {
        diagnostic_pnp_observations.reserve(armors.size());
        for (const auto& armor : armors) {
          diagnostic_pnp_observations.push_back(
            L3Estimation::toObservation(armor, timestamp));
          solver.single_pnp(diagnostic_pnp_observations.back());
        }
      }
      clock.lap("诊断 PnP", true);
      if (state_csv.is_open()) {
        state_csv << frame_index << ','
                  << std::chrono::duration<double>(timestamp.time_since_epoch()).count()
                  << ',' << static_cast<int>(tracker.state()) << ',' << armors.size()
                  << ',' << detection_frame.lights.size() << ','
                  << tracker.lastMatchCount() << ',' << '"'
                  << tracker.lastMatchedIdsString() << '"' << ','
                  << (target && target->jumped() ? 1 : 0) << ',';
        if (filter_estimate) {
          const auto& e = *filter_estimate;
          state_csv << e.center.x() << ',' << e.center.y() << ',' << e.center.z() << ','
                    << e.velocity.x() << ',' << e.velocity.y() << ',' << e.velocity.z()
                    << ',' << e.yaw * kRadToDeg << ',' << e.yaw_rate << ','
                    << e.radius1 << ',' << e.radius2.value_or(0.0) << ','
                    << e.height_offset.value_or(0.0) << ','
                    << e.roll.value_or(0.0) * kRadToDeg << ','
                    << e.pitch.value_or(0.0) * kRadToDeg << ',';
        } else {
          state_csv << ",,,,,,,,,,,,,";
        }
        state_csv << (target ? target->lastNis() : 0.0) << ','
                  << (target ? target->lastNisDof() : 0) << ','
                  << net_roi.x << ',' << net_roi.y << ',' << net_roi.width << ','
                  << net_roi.height << ','
                  << detector.lastRefine().refined << ','
                  << detector.lastRefine().network_kept << '\n';
      }

      const auto& observations = diagnostic_pnp_observations;
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
      if (target && target->last_id >= 0 &&
          static_cast<std::size_t>(target->last_id) <
            target_armor_poses.size()) {
        ekf_armor_yaw =
          target_armor_poses[static_cast<std::size_t>(target->last_id)].w();
      }

      /// 整车预测：把当前估计状态外推 predict_time 秒后重新展开所有装甲板
      std::optional<ReplayTarget> predicted;
      std::vector<Eigen::Vector4d> predicted_armor_poses;
      std::optional<double> predicted_armor_yaw;
      if (target && predict_time > 0.0) {
        predicted = *target;
        predicted->predict(predict_time);
        predicted_armor_poses = predicted->armor_xyza_list();
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
      clock.lap("CSV 导出", true);
      const auto plan =
        tracker.plan(planner, target, robot_state, plan_time, false);
      clock.lap("L4 规划");
      const int plan_armor_id =
        plan.fire.has_value() ? plan.fire->armor_id : -1;

      L5Control::FireInput fire_input;
      if (target) {
        fire_input.target_name = target->name;
      }
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
      if (plan.valid() && last_command_yaw &&
          plan_armor_id == last_plan_armor_id) {
        const double step =
          L6Telemetry::limit_rad(plan.aim.yaw - *last_command_yaw);
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
      clock.lap("L5 火控");

      if (plan.valid()) {
        ++plan_valid_frames;
        last_plan_armor_id = plan_armor_id;
        last_command_yaw = plan.aim.yaw;
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

      if (target && filter_estimate) {
        L6Telemetry::logDebugRaw(
          "[" + std::to_string(frame_index) + "] estimator=" +
          "ieskf+endpoint state=" +
          std::string(stateName(tracker.state())) + ' ' +
          filterKinematicsText(*filter_estimate) + ' ' +
          filterGeometryText(
            *filter_estimate, target->last_id, target->lastNis(),
            target->lastNisDof()));
      } else {
        L6Telemetry::logDebugRaw(
          "[" + std::to_string(frame_index) + "] estimator=" +
          "ieskf+endpoint state=" +
          std::string(stateName(tracker.state())) + " target=none");
      }

      if (full_view) {
        // full 调试视图仍保留原图四角点；单独的 recognition roi 窗口负责放大
        // 展示类别、颜色、置信度和外接 ROI。
        for (const auto& armor : armors) {
          const cv::Scalar color = armorDisplayColor(armor.color);
          for (std::size_t index = 0; index < armor.corners.size(); ++index) {
            cv::line(
              img, toPixel(armor.corners[index]),
              toPixel(armor.corners[(index + 1) % armor.corners.size()]),
              color, 2, cv::LINE_AA);
          }
        }

        // IESKF 消费的就是上面检测框的左右灯条端点，不再把 PnP 位姿框冒充成
        // 滤波观测；PnP 在这条路径只负责冷启动与候选有效性检查。
      }

      // 侧边灯条画在原图上：单独的 side lights 窗口看端点，这里看它到底长在
      // 车的哪一侧。只画滤波器真正吃下去的那几根——过了 matchLight 的长度、
      // 角度、卡方三道门，且这一帧 updateMulti 成功——标注关联到的物理板编号
      // 和左右。交给 L3 的候选大半会被门限毙掉，把它们一起画在原图上等于把
      // "检出了" 当成 "用上了"；被毙掉的和被判给已检出装甲板的都去 side
      // lights 窗口里看。灰框是灯条模型的搜索区 light_roi。
      if (light_roi) {
        cv::rectangle(img, *light_roi, {90, 90, 90}, 1, cv::LINE_AA);
      }
      {
        const cv::Scalar light_color{0, 255, 255};
        for (const auto& light : used_lights) {
          if (!light.isolated) {
            continue;
          }
          if (!std::isfinite(light.top.x) || !std::isfinite(light.bottom.x)) {
            continue;
          }
          const cv::Point top = toPixel(light.top);
          const cv::Point bottom = toPixel(light.bottom);
          cv::line(img, top, bottom, light_color, 3, cv::LINE_AA);
          cv::circle(img, top, 5, light_color, cv::FILLED, cv::LINE_AA);
          cv::circle(img, bottom, 5, light_color, 2, cv::LINE_AA);
          drawOutlinedText(
            img,
            cv::format(
              "#%zu id=%d %c", light.light_id, light.armor_id,
              light.is_left ? 'L' : 'R'),
            top + cv::Point{8, -8}, light_color, 0.5);
        }
      }

      // 绿色是当前估计器展开的全部物理装甲板，和 sp_vision 画的是同一个量：
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
          "frame=%d ieskf+endpoint state=%s target=%s det=%zu obs=%zu", frame_index,
          std::string(stateName(tracker.state())).c_str(),
          target ? armorClassName(target->name) : "-", armors.size(),
          observations.size()),
        {10, 32}, {255, 255, 255});
      drawOutlinedText(
        img,
        cv::format(
          "gimbal yaw=%.2fdeg",
          L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0)[0] *
            kRadToDeg),
        {10, 62}, {255, 255, 255});
      if (full_view && selected &&
          observations[*selected].name != L3Estimation::ArmorName::Unknown) {
        const auto& armor = observations[*selected];
        drawOutlinedText(
          img,
          cv::format(
            "PnP diagnostic yaw=%.1fdeg (init only)",
            armor.ypr_in_world[0] * kRadToDeg),
          {10, 92}, {0, 255, 0});
      }
      if (target && filter_estimate) {
        drawOutlinedText(
          img, filterKinematicsText(*filter_estimate),
          {10, full_view ? 122 : 92}, {0, 255, 0}, 0.55);
        drawOutlinedText(
          img,
          filterGeometryText(
            *filter_estimate, target->last_id, target->lastNis(),
            target->lastNisDof()),
          {10, full_view ? 152 : 122}, {0, 255, 0}, 0.5);
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
              plan_armor_id, plan_armor_id)
          : cv::format("CMD not sent (plan %s)", planErrorName(plan.reason)),
        {10, full_view ? 182 : 152},
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
      if (target && filter_estimate) {
        data["x"] = filter_estimate->center.x();
        data["vx"] = filter_estimate->velocity.x();
        data["y"] = filter_estimate->center.y();
        data["vy"] = filter_estimate->velocity.y();
        data["z"] = filter_estimate->center.z();
        data["vz"] = filter_estimate->velocity.z();
        data["a"] = filter_estimate->yaw * kRadToDeg;
        data["w"] = filter_estimate->yaw_rate;
        data["r"] = filter_estimate->radius1;
        data["yaw"] = filter_estimate->yaw * kRadToDeg;
        data["yaw_rate"] = filter_estimate->yaw_rate;
        data["radius1"] = filter_estimate->radius1;
        if (filter_estimate->radius2) {
          data["radius2"] = *filter_estimate->radius2;
        }
        if (filter_estimate->height_offset) {
          data["height_offset"] = *filter_estimate->height_offset;
        }
        if (filter_estimate->dz1 && filter_estimate->dz2) {
          data["dz1"] = *filter_estimate->dz1;
          data["dz2"] = *filter_estimate->dz2;
        }
        if (filter_estimate->roll && filter_estimate->pitch) {
          data["roll"] = *filter_estimate->roll * kRadToDeg;
          data["pitch"] = *filter_estimate->pitch * kRadToDeg;
        }
        data["last_id"] = target->last_id;
        data["nis"] = target->lastNis();
        data["nis_dof"] = target->lastNisDof();
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
        const FilterEstimate predicted_estimate = predicted->estimate();
        data["predict_time"] = predict_time;
        data["pred_x"] = predicted_estimate.center.x();
        data["pred_y"] = predicted_estimate.center.y();
        data["pred_z"] = predicted_estimate.center.z();
        data["pred_a"] = predicted_estimate.yaw * kRadToDeg;
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
      cv::imshow("recognition roi", recognition_panel);
      cv::imshow("used lights", used_light_panel);
      cv::imshow("side lights", side_light_panel);
      cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
      cv::imshow("reprojection", img);
      // waitKey 是人机交互的等待，和算法耗时无关，必须排除在外。
      clock.lap("绘图/显示", true);
      clock.flush();
      const int key = cv::waitKey(wait_ms);
      if (key == 'q' || key == 27) {
        break;
      }
      if (key == ' ') {
        paused = true;
      }
    }

    cv::destroyAllWindows();
    printStageTiming(clock);
    std::cout << "\n回放结束\n"
              << "估计器: ieskf+endpoint" << '\n'
              << "帧数: " << frames << '\n'
              << "有 PnP 观测的帧: " << observation_frames << '\n'
              << "PnP 成功的候选数: " << valid_pnp_observations
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
              << "同一装甲板内方向折返的帧(>0.05deg): "
              << same_armor_direction_reversal_frames << '\n'
              << "最大折返单步: " << largest_reversal_step * kRadToDeg
              << " deg\n"
              << "初始化门限: PnP 成功；逐帧校正: 完整板/独立灯条端点 + 单板深度差\n";
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
