// PnP 观测噪声 R 的离线标定工具。
//
// 在 Daedalus 仿真里跑 detect -> single_pnp，把每块装甲板的四维观测
// [方位角, 俯仰角, 距离, 板 yaw] 和同一帧的真值相减，按距离 × 斜视角分箱统计
// 残差协方差，输出 CSV 供进一步分析。
//
// 读之前先明白这个工具**不**能给你什么：仿真图像没有畸变、运动模糊、曝光和灯条
// 过曝，云台位姿是精确值也没有图像-IMU 时间偏移，所以这里量出来的方差在真车上
// 一定偏小 —— 尤其是距离和板 yaw 这两个敏感分量。它能给的是 R 随距离和斜视角
// 变化的**形状**，以及 R 的模型形式是否成立；标量尺度要在真车上用 NIS 反标。
//
// 三条纪律，工具本身会盯着：
//   1. 先看均值。坐标系链路错了会表现为偏置而不是噪声，均值不接近零就先查链路。
//   2. 板 yaw 的残差不是高斯的，PnP 跳解时是双峰。工具单独统计跳解率并给直方图。
//   3. 匹配用最近邻，会天然丢掉大误差。工具报告未匹配率，这个数不低就别信统计量。

#include "l1_sensor/daedalus_ground_truth.hpp"
#include "l1_sensor/daedalus_source.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/tracker.hpp"
#include "l3_estimation/types.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "runtime/auto_aim_config.hpp"

#include <Eigen/Geometry>

#include <opencv2/core/utility.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void requestStop(int) noexcept
{
  g_stop_requested = 1;
}

[[nodiscard]] double degrees(double radians) noexcept
{
  return radians * 180.0 / std::numbers::pi;
}

// PnP 把装甲板位置解算成**相对观测者原点**的向量，而真值是绝对 odom 坐标，
// 所以对比前必须减掉观测者原点。Daedalus 发布的 odom 是云台原点，muzzle/camera
// 是相对云台的偏移，而 DaedalusSource::calibration() 把 T_barrel_camera 的平移
// 取成 camera - muzzle，也就是说 xyz_in_world 的原点是**枪口**。
// 这里不赌哪一个对：三个候选原点同时统计，让距离残差的均值自己说话。
enum class Origin { Gimbal = 0, Muzzle = 1, Camera = 2 };
constexpr std::array<const char*, 3> kOriginNames{"gimbal", "muzzle", "camera"};

// 装甲板后仰 15°，再留一点余量：超过这个视角的板相机根本看不到，
// 所以它不可能是本次观测的对应板。
constexpr double kMaxVisibleViewAngle = 100.0 * std::numbers::pi / 180.0;

[[nodiscard]] Eigen::Vector3d observerOrigin(
  const L1Sensor::DaedalusFrame& frame,
  Origin origin)
{
  const Eigen::Matrix3d R_world_barrel = frame.gimbal.orientation.toRotationMatrix();
  switch (origin) {
    case Origin::Gimbal:
      return frame.odom.position;
    case Origin::Muzzle:
      return frame.odom.position + R_world_barrel * frame.muzzle.position;
    case Origin::Camera:
      return frame.odom.position + R_world_barrel * frame.camera.position;
  }
  return frame.odom.position;
}

[[nodiscard]] Origin parseOrigin(const std::string& value)
{
  for (std::size_t index = 0; index < kOriginNames.size(); ++index) {
    if (value == kOriginNames[index]) {
      return static_cast<Origin>(index);
    }
  }
  throw std::runtime_error("--origin must be gimbal, muzzle or camera");
}

[[nodiscard]] L1Sensor::EnemyColor parseEnemyColor(const std::string& value)
{
  if (value == "red") {
    return L1Sensor::EnemyColor::Red;
  }
  if (value == "blue") {
    return L1Sensor::EnemyColor::Blue;
  }
  throw std::runtime_error("enemy_color must be 'red' or 'blue'");
}

[[nodiscard]] L2Perception::ArmorColor armorColor(L1Sensor::EnemyColor color) noexcept
{
  switch (color) {
    case L1Sensor::EnemyColor::Red:
      return L2Perception::ArmorColor::Red;
    case L1Sensor::EnemyColor::Blue:
      return L2Perception::ArmorColor::Blue;
    case L1Sensor::EnemyColor::Unknown:
      return L2Perception::ArmorColor::Unknown;
  }
  return L2Perception::ArmorColor::Unknown;
}

// 真值里 team 0 = red、1 = blue，和 L1Sensor::EnemyColor 是两套编号。
[[nodiscard]] bool teamMatchesEnemy(int team, L1Sensor::EnemyColor enemy) noexcept
{
  return (team == 0 && enemy == L1Sensor::EnemyColor::Red) ||
    (team == 1 && enemy == L1Sensor::EnemyColor::Blue);
}

// 仿真的 ArmorLabel 与 newvision 的 ArmorClass 前七个编号是逐一对应的
// （Sentry/Guard=0 … Outpost=6）；仿真只有一个 Base=7，newvision 分成
// BaseSmall=7 和 BaseLarge=8。
//
// 识别只能定到**车**定不到**板**——一辆四板车的四块板贴的是同一个号，所以这一关
// 只用来排除跨车误配，同车内选哪块板仍然得靠几何。
[[nodiscard]] bool truthLabelMatchesClass(
  int truth_label,
  L3Estimation::ArmorName name) noexcept
{
  const int detected = static_cast<int>(name);
  if (truth_label == 7) {
    return detected == static_cast<int>(L2Perception::ArmorClass::BaseSmall) ||
      detected == static_cast<int>(L2Perception::ArmorClass::BaseLarge);
  }
  return truth_label >= 0 && truth_label <= 6 && truth_label == detected;
}

struct Sample {
  std::uint64_t frame_seq{0};
  int armor_label{-1};
  int plate_index{-1};
  int target_index{-1};

  // 分箱依据一律取真值，避免用被测量的估计值去决定它自己落进哪个箱。
  double truth_distance{0.0};
  double truth_view_angle{0.0};  // |板 yaw - 视线方位角|，0 = 正视
  double truth_radius{0.0};

  double e_azimuth{0.0};
  double e_elevation{0.0};
  double e_distance{0.0};
  double e_yaw{0.0};

  double reprojection_error{0.0};
  double match_distance{0.0};
  // 三种观测者原点各自的距离残差，用来判定哪个原点约定是对的。
  std::array<double, 3> e_distance_by_origin{};

  // 原始向量：观测的世界系位置、真值相对观测者原点的位置，以及本帧云台姿态。
  // 有了这三样才能把残差转回枪管系，用最小二乘解出常量外参偏移。
  Eigen::Vector3d observed{Eigen::Vector3d::Zero()};
  Eigen::Vector3d truth_relative{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q_world_barrel{Eigen::Quaterniond::Identity()};
};

struct Accumulator {
  std::size_t count{0};
  double sum{0.0};
  double sum_squares{0.0};

  void add(double value) noexcept
  {
    ++count;
    sum += value;
    sum_squares += value * value;
  }

  [[nodiscard]] double mean() const noexcept
  {
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
  }

  // 去均值后的方差。偏置属于坐标系问题，不该被算进观测噪声。
  [[nodiscard]] double variance() const noexcept
  {
    if (count < 2) {
      return 0.0;
    }
    const double m = mean();
    const double value =
      sum_squares / static_cast<double>(count) - m * m;
    return value > 0.0 ? value * static_cast<double>(count) /
        static_cast<double>(count - 1)
                       : 0.0;
  }

  [[nodiscard]] double sigma() const noexcept { return std::sqrt(variance()); }
};

struct BinStats {
  Accumulator azimuth;
  Accumulator elevation;
  Accumulator distance;
  Accumulator yaw;  // 只累计未跳解的样本
  std::size_t total{0};
  std::size_t flipped{0};

  void add(const Sample& sample, double flip_threshold) noexcept
  {
    ++total;
    azimuth.add(sample.e_azimuth);
    elevation.add(sample.e_elevation);
    distance.add(sample.e_distance);
    if (std::abs(sample.e_yaw) > flip_threshold) {
      ++flipped;
    } else {
      yaw.add(sample.e_yaw);
    }
  }
};

struct CalibConfig {
  L1Sensor::DaedalusSourceOptions source;
  L1Sensor::DaedalusGroundTruthOptions truth;
  runtime::AutoAimConfig auto_aim;
  std::size_t inference_threads{0};
  L1Sensor::EnemyColor enemy_color{L1Sensor::EnemyColor::Blue};
  std::chrono::milliseconds connect_timeout{10000};
  std::chrono::milliseconds poll_sleep{1};
};

template<typename T>
[[nodiscard]] T yamlOr(const YAML::Node& node, const char* key, T fallback)
{
  const YAML::Node value = node ? node[key] : YAML::Node{};
  return value ? value.as<T>() : std::move(fallback);
}

[[nodiscard]] CalibConfig loadConfig(const std::string& path)
{
  const YAML::Node root = YAML::LoadFile(path);
  CalibConfig config;

  const YAML::Node ipc = root["ipc"];
  config.source.meta_path =
    yamlOr<std::string>(ipc, "meta_path", config.source.meta_path);
  config.source.image_pool_path =
    yamlOr<std::string>(ipc, "image_pool_path", config.source.image_pool_path);
  config.source.producer_timeout =
    std::chrono::milliseconds(yamlOr<int>(ipc, "producer_timeout_ms", 1000));
  config.truth.path = yamlOr<std::string>(ipc, "truth_path", config.truth.path);
  config.truth.producer_timeout = config.source.producer_timeout;
  config.connect_timeout =
    std::chrono::milliseconds(yamlOr<int>(ipc, "connect_timeout_ms", 10000));
  config.poll_sleep =
    std::chrono::milliseconds(yamlOr<int>(ipc, "poll_sleep_ms", 1));

  config.auto_aim = runtime::loadAutoAimConfig(path);
  config.inference_threads = yamlOr<std::size_t>(
    root["inference"], "num_threads", config.inference_threads);
  config.enemy_color =
    parseEnemyColor(yamlOr<std::string>(root["robot"], "enemy_color", "blue"));
  return config;
}

[[nodiscard]] L2Perception::ArmorDetector makeDetector(const CalibConfig& config)
{
  auto backend = L2Perception::makeInferenceBackend(config.auto_aim.inference_backend);
  L2Perception::InferenceModelConfig model;
  model.model_path = config.auto_aim.model_path;
  model.device = config.auto_aim.inference_device;
  model.model_color_order = L2Perception::ModelColorOrder::Rgb;
  model.normalization_divisor = 255.0F;
  model.inference_num_threads = config.inference_threads;
  backend->load(model);
  if (!backend->ready()) {
    throw std::runtime_error("inference backend did not become ready");
  }
  return L2Perception::ArmorDetector(std::move(backend));
}

// 启动自检：把真值侧的几何和 config 的装甲板几何摆在一起。
//
// 观测减真值出现常量偏置时，嫌疑只有两边：外参链路（观测者原点摆错）或者真值
// 本身（MARKER mesh 根本不是板的四个角）。真值角点的边长能一次性排除后者 ——
// 如果量出来就是 config 的 135×56 mm，那真值是对的，问题在外参。
void reportGeometry(
  const L1Sensor::DaedalusGroundTruth& truth,
  const L1Sensor::DaedalusFrame& frame,
  const L3Estimation::ArmorConfig& armor_config)
{
  std::cout << "\n================ 启动自检 ================\n";
  if (truth.plates.empty()) {
    std::cout << "真值快照里没有装甲板，无法自检几何。\n";
    return;
  }

  Accumulator top;
  Accumulator bottom;
  Accumulator left;
  Accumulator right;
  Accumulator diagonal_gap;
  for (const auto& plate : truth.plates) {
    top.add((plate.corners[0] - plate.corners[1]).norm());
    bottom.add((plate.corners[3] - plate.corners[2]).norm());
    left.add((plate.corners[0] - plate.corners[3]).norm());
    right.add((plate.corners[1] - plate.corners[2]).norm());
    // 矩形的两条对角线等长；差得多说明这四个点根本不是一块矩形板的角。
    diagonal_gap.add(std::abs(
      (plate.corners[0] - plate.corners[2]).norm() -
      (plate.corners[1] - plate.corners[3]).norm()));
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "真值角点几何（" << truth.plates.size() << " 块板）：\n";
  std::cout << "  上边 |c0-c1| " << top.mean() << " m    下边 |c3-c2| "
            << bottom.mean() << " m\n";
  std::cout << "  左边 |c0-c3| " << left.mean() << " m    右边 |c1-c2| "
            << right.mean() << " m\n";
  std::cout << "  对角线长度差 " << diagonal_gap.mean() << " m\n";
  std::cout << "config 的装甲板几何：小板宽 " << armor_config.small_width
            << " m，大板宽 " << armor_config.big_width << " m，高 "
            << armor_config.height << " m\n";

  const double width = (top.mean() + bottom.mean()) / 2.0;
  const double height = (left.mean() + right.mean()) / 2.0;
  const double width_gap = std::min(
    std::abs(width - armor_config.small_width),
    std::abs(width - armor_config.big_width));
  const double height_gap = std::abs(height - armor_config.height);
  if (width_gap < 0.01 && height_gap < 0.01) {
    std::cout << "  => 真值角点就是装甲板的四个角，偏置不在真值侧。\n";
  } else {
    std::cout << "  => [警告] 真值角点与 config 的装甲板尺寸对不上（宽差 "
              << width_gap << " m，高差 " << height_gap
              << " m）。MARKER mesh 可能不是板角，或者 config 的板尺寸不对；"
                 "这种情况下所有残差都不可信。\n";
  }

  // 三个候选观测者原点，以及它们相对第一块板的高度差。xyz_in_world 的原点是
  // T_barrel_camera 的平移所隐含的那个点，这几行是用来对出它到底在哪。
  const Eigen::Matrix3d R_world_barrel =
    frame.gimbal.orientation.toRotationMatrix();
  const auto show = [](const char* name, const Eigen::Vector3d& value) {
    std::cout << "  " << std::left << std::setw(26) << name << std::right
              << "[" << std::setw(9) << value.x() << "," << std::setw(9)
              << value.y() << "," << std::setw(9) << value.z()
              << "]  |.| = " << value.norm() << '\n';
  };
  std::cout << "\n观测者原点相关量（world 系，z 朝上）：\n";
  show("odom（云台原点）", frame.odom.position);
  show("muzzle 偏移（云台系）", frame.muzzle.position);
  show("camera 偏移（云台系）", frame.camera.position);
  show("camera - muzzle", frame.camera.position - frame.muzzle.position);
  show("R_wb * muzzle", R_world_barrel * frame.muzzle.position);
  show("R_wb * camera", R_world_barrel * frame.camera.position);
  std::cout << "  第三行是 DaedalusSource::calibration() 拿去当 T_barrel_camera "
               "平移的向量；\n"
               "  后两行是本工具用来推观测者原点的。前者按云台系写、后者按枪管系"
               "用，\n  两者不一致就会变成一个跟着云台转的常量偏置。\n";

  const Eigen::Vector3d plate = truth.plates.front().center;
  std::cout << "\n第一块板中心 world z = " << plate.z()
            << " m，各候选原点的 z：\n";
  for (std::size_t index = 0; index < kOriginNames.size(); ++index) {
    const Eigen::Vector3d origin =
      observerOrigin(frame, static_cast<Origin>(index));
    std::cout << "  " << std::left << std::setw(8) << kOriginNames[index]
              << std::right << " z = " << std::setw(9) << origin.z()
              << "   板高于原点 " << std::setw(9) << plate.z() - origin.z()
              << " m\n";
  }
  std::cout << "==========================================\n";
}

// ---------------------------------------------------------------- 统计输出

void printBinTable(
  const std::string& title,
  const std::string& bin_header,
  const std::vector<std::string>& labels,
  const std::vector<BinStats>& bins)
{
  std::cout << '\n' << title << '\n';
  std::cout << std::left << std::setw(14) << bin_header << std::right
            << std::setw(8) << "N" << std::setw(12) << "σ方位[mrad]"
            << std::setw(12) << "σ俯仰[mrad]" << std::setw(12) << "σ距离[m]"
            << std::setw(12) << "σ板yaw[°]" << std::setw(10) << "跳解率"
            << std::setw(12) << "偏置距离[m]" << '\n';
  std::cout << std::string(92, '-') << '\n';

  for (std::size_t index = 0; index < bins.size(); ++index) {
    const BinStats& bin = bins[index];
    if (bin.total == 0) {
      continue;
    }
    const double flip_rate = static_cast<double>(bin.flipped) /
      static_cast<double>(bin.total) * 100.0;
    std::cout << std::left << std::setw(14) << labels[index] << std::right
              << std::setw(8) << bin.total << std::fixed
              << std::setw(12) << std::setprecision(3) << bin.azimuth.sigma() * 1000.0
              << std::setw(12) << std::setprecision(3) << bin.elevation.sigma() * 1000.0
              << std::setw(12) << std::setprecision(4) << bin.distance.sigma()
              << std::setw(12) << std::setprecision(3) << degrees(bin.yaw.sigma())
              << std::setw(9) << std::setprecision(1) << flip_rate << "%"
              << std::setw(12) << std::setprecision(4) << bin.distance.mean()
              << '\n';
  }
}

void printYawHistogram(const std::vector<Sample>& samples)
{
  // 板 yaw 残差是这四个分量里唯一可能双峰的，只报一个 σ 会把双峰平均掉。
  constexpr int kBins = 37;
  constexpr double kRange = 90.0;  // 度，超出的归到两端
  std::array<std::size_t, kBins> histogram{};
  for (const Sample& sample : samples) {
    const double value = std::clamp(degrees(sample.e_yaw), -kRange, kRange);
    const int index = static_cast<int>(
      std::lround((value + kRange) / (2.0 * kRange) * (kBins - 1)));
    ++histogram[static_cast<std::size_t>(std::clamp(index, 0, kBins - 1))];
  }

  const std::size_t peak = *std::max_element(histogram.begin(), histogram.end());
  if (peak == 0) {
    return;
  }
  std::cout << "\n板 yaw 残差直方图 (度，±90 外归入两端；看是不是单峰)\n";
  for (int index = 0; index < kBins; ++index) {
    const double centre = -kRange + 2.0 * kRange * index / (kBins - 1);
    const std::size_t width = histogram[static_cast<std::size_t>(index)] * 60 / peak;
    std::cout << std::right << std::setw(7) << std::fixed << std::setprecision(1)
              << centre << " | " << std::string(width, '#') << ' '
              << histogram[static_cast<std::size_t>(index)] << '\n';
  }
}

void printSummary(
  const std::vector<Sample>& samples,
  const L3Estimation::FilterEst::TargetConfig& target_config,
  Origin origin,
  std::size_t frames_seen,
  std::size_t frames_without_truth,
  std::size_t detections_total,
  std::size_t detections_unmatched,
  std::size_t detections_misclassified,
  double flip_threshold)
{
  std::cout << "\n================ PnP 观测噪声标定结果 ================\n";
  std::cout << "帧数 " << frames_seen << "，其中缺真值 " << frames_without_truth
            << "；PnP 成功观测 " << detections_total << "，未匹配上真值 "
            << detections_unmatched;
  if (detections_total > 0) {
    std::cout << " ("
              << std::fixed << std::setprecision(1)
              << static_cast<double>(detections_unmatched) /
        static_cast<double>(detections_total) * 100.0
              << "%)";
  }
  std::cout << "\n其中类别与真值对不上（误分类）" << detections_misclassified;
  if (detections_total > 0) {
    std::cout << " (" << std::fixed << std::setprecision(1)
              << static_cast<double>(detections_misclassified) /
        static_cast<double>(detections_total) * 100.0
              << "%)";
  }
  std::cout << "\n有效样本 " << samples.size() << "，观测者原点 = "
            << kOriginNames[static_cast<std::size_t>(origin)] << '\n';

  if (samples.empty()) {
    std::cout << "\n没有有效样本，统计量无意义。\n";
    return;
  }
  if (detections_total > 0 &&
      static_cast<double>(detections_unmatched) /
          static_cast<double>(detections_total) >
        0.2) {
    std::cout << "\n[警告] 未匹配率超过 20%。最近邻匹配会优先丢掉大误差，"
                 "这个比例下的方差是偏小的，先查匹配半径和坐标系。\n";
  }

  // ---- 1. 先看偏置 ----
  BinStats all;
  for (const Sample& sample : samples) {
    all.add(sample, flip_threshold);
  }

  std::cout << "\n--- 1. 残差均值（偏置）---\n";
  std::cout << "偏置是坐标系/标定问题，不是噪声。它显著大于 σ 就先别看后面的表。\n";
  const std::array<std::pair<const char*, const Accumulator*>, 4> components{{
    {"方位角 [mrad]", &all.azimuth},
    {"俯仰角 [mrad]", &all.elevation},
    {"距离   [m]   ", &all.distance},
    {"板 yaw [mrad]", &all.yaw},
  }};
  const std::array<double, 4> scales{1000.0, 1000.0, 1.0, 1000.0};
  for (std::size_t index = 0; index < components.size(); ++index) {
    const Accumulator& acc = *components[index].second;
    const double scale = scales[index];
    const double mean = acc.mean() * scale;
    const double sigma = acc.sigma() * scale;
    const double ratio = sigma > 0.0 ? std::abs(mean) / sigma : 0.0;
    std::cout << "  " << components[index].first << "  均值 " << std::fixed
              << std::setw(10) << std::setprecision(4) << mean << "   σ "
              << std::setw(10) << std::setprecision(4) << sigma
              << "   |均值|/σ " << std::setw(6) << std::setprecision(2) << ratio;
    if (ratio > 0.5) {
      std::cout << "   <-- 偏置显著";
    }
    std::cout << '\n';
  }

  // ---- 2. 观测者原点对比 ----
  std::cout << "\n--- 2. 观测者原点对比（选让距离偏置最接近 0 的那个）---\n";
  for (std::size_t index = 0; index < kOriginNames.size(); ++index) {
    Accumulator acc;
    for (const Sample& sample : samples) {
      acc.add(sample.e_distance_by_origin[index]);
    }
    std::cout << "  " << std::left << std::setw(8) << kOriginNames[index]
              << std::right << " 距离偏置 " << std::fixed << std::setw(10)
              << std::setprecision(5) << acc.mean() << " m   σ " << std::setw(10)
              << std::setprecision(5) << acc.sigma() << " m\n";
  }

  // ---- 2b. 把残差转回枪管系，解常量外参偏移 ----
  //
  // 观测和真值都在世界系，但外参错的话，误差是一个**固定在枪管系**的向量，
  // 随云台一起转，在世界系里看就成了方向乱变的"噪声"。转回枪管系一平均，
  // 常量的部分就现形了：如果散布远小于均值，那它根本不是噪声，是标定量。
  Eigen::Vector3d offset_sum = Eigen::Vector3d::Zero();
  for (const Sample& sample : samples) {
    offset_sum +=
      sample.q_world_barrel.conjugate() * (sample.observed - sample.truth_relative);
  }
  const Eigen::Vector3d offset = offset_sum / static_cast<double>(samples.size());
  Eigen::Vector3d scatter_sum = Eigen::Vector3d::Zero();
  for (const Sample& sample : samples) {
    const Eigen::Vector3d value =
      sample.q_world_barrel.conjugate() * (sample.observed - sample.truth_relative);
    scatter_sum += (value - offset).cwiseProduct(value - offset);
  }
  const Eigen::Vector3d scatter =
    (scatter_sum / static_cast<double>(samples.size())).cwiseSqrt();

  std::cout << "\n--- 2b. 残差在枪管系里的常量分量（外参偏移的最小二乘解）---\n";
  std::cout << std::fixed << std::setprecision(4);
  const std::array<const char*, 3> axes{"x (指向枪口)", "y (朝左)   ", "z (朝上)   "};
  for (int index = 0; index < 3; ++index) {
    const double ratio =
      scatter[index] > 0.0 ? std::abs(offset[index]) / scatter[index] : 0.0;
    std::cout << "  " << axes[static_cast<std::size_t>(index)] << "  均值 "
              << std::setw(9) << offset[index] << " m   散布 " << std::setw(9)
              << scatter[index] << " m   均值/散布 " << std::setw(6)
              << std::setprecision(2) << ratio << std::setprecision(4) << '\n';
  }
  std::cout << "  合计偏移 " << offset.norm() << " m\n";
  if (offset.norm() > 3.0 * scatter.norm() / std::sqrt(3.0)) {
    std::cout << "  => 这是一个固定在枪管系的常量偏移，属于**外参**，不是观测噪声。\n"
                 "     把它从 T_barrel_camera 的平移里减掉，再重新采一次。\n";
  } else {
    std::cout << "  => 常量分量不显著，残差主要是真的随机噪声。\n";
  }

  // ---- 3. 分箱 ----
  constexpr std::array<double, 6> kDistanceEdges{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<BinStats> distance_bins(kDistanceEdges.size() + 1);
  std::vector<std::string> distance_labels;
  distance_labels.push_back("< 1.0 m");
  for (std::size_t index = 0; index + 1 < kDistanceEdges.size(); ++index) {
    std::ostringstream label;
    label << std::fixed << std::setprecision(1) << kDistanceEdges[index] << " - "
          << kDistanceEdges[index + 1] << " m";
    distance_labels.push_back(label.str());
  }
  distance_labels.push_back("5.0 - 6.0 m");
  distance_labels.push_back(">= 6.0 m");

  constexpr std::array<double, 4> kAngleEdges{15.0, 30.0, 45.0, 60.0};
  std::vector<BinStats> angle_bins(kAngleEdges.size() + 1);
  const std::vector<std::string> angle_labels{
    "0 - 15°", "15 - 30°", "30 - 45°", "45 - 60°", ">= 60°"};

  for (const Sample& sample : samples) {
    std::size_t distance_bin = kDistanceEdges.size();
    for (std::size_t index = 0; index < kDistanceEdges.size(); ++index) {
      if (sample.truth_distance < kDistanceEdges[index]) {
        distance_bin = index;
        break;
      }
    }
    distance_bins[distance_bin].add(sample, flip_threshold);

    const double view_angle = degrees(sample.truth_view_angle);
    std::size_t angle_bin = kAngleEdges.size();
    for (std::size_t index = 0; index < kAngleEdges.size(); ++index) {
      if (view_angle < kAngleEdges[index]) {
        angle_bin = index;
        break;
      }
    }
    angle_bins[angle_bin].add(sample, flip_threshold);
  }

  printBinTable(
    "--- 3. 按真值距离分箱 ---", "距离", distance_labels, distance_bins);
  printBinTable(
    "--- 4. 按真值斜视角分箱（0° = 正视板面）---", "斜视角", angle_labels,
    angle_bins);

  // ---- 5. 与当前 config 对照 ----
  std::cout << "\n--- 5. 与 config 里的 R 基底对照 ---\n";
  std::cout << "config 的 distance/armor_yaw 还会在 update_ypda 里叠加 log1p 随动项，"
               "这里只比基底。\n";
  std::cout << std::left << std::setw(18) << "分量" << std::right
            << std::setw(16) << "config 方差" << std::setw(16) << "实测方差"
            << std::setw(12) << "比值" << '\n';
  const std::array<std::tuple<const char*, double, double>, 4> comparison{{
    {"方位角", target_config.angle_variance, all.azimuth.variance()},
    {"俯仰角", target_config.angle_variance, all.elevation.variance()},
    {"距离", target_config.distance_variance, all.distance.variance()},
    {"板 yaw", target_config.armor_yaw_variance, all.yaw.variance()},
  }};
  for (const auto& [name, configured, measured] : comparison) {
    std::cout << std::left << std::setw(18) << name << std::right
              << std::scientific << std::setw(16) << std::setprecision(3)
              << configured << std::setw(16) << std::setprecision(3) << measured
              << std::fixed << std::setw(12) << std::setprecision(1)
              << (measured > 0.0 ? configured / measured : 0.0) << '\n';
  }
  std::cout << "\n比值远大于 1 说明 config 给得过于保守（EKF 会不信观测），"
               "远小于 1 说明过于乐观（EKF 会追噪声）。\n"
               "但别直接把实测值填进 config：仿真没有畸变/运动模糊/灯条过曝，"
               "也没有图像-IMU 时间偏移，真车的 R 一定更大。\n";

  printYawHistogram(samples);
}

// ---------------------------------------------------------------- 主流程

[[nodiscard]] std::unique_ptr<L1Sensor::DaedalusSource> connectSource(
  const CalibConfig& config)
{
  const auto deadline = std::chrono::steady_clock::now() + config.connect_timeout;
  std::string last_error;
  while (g_stop_requested == 0 && std::chrono::steady_clock::now() < deadline) {
    std::string error;
    auto source = L1Sensor::DaedalusSource::connect(config.source, &error);
    if (source && source->producerAlive()) {
      return source;
    }
    last_error = source ? "Talos heartbeat is stale" : std::move(error);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error(
    "cannot connect to a live Daedalus simulator: " + last_error);
}

int run(
  const std::string& config_path,
  const std::string& csv_path,
  std::uint64_t max_frames,
  double match_radius,
  double flip_threshold,
  Origin origin)
{
  const CalibConfig config = loadConfig(config_path);
  L2Perception::ArmorDetector detector = makeDetector(config);
  auto source = connectSource(config);

  std::string truth_error;
  auto truth_source =
    L1Sensor::DaedalusGroundTruthSource::connect(config.truth, &truth_error);
  if (!truth_source) {
    throw std::runtime_error(
      "cannot open the Daedalus plate-truth channel (" + config.truth.path +
      "): " + truth_error +
      "\n提示：这个通道需要带单板真值发布的模拟器版本。");
  }
  L6Telemetry::logInfo("connected to Daedalus image and plate-truth channels");

  // 第一帧同时提供标定；没有标定就没有 PnP，直接停。
  std::optional<L1Sensor::DaedalusFrame> frame;
  std::optional<L1Sensor::CameraCalibration> calibration;
  const auto deadline = std::chrono::steady_clock::now() + config.connect_timeout;
  std::string calibration_error;
  while (g_stop_requested == 0 && std::chrono::steady_clock::now() < deadline) {
    frame = source->read();
    if (frame) {
      calibration = source->calibration(*frame, &calibration_error);
      if (calibration) {
        break;
      }
    }
    std::this_thread::sleep_for(config.poll_sleep);
  }
  if (!calibration) {
    throw std::runtime_error(
      "did not receive a calibrated Daedalus frame: " + calibration_error);
  }

  L3Estimation::PnpSolver solver(*calibration, config.auto_aim.armor);
  if (!solver.ready()) {
    throw std::runtime_error("PnpSolver rejected the Daedalus calibration");
  }
  const L2Perception::ArmorColor expected_color = armorColor(config.enemy_color);

  if (const auto first_truth = truth_source->latest()) {
    reportGeometry(*first_truth, *frame, config.auto_aim.armor);
  } else {
    std::cout << "\n[警告] 真值通道还没有可用快照，跳过启动自检。\n";
  }

  std::ofstream csv;
  if (!csv_path.empty()) {
    const std::filesystem::path path{csv_path};
    if (path.has_parent_path()) {
      std::error_code ignored;
      std::filesystem::create_directories(path.parent_path(), ignored);
    }
    csv.open(csv_path, std::ios::out | std::ios::trunc);
    if (!csv) {
      throw std::runtime_error("cannot open " + csv_path + " for writing");
    }
    // 后六列加上原始向量和云台姿态：常量偏置只有把残差转回枪管系才解得出来，
    // 光有标量残差只能看出"有偏置"，看不出偏置是哪个方向的哪个量。
    csv << "frame_seq,armor_label,plate_index,target_index,truth_distance,"
           "truth_view_angle_deg,truth_radius,e_azimuth,e_elevation,e_distance,"
           "e_yaw,reprojection_error,match_distance,e_distance_gimbal,"
           "e_distance_muzzle,e_distance_camera,"
           "obs_x,obs_y,obs_z,truth_x,truth_y,truth_z,qw,qx,qy,qz\n";
  }

  std::vector<Sample> samples;
  std::size_t frames_seen = 0;
  std::size_t frames_without_truth = 0;
  std::size_t detections_total = 0;
  std::size_t detections_unmatched = 0;
  std::size_t detections_misclassified = 0;

  std::optional<L1Sensor::DaedalusFrame> queued = std::move(frame);
  while (g_stop_requested == 0 &&
         (max_frames == 0 || frames_seen < max_frames)) {
    std::optional<L1Sensor::DaedalusFrame> current;
    if (queued) {
      current = std::move(queued);
      queued.reset();
    } else {
      current = source->read();
    }
    if (!current) {
      if (!source->producerAlive()) {
        L6Telemetry::logWarn("Daedalus heartbeat timed out; stopping");
        break;
      }
      std::this_thread::sleep_for(config.poll_sleep);
      continue;
    }

    ++frames_seen;
    const auto truth = truth_source->forFrame(current->frame_seq);
    if (!truth) {
      ++frames_without_truth;
      continue;
    }

    auto detections = detector.detect(current->bgr_image);
    std::erase_if(detections, [expected_color](const auto& detection) {
      return detection.color != expected_color;
    });
    if (detections.empty()) {
      continue;
    }

    // 与 Tracker::track 内部完全同一条路径：先设本帧云台姿态，再逐板 PnP。
    solver.set_R_world_barrel(current->gimbal.orientation);

    const std::array<Eigen::Vector3d, 3> origins{
      observerOrigin(*current, Origin::Gimbal),
      observerOrigin(*current, Origin::Muzzle),
      observerOrigin(*current, Origin::Camera)};
    const Eigen::Vector3d& active_origin =
      origins[static_cast<std::size_t>(origin)];

    for (const auto& detection : detections) {
      L3Estimation::Armor observation =
        L3Estimation::toArmorObservation(detection, current->timestamp);
      solver.single_pnp(observation);
      // name 保持 Unknown 就是"这一帧没解出位姿"，Tracker 也是这么筛的。
      if (observation.name == L3Estimation::ArmorName::Unknown) {
        continue;
      }
      ++detections_total;

      // 最近邻匹配。真值板之间相距约 2r（>0.3 m），匹配半径取得比它小就不会
      // 把相邻板配错；配不上的宁可丢掉也不硬凑。
      const L1Sensor::DaedalusPlateTruth* best = nullptr;
      double best_distance = match_radius;
      Eigen::Vector3d best_relative = Eigen::Vector3d::Zero();
      for (const auto& plate : truth->plates) {
        if (!teamMatchesEnemy(plate.team, config.enemy_color)) {
          continue;
        }
        // 类别对不上就不是同一辆车。少了这一关，观测误差一大就可能配到**另一台
        // 车**的板上去，残差看起来像是个巨大的随机误差。
        if (!truthLabelMatchesClass(plate.armor_label, observation.name)) {
          continue;
        }
        const Eigen::Vector3d relative = plate.center - active_origin;
        // 背对相机的板不可能被检测到，所以也不可能是这次观测的对应板。
        // 少了这道门，观测误差一大就会配到**相邻**板上去：四板车相邻板相距
        // 2r·sin45° ≈ 0.28 m，只比匹配半径的两倍多一点，配错的残差看起来像
        // 一个 90° 的 yaw 跳解，会把统计彻底带偏。
        const double view_angle = std::abs(L6Telemetry::limit_rad(
          L1Sensor::toNewvisionArmorYaw(plate.yaw) -
          std::atan2(relative.y(), relative.x())));
        if (view_angle > kMaxVisibleViewAngle) {
          continue;
        }
        const double distance = (relative - observation.xyz_in_world).norm();
        if (distance < best_distance) {
          best_distance = distance;
          best = &plate;
          best_relative = relative;
        }
      }
      if (best == nullptr) {
        ++detections_unmatched;
        // 该帧真值里根本没有同类别的板 => 这次是误分类，而不是几何配不上。
        // 按类别筛会把这些样本丢掉，统计因此偏向"分类正确"的板，比例要报出来。
        const bool any_same_class = std::any_of(
          truth->plates.begin(), truth->plates.end(),
          [&](const auto& plate) {
            return teamMatchesEnemy(plate.team, config.enemy_color) &&
              truthLabelMatchesClass(plate.armor_label, observation.name);
          });
        if (!any_same_class) {
          ++detections_misclassified;
        }
        continue;
      }

      const Eigen::Vector3d truth_ypd = L6Telemetry::xyz2ypd(best_relative);
      // 真值发的是朝外法线，newvision 的板 yaw 是朝内法线，差 π。
      const double truth_yaw = L1Sensor::toNewvisionArmorYaw(best->yaw);

      Sample sample;
      sample.frame_seq = current->frame_seq;
      sample.armor_label = best->armor_label;
      sample.plate_index = best->plate_index;
      sample.target_index = best->target_index;
      sample.truth_distance = truth_ypd.z();
      // 与 update_ypda 里的 delta_angle 同一个定义，只是全部取真值。
      sample.truth_view_angle = std::abs(L6Telemetry::limit_rad(
        truth_yaw - std::atan2(best_relative.y(), best_relative.x())));
      sample.truth_radius = best->radius;
      sample.e_azimuth =
        L6Telemetry::limit_rad(observation.ypd_in_world.x() - truth_ypd.x());
      sample.e_elevation =
        L6Telemetry::limit_rad(observation.ypd_in_world.y() - truth_ypd.y());
      sample.e_distance = observation.ypd_in_world.z() - truth_ypd.z();
      sample.e_yaw =
        L6Telemetry::limit_rad(observation.ypr_in_world[0] - truth_yaw);
      sample.reprojection_error = observation.reprojection_error;
      sample.match_distance = best_distance;
      for (std::size_t index = 0; index < origins.size(); ++index) {
        const Eigen::Vector3d relative = best->center - origins[index];
        sample.e_distance_by_origin[index] =
          observation.ypd_in_world.z() - relative.norm();
      }
      sample.observed = observation.xyz_in_world;
      sample.truth_relative = best_relative;
      sample.q_world_barrel = current->gimbal.orientation;

      if (csv) {
        csv << sample.frame_seq << ',' << sample.armor_label << ','
            << sample.plate_index << ',' << sample.target_index << ','
            << sample.truth_distance << ',' << degrees(sample.truth_view_angle)
            << ',' << sample.truth_radius << ',' << sample.e_azimuth << ','
            << sample.e_elevation << ',' << sample.e_distance << ','
            << sample.e_yaw << ',' << sample.reprojection_error << ','
            << sample.match_distance << ','
            << sample.e_distance_by_origin[0] << ','
            << sample.e_distance_by_origin[1] << ','
            << sample.e_distance_by_origin[2] << ','
            << sample.observed.x() << ',' << sample.observed.y() << ','
            << sample.observed.z() << ',' << sample.truth_relative.x() << ','
            << sample.truth_relative.y() << ',' << sample.truth_relative.z()
            << ',' << sample.q_world_barrel.w() << ','
            << sample.q_world_barrel.x() << ',' << sample.q_world_barrel.y()
            << ',' << sample.q_world_barrel.z() << '\n';
      }
      samples.push_back(sample);
    }

    if (frames_seen % 200 == 0) {
      std::cout << "\r已处理 " << frames_seen << " 帧，样本 " << samples.size()
                << " 个 " << std::flush;
      if (csv) {
        csv.flush();
      }
    }
  }

  std::cout << '\n';
  if (csv) {
    csv.flush();
  }
  printSummary(
    samples, config.auto_aim.filter, origin, frames_seen, frames_without_truth,
    detections_total, detections_unmatched, detections_misclassified,
    flip_threshold);
  if (!csv_path.empty()) {
    std::cout << "\n逐样本残差已写入 " << csv_path << '\n';
  }
  return samples.empty() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv)
{
  // cv::CommandLineParser 只认 --key=value；写成 -k value 会把 value 当成位置参数。
  const cv::String keys =
    "{help h        |                          | 打印本帮助}"
    "{config        |config/daedalus.yaml      | 仿真运行配置}"
    "{out           |logs/pnp_residuals.csv    | 逐样本残差 CSV，空字符串则不写}"
    "{frames        |0                         | 处理多少帧后停止，0 = 直到 Ctrl-C}"
    "{match-radius  |0.10                      | 观测与真值板的最近邻匹配半径，米}"
    "{flip-deg      |45.0                      | 板 yaw 残差超过此值判为 PnP 跳解，度}"
    "{origin        |muzzle                    | 观测者原点：gimbal/muzzle/camera}";

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Daedalus PnP 观测噪声 R 标定工具");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);
  L6Telemetry::initLogger();

  int exit_code = 0;
  try {
    if (!parser.check()) {
      parser.printErrors();
      throw std::runtime_error("invalid command line");
    }
    const double match_radius = parser.get<double>("match-radius");
    const double flip_deg = parser.get<double>("flip-deg");
    if (!(match_radius > 0.0) || !(flip_deg > 0.0)) {
      throw std::runtime_error("--match-radius and --flip-deg must be positive");
    }
    exit_code = run(
      parser.get<std::string>("config"), parser.get<std::string>("out"),
      parser.get<std::uint64_t>("frames"), match_radius,
      flip_deg * std::numbers::pi / 180.0,
      parseOrigin(parser.get<std::string>("origin")));
  } catch (const std::exception& exception) {
    L6Telemetry::logError("daedalus_noise_calib stopped", exception.what());
    std::cerr << "daedalus_noise_calib stopped: " << exception.what() << '\n';
    exit_code = 1;
  }
  L6Telemetry::flushLogger();
  return exit_code;
}
