#pragma once

#include "l2_perception/armor.hpp"

#include <opencv2/core/types.hpp>

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <string_view>

namespace L3Estimation {

// L3 中的时间戳统一使用单调时钟，避免系统时间校准造成负时间差。
using TimePoint = std::chrono::steady_clock::time_point;
using ArmorName = L2Perception::ArmorClass;

// 物理装甲板板型只决定 PnP 几何尺寸，车辆类别由 Armor::name 单独表示。
enum class ArmorType : std::uint8_t {
  Small,  // 小装甲板
  Big     // 大装甲板
};

// 识别类别 → 实际板型。场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，
// 基地虽然有 Bs/Bb 两个类别但装甲板实物都是小板，所以只有 Hero 走 Big 分支。
//
// 未知类别返回 nullopt，不猜板型：猜错会同时污染 PnP 几何和火控的角度容差。
// L3 的 PnpSolver 和 L5 的 FireDecider 共用这一份映射。
[[nodiscard]] constexpr std::optional<ArmorType> armorTypeOf(ArmorName name) noexcept
{
  switch (name) {
    case ArmorName::Hero:
      return ArmorType::Big;

    case ArmorName::Guard:
    case ArmorName::Engineer:
    case ArmorName::Infantry3:
    case ArmorName::Infantry4:
    case ArmorName::Infantry5:
    case ArmorName::Outpost:
    case ArmorName::BaseSmall:
    case ArmorName::BaseLarge:
      return ArmorType::Small;

    case ArmorName::Unknown:
      break;
  }
  return std::nullopt;
}

// Tracker 的四态生命周期。
enum class TrackState : std::uint8_t {
  Lost,       // 当前没有可用目标
  Detecting,  // 已发现目标，等待连续帧确认
  Tracking,   // 稳定跟踪
  TempLost    // 短时丢失，继续输出预测状态
};

// L3 对 L2 输出的 Armor 执行单板 PnP 和坐标变换后得到的观测。
struct Armor {
  // 分类信息。name 是车辆类别，type 是实际采用的物理板型。
  ArmorName name{ArmorName::Unknown};
  ArmorType type{ArmorType::Small};
  int class_id{-1};

  // 图像角点顺序固定为左上、右上、右下、左下，单位为 pixel。
  std::array<cv::Point2f, 4> points{};
  // L2 检测得到的四角点几何中心，单位为 pixel。
  cv::Point2f center{};

  // 平移量单位均为 meter。
  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};
  // 固定顺序为 [yaw, pitch, roll]，采用 Rz(yaw)Ry(pitch)Rx(roll)。
  Eigen::Vector3d ypr_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_world{Eigen::Vector3d::Zero()};
  // IPPE 原始姿态。GTSAM 的 Pose3 初值和板号关联都必须用同一份位姿；
  // ypr_in_world[0] 会在共享 PnP 的 yaw 搜索中被覆盖，不能再当作 JLU 原始观测。
  Eigen::Vector3d ypr_raw_in_world{Eigen::Vector3d::Zero()};
  // [方位角, 俯仰角, 距离]，角度单位为 radian，距离单位为 meter。
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};

  // 四个角点的二维像素 RMSE，取自 IPPE 在相机系的原始解，与 yaw 优化无关。
  // 只作诊断输出，不参与观测门限。
  double reprojection_error{std::numeric_limits<double>::infinity()};
  // 检测置信度和四边形像素面积从 L2 原样传入。
  double confidence{0.0};
  double area{0.0};
  // yaw 优化前的单板 PnP 原始朝向角，等于 ypr_raw_in_world[0]。
  double yaw_raw{0.0};

  // 对应原始图像的曝光时刻。
  TimePoint timestamp{};
};

// 装甲板几何与 PnP yaw 搜索参数，由 config/auto_aim.yaml 的 armor 段填充。
struct ArmorConfig {
  // 装甲板几何尺寸，单位为 meter。
  double small_width{0.135};
  double big_width{0.230};
  double height{0.056};

  // 装甲板安装倾角，单位 radian。yaw 搜索假设只有 yaw 是自由量，倾角固定。
  double mount_pitch{15.0 * std::numbers::pi / 180.0};
  double outpost_mount_pitch{-15.0 * std::numbers::pi / 180.0};

  // yaw 搜索分两段：先在以枪管 yaw 为中心、总宽 yaw_search_range 的窗口里按
  // yaw_coarse_step 粗扫定位代价盆地，再在粗扫最优点附近 ±yaw_fine_range 内按
  // yaw_fine_step 细扫。单段扫描要同时满足"窗口够宽"和"分辨率够细"，步数就会
  // 线性膨胀；两段式用 47 + 60 次评估换到 0.1° 的分辨率。单位 radian。
  //
  // yaw_fine_range 必须 >= yaw_coarse_step，否则粗扫格点之间的空隙不会被细扫
  // 覆盖，真正的极小值可能落在两段搜索都够不着的地方。
  double yaw_search_range{140.0 * std::numbers::pi / 180.0};
  double yaw_coarse_step{3.0 * std::numbers::pi / 180.0};
  double yaw_fine_range{3.0 * std::numbers::pi / 180.0};
  double yaw_fine_step{0.1 * std::numbers::pi / 180.0};

  // yaw 搜索代价：每个角点的像素距离过 Huber，再加上相邻边夹角误差过 Huber
  // 后按 shape_weight 加权的形状项。
  //
  // 位置项用 Huber 而不是裸的距离和：一个角点被灯条反光或遮挡带偏时，平方或
  // 线性代价都会让它主导整条曲线，Huber 把它的影响截断在 delta 上。
  // 形状项是给正视姿态用的：正视时角点位置对 yaw 几乎不敏感，代价曲线在极小
  // 值附近是平的，但边的方向仍然敏感，靠它把盆地压出来。
  // 夹角误差以**度**参与计算，所以 huber_deg 也写成度。
  double yaw_cost_huber_px{5.0};
  double yaw_cost_huber_deg{10.0};
  double yaw_cost_shape_weight{0.3};
};

// 两个后端共享的整车物理模型参数，由 config/auto_aim.yaml 的 target 段填充。
struct TargetConfig {
  // 过程噪声：分段白噪声模型的加速度方差和角加速度方差。
  // 前哨站运动模式固定，因此给更小的值。
  double accel_variance{100.0};
  double yaw_accel_variance{400.0};
  double outpost_accel_variance{10.0};
  double outpost_yaw_accel_variance{0.1};

  // 各车型的初始旋转半径，单位 meter。
  double radius{0.2};
  double outpost_radius{0.2765};
  double base_radius{0.3205};
  // 半径的物理范围，越界即判定估计发散。
  double min_radius{0.05};
  double max_radius{0.5};
};

// 整车状态估计的后端实现。两种后端估的是**同一组量**（旋转中心的位置与速度、
// 整车 yaw 与角速度、两组半径和高度差），所以 L3 交给 L4 的 TrackedTarget 与
// 后端无关，差别只在于这组量是怎么被观测约束出来的。
enum class EstimatorBackend : std::uint8_t {
  // 整车扩展卡尔曼滤波，见 l3_estimation/filter_est/。默认后端。
  Filter,
  // GTSAM 因子图 + ISAM2 增量优化，见 l3_estimation/gtsam_est/。
  // 需要 xmake f --use_gtsam=y 才会编进来，否则构造时抛异常。
  Gtsam
};

[[nodiscard]] constexpr std::string_view toString(EstimatorBackend backend) noexcept
{
  switch (backend) {
    case EstimatorBackend::Gtsam:
      return "gtsam";
    case EstimatorBackend::Filter:
      break;
  }
  return "filter";
}

// YAML 里写的后端名。拼错时返回 nullopt，由调用方决定是报错还是留默认值——
// 这里不静默回退，选错后端和选错噪声参数一样会让整条估计链路对不上。
[[nodiscard]] constexpr std::optional<EstimatorBackend> estimatorBackendFromString(
  std::string_view name) noexcept
{
  if (name == "filter" || name == "ekf") {
    return EstimatorBackend::Filter;
  }
  if (name == "gtsam" || name == "fgo") {
    return EstimatorBackend::Gtsam;
  }
  return std::nullopt;
}

struct TrackerConfig {
  // 从 Detecting 转入 Tracking 所需的连续有效观测帧数。
  int min_detect_count{5};
  // 非 Lost 状态允许的最大相邻帧间隔，超过即认为相机掉线，丢弃旧运动状态。
  std::chrono::milliseconds max_frame_interval{100};
  // 临时丢失按连续帧数计数；前哨站允许更长的无观测预测窗口。
  int max_temp_lost_count{15};
  int outpost_max_temp_lost_count{75};
};

}  // namespace L3Estimation
