#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/association.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/types.hpp"

#include <Eigen/Geometry>

#include <array>
#include <optional>
#include <string>
#include <vector>

// 误差状态整车跟踪器。照搬 awakening 的 ArmorTracker，是 Tracker 的 ESEKF
// 替代品，公共接口刻意与之对齐，便于在同一段回放上做 A/B。
namespace L3Estimation {

// L2 -> L3 的显式转换入口。这里只复制检测字段，三维位姿由调用方用当帧的
// PnpSolver 补充——L3 需要的是"哪些像素角点属于哪块板"，位姿是后一步的事。
Armor toArmorObservation(
  const L2Perception::Armor& detection, TimePoint timestamp);

struct EskfTrackerConfig
{
  // Detecting 连续多少帧有效关联才转 Tracking。
  int tracking_thres{5};
  // TempLost 允许的无观测预测时长，超时放弃。前哨站转速固定、轨迹规整，
  // 可以撑更久。
  // awakening 的主相机配置为 1.5 s；按真实时间而不是丢帧计数，回放倍速不影响。
  double lost_time_thres{1.5};
  double lost_time_thres_outpost{2.0};
};

// 本帧真正送进 IESKF 多观测更新的一根灯条。完整装甲板会拆成左右两根；
// isolated=true 表示它来自独立灯条检测并通过整车几何关联，而不是装甲板角点。
// 这份结构只承载调试显示所需的信息，不参与滤波计算。
struct UvlUpdateLight
{
  cv::Point2f top{};
  cv::Point2f bottom{};
  int armor_id{-1};
  bool is_left{false};
  bool isolated{false};
};

class EskfTracker
{
public:
  EskfTracker(
    const L1Sensor::CameraCalibration & calibration, ArmorConfig armor_config = {},
    EskfTrackerConfig tracker_config = {}, EskfTargetConfig target_config = {});

  bool ready() const noexcept { return ready_; }
  TrackState state() const noexcept { return buffer_[current_].lifecycle.state; }

  // 处理一帧检测。枪管姿态必须对应 timestamp 所表示的图像曝光时刻。
  // Lost 或初始化失败时返回空；TempLost 时返回纯预测状态。
  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor> & detections,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp);

  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::vector<L2Perception::Light>& lights,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp);

  // 用上一帧目标预测整车全部灯条的包围框，再按 Awakening 的 1.6 倍扩张。
  // 仅 Tracking/TempLost 且开启独立灯条观测时返回 ROI。
  std::optional<cv::Rect> lightDetectionRoi(
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp, const cv::Size& image_size) const;

  // 送给**网络**的检测 ROI。照搬 awakening 的 get_net_focus_roi。
  //
  // 与 lightDetectionRoi 的分工：那个只服务传统灯条检测，越紧越好；这个要喂
  // 进固定尺寸输入的网络，所以多三步——按网络输入宽高比修正（减少 letterbox
  // padding）、扩成方形、并随"距上次更新的时长"线性膨胀，超时直接退化为整图。
  //
  // 收益在远距小目标：ROI 裁剪后再 resize 到网络输入，相当于对目标区域局部
  // 放大，保留灯条边缘与数字结构。目标不可聚焦时返回整图而不是空——调用方
  // 拿到的永远是一个可直接使用的矩形。
  cv::Rect netFocusRoi(
    const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
    const cv::Size& image_size, double target_wh_ratio = 1.0) const;

  // 本帧关联到的完整装甲板数量，以及关联到的物理板编号（形如 "0|2"）。
  // 关联在编号间来回跳会让整车 yaw 每帧偏 2π/N，是抖动最常见的来源，所以
  // 这两项要能逐帧导出比对。
  int lastMatchCount() const noexcept { return last_match_count_; }
  const std::string& lastMatchedIds() const noexcept { return last_matched_ids_; }

  // 最近一帧中实际参与 updateMulti() 的全部 UVL 灯条。初始化帧、无关联帧
  // 和纯预测帧均为空，避免把“检测候选”误画成“已用于更新”。
  const std::vector<UvlUpdateLight>& lastUvlUpdateLights() const noexcept
  {
    return buffer_[current_].uvl_update_lights;
  }

  // 本帧全部观测（含被质量门限拒绝的），供 L6 调试。
  const std::vector<Armor> & observations() const noexcept { return observations_; }
  std::vector<Eigen::Vector4d> targetArmorPoses() const;

  void reset() noexcept;

private:
  // 双缓冲的一个槽位：一个目标加它自己的状态机计数。
  struct Slot
  {
    EskfTarget target;
    TrackLifecycle lifecycle;
    TimePoint last_update{};
    std::vector<UvlUpdateLight> uvl_update_lights;
  };

  bool initializeTarget(
    Slot& slot, const std::vector<Armor>& candidates, TimePoint timestamp,
    const Eigen::Isometry3d & camera_in_world);
  bool updateTarget(
    Slot& slot, const std::vector<Armor>& candidates,
    const std::vector<L2Perception::Light>& lights, TimePoint timestamp,
    const Eigen::Isometry3d & camera_in_world);

  // 当前目标所有装甲板灯条端点的预测包围盒。两个 ROI 共用这一步。
  // 目标不可聚焦（未初始化、非跟踪态、超时）时返回空。
  std::optional<cv::Rect> predictedLightBounds(
    const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
    const cv::Size& image_size, bool require_light_measurements) const;

  bool semanticObservationUsable(const Armor& armor) const noexcept;
  bool pnpObservationUsable(const Armor& armor) const noexcept;
  std::vector<Armor> initializationObservations();
  double lostTimeThreshold(const EskfTarget & target) const noexcept;

  L1Sensor::CameraCalibration calibration_;
  ArmorConfig armor_config_;
  EskfTrackerConfig tracker_config_;
  EskfTargetConfig target_config_;
  PnpSolver pnp_solver_;
  cv::Point2f image_center_{};
  bool ready_{false};

  // 双缓冲。当前目标进 TempLost 时，另一个槽位同时尝试初始化新目标；若新目标
  // 先转成 Tracking 就顶上。这样操作手切目标、或目标被短暂完全遮挡后重新出现
  // 都能快速接上，而不必等当前目标超时。
  std::array<Slot, 2> buffer_{};
  std::size_t current_{0};
  std::size_t previous_{1};

  std::vector<Armor> observations_;
  int last_match_count_{0};
  std::string last_matched_ids_;
};

}  // namespace L3Estimation
