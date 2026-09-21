#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/armor_matcher.hpp"
#include "l3_estimation/armor/armor_observation.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/track_roi.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/tracking/association.hpp"

#include <Eigen/Geometry>

#include <array>
#include <optional>
#include <string>
#include <vector>

// 误差状态整车跟踪器：管一个（双缓冲下是两个）整车目标的生命周期，把每帧的
// 装甲板和灯条关联到目标上去做 IESKF 更新。
//
// 这一层只管"哪个目标、活着还是死了"。关联怎么算在 armor_matcher，滤波怎么走
// 在 EskfTarget，ROI 怎么画在 track_roi。
namespace L3Estimation {

// L2 -> L3 的显式转换：只搬类别、颜色、四角点、置信度这些检测字段，位姿留空
// 由调用方用当帧的 PnpSolver 填。
Armor toObservation(const L2Perception::Armor & detection, TimePoint timestamp);

struct EskfTrackerConfig
{
  // Detecting 要连续关联上多少帧才转 Tracking。
  int tracking_thres{5};
  // TempLost 最多靠预测撑多少秒，超时退回 Lost。按真实时间算而不是数丢帧，
  // 回放倍速不影响判定。前哨转速固定、轨迹规整，可以撑更久。
  double lost_time_thres{1.5};
  double lost_time_thres_outpost{2.0};
  // 相邻两次 track() 的时间差超过它（秒）就当作断过流：相机掉线重连、回放
  // 拼接都会这样。两个槽全部复位，用当前帧重建。按旧速度外推这么久没有意义，
  // 外推出来的目标还会照常交给 L4。
  double max_frame_gap{0.1};
  // 距上次成功更新超过它（秒）后，预测不再按速度外推，目标停在这一刻的位置
  // 和姿态上，协方差照常增长。速度或角速度一旦被错关联带偏，TempLost 期间
  // 匀速外推能把目标推出画面，按预测位置做的关联也就再也接不上真实的板。
  double temp_lost_predict_time{0.1};
};

// 本帧真正作为端点观测送进 IESKF 多观测更新的一根灯条。完整装甲板会拆成
// 左右两根，isolated 为 true 表示这根来自独立灯条检测、靠整车几何关联上，
// 而不是装甲板的角点。只用于调试显示，不参与滤波计算。
struct UsedLight
{
  cv::Point2f top{};
  cv::Point2f bottom{};
  int armor_id{-1};
  bool is_left{false};
  bool isolated{false};
  // isolated 为 true 时，是这根灯条在本帧 L2 侧边灯条候选数组（lastLights）
  // 里的下标，调试显示靠它把"滤波器真正吃下去的"回查到检出候选上；来自装甲
  // 板角点的那两根没有下标，恒为 0。
  std::size_t light_id{0};
};

class EskfTracker
{
public:
  EskfTracker(
    const L1Sensor::CameraCalibration & calibration, ArmorConfig armor_config = {},
    EskfTrackerConfig tracker_config = {}, EskfTargetConfig target_config = {});

  bool ready() const noexcept { return ready_; }
  TrackState state() const noexcept { return buffer_[current_].lifecycle.state; }

  // 处理一帧：把检测转成观测 → Lost 时挑候选初始化，否则关联并更新 → 推进
  // 状态机 → 返回当前目标的快照。q_world_barrel 必须对应 timestamp 这一时刻
  // 的枪管姿态。Lost 或初始化失败返回空，TempLost 返回纯预测状态。
  //
  // 第二个重载多收一组独立灯条，它们会作为额外的端点观测参与同一次更新。
  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor> & detections,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp);

  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor> & detections,
    const std::vector<L2Perception::Light> & lights,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp);

  // 独立灯条的搜索 ROI。只在 Tracking/TempLost、开了独立灯条观测、且目标不是
  // 基地时返回；形状见 Roi::light。
  std::optional<cv::Rect> lightRoi(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    const cv::Size & image_size) const;

  // 送给网络的检测 ROI，形状见 Roi::net。目标不可聚焦时返回整图而不是空，
  // 调用方拿到的永远是一个能直接用的矩形。
  cv::Rect netFocusRoi(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    const cv::Size & image_size, double target_wh_ratio = 1.0) const;

  // 本帧关联到的完整装甲板数量，以及关联到的物理板编号（形如 "0|2"）。
  // 关联在编号间来回跳会让整车 yaw 每帧偏 2π/N，是抖动最常见的来源，所以这
  // 两项要能逐帧导出比对。
  int lastMatchCount() const noexcept { return last_match_count_; }
  const std::string & lastMatchedIds() const noexcept { return last_matched_ids_; }

  // 最近一帧真正进了 update 的灯条。初始化帧、无关联帧和纯预测帧都是空的，
  // 免得把"检测候选"误画成"已用于更新"。
  const std::vector<UsedLight> & usedLights() const noexcept
  {
    return buffer_[current_].used_lights;
  }

  // matchLight 各道门的累计拒绝数，从进程开始一直累加，不随目标复位清零。
  // 侧边灯条采纳数为零时，靠它区分"候选槽位没开出来"和"某道门太紧"。
  const LightMatchStats & lightMatchStats() const noexcept { return light_match_stats_; }

  // 当前目标被丢弃的累计次数（超时、发散、Detecting 丢帧、断流）。丢弃后同一帧
  // 就可能重建，只看 state() 数不出这些，诊断工具按这个计数。
  std::size_t dropCount() const noexcept { return drop_count_; }

  // 本帧全部观测，含被质量门限拒掉的，供 L6 调试显示。
  const std::vector<Armor> & observations() const noexcept { return observations_; }
  std::vector<Eigen::Vector4d> armorPoses() const;

  void reset() noexcept;

private:
  // 双缓冲的一个槽位：一个目标，加它自己的状态机计数和调试快照。
  struct Slot
  {
    EskfTarget target;
    TrackLifecycle lifecycle;
    TimePoint last_update{};
    std::vector<UsedLight> used_lights;
  };

  // 一帧里与槽位无关的输入，拼一次传给两个槽。
  struct Frame
  {
    TimePoint timestamp{};
    Eigen::Isometry3d camera_in_world{Eigen::Isometry3d::Identity()};
    const std::vector<L2Perception::Light> * lights{nullptr};
  };

  // --- track() 的各个步骤，按调用顺序排 ----------------------------------

  // 清掉上一帧留下的逐帧量。早退路径也要先走这一步，免得下游读到隔帧的结果。
  void clearFrame() noexcept;
  // 断流（间隔过长或时间倒退）后旧目标的外推不可信，两个槽都清掉，这一帧按
  // Lost 重新挑候选初始化。
  void dropOnGap(TimePoint timestamp);
  // L2 -> L3：正常更新只搬类别和四角点，不拿 PnP 成功当入口门限。PnP 只在
  // Lost 初始化和单块完整板求深度差时才跑。
  void readDetections(
    const std::vector<L2Perception::Armor> & detections, TimePoint timestamp,
    const std::optional<Eigen::Quaterniond> & q_world_barrel);
  // 推进一个槽一帧：初始化或更新 → 状态机 → 发散检查。返回本帧是否关联上。
  bool advance(Slot & slot, const Frame & frame, std::optional<ArmorName> prefer);
  // 双缓冲：当前目标进 TempLost 时让另一个槽同时抓新目标，新目标一转成
  // Tracking 就交换上来，不必等当前目标超时。
  void runBackup(const Frame & frame);

  // prefer 给定时优先取同类别的候选，没有才退回最靠近图像中心的那块。
  bool initTarget(
    Slot & slot, const std::vector<Armor> & candidates, const Frame & frame,
    std::optional<ArmorName> prefer);
  bool updateTarget(Slot & slot, const Frame & frame);

  // 本帧的初始化候选，按需算一次、缓存到帧末。正常 Tracking 一次 PnP 都不跑；
  // 当前槽恰好在这一帧转进 TempLost 时，紧接着处理备用槽还能当帧完成初始化，
  // 不用白等一帧。
  const std::vector<Armor> & initCandidates();

  // 观测可用性的两道门：前者要求类别、颜色这些语义字段齐全，后者再要求
  // 这一帧的 PnP 真的解出了位姿。
  bool semanticUsable(const Armor & armor) const noexcept;
  bool pnpUsable(const Armor & armor) const noexcept;

  // ROI 的状态机门限。不在跟踪态、未初始化或已超时时返回空；
  // require_light_measurements 再额外要求开了独立灯条观测且目标不是基地。
  std::optional<Roi::Focus> roiFocus(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    bool require_light_measurements) const;

  double lostThreshold(const EskfTarget & target) const noexcept;
  // 这个槽的运动外推截止时刻：上次成功更新后再过 temp_lost_predict_time。
  TimePoint holdFrom(const Slot & slot) const noexcept;

  L1Sensor::CameraCalibration calibration_;
  ArmorConfig armor_config_;
  EskfTrackerConfig tracker_config_;
  EskfTargetConfig target_config_;
  PnpSolver pnp_solver_;
  cv::Point2f image_center_{};
  bool ready_{false};

  // 双缓冲：当前目标进 TempLost 时，另一个槽位同时尝试初始化新目标，新目标
  // 先转成 Tracking 就交换上来。这样切目标或目标被短暂完全遮挡后重新出现都
  // 能马上接上，不必等当前目标超时。备用槽优先抓与当前目标同类别的板；当前
  // 目标的外推已经停住（见 temp_lost_predict_time）时，同类别的新目标不必等
  // 转 Tracking，带着自己的 Detecting 计数直接换上来。
  std::array<Slot, 2> buffer_{};
  std::size_t current_{0};
  std::size_t previous_{1};

  // 上一次 track() 的帧时间，用来判断断流。
  std::optional<TimePoint> last_frame_;
  std::size_t drop_count_{0};

  // 以下都是逐帧量，每帧由 clearFrame() 清空。
  std::vector<Armor> observations_;
  std::optional<std::vector<Armor>> init_candidates_;
  int last_match_count_{0};
  std::string last_matched_ids_;
  // 这一项例外：跨帧累加，不随目标复位清零。
  LightMatchStats light_match_stats_{};
};

}  // namespace L3Estimation
