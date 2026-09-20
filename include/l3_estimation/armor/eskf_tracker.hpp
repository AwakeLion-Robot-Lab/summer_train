#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/tracking/association.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/types.hpp"

#include <Eigen/Geometry>

#include <array>
#include <optional>
#include <string>
#include <vector>

// 误差状态整车跟踪器：管一个（双缓冲下是两个）整车目标的生命周期，把每帧的
// 装甲板和灯条关联到目标上去做 IESKF 更新。
namespace L3Estimation {

// L2 -> L3 的显式转换：只搬类别、颜色、四角点、置信度这些检测字段，位姿留空
// 由调用方用当帧的 PnpSolver 填。
Armor toObservation(
  const L2Perception::Armor& detection, TimePoint timestamp);

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
};

class EskfTracker
{
public:
  EskfTracker(
    const L1Sensor::CameraCalibration & calibration, ArmorConfig armor_config = {},
    EskfTrackerConfig tracker_config = {}, EskfTargetConfig target_config = {});

  bool ready() const noexcept { return ready_; }
  TrackState state() const noexcept { return buffer_[current_].lifecycle.state; }

  // 处理一帧：把检测转成观测并补 PnP → Lost 时挑候选初始化，否则关联并更新
  // → 推进状态机 → 返回当前目标的快照。q_world_barrel 必须对应 timestamp
  // 这一时刻的枪管姿态。Lost 或初始化失败返回空，TempLost 返回纯预测状态。
  //
  // 第二个重载多收一组独立灯条，它们会作为额外的端点观测参与同一次更新。
  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor> & detections,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp);

  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::vector<L2Perception::Light>& lights,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp);

  // 独立灯条的搜索 ROI：把整车全部灯条的预测包围框放大 1.6 倍。只在
  // Tracking/TempLost、开了独立灯条观测、且目标不是基地时返回。
  std::optional<cv::Rect> lightRoi(
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp, const cv::Size& image_size) const;

  // 送给网络的检测 ROI。同样从预测包围框出发，但比 lightRoi 多三步：按
  // target_wh_ratio（网络输入宽高比）修正形状以减少 letterbox 填充、扩成方形、
  // 再随距上次更新的时长线性膨胀，超时退化为整图。
  //
  // 目标不可聚焦时返回整图而不是空，调用方拿到的永远是一个能直接用的矩形。
  cv::Rect netFocusRoi(
    const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
    const cv::Size& image_size, double target_wh_ratio = 1.0) const;

  // 本帧关联到的完整装甲板数量，以及关联到的物理板编号（形如 "0|2"）。
  // 关联在编号间来回跳会让整车 yaw 每帧偏 2π/N，是抖动最常见的来源，所以这
  // 两项要能逐帧导出比对。
  int lastMatchCount() const noexcept { return last_match_count_; }
  const std::string& lastMatchedIds() const noexcept { return last_matched_ids_; }

  // 最近一帧真正进了 updateMulti 的灯条。初始化帧、无关联帧和纯预测帧都是
  // 空的，免得把“检测候选”误画成“已用于更新”。
  const std::vector<UsedLight>& usedLights() const noexcept
  {
    return buffer_[current_].used_lights;
  }

  // matchLight 各道门的累计拒绝数，从进程开始一直累加，不随目标复位清零。
  // 侧边灯条采纳数为零时，靠它区分“候选槽位没开出来”和“某道门太紧”。
  const LightMatchStats& lightMatchStats() const noexcept { return light_match_stats_; }

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

  // prefer 给定时优先取同类别的候选，没有才退回最靠近图像中心的那块。
  bool initTarget(
    Slot& slot, const std::vector<Armor>& candidates, TimePoint timestamp,
    const Eigen::Isometry3d & camera_in_world,
    std::optional<ArmorName> prefer = std::nullopt);
  bool updateTarget(
    Slot& slot, const std::vector<Armor>& candidates,
    const std::vector<L2Perception::Light>& lights, TimePoint timestamp,
    const Eigen::Isometry3d & camera_in_world);

  // 把当前目标预测到 timestamp，投影出所有装甲板的灯条端点，取包围盒并与
  // 图像相交。两个 ROI 都从这里出发。目标未初始化、不在跟踪态或已超时时返回
  // 空；require_light_measurements 再额外要求开了独立灯条观测且目标不是基地。
  std::optional<cv::Rect> lightBounds(
    const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
    const cv::Size& image_size, bool require_light_measurements) const;

  // 观测可用性的两道门：前者要求类别、颜色这些语义字段齐全，后者再要求
  // 这一帧的 PnP 真的解出了位姿。
  bool semanticUsable(const Armor& armor) const noexcept;
  bool pnpUsable(const Armor& armor) const noexcept;
  // 挑出能用来初始化新目标的观测：过语义门 → 逐个做 single_pnp → 过 PnP 门，
  // 最后按离图像中心的距离排序，近的在前。
  std::vector<Armor> initCandidates();
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

  std::vector<Armor> observations_;
  int last_match_count_{0};
  std::string last_matched_ids_;
  LightMatchStats light_match_stats_{};
};

}  // namespace L3Estimation
