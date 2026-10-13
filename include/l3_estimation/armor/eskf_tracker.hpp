#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/armor_matcher.hpp"
#include "l3_estimation/armor/armor_observation.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/track_roi.hpp"
#include "l3_estimation/armor/track_slot.hpp"
#include "l3_estimation/armor/types.hpp"

#include <Eigen/Geometry>

#include <optional>
#include <string>
#include <vector>

namespace L3Estimation {

/**
 * @brief L2 -> L3 的显式转换
 *
 * 只搬类别、颜色、四角点、置信度这些检测字段，位姿留空由调用方用当帧的
 * PnpSolver 填。
 *
 * @param detection L2 检出的装甲板
 * @return 未解 PnP 的 L3 观测
 */
Armor toObservation(const L2Perception::Armor & detection);

/**
 * @brief 跟踪器的状态机与 ROI 参数
 */
struct EskfTrackerConfig
{
  /// Detecting 要连续关联上多少帧才转 Tracking。
  int tracking_thres{5};

  /// TempLost 最多靠预测撑多少秒，超时退回 Lost。按真实时间算而不是数丢帧，
  /// 回放倍速不影响判定。
  double lost_time_thres{1.5};

  /// 前哨的 TempLost 超时，秒。前哨转速固定、轨迹规整，可以撑更久。
  double lost_time_thres_outpost{2.0};

  /// 相邻两次 track() 的时间差超过它（秒）就当作断过流：相机掉线重连、回放
  /// 拼接都会这样。两个槽全部复位，用当前帧重建。按旧速度外推这么久没有意义，
  /// 外推出来的目标还会照常交给 L4。
  double max_frame_gap{0.1};

  /// 距上次成功更新超过它（秒）后，预测不再按速度外推，目标停在这一刻的位置
  /// 和姿态上，协方差照常增长。速度或角速度一旦被错关联带偏，TempLost 期间
  /// 匀速外推能把目标推出画面，按预测位置做的关联也就再也接不上真实的板。
  double temp_lost_predict_time{0.1};

  /// 冷启动也用网络 ROI。整图缩放和裁剪放大后检出的板尺寸有系统差（回放实测
  /// ROI 里的板窄约 4.5%），原先进 Tracking 才开 ROI，切换那一刻深度跳一下；
  /// 刚初始化时速度不确定度大，这一跳会被当成径向速度（静止目标冷启动 vx 冲到
  /// 0.5 m/s）。打开后初始化那一帧在候选周围按跟踪时同样的 ROI 再检一次，从
  /// 这份检测初始化，Detecting 起就一直用 ROI，全程同一尺度，代价是初始化帧多
  /// 一次推理，见 initRoi()。只管网络 ROI，独立灯条的 ROI 仍按 Tracking 开。
  ///
  /// 另一种做法"等累计观测数够了再开 ROI"回放比过：只是把跳变推后，径向残差
  /// 拖得更久，最坏收敛帧数还变长，已去掉。数字见 docs/replay_ab_log.md。
  bool init_roi{false};
};

/**
 * @brief 误差状态整车跟踪器
 *
 * 调度两个跟踪槽（当前 + 备用），把每帧的装甲板和灯条喂给它们。
 *
 * 一帧经过的几层，各管一件事：
 *   EskfTracker    这一帧怎么喂：断流复位；槽是 Lost 就挑板建目标，否则外推、
 *                  关联、更新；备用槽什么时候开始抓、什么时候顶上来；由当前
 *                  目标算 ROI。
 *   TrackSlot      一个槽的状态机和滤波器怎么一起变。
 *   EskfTarget     滤波本身：reset → predictEkf → update。
 *   armor_matcher  观测挂到哪块板、哪根灯条上。
 *   track_roi      ROI 长什么样。
 */
class EskfTracker
{
public:
  /**
   * @brief 构造跟踪器和内部的 PnpSolver
   * @param calibration 相机标定，需带 T_barrel_camera
   * @param armor_config 装甲板几何尺寸，PnP 物点和灯条端点观测共用这一份
   * @param tracker_config 状态机与 ROI 参数
   * @param target_config 整车滤波器配置；其中的 armor 会被 armor_config 覆盖
   */
  EskfTracker(
    const L1Sensor::CameraCalibration & calibration, ArmorConfig armor_config = {},
    EskfTrackerConfig tracker_config = {}, EskfTargetConfig target_config = {});

  /**
   * @brief 标定与配置是否可用；为 false 时不应调用 track()
   * @return 可用返回 true
   */
  bool ready() const noexcept { return ready_; }

  /**
   * @brief 当前目标的跟踪状态
   * @return 跟踪状态
   */
  TrackState state() const noexcept { return active_.state(); }

  /**
   * @brief 处理一帧
   *
   * 把检测转成观测 → Lost 时挑候选初始化，否则关联并更新 → 推进状态机 → 返回
   * 当前目标的快照。
   *
   * @param detections 本帧检出的装甲板，已按敌方颜色筛过
   * @param q_world_barrel 必须对应 timestamp 这一时刻的枪管姿态
   * @param timestamp 本帧曝光时刻
   * @return 当前目标的快照；Lost 或初始化失败返回空，TempLost 返回纯预测状态
   */
  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor> & detections,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp);

  /**
   * @brief 处理一帧，多收一组独立灯条
   *
   * 独立灯条会作为额外的端点观测参与同一次更新。
   *
   * @param detections 本帧检出的装甲板，已按敌方颜色筛过
   * @param lights 本帧的侧边灯条
   * @param q_world_barrel 必须对应 timestamp 这一时刻的枪管姿态
   * @param timestamp 本帧曝光时刻
   * @return 当前目标的快照，同上
   */
  std::optional<EskfTarget> track(
    const std::vector<L2Perception::Armor> & detections,
    const std::vector<L2Perception::Light> & lights,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp);

  /**
   * @brief 独立灯条的搜索 ROI
   *
   * 只在 Tracking/TempLost、开了独立灯条观测、且目标不是基地时返回；形状见
   * Roi::light。
   *
   * @param q_world_barrel 曝光时刻的枪管姿态
   * @param timestamp 曝光时刻
   * @param image_size 图像尺寸
   * @return ROI；不满足条件时为空
   */
  std::optional<cv::Rect> lightRoi(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    const cv::Size & image_size) const;

  /**
   * @brief 剖面搜索的侧边灯条位置，见 Roi::hints
   * @param q_world_barrel 曝光时刻的枪管姿态
   * @param timestamp 曝光时刻
   * @return 预测的灯条端点；门限与 lightRoi 相同，不满足时为空
   */
  std::vector<L2Perception::LightHint> lightHints(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp) const;

  /**
   * @brief 送给网络的检测 ROI，形状见 Roi::net
   * @param q_world_barrel 曝光时刻的枪管姿态
   * @param timestamp 曝光时刻
   * @param image_size 图像尺寸
   * @param target_wh_ratio 网络输入的宽高比
   * @return ROI；目标不可聚焦时返回整图而不是空，调用方拿到的永远是一个能直接
   *         用的矩形
   */
  cv::Rect netFocusRoi(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    const cv::Size & image_size, double target_wh_ratio = 1.0) const;

  /**
   * @brief 冷启动那一帧给网络的 ROI（init_roi 打开时）
   *
   * 当前目标是 Lost、这一帧会从 detections 里初始化，就按 initTarget 会挑的那块
   * 板临时建一个目标，用跟踪时同一个 Roi::net 算出窗口。调用方在同一帧里用它
   * 再检一次、把新结果交给 track()，初始化的观测就和之后 ROI 帧的尺度一致。
   *
   * @param detections 本帧检出的装甲板，要和交给 track() 的一样先按颜色筛过
   * @param q_world_barrel 曝光时刻的枪管姿态
   * @param timestamp 曝光时刻
   * @param image_size 图像尺寸
   * @param target_wh_ratio 网络输入的宽高比
   * @return ROI；不适用时返回空
   */
  std::optional<cv::Rect> initRoi(
    const std::vector<L2Perception::Armor> & detections,
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    const cv::Size & image_size, double target_wh_ratio = 1.0);

  /**
   * @brief 当前目标本帧进了更新的完整装甲板数量
   *
   * 关联在编号间来回跳会让整车 yaw 每帧偏 2π/N，是抖动最常见的来源，所以这
   * 一项和 lastMatchedIds() 要能逐帧导出比对。两项都从 usedLights() 推出来，
   * 三者说的必然是同一个目标的同一次更新。
   *
   * @return 完整板块数
   */
  int lastMatchCount() const noexcept;

  /**
   * @brief 当前目标本帧进了更新的完整装甲板的物理板编号，见 lastMatchCount
   * @return 形如 "0|2" 的编号串；没有时为空串
   */
  std::string lastMatchedIds() const;

  /**
   * @brief 最近一帧真正进了 update 的灯条
   *
   * 初始化帧、无关联帧和纯预测帧都是空的，免得把"检测候选"误画成"已用于更新"。
   *
   * @return 采纳清单
   */
  const std::vector<UsedLight> & usedLights() const noexcept { return active_.usedLights(); }

  /**
   * @brief matchLight 各道门的累计拒绝数
   *
   * 从进程开始一直累加，不随目标复位清零。侧边灯条采纳数为零时，靠它区分
   * "候选槽位没开出来"和"某道门太紧"。
   *
   * @return 累计统计
   */
  const LightMatchStats & lightMatchStats() const noexcept { return light_match_stats_; }

  /**
   * @brief 当前目标被丢弃的累计次数（超时、发散、Detecting 丢帧、断流）
   *
   * 丢弃后同一帧就可能重建，只看 state() 数不出这些，诊断工具按这个计数。
   *
   * @return 丢弃次数
   */
  std::size_t dropCount() const noexcept { return drop_count_; }

  /**
   * @brief 当前目标各块装甲板在世界系的位姿
   * @return 每块板一个 [x, y, z, yaw]；没有目标时为空
   */
  std::vector<Eigen::Vector4d> armorPoses() const;

  /**
   * @brief 两个槽都清空、忘掉上一帧时间，回到初始状态
   */
  void reset() noexcept;

private:
  /**
   * @brief 一帧里与槽位无关的输入，以及由它派生、只活到帧末的量
   *
   * 拼一次传给两个槽。放在这里而不是成员变量里，就不用每帧手动清空，也不会被
   * 下一帧读到。
   */
  struct Frame
  {
    /// 本帧曝光时刻
    TimePoint timestamp{};

    /// 曝光时刻相机光学系在世界系的位姿
    Eigen::Isometry3d camera_in_world{Eigen::Isometry3d::Identity()};

    /// 本帧检测转成的观测，只搬了类别和角点，没跑 PnP。
    std::vector<Armor> armors;

    /// 本帧的侧边灯条，不持有；没有时为 nullptr
    const std::vector<L2Perception::Light> * lights{nullptr};

    /// initCandidates() 的缓存，第一次用到时才算。
    std::optional<std::vector<Armor>> init_candidates;
  };

  // --- track() 的各个步骤，按调用顺序排 ----------------------------------

  /**
   * @brief 断流（间隔过长或时间倒退）后两个槽都放弃
   *
   * 旧目标的外推不可信，这一帧按 Lost 重新挑候选初始化。
   *
   * @param timestamp 本帧曝光时刻
   */
  void dropOnGap(TimePoint timestamp);

  /**
   * @brief 推进一个槽一帧
   *
   * Lost 就挑候选建目标，否则外推、关联、更新；再按本帧是否关联上推进它的
   * 状态机。
   *
   * @param slot 要推进的槽
   * @param frame 本帧输入
   * @param prefer 建目标时优先的车辆类别
   */
  void advance(TrackSlot & slot, Frame & frame, std::optional<ArmorName> prefer);

  /**
   * @brief 双缓冲：当前目标进 TempLost 时让备用槽同时抓新目标
   *
   * 新目标一转成 Tracking 就交换上来，不必等当前目标超时。
   *
   * @param frame 本帧输入
   */
  void runBackup(Frame & frame);

  // --- 单个槽：建目标与更新 ----------------------------------------------

  /**
   * @brief 从本帧候选里挑一块板建目标
   * @param slot 要建目标的槽
   * @param frame 本帧输入
   * @param prefer 给定时优先取同类别的候选，没有才退回最靠近图像中心的那块
   * @return 没有候选时返回 false，槽留在 Lost
   */
  bool initTarget(TrackSlot & slot, Frame & frame, std::optional<ArmorName> prefer);

  /**
   * @brief 外推到本帧 → 关联装甲板和侧边灯条 → 更新
   * @param slot 要更新的槽
   * @param frame 本帧输入
   * @return 是否真正更新了
   */
  bool updateTarget(TrackSlot & slot, const Frame & frame);

  /**
   * @brief 本帧的初始化候选，按需算一次、缓存在 frame 里
   *
   * 正常 Tracking 一次 PnP 都不跑；当前槽恰好在这一帧转进 TempLost 时，紧接着
   * 处理备用槽还能当帧完成初始化，不用白等一帧。
   *
   * @param frame 本帧输入
   * @return 语义完整且 PnP 解出位姿的候选板，按离图像中心由近到远排序
   */
  const std::vector<Armor> & initCandidates(Frame & frame);

  /**
   * @brief 观测可用性的第一道门：类别、颜色这些语义字段齐全，角点有限
   * @param armor 候选板
   * @return 可用返回 true
   */
  bool semanticUsable(const Armor & armor) const noexcept;

  /**
   * @brief 观测可用性的第二道门：这一帧的 PnP 真的解出了位姿
   * @param armor 候选板
   * @return 可用返回 true
   */
  bool pnpUsable(const Armor & armor) const noexcept;

  // --- ROI ----------------------------------------------------------------

  /**
   * @brief ROI 的状态机门限
   * @param q_world_barrel 曝光时刻的枪管姿态
   * @param timestamp 曝光时刻
   * @param for_lights true 是独立灯条的 ROI：要求在跟踪态、开了独立灯条观测且
   *                   目标不是基地；否则是网络 ROI，init_roi 打开时从 Detecting
   *                   起就给，否则和灯条一样按跟踪态
   * @return ROI 输入；未初始化或已超时时返回空
   */
  std::optional<Roi::Focus> roiFocus(
    const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
    bool for_lights) const;

  /**
   * @brief 拿一个目标拼一份 ROI 输入
   *
   * 当前目标和冷启动的临时目标都走它，两边的窗口因此按同一套参数算。
   *
   * @param target 要聚焦的目标
   * @param camera_in_world 曝光时刻相机光学系在世界系的位姿
   * @param motion_end 运动外推的截止时刻
   * @param lost_time 距上次成功更新的时长，s
   * @return ROI 输入
   */
  Roi::Focus focusOn(
    const EskfTarget & target, const Eigen::Isometry3d & camera_in_world,
    TimePoint motion_end, double lost_time) const;

  /**
   * @brief 按类别取 TempLost 超时门限
   * @param name 车辆类别
   * @return 前哨取 lost_time_thres_outpost，其余取 lost_time_thres，s
   */
  double lostThreshold(ArmorName name) const noexcept;

  /// 相机标定
  L1Sensor::CameraCalibration calibration_;

  /// 状态机与 ROI 参数
  EskfTrackerConfig tracker_config_;

  /// 整车滤波器配置
  EskfTargetConfig target_config_;

  /// 逐帧解 PnP
  PnpSolver pnp_solver_;

  /// 图像中心，初始化时挑离它最近的板
  cv::Point2f image_center_{};

  /// 标定与配置是否可用
  bool ready_{false};

  /// 当前槽。
  ///
  /// 双缓冲：当前目标进 TempLost 时，另一个槽位同时尝试初始化新目标，新目标
  /// 先转成 Tracking 就交换上来。这样切目标或目标被短暂完全遮挡后重新出现都
  /// 能马上接上，不必等当前目标超时。备用槽优先抓与当前目标同类别的板；当前
  /// 目标的外推已经停住（见 temp_lost_predict_time）时，同类别的新目标不必等
  /// 转 Tracking，带着自己的 Detecting 计数直接换上来。
  ///
  /// 两个槽的角色固定：换目标交换的是槽的内容，不是哪个槽算"当前"。
  TrackSlot active_{};

  /// 备用槽，见 active_
  TrackSlot backup_{};

  /// 上一次 track() 的帧时间，用来判断断流。
  std::optional<TimePoint> last_frame_;

  /// 当前目标被丢弃的累计次数
  std::size_t drop_count_{0};

  /// matchLight 的累计统计，跨帧累加，不随目标复位清零。
  LightMatchStats light_match_stats_{};
};

}  // namespace L3Estimation
