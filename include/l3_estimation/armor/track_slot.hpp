#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/armor_observation.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/tracking/lifecycle.hpp"

#include <opencv2/core/types.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace L3Estimation {

/**
 * @brief 本帧真正作为端点观测送进 IESKF 多观测更新的一根灯条
 *
 * 完整装甲板会拆成左右两根，isolated 为 true 表示这根来自独立灯条检测、靠整车
 * 几何关联上，而不是装甲板的角点。只用于调试显示，不参与滤波计算。
 */
struct UsedLight
{
  /// 上端点，像素
  cv::Point2f top{};

  /// 下端点，像素
  cv::Point2f bottom{};

  /// 物理板编号
  int armor_id{-1};

  /// 是否左灯条
  bool is_left{false};

  /// 是否来自独立灯条检测
  bool isolated{false};

  /// isolated 为 true 时，是这根灯条在本帧 L2 侧边灯条候选数组（lastLights）
  /// 里的下标，调试显示靠它把"滤波器真正吃下去的"回查到检出候选上；来自装甲
  /// 板角点的那两根没有下标，恒为 0。
  std::size_t light_id{0};
};

/**
 * @brief 跟踪器的一个槽：一个整车目标，加它的生命周期
 *
 * 滤波器和生命周期是两套状态，却必须一起改：建目标要同时置回 Detecting、记下
 * 更新时刻；更新成功要同时刷新更新时刻和采纳清单；放弃目标要同时清掉清单。
 * 这个类是两者唯一的交汇处，对外只有 start / predict / update / step / drop
 * 几个改动入口，目标本身只读。跟踪器只决定什么时候调哪一个、喂什么观测，
 * 不直接碰状态机，也不直接碰滤波器。
 *
 * 不变式：alive() 为真时 target() 一定建过滤波器。Lost 的槽里旧目标不清，
 * 但也没有人读它。
 */
class TrackSlot
{
public:
  /**
   * @brief 当前跟踪状态
   * @return 跟踪状态
   */
  TrackState state() const noexcept { return life_.state(); }

  /**
   * @brief 槽里是否有目标（不是 Lost）
   * @return 非 Lost 时返回 true
   */
  bool alive() const noexcept { return life_.alive(); }

  /**
   * @brief 目标是否已过 Detecting 的确认
   * @return Tracking 或 TempLost 时返回 true
   */
  bool confirmed() const noexcept { return life_.confirmed(); }

  /**
   * @brief 槽里的整车目标，只读
   * @return 整车目标
   */
  const EskfTarget & target() const noexcept { return target_; }

  /**
   * @brief 目标的车辆类别
   * @return 车辆类别
   */
  ArmorName name() const noexcept { return target_.name; }

  /**
   * @brief 距上次成功更新（或建立）的秒数
   * @param now 当前帧时刻
   * @return 秒数，非负
   */
  double idle(TimePoint now) const noexcept { return elapsedSeconds(last_update_, now); }

  /**
   * @brief 运动外推的截止时刻：上次成功更新后再过 hold 秒
   *
   * 滤波器预测和 ROI 用的是同一个时刻，过了它两者都原地不动。
   *
   * @param hold 秒数
   * @return 截止时刻
   */
  TimePoint holdFrom(double hold) const noexcept;

  /**
   * @brief 本帧真正进了滤波更新的灯条
   *
   * 建目标、没关联上和纯预测的帧都是空的，免得把"检测候选"误画成"已用于更新"。
   *
   * @return 采纳清单
   */
  const std::vector<UsedLight> & usedLights() const noexcept { return used_lights_; }

  // --- 改动入口 -----------------------------------------------------------

  /**
   * @brief Lost → Detecting：从这块板重建目标，更新时刻记为 now
   * @param armor 已解出位姿的装甲板
   * @param config 整车滤波器配置
   * @param now 当前帧时刻
   */
  void start(const Armor & armor, const EskfTargetConfig & config, TimePoint now);

  /**
   * @brief 滤波器推进到 now，运动外推截止在 holdFrom(hold)
   * @param now 当前帧时刻
   * @param hold 上次成功更新之后还允许外推的秒数
   */
  void predict(TimePoint now, double hold);

  /**
   * @brief 吸收本帧的关联结果
   *
   * 至少关联上一块完整板才会真正更新；成功时刷新更新时刻和采纳清单。
   *
   * @param matched 关联上的完整装甲板
   * @param matched_lights 关联上的侧边灯条
   * @param depth_diff 单块完整板时的左右灯条深度差，meter
   * @param now 当前帧时刻
   * @param ctx 本帧的观测上下文
   * @return 是否成功更新
   */
  bool update(
    const std::vector<MatchedArmor> & matched, const std::vector<MatchedLight> & matched_lights,
    const std::optional<double> & depth_diff, TimePoint now, const ObsContext & ctx);

  /**
   * @brief 按本帧是否关联上推进状态机
   *
   * 已确认的目标若发散就直接放弃，别让它把下游一起带歪。
   *
   * @param found 本帧是否关联上
   * @param now 当前帧时刻
   * @param tracking_thres Detecting 转 Tracking 要连续关联的帧数
   * @param lost_thres TempLost 的超时门限，s
   */
  void step(bool found, TimePoint now, int tracking_thres, double lost_thres);

  /**
   * @brief 任何状态 → Lost，并清掉采纳清单
   */
  void drop() noexcept;

  /**
   * @brief 新的一帧开始：清掉上一帧的采纳清单
   *
   * 早退或本帧没轮到更新的槽也要清，免得下游读到隔帧的结果。
   */
  void clearFrame() noexcept { used_lights_.clear(); }

private:
  /// 整车目标
  EskfTarget target_;

  /// 生命周期状态机
  TrackLifecycle life_;

  /// 上次成功更新（或建立）的时刻
  TimePoint last_update_{};

  /// 本帧的采纳清单
  std::vector<UsedLight> used_lights_;
};

}  // namespace L3Estimation
