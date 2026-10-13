#pragma once

#include "l3_estimation/types.hpp"

/**
 * @file
 * @brief 一个目标的四态生命周期，装甲板和符共用
 */
namespace L3Estimation {

/**
 * @brief 四态生命周期状态机
 *
 * 状态只能经 start / step / drop 改，外面读得到、写不了：迁移规则全在这一个
 * 类里，不会有哪处代码绕开它直接改状态。
 *
 *   start  Lost → Detecting，确认计数清零。新目标怎么建由调用方负责。
 *   step   按本帧是否关联上观测（found）推进一步：
 *     Detecting  关联上就累加计数，超过 tracking_thres 转 Tracking；丢一帧
 *                直接退回 Lost，重新开始计数；
 *     Tracking   丢帧转 TempLost；
 *     TempLost   再次关联上就回 Tracking，距上次更新超过 lost_time_thres 秒
 *                （由调用方算好传进来的 lost_time）则退回 Lost；
 *     Lost       什么都不做。
 *   drop   任何状态 → Lost。超时以外的放弃理由（发散、断流、被备用目标顶掉）
 *          都走这里。
 *
 * Detecting 丢一帧就重来是防误初始化：一次偶然的错误关联不该建立一个目标。
 * TempLost 的超时按真实时间算而不是数丢帧，回放倍速不影响判定。
 */
class TrackLifecycle
{
public:
  /**
   * @brief 当前状态
   * @return 跟踪状态
   */
  TrackState state() const noexcept { return state_; }

  /**
   * @brief 目标是否还在（不是 Lost）
   * @return 非 Lost 时返回 true
   */
  bool alive() const noexcept { return state_ != TrackState::Lost; }

  /**
   * @brief 是否已经过了 Detecting 的确认
   * @return Tracking，或丢了但还在靠预测撑的 TempLost 时返回 true
   */
  bool confirmed() const noexcept
  {
    return state_ == TrackState::Tracking || state_ == TrackState::TempLost;
  }

  /**
   * @brief Lost → Detecting，确认计数清零
   */
  void start() noexcept
  {
    state_ = TrackState::Detecting;
    count_ = 0;
  }

  /**
   * @brief 按本帧是否关联上观测推进一步，规则见类说明
   * @param found 本帧是否关联上观测
   * @param tracking_thres Detecting 连续关联超过这么多帧转 Tracking
   * @param lost_time 距上次成功更新的时长，s
   * @param lost_time_thres TempLost 的超时门限，s
   */
  void step(bool found, int tracking_thres, double lost_time, double lost_time_thres) noexcept
  {
    switch (state_) {
      case TrackState::Detecting:
        if (!found) {
          drop();
          break;
        }
        if (++count_ > tracking_thres) {
          state_ = TrackState::Tracking;
          count_ = 0;
        }
        break;

      case TrackState::Tracking:
        if (!found) {
          state_ = TrackState::TempLost;
        }
        break;

      case TrackState::TempLost:
        if (found) {
          state_ = TrackState::Tracking;
          break;
        }
        if (lost_time > lost_time_thres) {
          drop();
        }
        break;

      case TrackState::Lost:
        break;
    }
  }

  /**
   * @brief 任何状态 → Lost
   */
  void drop() noexcept
  {
    state_ = TrackState::Lost;
    count_ = 0;
  }

private:
  /// 当前状态
  TrackState state_{TrackState::Lost};

  /// Detecting 已连续关联上的帧数。
  int count_{0};
};

}  // namespace L3Estimation
