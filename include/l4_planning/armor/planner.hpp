#pragma once

#include "l3_estimation/armor/eskf_target.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

namespace L4Planning {

/**
 * @brief 一次规划的输入
 */
struct PlanInput {
  /// L3 给出的目标快照；没有目标时为空
  std::optional<L3Estimation::EskfTarget> target;

  /// 下位机回传的弹速，m/s。没发（0）或不可信时 Planner 换成 default_bullet_speed。
  double bullet_speed{0.0};

  /// 本次规划开始的 steady_clock 时间
  TimePoint plan_time{};

  /// 是否补偿 target.t() 到 plan_time 的已发生延迟；false 时按固定 5 ms（离线入口）
  bool to_now{true};

  /// 上一帧实测的"规划结束 -> 串口发出"耗时，单位秒。本帧的值要等规划做完
  /// 才知道，所以只能用上一帧的量代入；帧间这一段基本恒定。
  double plan_to_send{0.0};

  /// 规划时刻的枪管姿态。只在进自瞄后的头一次选板用：两块板都能打时挑离枪口
  /// 最近的那块。缺省时照常选更正对的板。
  std::optional<Eigen::Quaterniond> q_world_barrel;
};

/**
 * @brief 定点规划器：预测命中时刻、选择实体装甲板并解算 yaw/pitch
 */
class Planner {
public:
  /**
   * @brief 用给定参数构造，构造即算一轮新的自瞄
   * @param config 规划器参数
   */
  explicit Planner(PlanConfig config = {});

  /**
   * @brief 规划一帧
   *
   * 把目标外推到发射时刻、选板，再在弹道解算与命中时刻预测之间迭代，直到飞行
   * 时间收敛。
   *
   * @param input 目标、弹速和时间信息
   * @return 规划结果；失败时 error 说明原因
   */
  [[nodiscard]] Plan plan(const PlanInput& input);

  /**
   * @brief 标记一轮新的自瞄：下一次成功规划改按离枪口最近选板，见 choose
   */
  void reset() noexcept;

private:
  /**
   * @brief 选中的瞄准板
   */
  struct AimPoint {
    /// 是否选到了板
    bool valid{false};

    /// armor_xyza_list() 中的物理板编号
    int armor_id{-1};

    /// [x, y, z, normal_yaw]
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};
  };

  /**
   * @brief 在外推后的目标上选一块可击打的板
   * @param target 已外推到目标时刻的整车目标
   * @param muzzle 枪口在世界系的单位指向，只在进自瞄后的头一次选板给
   * @return 选中的板；没有板在窗口内时 valid 为 false
   */
  AimPoint choose(
    const L3Estimation::EskfTarget& target,
    const std::optional<Eigen::Vector3d>& muzzle);

  /**
   * @brief 下位机弹速不可信时换成缺省值；只在切换时打一条日志
   * @param reported 下位机回传的弹速，m/s
   * @return 实际用于解弹道的弹速，m/s
   */
  double bulletSpeed(double reported);

  /// 规划器参数
  PlanConfig config_;

  /// 当前锁定的物理板编号，-1 表示未锁定
  int locked_id_{-1};

  /// 构造即算一轮新的自瞄，reset() 再置回；第一次成功规划后清掉。
  bool entering_{true};

  /// 上一帧是否在用缺省弹速，用于只在切换时打日志
  bool default_speed_{false};
};

}  // namespace L4Planning
