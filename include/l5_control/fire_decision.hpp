#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/reject_reason.hpp"

#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace L5Control {
using TimePoint = std::chrono::steady_clock::time_point;
struct FireConfig {
  // 第一阶段必须保持 false；完成验收后由实车配置显式修改。
  bool shoot_enable{false};
//弹丸和热量限制
  std::optional<double> bullet_diameter;  // meter; 17 mm projectile = 0.017
  std::optional<double> min_bullet_speed;  // meter per second
  std::optional<double> max_bullet_speed;  // meter per second
  std::optional<double> heat_limit;
  std::optional<double> heat_per_shot;
 
//云台机械范围
  std::optional<double> min_yaw;   //目标装甲板最小 yaw（机械范围）
  std::optional<double> max_yaw;    //目标装甲板最大 yaw
  std::optional<double> min_pitch;   //目标装甲板最小 pitch
  std::optional<double> max_pitch;   //目标装甲板最大 pitch


// 根据目标距离选择 yaw 误差和跳变限制。代码内部角度统一使用 rad。
  std::optional<double> yaw_distance_boundary;          // meter
  std::optional<double> near_max_yaw_command_jump;      // rad
  std::optional<double> near_max_yaw_error;             // rad
  std::optional<double> far_max_yaw_command_jump;       // rad
  std::optional<double> far_max_yaw_error;              // rad

// pitch 暂时继续使用固定限制。
  std::optional<double> max_aim_pitch_error;             // rad
  std::optional<double> max_pitch_command_jump;          // rad
  std::optional<double> max_pitch_error;                 // rad
  std::chrono::milliseconds max_robot_state_age{50};
  std::chrono::milliseconds max_gimbal_pose_age{20};
  std::chrono::milliseconds max_plan_age{30};


  [[nodiscard]] bool parametersReady() const noexcept
  {
    const auto finite = [](const std::optional<double>& value) {
      return value.has_value() && std::isfinite(*value);
    };

    return finite(bullet_diameter) && *bullet_diameter > 0.0 &&
           finite(min_bullet_speed) && finite(max_bullet_speed) &&
           *min_bullet_speed > 0.0 && *min_bullet_speed < *max_bullet_speed &&
           finite(heat_limit) && *heat_limit > 0.0 &&
           finite(heat_per_shot) && *heat_per_shot > 0.0 &&
           finite(min_yaw) && finite(max_yaw) && *min_yaw < *max_yaw &&
           finite(min_pitch) && finite(max_pitch) && *min_pitch < *max_pitch &&
           finite(yaw_distance_boundary) && *yaw_distance_boundary > 0.0 &&
           finite(near_max_yaw_command_jump) && *near_max_yaw_command_jump > 0.0 &&
           finite(near_max_yaw_error) && *near_max_yaw_error > 0.0 &&
           finite(far_max_yaw_command_jump) && *far_max_yaw_command_jump > 0.0 &&
           finite(far_max_yaw_error) && *far_max_yaw_error > 0.0 &&
           finite(max_aim_pitch_error) && *max_aim_pitch_error > 0.0 &&
           finite(max_pitch_command_jump) && *max_pitch_command_jump > 0.0 &&
           finite(max_pitch_error) && *max_pitch_error > 0.0 &&
           max_robot_state_age.count() > 0 &&
           max_gimbal_pose_age.count() > 0 &&
           max_plan_age.count() > 0;
  }
};

// 从 YAML 文件读取火控参数。YAML 中角度使用 degree，读取后转换为 rad。
FireConfig loadFireConfig(const std::string& config_path);

struct FireInput {
  std::optional<L3Estimation::TargetState> target;
  L4Planning::AimPlan plan;
  L1Sensor::RobotState robot_state;

  TimePoint now{};
 
  bool calibration_ready{false};
  
};
struct FireDecision {
  // fire_feasible 记录理论窗口；shoot 是考虑 shoot_enable 后的实际下发值。
  bool fire_feasible{false};
  bool shoot{false};
  std::vector<RejectReason> reasons;
};

// 记录上一帧瞄准状态，用于判断目标或装甲板是否发生切换。
class FireEvaluator {
public:
  explicit FireEvaluator(FireConfig config = {});//火控判断器
  [[nodiscard]] FireDecision evaluate(const FireInput& input);

  void reset() noexcept;

private:
  FireConfig config_;

  int last_target_id_{-1};

  std::optional<double> last_command_yaw_;
  std::optional<double> last_command_pitch_;
  std::optional<double> last_fly_time_;

  std::size_t stable_tracking_frames_{0};//表示稳定追踪多少帧
  TimePoint last_switch_time_{};//表示上一次切换目标或装甲板的时间
};


}  
// namespace L5Control
