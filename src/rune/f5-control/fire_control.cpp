// Idle/shoot timing reference: Alliance-Algorithm/rmcs_auto_aim_v2.
// See README.md and ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f5-control/fire_control.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace Rune {
namespace {
double seconds(TimePoint a,TimePoint b) {
  return std::chrono::duration<double>(a-b).count();
}
double yawError(double a,double b) { return std::abs(std::remainder(a-b,2*std::numbers::pi)); }
}

ControlStage::ControlStage(FireControlConfig config) : config_(std::move(config)) {
  for (double v : {config_.max_yaw_error_rad,config_.max_pitch_error_rad,
                  config_.shoot_duration_s,config_.max_plan_age_s})
    if (!std::isfinite(v) || v<=0) throw std::invalid_argument("Invalid rune fire configuration");
  for (double v : {config_.aim_settle_s,config_.idle_duration_s,
                  config_.min_shot_interval_s,config_.command_jump_threshold_rad})
    if (!std::isfinite(v) || v<0) throw std::invalid_argument("Invalid rune fire timing");
  const double period=config_.idle_duration_s+config_.shoot_duration_s;
  if (!std::isfinite(period) || period<1e-6)
    throw std::invalid_argument("Invalid rune firing period");
}
void ControlStage::suspend() noexcept {
  cycle_started_={}; aligned_since_={}; fired_cycle_=-1;
  sector_=-1;
}
void ControlStage::reset() noexcept {
  suspend(); mode_.reset(); epoch_=0; last_shot_={}; previous_command_={};
  latched_=false; last_command_.reset();
}
void ControlStage::rearm() noexcept { latched_=false; suspend(); }

Command ControlStage::update(const AimPlan& plan,const ControlInput& control) {
  Command out; out.mode=plan.mode; out.status=plan.status;
  const auto now=control.command_at;
  if (!plan.valid || now==TimePoint{} || plan.generated_at==TimePoint{} ||
      plan.captured_at==TimePoint{} || seconds(now,plan.generated_at)<0 ||
      seconds(now,plan.generated_at)>config_.max_plan_age_s ||
      seconds(now,plan.captured_at)<0 || seconds(now,plan.captured_at)>config_.max_plan_age_s) {
    suspend(); if (plan.valid) out.status=Status::PlanStale; return out;
  }
  if (!std::isfinite(plan.yaw) || !std::isfinite(plan.pitch) ||
      !std::isfinite(plan.ballistic_yaw_rad) || !std::isfinite(plan.ballistic_pitch_rad) ||
      !std::isfinite(plan.fly_time) || plan.fly_time<=0 || plan.impact_time<now ||
      (plan.using_MPC && (plan.samples.empty() || plan.samples.front().execute_time!=plan.generated_at))) {
    suspend(); out.status=Status::InvalidInput; return out;
  }
  out.yaw_rad=plan.yaw; out.pitch_rad=plan.pitch; out.flight_time_s=plan.fly_time;
  out.found=true; out.center_hold=plan.center_hold; out.using_mpc=plan.using_MPC;
  out.target_sector=plan.armor_id; out.yaw_rate_rad_s=plan.yaw_rate;
  out.pitch_rate_rad_s=plan.pitch_rate; out.yaw_acceleration_rad_s2=plan.yaw_acceleration;
  out.pitch_acceleration_rad_s2=plan.pitch_acceleration; out.samples=plan.samples;
  const auto serial=makeSerialCommand(out);
  if (!serial) { suspend(); out.found=false; out.status=Status::InvalidInput; return out; }
  const bool jump=last_command_ && config_.command_jump_threshold_rad>0 &&
    (yawError(serial->yaw,last_command_->yaw)>config_.command_jump_threshold_rad ||
     std::abs(serial->pitch-last_command_->pitch)>config_.command_jump_threshold_rad);
  last_command_=serial;
  if (previous_command_!=TimePoint{} && now<previous_command_) {
    suspend(); out.status=Status::InvalidInput; return out;
  }
  if (previous_command_!=TimePoint{} && seconds(now,previous_command_)>config_.max_plan_age_s)
    suspend(); // A gap in feedback cannot count as continuous alignment.
  previous_command_=now;
  if (!mode_ || *mode_!=plan.mode || epoch_!=plan.tracking_epoch || sector_!=plan.armor_id) {
    suspend(); mode_=plan.mode; epoch_=plan.tracking_epoch; sector_=plan.armor_id;
  }
  if (plan.center_hold || !plan.fire_permitted || plan.armor_id<0 || plan.armor_id>=5) {
    suspend(); return out;
  }
  if (cycle_started_==TimePoint{}) cycle_started_=now;
  if (!std::isfinite(control.actual_yaw_rad) || !std::isfinite(control.actual_pitch_rad) ||
      (control.feedback_at!=TimePoint{} && (seconds(now,control.feedback_at)<0 ||
        seconds(now,control.feedback_at)>config_.max_plan_age_s))) {
    aligned_since_={}; out.status=Status::InvalidFeedback; return out;
  }
  // Check BOTH the executable trajectory and the true ballistic intercept.
  // A lagging MPC reference must not open a false firing window.
  if (jump || yawError(plan.yaw,control.actual_yaw_rad)>config_.max_yaw_error_rad ||
      yawError(plan.ballistic_yaw_rad,control.actual_yaw_rad)>config_.max_yaw_error_rad ||
      std::abs(plan.pitch-control.actual_pitch_rad)>config_.max_pitch_error_rad ||
      std::abs(plan.ballistic_pitch_rad-control.actual_pitch_rad)>config_.max_pitch_error_rad) {
    aligned_since_={}; out.status=jump ? Status::CommandJump : Status::AimNotReady; return out;
  }
  if (aligned_since_==TimePoint{}) aligned_since_=now;
  if (seconds(now,aligned_since_)+1e-9<config_.aim_settle_s) {
    out.status=Status::AimNotReady; return out;
  }
  const double period=config_.idle_duration_s+config_.shoot_duration_s;
  const double elapsed=seconds(now,cycle_started_);
  const auto cycle=static_cast<std::int64_t>(std::floor(elapsed/period));
  const double position=elapsed-cycle*period;
  if (position+1e-9<config_.idle_duration_s) { out.status=Status::Idle; return out; }
  if (fired_cycle_==cycle || (config_.require_rearm && latched_)) {
    out.status=Status::AlreadyFired; return out;
  }
  if (last_shot_!=TimePoint{} && seconds(now,last_shot_)+1e-9<config_.min_shot_interval_s) {
    out.status=Status::ShotCooldown; return out;
  }
  out.fire_feasible=true;
  if (!config_.shoot_enable || !control.shoot_requested) {
    out.status=Status::ShootDisabled; return out;
  }
  out.fire=true; out.status=Status::Ready;
  fired_cycle_=cycle; last_shot_=now; latched_=true;
  last_command_->shoot=true;
  return out;
}

std::optional<L1Sensor::SerialCommand> ControlStage::makeSerialCommand(const Command& command) {
  if (!command.found || !std::isfinite(command.yaw_rad) || !std::isfinite(command.pitch_rad) ||
      !std::isfinite(command.yaw_rate_rad_s) || !std::isfinite(command.pitch_rate_rad_s) ||
      !std::isfinite(command.yaw_acceleration_rad_s2) || !std::isfinite(command.pitch_acceleration_rad_s2)) return {};
  L1Sensor::SerialCommand result;
  result.yaw=command.yaw_rad; result.pitch=-command.pitch_rad; result.shoot=command.fire;
  result.yaw_velocity=command.yaw_rate_rad_s;
  result.pitch_velocity=-command.pitch_rate_rad_s;
  result.yaw_acceleration=command.yaw_acceleration_rad_s2;
  result.pitch_acceleration=-command.pitch_acceleration_rad_s2;
  return result;
}
std::optional<L1Sensor::SerialCommand> ControlStage::safeHold() const {
  if (!last_command_) return {};
  auto hold=*last_command_; hold.shoot=false;
  hold.yaw_velocity=hold.pitch_velocity=hold.yaw_acceleration=hold.pitch_acceleration=0;
  return hold;
}

} // namespace Rune
