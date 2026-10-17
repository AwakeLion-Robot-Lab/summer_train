// Target eligibility and warmup reference: Alliance-Algorithm/rmcs_auto_aim_v2.
// See README.md and ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f4-planning/planning_detail.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace Rune {

PlanningStage::PlanningStage(PlanningConfig config) : config_(std::move(config)) {
  for (double value : {config_.gravity_m_s2,config_.max_observation_age_s,
      config_.inactive_timeout_s,config_.max_prediction_s,config_.flight_time_tolerance_s})
    if (!std::isfinite(value) || value<=0) throw std::invalid_argument("Invalid rune planning configuration");
  for (double value : {config_.small_warmup_s,config_.big_warmup_s})
    if (!std::isfinite(value) || value<0) throw std::invalid_argument("Invalid rune warmup");
  if (config_.max_iterations<1 || config_.max_iterations>100 ||
      !std::isfinite(config_.yaw_offset_rad) || !std::isfinite(config_.pitch_offset_rad) ||
      (config_.enable_air_resistance && (!std::isfinite(config_.linear_drag_coefficient) ||
                                       config_.linear_drag_coefficient<=0)))
    throw std::invalid_argument("Invalid rune ballistic configuration");
}

void PlanningStage::reset() noexcept {
  blades_={}; mode_.reset(); started_at_={}; previous_capture_={};
  selected_sector_=-1; ++epoch_;
}

AimPlan PlanningStage::plan(const Motion& motion,Mode mode,const ControlInput& control) {
  AimPlan plan; plan.mode=mode; plan.captured_at=motion.pose.captured_at;
  plan.generated_at=control.command_at;
  const auto& pose=motion.pose;
  const double age=std::chrono::duration<double>(control.command_at-pose.captured_at).count();
  if (pose.captured_at==TimePoint{} || control.command_at==TimePoint{} ||
      age<0 || age>config_.max_observation_age_s ||
      !std::isfinite(control.fire_delay_s) || control.fire_delay_s<0) {
    reset(); plan.status=Status::PlanStale; return plan;
  }
  if (!pose.center_barrel.allFinite() || pose.center_barrel.x()<=0 ||
      !pose.target_barrel.allFinite() || pose.target_barrel.x()<=0 ||
      !pose.normal_barrel.allFinite() || pose.normal_barrel.norm()<0.9 ||
      (pose.target_barrel-pose.center_barrel).norm()<1e-6 ||
      (pose.filter_state && (pose.filter_state->target_sector<0 ||
        pose.filter_state->target_sector>=5 || !pose.filter_state->x.allFinite()))) {
    reset(); plan.status=Status::InvalidInput; return plan;
  }
  const bool new_frame=pose.captured_at!=previous_capture_;
  const double gap=std::chrono::duration<double>(pose.captured_at-previous_capture_).count();
  if ((mode_ && *mode_!=mode) || (new_frame && (pose.history_reset ||
      (previous_capture_!=TimePoint{} && (gap<0 || gap>config_.max_observation_age_s)) ||
      (!pose.filter_state && pose.target_switched)))) reset();
  if (!mode_) { mode_=mode; started_at_=pose.captured_at; }
  plan.tracking_epoch=epoch_;
  const int reference=pose.filter_state ? pose.filter_state->target_sector : 0;
  if (pose.captured_at!=previous_capture_) {
    if (pose.blades.empty()) blades_[reference]={!pose.activated,pose.captured_at};
    else for (const auto& blade : pose.blades) {
      if (blade.sector<0 || blade.sector>=5) continue;
      blades_[(reference+blade.sector)%5]={!blade.activated,pose.captured_at};
    }
    previous_capture_=pose.captured_at;
  }
  for (auto& blade : blades_) {
    if (std::chrono::duration<double>(control.command_at-blade.seen_at).count()>
        config_.inactive_timeout_s) blade.inactive=false;
  }
  if (selected_sector_<0 || !blades_[selected_sector_].inactive) {
    selected_sector_=-1;
    for (int j=0;j<5;++j) if (blades_[j].inactive) { selected_sector_=j; break; }
  }
  Eigen::Quaterniond orientation=Eigen::Quaterniond::Identity();
  if (pose.q_world_barrel) {
    orientation=*pose.q_world_barrel;
    if (!orientation.coeffs().allFinite() || std::abs(orientation.squaredNorm()-1)>1e-3) {
      reset(); plan.status=Status::InvalidInput; return plan;
    }
    orientation.normalize();
  }
  plan.center_aim=orientation*pose.center_barrel;
  const Eigen::Vector3d normal=(orientation*pose.normal_barrel).normalized();
  const double warmup=mode==Mode::Small ? config_.small_warmup_s : config_.big_warmup_s;
  const bool motion_valid=motion.ready && std::isfinite(motion.angular_speed_rad_s) &&
    (mode!=Mode::Big || static_cast<bool>(motion.phase_delta_rad));
  const bool warmed=std::chrono::duration<double>(control.command_at-started_at_).count()>=warmup;
  const bool eligible=motion_valid && warmed && selected_sector_>=0;
  plan.center_hold=!eligible;
  if (!eligible && !config_.center_hold) return plan;
  if (!std::isfinite(control.bullet_speed_m_s) || control.bullet_speed_m_s<=0) {
    plan.status=Status::BallisticFailed; return plan;
  }
  Eigen::Vector3d target=plan.center_aim;
  std::optional<detail::Intercept> solution;
  if (eligible) {
    // Keep the selected leaf first; if its ballistic intercept is unreachable,
    // try the remaining eligible identities before abandoning this frame.
    const int preferred=selected_sector_;
    for (int attempt=0;attempt<6;++attempt) {
      const int sector=attempt==0 ? preferred : attempt-1;
      if ((attempt>0 && sector==preferred) || !blades_[sector].inactive) continue;
      const double angle=(sector-reference)*2*std::numbers::pi/5;
      target=plan.center_aim+Eigen::AngleAxisd(angle,normal)*
        (orientation*(pose.target_barrel-pose.center_barrel));
      for (const auto& blade : pose.blades)
        if ((reference+blade.sector)%5==sector && blade.target_barrel.allFinite() &&
            blade.target_barrel.x()>0)
          target=orientation*blade.target_barrel;
      solution=detail::intercept(motion,target,plan.center_aim,normal,false,control,config_);
      if (solution) { selected_sector_=sector; break; }
    }
  } else {
    solution=detail::intercept(motion,target,plan.center_aim,normal,true,control,config_);
  }
  if (!solution) { plan.status=Status::BallisticFailed; return plan; }
  plan.yaw=plan.ballistic_yaw_rad=std::remainder(
    solution->ballistic.yaw+config_.yaw_offset_rad,2*std::numbers::pi);
  plan.pitch=plan.ballistic_pitch_rad=solution->ballistic.pitch+config_.pitch_offset_rad;
  plan.fly_time=solution->ballistic.flight_time;
  plan.prediction_horizon_s=solution->horizon_s;
  plan.impact_time=control.command_at+std::chrono::duration_cast<TimePoint::duration>(
    std::chrono::duration<double>(control.fire_delay_s+plan.fly_time));
  plan.aim_point_barrel=orientation.conjugate()*solution->point;
  plan.aim_point_world=solution->point; // absolute axes, relative to current muzzle
  plan.used_air_resistance=solution->ballistic.used_air_resistance;
  plan.armor_id=eligible ? selected_sector_ : -1;
  plan.target_id=0; plan.valid=true; plan.tracked=true;
  plan.tracked_ready=eligible; plan.within_firing_window=eligible; plan.fire_permitted=eligible;
  plan.status=eligible ? Status::AimNotReady : (!motion_valid ? Status::MotionNotReady :
    (!warmed ? Status::WarmingUp : Status::CenterHold));
  detail::optimizeTrajectory(plan,motion,target,normal,control,config_);
  return plan;
}

} // namespace Rune
