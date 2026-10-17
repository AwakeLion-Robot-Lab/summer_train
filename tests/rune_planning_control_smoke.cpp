#include "rune/f4-planning/planning.hpp"
#include "rune/f5-control/fire_control.hpp"
#include "armor/l3_planning/ballistic_solver.hpp"
#include "armor/l3_planning/tiny_mpc.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace {
using namespace Rune;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
TimePoint stamp(double t) {
  return TimePoint{}+std::chrono::seconds(10)+
    std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(t));
}
PlanningConfig planningConfig() {
  PlanningConfig config;
  config.small_warmup_s=config.big_warmup_s=0;
  config.mpc.enable_mpc=false;
  return config;
}
Motion scene(double t=0,double speed=1,int reference=0) {
  Motion motion;
  auto& p=motion.pose;
  p.center_barrel={8,.2,.3}; p.normal_barrel={1,0,0}; p.captured_at=stamp(t);
  const auto target=[&](int j) {
    const double a=speed*t+j*2*std::numbers::pi/5;
    return (p.center_barrel+Eigen::Vector3d(0,-.7*std::sin(a),.7*std::cos(a))).eval();
  };
  p.target_barrel=target(reference);
  p.filter_state=FilterState{}; p.filter_state->target_sector=reference;
  p.filter_state->x[5]=speed*t;
  for (int j=0;j<5;++j) {
    BladeMatch blade; blade.sector=(j-reference+5)%5;
    blade.activated=j!=0 && j!=2; blade.target_barrel=target(j);
    p.blades.push_back(blade);
  }
  motion.angular_speed_rad_s=speed; motion.ready=true; motion.model=MotionModel::Linear;
  motion.phase_delta_rad=[speed](double dt)->std::optional<double> {
    if (dt<0 || !std::isfinite(dt)) return {};
    return speed*dt;
  };
  motion.phase_delta_signed_rad=[speed](double dt)->std::optional<double>{return speed*dt;};
  return motion;
}
ControlInput input(double t=0) {
  ControlInput c; c.command_at=stamp(t+.02); c.bullet_speed_m_s=28; c.fire_delay_s=.02;
  return c;
}
void prediction() {
  auto config=planningConfig();
  for (double speed : {-1.0,1.0}) {
    auto motion=scene(0,speed);
    const auto c=input();
    const auto plan=PlanningStage(config).plan(motion,Mode::Small,c);
    check(plan.valid && plan.fire_permitted && !plan.center_hold,"direct intercept failed");
    const Eigen::Vector3d expected=motion.pose.center_barrel+
      Eigen::AngleAxisd(speed*plan.prediction_horizon_s,Eigen::Vector3d::UnitX())*
        (motion.pose.target_barrel-motion.pose.center_barrel);
    check((expected-plan.aim_point_world).norm()<1e-10,"delay/rotation prediction mismatch");
    check(std::abs(plan.prediction_horizon_s-(.04+plan.fly_time))<config.flight_time_tolerance_s,
          "flight-time fixed point did not converge");
    L4Planning::BallisticRequest request;
    request.target_position_barrel=expected; request.bullet_speed=28;
    const auto oracle=L4Planning::BallisticSolver{}.solve(request);
    check(oracle.valid && std::abs(oracle.pitch-plan.pitch)<1e-10 &&
          std::abs(oracle.yaw-plan.yaw)<1e-10,"shared ballistic solution changed");
    check(std::isfinite(plan.yaw_rate) && std::abs(plan.yaw_rate)>.01,"direct feedforward missing");
  }
  auto slow=input(); slow.bullet_speed_m_s=8.9;
  const auto reachable=PlanningStage(config).plan(scene(0,0),Mode::Small,slow);
  check(reachable.valid && reachable.armor_id==2,
        "unreachable preferred leaf hid another feasible target");
  auto m=scene();
  m.pose.q_world_barrel=Eigen::Quaterniond(
    Eigen::AngleAxisd(3.13,Eigen::Vector3d::UnitZ())*
    Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitY()));
  const auto tilted=PlanningStage(config).plan(m,Mode::Big,input());
  check(tilted.valid && std::abs(tilted.yaw)>2.9,"absolute yaw / gravity frame lost");
  const Eigen::Vector3d absolute=m.pose.q_world_barrel.value()*tilted.aim_point_barrel;
  check((absolute-tilted.aim_point_world).norm()<1e-10,"barrel/absolute point transforms inconsistent");
  auto drag=config; drag.enable_air_resistance=true; drag.linear_drag_coefficient=.12;
  const auto vacuum=PlanningStage(config).plan(scene(0,0),Mode::Small,input());
  const auto resisted=PlanningStage(drag).plan(scene(0,0),Mode::Small,input());
  check(resisted.valid && resisted.used_air_resistance && resisted.fly_time>vacuum.fly_time &&
        resisted.pitch>vacuum.pitch,"existing linear-drag ballistic model not used");
  m=scene();
  m.phase_delta_rad=[](double h)->std::optional<double> { return 1.2*h-.8/1.94*(std::cos(1.94*h+.7)-std::cos(.7)); };
  const auto big=PlanningStage(config).plan(m,Mode::Big,input());
  const Eigen::Vector3d expected=m.pose.center_barrel+Eigen::AngleAxisd(*m.phase_delta_rad(big.prediction_horizon_s),
    Eigen::Vector3d::UnitX())*(m.pose.target_barrel-m.pose.center_barrel);
  check(big.valid && (expected-big.aim_point_world).norm()<1e-10,"big-rune analytic predictor bypassed");
}
void selectionAndWarmup() {
  auto config=planningConfig();
  PlanningStage planner(config);
  auto p=planner.plan(scene(0,1,2),Mode::Big,input());
  check(p.armor_id==0,"initial selection did not use persistent identity order");
  auto m=scene(.02,1,2);
  std::reverse(m.pose.blades.begin(),m.pose.blades.end());
  p=planner.plan(m,Mode::Big,input(.02));
  check(p.armor_id==0,"detection reordering switched blade identity");
  m=scene(.04,1,2);
  m.pose.blades.erase(m.pose.blades.begin());
  p=planner.plan(m,Mode::Big,input(.04));
  check(p.armor_id==0,"short occlusion lost the held inactive blade");
  m=scene(.13,1,2); m.pose.blades.erase(m.pose.blades.begin());
  p=planner.plan(m,Mode::Big,input(.13));
  check(p.armor_id==2,"expired inactive blade remained selected");
  m=scene(.15,1,2); for (auto& blade : m.pose.blades) blade.activated=true;
  p=planner.plan(m,Mode::Big,input(.15));
  check(p.valid && p.center_hold && !p.fire_permitted && p.armor_id==-1,
        "activated blades did not produce non-firing center hold");
  m=scene(.17,1,2); m.pose.blades[0].activated=true;
  p=planner.plan(m,Mode::Big,input(.17));
  check(p.armor_id==2,"confirmed active blade was selected again");
  PlanningStage repeated(config);
  m=scene(); m.pose.history_reset=true;
  const auto first=repeated.plan(m,Mode::Small,input());
  auto c=input(); c.command_at=stamp(.11);
  const auto old=repeated.plan(m,Mode::Small,c);
  check(old.valid && old.center_hold && old.tracking_epoch==first.tracking_epoch,
        "repeated frame refreshed its inactive timeout or reset history");
  config.small_warmup_s=.2; config.big_warmup_s=.4;
  PlanningStage warming(config);
  for (int i=0;i<=25;++i) {
    p=warming.plan(scene(.02*i),Mode::Small,input(.02*i));
    if (i<8) check(p.center_hold && !p.fire_permitted,"small warmup bypassed");
  }
  check(p.fire_permitted,"small warmup never completed");
  auto epoch=p.tracking_epoch;
  p=warming.plan(scene(.52),Mode::Big,input(.52));
  check(p.center_hold && p.tracking_epoch!=epoch,"mode switch retained readiness");
  for (int i=27;i<=50;++i) p=warming.plan(scene(.02*i),Mode::Big,input(.02*i));
  check(p.fire_permitted,"big warmup never completed");
  auto reset=scene(1.02); reset.pose.history_reset=true;
  p=warming.plan(reset,Mode::Big,input(1.02));
  check(p.center_hold && !p.fire_permitted,"tracking reset retained warmup");
}
void invalidInputs() {
  auto config=planningConfig(); auto m=scene(); auto c=input();
  c.bullet_speed_m_s=0;
  check(!PlanningStage(config).plan(m,Mode::Small,c).valid,"invalid speed accepted");
  c=input(); c.bullet_speed_m_s=1;
  check(!PlanningStage(config).plan(m,Mode::Small,c).valid,"unreachable target accepted");
  c=input(); c.command_at=stamp(.2);
  check(!PlanningStage(config).plan(m,Mode::Small,c).valid,"stale observation accepted");
  c=input(); c.command_at=stamp(-.1);
  check(!PlanningStage(config).plan(m,Mode::Small,c).valid,"future observation accepted");
  c=input(); m.pose.normal_barrel.setZero();
  check(!PlanningStage(config).plan(m,Mode::Small,c).valid,"invalid plane accepted");
  m=scene(); m.phase_delta_rad=[](double)->std::optional<double>{return {};};
  check(!PlanningStage(config).plan(m,Mode::Big,input()).valid,"failed phase prediction accepted");
  m=scene(); m.ready=false;
  auto hold=PlanningStage(config).plan(m,Mode::Big,input());
  check(hold.valid && hold.center_hold && !hold.fire_permitted,"unready motion was armed");
  config.max_iterations=1; config.flight_time_tolerance_s=1e-15;
  check(!PlanningStage(config).plan(scene(),Mode::Small,input()).valid,"unconverged intercept accepted");
  bool threw=false;
  try { auto bad=planningConfig(); bad.inactive_timeout_s=-1; PlanningStage stage(bad); }
  catch(const std::invalid_argument&) { threw=true; }
  check(threw,"invalid configuration accepted");
}
void mpc() {
  auto config=planningConfig(); config.mpc.enable_mpc=true;
  const auto plan=PlanningStage(config).plan(scene(0,.5),Mode::Big,input());
  check(plan.valid && plan.using_MPC && plan.samples.size()==L4Planning::kTinyMpcHorizon-L4Planning::kTinyMpcControlIndex,
        "existing TinyMPC did not produce a centered trajectory");
  check(plan.samples.front().execute_time==input().command_at &&
        std::abs(std::chrono::duration<double>(plan.samples.back().execute_time-
          plan.samples.front().execute_time).count()-.49)<1e-8,"MPC execution times shifted");
  for (std::size_t i=1;i<plan.samples.size();++i) {
    const auto& a=plan.samples[i-1]; const auto& b=plan.samples[i];
    check(std::abs(std::remainder(b.yaw-a.yaw,2*std::numbers::pi)-a.yaw_rate*.01)<1e-8 &&
          std::abs(b.pitch-a.pitch-a.pitch_rate*.01)<1e-8,"MPC discrete dynamics broken");
    check(b.yaw_acceleration>=config.mpc.min_yaw_acceleration &&
          b.yaw_acceleration<=config.mpc.max_yaw_acceleration,"MPC acceleration bound violated");
  }
  auto wrap=scene(0,1);
  wrap.pose.q_world_barrel=Eigen::Quaterniond(Eigen::AngleAxisd(3.13,Eigen::Vector3d::UnitZ()));
  const auto wrapped=PlanningStage(config).plan(wrap,Mode::Big,input());
  check(wrapped.using_MPC && std::abs(wrapped.yaw_rate)<1,"yaw wrapping corrupted MPC reference");
  auto failed=config; failed.mpc.min_yaw_acceleration=1; failed.mpc.max_yaw_acceleration=-1;
  const auto fallback=PlanningStage(failed).plan(scene(),Mode::Big,input());
  check(fallback.valid && !fallback.using_MPC && fallback.samples.empty() &&
        fallback.mpc_diagnostics.status==MpcStatus::NotConverged,"MPC failure did not preserve direct aim");
  auto short_horizon=config; short_horizon.max_prediction_s=.5;
  const auto limited=PlanningStage(short_horizon).plan(scene(),Mode::Big,input());
  check(limited.valid && !limited.using_MPC &&
        limited.mpc_diagnostics.status==MpcStatus::InvalidReference,"invalid MPC reference lost direct aim");
  auto external=scene(); external.model=MotionModel::External;
  external.phase_delta_signed_rad={};
  const auto forward_only=PlanningStage(config).plan(external,Mode::Big,input());
  check(forward_only.valid && !forward_only.using_MPC,
        "forward-only external model was extrapolated backwards");
  L4Planning::TinyMpcReference reference; reference.valid=true;
  const auto solution=L4Planning::solveTinyMpc(reference,config.mpc);
  const auto old_samples=L4Planning::makeTinyMpcSamples(reference,solution,stamp(0));
  check(old_samples.size()==50,"armor MPC default control index changed");
}
AimPlan firePlan(double t) {
  auto p=PlanningStage(planningConfig()).plan(scene(0,0),Mode::Small,input());
  p.generated_at=p.captured_at=stamp(t);
  p.impact_time=p.generated_at+std::chrono::duration_cast<TimePoint::duration>(
    std::chrono::duration<double>(.02+p.fly_time));
  p.tracking_epoch=1;
  return p;
}
ControlInput fireInput(double t,const AimPlan& p) {
  auto c=input(); c.command_at=stamp(t);
  c.actual_yaw_rad=p.ballistic_yaw_rad; c.actual_pitch_rad=p.ballistic_pitch_rad;
  return c;
}
void firing() {
  FireControlConfig config; config.shoot_enable=true;
  config.aim_settle_s=.03; config.idle_duration_s=.2; config.shoot_duration_s=.1;
  config.min_shot_interval_s=.25;
  ControlStage stage(config);
  const auto tick=[&](double t) { auto p=firePlan(t); return stage.update(p,fireInput(t,p)); };
  check(!tick(0).fire,"settling was skipped");
  check(tick(.04).status==Status::Idle,"idle timing missing");
  check(!tick(.1).fire && !tick(.15).fire,"idle emitted an early pulse");
  check(tick(.201).fire,"shoot window did not emit a pulse");
  check(!tick(.201).fire && !tick(.22).fire,"duplicate cycle emitted a second pulse");
  check(!tick(.32).fire && !tick(.4).fire && !tick(.45).fire && tick(.511).fire,
        "next idle/shoot cycle failed");
  auto p=firePlan(.52); p.armor_id=2;
  check(!stage.update(p,fireInput(.52,p)).fire,"target switch skipped idle");
  for (double t : {.6,.68}) {
    p=firePlan(t); p.armor_id=2;
    check(!stage.update(p,fireInput(t,p)).fire,"new target skipped idle");
  }
  p=firePlan(.73); p.armor_id=2;
  check(stage.update(p,fireInput(.73,p)).status==Status::ShotCooldown,"cross-target shot cooldown lost");
  p=firePlan(.77); p.armor_id=2;
  check(stage.update(p,fireInput(.77,p)).fire,"new target never became shootable");
  check(stage.safeHold() && !stage.safeHold()->shoot && stage.safeHold()->yaw_velocity==0,
        "safe hold retained firing or feedforward");
  config.aim_settle_s=config.idle_duration_s=0; config.min_shot_interval_s=.05;
  config.require_rearm=true;
  ControlStage latched(config);
  p=firePlan(0); check(latched.update(p,fireInput(0,p)).fire,"manual latch initial shot failed");
  p=firePlan(.21); check(!latched.update(p,fireInput(.21,p)).fire,"manual latch rearmed itself");
  latched.rearm(); check(latched.update(p,fireInput(.21,p)).fire,"explicit rearm failed");
  config.require_rearm=false; config.shoot_enable=false;
  ControlStage disabled(config); p=firePlan(0);
  const auto dry=disabled.update(p,fireInput(0,p));
  check(dry.fire_feasible && !dry.fire && dry.status==Status::ShootDisabled,"dry-run feasibility missing");
  config.shoot_enable=true;
  ControlStage requested(config); auto request=fireInput(0,p); request.shoot_requested=false;
  const auto inhibited=requested.update(p,request);
  check(inhibited.fire_feasible && !inhibited.fire,"operator request gate bypassed");
}
void controlRejections() {
  FireControlConfig config; config.shoot_enable=true;
  config.aim_settle_s=config.idle_duration_s=0;
  ControlStage stage(config);
  auto p=firePlan(0); auto c=fireInput(0,p);
  c.actual_pitch_rad=NAN;
  check(stage.update(p,c).status==Status::InvalidFeedback,"missing feedback allowed firing");
  c=fireInput(0,p); c.feedback_at=stamp(-.2);
  check(stage.update(p,c).status==Status::InvalidFeedback,"stale gimbal feedback accepted");
  c=fireInput(0,p); c.actual_yaw_rad+=.1;
  check(!stage.update(p,c).fire,"misaligned gimbal allowed firing");
  auto fake=p; fake.yaw+=.1; fake.using_MPC=true;
  L4Planning::AimSample sample; sample.execute_time=fake.generated_at; sample.yaw=fake.yaw;
  fake.samples.push_back(sample); c=fireInput(0,fake); c.actual_yaw_rad=fake.yaw;
  check(!stage.update(fake,c).fire,"MPC lag was mistaken for ballistic alignment");
  c=fireInput(.2,p);
  check(!stage.update(p,c).found,"stale plan accepted");
  p=firePlan(.21); p.center_hold=true; p.fire_permitted=false;
  check(!stage.update(p,fireInput(.21,p)).fire,"center hold allowed firing");
  ControlStage jumped(config);
  p=firePlan(0); (void)jumped.update(p,fireInput(0,p));
  p=firePlan(.01); p.yaw+=.4; p.ballistic_yaw_rad=p.yaw;
  check(jumped.update(p,fireInput(.01,p)).status==Status::CommandJump,
        "command discontinuity allowed immediate firing");
  config.aim_settle_s=.03;
  ControlStage interrupted(config);
  p=firePlan(0); (void)interrupted.update(p,fireInput(0,p));
  p=firePlan(.2);
  check(!interrupted.update(p,fireInput(.2,p)).fire,"missing feedback counted as stable alignment");
  Command command; command.found=true; command.pitch_rad=.3; command.pitch_rate_rad_s=.2;
  command.pitch_acceleration_rad_s2=.4;
  const auto serial=ControlStage::makeSerialCommand(command);
  check(serial && serial->pitch==-.3 && serial->pitch_velocity==-.2 && serial->pitch_acceleration==-.4,
        "serial pitch/feedforward sign mismatch");
  command.fire=true; command.found=false;
  check(!ControlStage::makeSerialCommand(command),"invalid command reached serial adapter");
}
}
int main() {
  try { prediction(); selectionAndWarmup(); invalidInputs(); mpc(); firing(); controlRejections();
    std::cout<<"rune_planning_control_smoke passed\n"; }
  catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
