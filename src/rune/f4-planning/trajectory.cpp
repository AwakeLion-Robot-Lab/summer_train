#include "rune/f4-planning/planning_detail.hpp"
#include "armor/l3_planning/tiny_mpc.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <utility>

namespace Rune::detail {

void optimizeTrajectory(AimPlan& plan, const Motion& motion,
    const Eigen::Vector3d& point, const Eigen::Vector3d& normal,
    const ControlInput& control, const PlanningConfig& config) {
  using namespace L4Planning;
  constexpr double dt=kTinyMpcStepSeconds;
  const auto angles=[&](double offset)->std::optional<Eigen::Vector2d> {
    if (offset==0) return Eigen::Vector2d(plan.ballistic_yaw_rad,plan.ballistic_pitch_rad);
    const auto future=intercept(motion,point,plan.center_aim,normal,plan.center_hold,
                               control,config,offset);
    if (!future) return {};
    return Eigen::Vector2d(future->ballistic.yaw+config.yaw_offset_rad,
                          future->ballistic.pitch+config.pitch_offset_rad);
  };
  // Direct fallback still carries useful feedforward, even if the full MPC
  // reference cannot be built (for example an external forward-only model).
  const auto next=angles(dt),after=angles(2*dt);
  if (next && after) {
    const double y1=std::remainder(next->x()-plan.ballistic_yaw_rad,2*std::numbers::pi);
    const double y2=y1+std::remainder(after->x()-next->x(),2*std::numbers::pi);
    plan.yaw_rate=(4*y1-y2)/(2*dt);
    plan.pitch_rate=(-3*plan.ballistic_pitch_rad+4*next->y()-after->y())/(2*dt);
    plan.yaw_acceleration=(y2-2*y1)/(dt*dt);
    plan.pitch_acceleration=(after->y()-2*next->y()+plan.ballistic_pitch_rad)/(dt*dt);
  }
  if (!config.mpc.enable_mpc || plan.center_hold) return;
  plan.mpc_diagnostics.status=MpcStatus::InvalidReference;
  TinyMpcReference reference;
  reference.yaw_origin=plan.ballistic_yaw_rad;
  double previous_yaw=reference.yaw_origin;
  for (std::size_t i=0;i<kTinyMpcHorizon;++i) {
    const double offset=(static_cast<double>(i)-kTinyMpcControlIndex)*dt;
    const auto sample=angles(offset);
    if (!sample) return;
    const double delta=std::remainder(sample->x()-previous_yaw,2*std::numbers::pi);
    reference.yaw.angle[i]=(i ? reference.yaw.angle[i-1] : 0)+delta;
    reference.pitch.angle[i]=sample->y();
    previous_yaw=sample->x();
  }
  const auto differentiate=[](TinyMpcAxisReference& axis) {
    axis.angular_velocity[0]=(axis.angle[1]-axis.angle[0])/kTinyMpcStepSeconds;
    for (std::size_t i=1;i+1<kTinyMpcHorizon;++i)
      axis.angular_velocity[i]=(axis.angle[i+1]-axis.angle[i-1])/(2*kTinyMpcStepSeconds);
    axis.angular_velocity.back()=(axis.angle.back()-axis.angle[kTinyMpcHorizon-2])/kTinyMpcStepSeconds;
  };
  differentiate(reference.yaw); differentiate(reference.pitch);
  reference.valid=true;
  const auto solution=solveTinyMpc(reference,config.mpc);
  auto& diagnostic=plan.mpc_diagnostics;
  diagnostic.status=MpcStatus::NotConverged;
  diagnostic.yaw_iterations=solution.yaw.info.iteration_count;
  diagnostic.pitch_iterations=solution.pitch.info.iteration_count;
  diagnostic.yaw_primal_residual=solution.yaw.info.primal_residual;
  diagnostic.pitch_primal_residual=solution.pitch.info.primal_residual;
  diagnostic.yaw_dual_residual=solution.yaw.info.dual_residual;
  diagnostic.pitch_dual_residual=solution.pitch.info.dual_residual;
  auto samples=makeTinyMpcSamples(reference,solution,control.command_at);
  if (samples.empty()) return;
  diagnostic.status=MpcStatus::Applied;
  plan.samples=std::move(samples); plan.using_MPC=true;
  const auto& current=plan.samples.front();
  plan.yaw=current.yaw; plan.pitch=current.pitch;
  plan.yaw_rate=current.yaw_rate; plan.pitch_rate=current.pitch_rate;
  plan.yaw_acceleration=current.yaw_acceleration;
  plan.pitch_acceleration=current.pitch_acceleration;
}

} // namespace Rune::detail
