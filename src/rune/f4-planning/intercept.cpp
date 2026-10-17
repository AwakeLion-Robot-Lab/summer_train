#include "rune/f4-planning/planning_detail.hpp"

#include <Eigen/Geometry>
#include <cmath>

namespace Rune::detail {

std::optional<Intercept> intercept(
    const Motion& motion, const Eigen::Vector3d& point,
    const Eigen::Vector3d& center, const Eigen::Vector3d& normal,
    bool center_hold, const ControlInput& control, const PlanningConfig& config,
    double command_offset_s) {
  const double age=std::chrono::duration<double>(control.command_at-motion.pose.captured_at).count();
  const double delay=age+control.fire_delay_s+command_offset_s;
  double flight=point.norm()/control.bullet_speed_m_s;
  if (!std::isfinite(flight) || flight<=0 || !std::isfinite(delay)) return {};
  for (int i=0;i<config.max_iterations;++i) {
    const double horizon=delay+flight;
    if (std::abs(horizon)>config.max_prediction_s) return {};
    Eigen::Vector3d future=point;
    if (!center_hold) {
      std::optional<double> phase;
      if (horizon>=0)
        phase=motion.phase_delta_rad ? motion.phase_delta_rad(horizon)
          : std::optional<double>(motion.angular_speed_rad_s*horizon);
      else if (motion.phase_delta_signed_rad) phase=motion.phase_delta_signed_rad(horizon);
      else if (motion.model==MotionModel::Linear) phase=motion.angular_speed_rad_s*horizon;
      if (!phase || !std::isfinite(*phase)) return {};
      future=center+Eigen::AngleAxisd(*phase,normal)*(point-center);
    }
    const auto ballistic=solveBallistic(future,control.bullet_speed_m_s,
      config.gravity_m_s2,config.enable_air_resistance,config.linear_drag_coefficient);
    if (!ballistic) return {};
    if (std::abs(ballistic->flight_time-flight)<config.flight_time_tolerance_s)
      return Intercept{*ballistic,future,horizon};
    flight=ballistic->flight_time;
  }
  return {};
}

} // namespace Rune::detail
