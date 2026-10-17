#include "rune/f3-estimator/estimator.hpp"

namespace Rune {

Motion EstimatorStage::updateSmallMotion(const Pose& pose) {
  const auto linear=fitter_.fitLinear();
  Motion motion;
  if (linear) motion=fittedMotion(pose,*linear,MotionModel::Linear);
  else motion.pose=pose;
  motion.linear_cost_rad2=linear ? linear->cost : std::numeric_limits<double>::infinity();
  // Small rune uses constant angular speed and never fits a sinusoid.
  motion.sine_cost_rad2=std::numeric_limits<double>::infinity();
  return motion;
}

} // namespace Rune
