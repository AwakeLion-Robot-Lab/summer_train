#include "l4_planning/predictor.hpp"

namespace L4Planning {

L3Estimation::TargetState Predictor::predict(const L3Estimation::TargetState& target, double dt) const
{
  auto predicted = target;
  predicted.center += target.velocity * dt;
  return predicted;
}

}  // namespace L4Planning
