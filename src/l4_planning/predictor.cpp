#include "l4_planning/predictor.hpp"

namespace L4Planning {

L3Estimation::TargetState Predictor::predict(const L3Estimation::TargetState& target, double dt) const
{
  auto predicted = target;
  predicted.position.x += target.velocity.x * dt;
  predicted.position.y += target.velocity.y * dt;
  predicted.position.z += target.velocity.z * dt;
  return predicted;
}

}  // namespace L4Planning
