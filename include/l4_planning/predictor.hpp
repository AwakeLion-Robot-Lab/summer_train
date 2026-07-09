#pragma once

#include "l3_estimation/target_estimator.hpp"

namespace L4Planning {

class Predictor {
public:
  L3Estimation::TargetState predict(const L3Estimation::TargetState& target, double dt) const;
};

}  // namespace L4Planning
