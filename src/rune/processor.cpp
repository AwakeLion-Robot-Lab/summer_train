#include "rune/processor.hpp"

#include <utility>

namespace Rune {

Processor::Processor(Geometry geometry, EstimatorConfig estimator,
                     PlanningConfig planning,
                     std::unique_ptr<IBigMotionModel> big_model)
  : building_(geometry), estimator_(estimator, std::move(big_model)),
    planning_(planning) {}

void Processor::reset() {
  estimator_.reset();
  planning_.resetFireLatch();
  previous_mode_.reset();
}

Command Processor::process(const Frame& frame, const ControlInput& control) {
  Command result;
  result.mode = frame.mode;
  if (previous_mode_ && *previous_mode_ != frame.mode) reset();
  previous_mode_ = frame.mode;
  const auto region = detection_.run(frame);
  if (!region) return result;
  const auto observation = observation_.run(*region, frame.color);
  if (!observation) { result.status = Status::NoContour; return result; }
  const auto pose = building_.run(*observation, frame);
  if (!pose) { result.status = Status::PoseFailed; return result; }
  return planning_.plan(estimator_.update(*pose, frame.mode), frame.mode, control);
}

}  // namespace Rune
