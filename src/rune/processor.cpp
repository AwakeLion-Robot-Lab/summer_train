#include "rune/processor.hpp"

#include <utility>
#include <algorithm>

namespace Rune {

Processor::Processor(Geometry geometry, EstimatorConfig estimator,
                     PlanningConfig planning,
                     std::unique_ptr<IBigMotionModel> big_model,
                     std::optional<FireControlConfig> fire)
  : building_(geometry), filter_(geometry, estimator.filter),
    estimator_(estimator, std::move(big_model)),
    planning_(planning),
    control_(fire.value_or(FireControlConfig{
      .shoot_enable=planning.shoot_enable,
      .max_yaw_error_rad=planning.max_yaw_error_rad,
      .max_pitch_error_rad=planning.max_pitch_error_rad})) {}

void Processor::reset() {
  building_.reset();
  filter_.reset();
  estimator_.reset();
  planning_.reset();
  control_.reset();
  previous_mode_.reset();
  last_motion_.reset();
  last_plan_.reset();
}

Command Processor::process(const Frame& frame, const ControlInput& control) {
  Command result;
  result.mode = frame.mode;
  last_motion_.reset();
  last_plan_.reset();
  const auto lost = [&] {
    building_.reset(); filter_.reset(); estimator_.reset();
    planning_.reset(); control_.suspend();
  };
  if (previous_mode_ && *previous_mode_ != frame.mode) lost();
  previous_mode_ = frame.mode;
  const auto regions = detection_.runAll(frame);
  if (regions.empty() || std::none_of(regions.begin(), regions.end(),
      [](const auto& region) { return region.keypoints.class_id == 0; })) {
    lost(); return result;
  }
  std::vector<Observation> observations;
  for (const auto& region : regions)
    if (auto observation = observation_.run(region, frame.color, frame.mode))
      observations.push_back(std::move(*observation));
  if (observations.empty() || std::none_of(observations.begin(), observations.end(),
      [](const auto& observation) { return observation.keypoints.class_id == 0; })) {
    lost(); result.status = Status::NoContour; return result;
  }
  const auto pose = building_.run(observations, frame);
  if (!pose) { lost(); result.status = Status::PoseFailed; return result; }
  const auto filtered=filter_.update(*pose,observations,frame);
  if (!filtered) { lost(); result.status=Status::PoseFailed; return result; }
  last_motion_=estimator_.update(*filtered, frame.mode);
  last_plan_=planning_.plan(*last_motion_, frame.mode, control);
  return control_.update(*last_plan_,control);
}

}  // namespace Rune
