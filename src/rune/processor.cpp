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
      .max_pitch_error_rad=planning.max_pitch_error_rad})),
    continuity_gap_s_(std::min({estimator.max_gap_s, estimator.filter.max_gap_s,
                               geometry.plane_window_max_gap_s})) {}

void Processor::reset() {
  building_.reset();
  filter_.reset();
  estimator_.reset();
  planning_.reset();
  control_.reset();
  previous_mode_.reset();
  previous_color_.reset();
  previous_frame_time_={}; last_observation_time_={};
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
    last_observation_time_={};
  };
  const bool invalid_image=frame.bgr.empty() || frame.bgr.type()!=CV_8UC3;
  const bool invalid_time=frame.captured_at==TimePoint{} ||
      (previous_frame_time_!=TimePoint{} && frame.captured_at<=previous_frame_time_);
  if ((previous_mode_ && *previous_mode_ != frame.mode) ||
      (previous_color_ && *previous_color_ != frame.color) ||
      invalid_image || invalid_time) lost();
  previous_mode_ = frame.mode;
  previous_color_=frame.color;
  previous_frame_time_=frame.captured_at;
  if (invalid_image || invalid_time) return result;
  // Missing image evidence is not a measurement. Stop the current command and
  // firing alignment immediately, but let a brief blink retain estimator history.
  // Compare against the last SUCCESSFUL observation, so repeated empty frames
  // cannot extend this timeout indefinitely.
  const auto missed = [&] {
    const double gap=std::chrono::duration<double>(frame.captured_at-last_observation_time_).count();
    if (last_observation_time_==TimePoint{} || gap<=0 || gap>continuity_gap_s_) lost();
    else control_.suspend();
  };
  const auto regions = detection_.runAll(frame);
  if (regions.empty() || std::none_of(regions.begin(), regions.end(),
      [](const auto& region) { return region.keypoints.class_id == 0; })) {
    missed(); return result;
  }
  std::vector<Observation> observations;
  for (const auto& region : regions)
    if (auto observation = observation_.run(region, frame.color, frame.mode))
      observations.push_back(std::move(*observation));
  if (observations.empty() || std::none_of(observations.begin(), observations.end(),
      [](const auto& observation) { return observation.keypoints.class_id == 0; })) {
    missed(); result.status = Status::NoContour; return result;
  }
  const auto pose = building_.run(observations, frame);
  if (!pose) { lost(); result.status = Status::PoseFailed; return result; }
  const auto filtered=filter_.update(*pose,observations,frame);
  if (!filtered) { lost(); result.status=Status::PoseFailed; return result; }
  last_observation_time_=frame.captured_at;
  last_motion_=estimator_.update(*filtered, frame.mode);
  last_plan_=planning_.plan(*last_motion_, frame.mode, control);
  return control_.update(*last_plan_,control);
}

}  // namespace Rune
