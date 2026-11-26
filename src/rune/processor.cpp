#include "rune/f6_telemetry/processor.hpp"
#include "geometry/pose_projection.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Rune {
namespace {
bool validGimbalPose(const Frame& frame) {
  const auto& attitude=frame.q_world_barrel;
  if (attitude && (!attitude->coeffs().allFinite() || std::abs(attitude->squaredNorm()-1)>1e-3))
    return false;
  const auto& world_pose=frame.T_world_barrel;
  if (!world_pose) return attitude.has_value();
  return tools::geometry::rigid(*world_pose) && (!attitude ||
      (attitude->normalized().toRotationMatrix()-world_pose->linear()).norm()<=1e-3);
}
} // namespace

Processor::Processor(Geometry geometry, EstimatorConfig estimator,
                     PlanningConfig planning,
                     std::unique_ptr<IBigMotionModel> big_model,
                     std::optional<FireControlConfig> fire, bool gimbal_only)
  : building_(geometry, !(fire ? fire->shoot_enable : planning.shoot_enable), gimbal_only),
    filter_(geometry, estimator.filter, gimbal_only),
    estimator_(estimator, std::move(big_model)),
    planning_(planning),
    control_(fire.value_or(FireControlConfig{
      .shoot_enable=planning.shoot_enable,
      .max_yaw_error_rad=planning.max_yaw_error_rad,
      .max_pitch_error_rad=planning.max_pitch_error_rad})),
    continuity_gap_s_(std::min({estimator.max_gap_s, estimator.filter.max_gap_s,
                               geometry.plane_window_max_gap_s})),
    gimbal_only_(gimbal_only) {
  if (gimbal_only_ && (planning.shoot_enable || (fire && fire->shoot_enable)))
    throw std::invalid_argument("gimbal-only rune tracking requires disabled firing output");
}

void Processor::resetTracking() {
  building_.reset();
  filter_.reset();
  estimator_.reset();
  planning_.reset();
  control_.suspend();
  last_observation_time_={};
  last_motion_.reset();
  last_plan_.reset();
  follow_center_=false;
}

void Processor::reset() {
  resetTracking();
  control_.reset();
  previous_mode_.reset();
  previous_color_.reset();
  previous_frame_time_={};
  gimbal_hold_.reset();
}

Command Processor::process(const Frame& frame, const ControlInput& control) {
  Command result;
  result.mode = frame.mode;
  if (gimbal_only_ && (!validGimbalPose(frame) ||
      !frame.T_barrel_camera || frame.camera_matrix.empty())) {
    resetTracking();
    result.status=Status::InvalidFeedback;
    result.single_shoot=false;
    return result;
  }
  result.status = observe(frame);
  if (gimbal_only_) return followGimbal(std::move(result), frame, control);
  if (!last_motion_) {
    const int sector=control_.predictionSector();
    const double age=std::chrono::duration<double>(control.command_at-last_observation_time_).count();
    if (sector>=0 && age<=continuity_gap_s_) {
      if (const auto prediction=filter_.predictTarget(frame,control.command_at,sector)) {
        return control_.followPrediction(result.mode,result.status,prediction->observed_direction,
                                         prediction->predicted_direction,control);
      }
    }
    return result;
  }
  last_plan_ = planning_.plan(*last_motion_, frame.mode, control);
  return control_.update(*last_plan_, control);
}

Command Processor::followGimbal(Command result, const Frame& frame, const ControlInput& control) {
  result.single_shoot=false;
  if (last_motion_) follow_center_=last_motion_->pose.activated;
  const double age=std::chrono::duration<double>(control.command_at-last_observation_time_).count();
  if (last_observation_time_==TimePoint{} || age<0 || age>continuity_gap_s_) return result;
  const auto prediction=filter_.predictGimbal(frame,control.command_at,follow_center_);
  if (!prediction) return result;
  const auto& direction=prediction->direction;
  const double horizontal=std::hypot(direction.x(),direction.y());
  if (horizontal<1e-6) return result;
  result.yaw_rad=std::atan2(direction.y(),direction.x());
  result.pitch_rad=std::atan2(direction.z(),horizontal);
  result.target_sector=prediction->sector;
  result.center_hold=follow_center_;
  result.found=true;
  if (last_motion_) result.status=Status::Ready;
  gimbal_hold_=ControlStage::makeSerialCommand(result);
  return result;
}

Status Processor::observe(const Frame& frame) {
  last_motion_.reset();
  last_plan_.reset();
  const bool invalid_image=frame.bgr.empty() || frame.bgr.type()!=CV_8UC3;
  const bool invalid_time=frame.captured_at==TimePoint{} ||
      (previous_frame_time_!=TimePoint{} && frame.captured_at<=previous_frame_time_);
  if ((previous_mode_ && *previous_mode_ != frame.mode) ||
      (previous_color_ && *previous_color_ != frame.color) ||
      invalid_image || invalid_time) resetTracking();
  previous_mode_ = frame.mode;
  previous_color_=frame.color;
  previous_frame_time_=frame.captured_at;
  if (invalid_image || invalid_time) return Status::NoDetection;
  // 从最后一帧有效观测计算允许保留状态的时间。
  const auto missed = [&] {
    const double gap=std::chrono::duration<double>(frame.captured_at-last_observation_time_).count();
    if (last_observation_time_==TimePoint{} || gap<=0 || gap>continuity_gap_s_) resetTracking();
    else if (!gimbal_only_) control_.missObservation();
    else if (validGimbalPose(frame)) filter_.updateReobservedPhase(frame);
  };
  const auto regions = detection_.runAll(frame);
  const auto usable = [this](const auto& item) {
    return gimbal_only_ || item.keypoints.class_id == 0;
  };
  if (std::none_of(regions.begin(), regions.end(), usable)) {
    filter_.noteObservationGap();
    missed(); return Status::NoDetection;
  }
  std::vector<Observation> observations;
  for (const auto& region : regions) {
    // 靶环校验失败视为缺少图像观测，沿用短时漏检的处理。
    if (region.keypoints.target_ring_status==TargetRingStatus::Rejected) continue;
    if (auto observation = observation_.run(region, frame.color, frame.mode))
      observations.push_back(std::move(*observation));
  }
  if (std::none_of(observations.begin(), observations.end(), usable)) {
    filter_.noteObservationGap();
    missed(); return Status::NoContour;
  }
  const auto pose = building_.run(observations, frame);
  if (!pose) {
    filter_.noteObservationGap();
    if (gimbal_only_ && building_.hasHistory()) missed();
    else resetTracking();
    return Status::PoseFailed;
  }
  const auto filtered=filter_.update(*pose,observations,frame);
  if (!filtered) {
    if (gimbal_only_ && filter_.hasHistory()) missed();
    else resetTracking();
    return Status::PoseFailed;
  }
  last_observation_time_=frame.captured_at;
  last_motion_=estimator_.update(*filtered, frame.mode);
  return last_motion_->ready ? Status::Ready : Status::MotionNotReady;
}

}  // namespace Rune
