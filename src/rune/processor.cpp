#include "rune/f6_telemetry/processor.hpp"
#include "rune/f1-observation/target_gap_refiner.hpp"
#include "geometry/pose_projection.hpp"
#include "f1-observation/ring_reobservation.hpp"

#include <algorithm>
#include <cmath>
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
                     std::optional<FireControlConfig> fire)
  : follow_only_(!(fire ? fire->shoot_enable : planning.shoot_enable)),
    building_(geometry, true, follow_only_),
    filter_(geometry, estimator.filter, follow_only_),
    estimator_(estimator, std::move(big_model)),
    planning_(planning),
    control_(fire.value_or(FireControlConfig{
      .shoot_enable=planning.shoot_enable,
      .max_yaw_error_rad=planning.max_yaw_error_rad,
      .max_pitch_error_rad=planning.max_pitch_error_rad})),
    continuity_gap_s_(std::min({estimator.max_gap_s, estimator.filter.max_gap_s,
                               geometry.plane_window_max_gap_s})),
    radius_m_(geometry.radius_m) {}

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
  if (follow_only_ && (!validGimbalPose(frame) ||
      !frame.T_barrel_camera || frame.camera_matrix.empty())) {
    resetTracking();
    result.status=Status::InvalidFeedback;
    result.single_shoot=false;
    return result;
  }
  result.status = observe(frame);
  if (follow_only_) return followGimbal(std::move(result), frame, control);
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
    if (last_observation_time_==TimePoint{} || gap<=0 || gap>continuity_gap_s_) {
      resetTracking();
      return;
    }
    if (!follow_only_) {
      control_.missObservation();
      return;
    }
    if (!validGimbalPose(frame)) return;
    // 先利用三维先验投影在原图主动再观测靶面8个关键点，缺失时再回退单靶环重检。
    bool updated_by_gaps = false;
    const int active_sector = filter_.targetSector();
    if (active_sector >= 0) {
      for (const auto& prior : filter_.predictTargetPriors(frame)) {
        if (prior.sector != active_sector) continue;
        const auto reobs = refineTargetGapsFromImage(
            frame.bgr, frame.color, prior.projected_ellipse, prior.outward_direction,
            prior.homography_target_to_image, &prior.gap_search_sigma_px, false);
        if (reobs.measured_spokes >= 2 && (reobs.measured_spokes + reobs.measured_gaps) >= 4) {
          updated_by_gaps = filter_.updateReobservedTargetGaps(
              frame, prior.sector, reobs.gap_points, reobs.gap_point_variances);
        }
        break;
      }
    }
    if (!updated_by_gaps) filter_.updateReobservedPhase(frame);
  };
  const auto regions = detection_.runAll(frame);
  const auto usable = [this](const auto& item) {
    return follow_only_ || item.keypoints.class_id == 0;
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
  // 利用上一帧七状态三维先验单应性矩阵引导原图亚像素精修与缺失槽位主动补测。
  if (filter_.hasHistory() && !frame.bgr.empty() && frame.bgr.type() == CV_8UC3) {
    const auto priors = filter_.predictTargetPriors(frame);
    for (auto& obs : observations) {
      if (obs.keypoints.class_id != 0 || !obs.keypoints.image_contours ||
          obs.keypoints.target_ring_status == TargetRingStatus::Rejected) continue;
      const auto center = obs.keypoints.armorCenter();
      if (!center || !obs.keypoints.image_contours->target_ellipse) continue;
      const auto& ellipse = *obs.keypoints.image_contours->target_ellipse;
      const double diameter = std::max(ellipse.size.width, ellipse.size.height);
      const FeatureFilter::TargetPriorProjection* matched_prior = nullptr;
      double best_dist = std::max(14.0, 0.35 * diameter);
      for (const auto& prior : priors) {
        const double d = cv::norm(prior.center_pixel - *center);
        if (d < best_dist) {
          best_dist = d;
          matched_prior = &prior;
        }
      }
      if (!matched_prior) continue;
      const auto refined = refineTargetGapsFromImage(
          frame.bgr, frame.color, ellipse, matched_prior->outward_direction,
          matched_prior->homography_target_to_image, &matched_prior->gap_search_sigma_px, false);
      if (refined.measured_spokes < 2) continue;
      auto updated = std::make_shared<DetectedContours>(*obs.keypoints.image_contours);
      bool changed = false;
      for (std::size_t k = 0; k < 8; ++k) {
        if (!refined.gap_points[k]) continue;
        const double gate = std::max(6.0, 3.5 * matched_prior->gap_search_sigma_px[k]);
        if (cv::norm(*refined.gap_points[k] - matched_prior->gap_pixels[k]) > gate) continue;
        if (!updated->gap_points[k]) {
          updated->gap_points[k] = refined.gap_points[k];
          updated->gap_point_variances[k] = refined.gap_point_variances[k];
          changed = true;
        } else {
          const double d = cv::norm(*refined.gap_points[k] - *updated->gap_points[k]);
          if (d <= std::max(3.5, 0.08 * diameter)) {
            const double v_old = updated->gap_point_variances[k] > 0.0F
                ? double(updated->gap_point_variances[k]) : 4.0;
            const double v_new = refined.gap_point_variances[k] > 0.0F
                ? double(refined.gap_point_variances[k]) : 4.0;
            const double w_new = v_old / (v_old + v_new);
            updated->gap_points[k] = float(w_new) * (*refined.gap_points[k]) +
                                     float(1.0 - w_new) * (*updated->gap_points[k]);
            updated->gap_point_variances[k] = float(std::max(0.36, (v_old * v_new) / (v_old + v_new)));
            changed = true;
          }
        }
      }
      if (changed) obs.keypoints.image_contours = std::move(updated);
    }
  }
  auto pose = building_.run(observations, frame);
  if (!pose) {
    filter_.noteObservationGap();
    if (follow_only_ && building_.hasHistory()) missed();
    else resetTracking();
    return Status::PoseFailed;
  }
  if (follow_only_) {
    for (const auto& blade : pose->blades) {
      auto& target = observations[blade.observation_index].keypoints;
      if (const auto ring = detail::reobserveInactiveArrowRing(frame, *pose, target, radius_m_)) {
        const auto delta = ring->ellipse.center - *target.armorCenter();
        target.top += delta;
        target.left += delta;
        target.right += delta;
        target.bottom += delta;
        target.target_variance_scale = std::max(target.target_variance_scale, 1 / ring->arc_coverage);
        if (target.image_contours) {
          auto updated = std::make_shared<DetectedContours>(*target.image_contours);
          updated->target_ellipse = ring->ellipse;
          updated->gap_points = {};
          updated->gap_point_variances = {};
          updated->gap_display_predictions = {};
          const cv::Point2f outward = ring->ellipse.center - target.center_r;
          if (target.point_valid[4] && cv::norm(outward) >= 2.0) {
            const auto reobs = refineTargetGapsFromImage(
                frame.bgr, frame.color, ring->ellipse, outward, std::nullopt, nullptr, false);
            if (reobs.measured_spokes >= 2) {
              const double coverage_scale = std::max(1.0, 1.0 / ring->arc_coverage);
              for (std::size_t k = 0; k < updated->gap_points.size(); ++k) {
                if (!reobs.gap_points[k]) continue;
                updated->gap_points[k] = reobs.gap_points[k];
                const double base_var = reobs.gap_point_variances[k] > 0.0F
                    ? double(reobs.gap_point_variances[k]) : 4.0;
                updated->gap_point_variances[k] = float(base_var * coverage_scale);
              }
            }
          }
          target.image_contours = std::move(updated);
        }
        ++pose->arrow_refined_targets;
      }
    }
  }
  const auto filtered=filter_.update(*pose,observations,frame);
  if (!filtered) {
    if (follow_only_ && filter_.hasHistory()) missed();
    else resetTracking();
    return Status::PoseFailed;
  }
  last_observation_time_=frame.captured_at;
  last_motion_=estimator_.update(*filtered, frame.mode);
  return last_motion_->ready ? Status::Ready : Status::MotionNotReady;
}

}  // namespace Rune
