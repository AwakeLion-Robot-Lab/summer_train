// Algorithm reference: SZURPVision/RP-26Rune, PowerRunePlane.
// See README.md and RP-26Rune-LICENSE in this directory.
// R/inactive-seed layout selection also references Alliance-Algorithm/
// rmcs_auto_aim_v2 (MIT); see ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f2-building/builder.hpp"
#include "rune/f2-building/builder_detail.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

namespace Rune {
using namespace building_detail;

void BuildingStage::reset() {
  plane_window_.reset(); previous_.reset(); previous_frame_.reset();
}

std::optional<Pose> BuildingStage::run(const Observation& observation,const Frame& frame) {
  return run(std::vector<Observation>{observation},frame);
}

std::optional<Pose> BuildingStage::run(const std::vector<Observation>& observations,const Frame& frame) {
  const auto fail=[&]() -> std::optional<Pose> { reset(); return {}; };
  try {
    const auto& g=geometry_;
    if (!validGeometry(g) || observations.empty() || observations.size()>32) return fail();
    std::optional<Eigen::Matrix3d> orientation;
    if (!worldRotation(frame,orientation)) return fail();
    if (previous_frame_) {
      const auto& before=*previous_frame_;
      const double dt=std::chrono::duration<double>(frame.captured_at-before.captured_at).count();
      const auto sameMat=[](const cv::Mat& a,const cv::Mat& b) {
        if (a.empty() || b.empty()) return a.empty() && b.empty();
        return a.size()==b.size() && a.type()==b.type() && cv::norm(a,b,cv::NORM_INF)<1e-12;
      };
      if (dt<=0 || dt>g.plane_window_max_gap_s || frame.mode!=before.mode || frame.color!=before.color ||
          frame.world_frame_id!=before.world_frame_id ||
          bool(frame.T_world_barrel)!=bool(before.T_world_barrel) ||
          bool(frame.q_world_barrel)!=bool(before.q_world_barrel) ||
          !sameMat(frame.camera_matrix,before.camera_matrix) || !sameMat(frame.distortion,before.distortion) ||
          !frame.T_barrel_camera || !before.T_barrel_camera ||
          (frame.T_barrel_camera->matrix()-before.T_barrel_camera->matrix()).norm()>1e-12) reset();
    }
    std::size_t reference=observations.size();
    double best_score=-1e30;
    for (std::size_t i=0;i<observations.size();++i) {
      const auto& o=observations[i];
      if (o.keypoints.class_id!=0 || !validObservation(o,false)) continue;
      const double score=o.keypoints.confidence;
      if (score>best_score) { best_score=score; reference=i; }
    }
    // Preserve the old single-small-activated diagnostic API; Processor itself
    // requires an inactive target before invoking the solver.
    if (reference==observations.size() && observations.size()==1 && frame.mode==Mode::Small &&
        observations[0].keypoints.class_id==1) reference=0;
    if (reference==observations.size()) return fail();
    // In big rune, confidence order may change between the two inactive blades.
    // Retain the previous physical target when it remains near its last position.
    if (previous_ && previous_frame_ && frame.T_world_barrel && previous_frame_->T_world_barrel && frame.T_barrel_camera) {
      const Eigen::Vector3d target_world=*previous_frame_->T_world_barrel*previous_->target_barrel;
      const Eigen::Vector3d camera_point=frame.T_barrel_camera->inverse()*frame.T_world_barrel->inverse()*target_world;
      if (camera_point.z()>0) {
        cv::Mat camera,distortion;
        if (!validCamera(frame,camera,distortion)) return fail();
        const auto pixel=project(std::vector<cv::Point3d>{{camera_point.x(),camera_point.y(),camera_point.z()}},
                                 Parameters::Zero(),camera,distortion)[0];
        double nearest=1e30;
        for (std::size_t i=0;i<observations.size();++i) {
          const auto& o=observations[i];
          if (o.keypoints.class_id!=0 || !validObservation(o,false)) continue;
          const auto center=(cv::Point2d(o.keypoints.top)+cv::Point2d(o.keypoints.left)+
                             cv::Point2d(o.keypoints.right)+cv::Point2d(o.keypoints.bottom))*0.25;
          const double radius=cv::norm(center-cv::Point2d(o.keypoints.center_r));
          const double distance=cv::norm(center-pixel)/std::max(2.0,radius);
          if (distance<0.5 && distance<nearest) { nearest=distance; reference=i; }
        }
      }
    }
    // Try every inactive seed and distinct R hypothesis on initialization.
    // Compare their full five-blade layouts before trusting NN confidence.
    // Each trial starts from the SAME plane window: rejected PnP hypotheses
    // must not contaminate temporal geometry history.
    const PlaneWindow original_window=plane_window_;
    PlaneWindow selected_window=original_window;
    std::optional<Pose> pose;
    double selected_confidence=-1;
    const auto inactive_count=[&](const Pose& p) {
      return std::count_if(p.blades.begin(),p.blades.end(),[](const auto& b){return !b.activated;});
    };
    std::vector<std::size_t> seeds;
    for (std::size_t i=0;i<observations.size();++i)
      if (observations[i].keypoints.class_id==0 && validObservation(observations[i],false)) seeds.push_back(i);
    if (seeds.empty()) seeds.push_back(reference); // legacy activated diagnostic
    std::stable_sort(seeds.begin(),seeds.end(),[&](auto a,auto b) {
      return observations[a].keypoints.confidence>observations[b].keypoints.confidence;
    });
    if (previous_) {
      const auto it=std::find(seeds.begin(),seeds.end(),reference);
      if (it!=seeds.end()) std::rotate(seeds.begin(),it,it+1);
    }
    std::vector<std::size_t> icons;
    for (std::size_t i=0;i<observations.size();++i) {
      const auto& pixel=observations[i].keypoints.center_r;
      if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) continue;
      if (std::none_of(icons.begin(),icons.end(),[&](auto j) {
        return cv::norm(pixel-observations[j].keypoints.center_r)<2;
      })) icons.push_back(i);
    }
    unsigned trials=0;
    for (const auto seed : seeds) {
      for (const auto icon : icons) {
        if (++trials>64) break;
        auto hypothesis=observations;
        hypothesis[seed].keypoints.center_r=observations[icon].keypoints.center_r;
        hypothesis[seed].center_r=observations[icon].center_r;
        plane_window_=original_window;
        auto candidate=solve(hypothesis,frame,seed);
        if (!candidate) continue;
        const double error=candidate->reprojection_error_px+candidate->chamfer_error_px;
        const double best_error=pose ? pose->reprojection_error_px+pose->chamfer_error_px : 1e30;
        const bool same_layout=pose && inactive_count(*candidate)==inactive_count(*pose) &&
          candidate->blades.size()==pose->blades.size();
        const double confidence=observations[seed].keypoints.confidence;
        if (!pose || inactive_count(*candidate)>inactive_count(*pose) ||
            (inactive_count(*candidate)==inactive_count(*pose) && candidate->blades.size()>pose->blades.size()) ||
            (same_layout && (error<best_error-0.25 ||
              (std::abs(error-best_error)<=0.25 && confidence>selected_confidence)))) {
          pose=std::move(candidate); selected_window=plane_window_; selected_confidence=confidence;
        }
      }
      // Retain a valid previously selected physical target through confidence
      // reversal. New target selection occurs only after that seed disappears.
      if (previous_ && seed==reference && pose) break;
      if (trials>=64) break;
    }
    plane_window_=selected_window;
    if (!pose) return fail();
    updatePhase(*pose);
    pose->normal_vertical_component=std::abs(pose->normal_barrel.dot(pose->up_barrel));
    if (!std::isfinite(pose->phase_rad) || pose->target_barrel.x()<=0) return fail();
    if (previous_) {
      Eigen::Vector3d previous_radial=previous_->target_barrel-previous_->center_barrel;
      Eigen::Vector3d radial=pose->target_barrel-pose->center_barrel;
      std::optional<Eigen::Matrix3d> old_orientation;
      if (previous_frame_) worldRotation(*previous_frame_,old_orientation);
      if (orientation && old_orientation) { radial=(*orientation)*radial; previous_radial=(*old_orientation)*previous_radial; }
      pose->target_switched=radial.normalized().dot(previous_radial.normalized())<std::cos(std::numbers::pi/5);
    }
    pose->history_reset=!previous_.has_value();
    previous_=*pose;
    // Retain only metadata needed for continuity, never a full image/detection list.
    Frame metadata;
    metadata.captured_at=frame.captured_at; metadata.mode=frame.mode; metadata.color=frame.color;
    metadata.T_barrel_camera=frame.T_barrel_camera; metadata.T_world_barrel=frame.T_world_barrel;
    metadata.q_world_barrel=frame.q_world_barrel; metadata.world_frame_id=frame.world_frame_id;
    metadata.camera_matrix=frame.camera_matrix.clone(); metadata.distortion=frame.distortion.clone();
    previous_frame_=std::move(metadata);
    return pose;
  } catch (const cv::Exception&) { return fail(); }
}
} // namespace Rune
