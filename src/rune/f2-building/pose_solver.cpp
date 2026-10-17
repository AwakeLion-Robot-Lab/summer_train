// Algorithm reference: SZURPVision/RP-26Rune, PowerRunePlane.
// See README.md and RP-26Rune-LICENSE in this directory.
// R/inactive-seed layout selection also references Alliance-Algorithm/
// rmcs_auto_aim_v2 (MIT); see ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f2-building/builder.hpp"
#include "rune/f2-building/builder_detail.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <utility>

namespace Rune {
using namespace building_detail;

std::optional<Pose> BuildingStage::solve(const std::vector<Observation>& observations,
                                       const Frame& frame,std::size_t reference,bool allow_pca) {
  const auto& g=geometry_;
  cv::Mat camera,distortion;
  if (!validGeometry(g) || !validCamera(frame,camera,distortion)) return {};
  std::optional<Eigen::Matrix3d> world_rotation;
  if (!worldRotation(frame,world_rotation)) return {};
  std::optional<Eigen::Vector3d> vertical;
  if (world_rotation) vertical=frame.T_barrel_camera->linear().transpose()*world_rotation->transpose()*Eigen::Vector3d::UnitZ();
  for (const auto& model : models_) {
    if (!model) continue;
    for (const auto& boundary : model->boundaries)
      if (!std::all_of(boundary.begin(),boundary.end(),finite)) return {};
    if (!validPcaBox(model->pca_box)) return {};
  }
  std::vector<Prepared> prepared;
  std::size_t distance_pixels=0;
  std::vector<std::size_t> order(observations.size());
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b) {
    if (a==reference || b==reference) return a==reference && b!=reference;
    const float x=observations[a].keypoints.confidence,y=observations[b].keypoints.confidence;
    return (std::isfinite(x) ? x : -1.0F)>(std::isfinite(y) ? y : -1.0F);
  });
  for (const std::size_t i : order) {
    const auto& o=observations[i];
    const bool arm_only=o.keypoints.class_id==1 && frame.mode==Mode::Big;
    if (!validObservation(o,arm_only)) continue;
    Prepared p; p.index=i;
    const int state=o.keypoints.class_id==0 ? 0 : (frame.mode==Mode::Small ? 1 : 2);
    const auto& model=models_[state];
    if (!arm_only) {
      p.anchors=anchors_; p.pixels=networkPoints(o);
      if (allow_pca && g.use_pca_anchors && model && !model->pca_box.empty()) {
        if (const auto box=pcaEnvelope(o,camera,distortion)) {
          p.anchors=model->pca_box; p.anchors.emplace_back(0,0,0);
          p.pixels=*box; p.pixels.emplace_back(o.keypoints.center_r);
          p.pca=true;
        }
      }
    }
    if (g.refinement_iterations>0 && model) {
      const std::array<const std::vector<cv::Point>*,3> contours{&o.armor,&o.arm,&o.center_r};
      bool usable=true;
      std::size_t local_pixels=0;
      for (std::size_t j=0;j<3;++j) {
        // The hub is shared, not five rotated copies of the R glyph. In joint
        // mode use its shared center anchors; its in-plane orientation is unknown.
        if (j==2 && observations.size()>1) continue;
        if (model->boundaries[j].empty()) continue;
        const auto field=makeField(*contours[j],model->boundaries[j],g);
        if (!field) { usable=false; break; }
        local_pixels+=field->image.distances.total();
        if (distance_pixels+local_pixels>3*1024*1024) { usable=false; break; }
        p.fields.push_back(*field);
      }
      if (!usable) continue;
      distance_pixels+=local_pixels;
    }
    // Activated big blades are arm-only support; never fabricate armor anchors.
    if (p.anchors.empty() && p.fields.empty()) continue;
    prepared.push_back(std::move(p));
    if (prepared.size()==10) break;
  }
  const auto ref=std::find_if(prepared.begin(),prepared.end(),[&](const auto& p){return p.index==reference;});
  if (ref==prepared.end() || ref->anchors.size()<4) return {};
  std::vector<cv::Mat> seeds_r,seeds_t;
  if (!cv::solvePnPGeneric(ref->anchors,ref->pixels,camera,distortion,seeds_r,seeds_t,false,cv::SOLVEPNP_IPPE)) return {};
  Parameters best;
  Evaluation best_eval;
  std::vector<BladeMatch> best_matches;
  bool best_refined=false;
  std::vector<std::array<std::size_t,5>> seen_assignments;
  for (std::size_t seed=0;seed<seeds_r.size();++seed) {
    cv::Vec3d rv,tv;
    seeds_r[seed].copyTo(rv); seeds_t[seed].copyTo(tv);
    Parameters initial; initial<<rv[0],rv[1],rv[2],tv[0],tv[1],tv[2];
    if (!inFront(ref->anchors,initial)) continue;
    const auto R=rotation(initial);
    const cv::Vec3d normal(R(0,2),R(1,2),R(2,2));
    const double numerator=normal.dot(tv);
    const auto center_pixel=project(std::vector<cv::Point3d>{{0,0,0}},initial,camera,distortion)[0];
    std::array<const Prepared*,5> slots{};
    std::array<double,5> scores; scores.fill(1e30);
    slots[0]=&*ref; scores[0]=-1;
    for (const auto& candidate : prepared) {
      if (candidate.index==reference) continue;
      const auto& o=observations[candidate.index];
      if (cv::norm(cv::Point2d(o.keypoints.center_r)-center_pixel)>g.max_center_disagreement_px) continue;
      cv::Point2d target=(cv::Point2d(o.keypoints.top)+cv::Point2d(o.keypoints.left)+
                         cv::Point2d(o.keypoints.right)+cv::Point2d(o.keypoints.bottom))*0.25;
      const bool arm_only=frame.mode==Mode::Big && o.keypoints.class_id==1;
      if (arm_only) {
        const auto moments=cv::moments(o.arm);
        if (std::abs(moments.m00)<1) continue;
        target={moments.m10/moments.m00,moments.m01/moments.m00};
      }
      std::vector<cv::Point2d> ray_points;
      cv::undistortPoints(std::vector<cv::Point2d>{target},ray_points,camera,distortion);
      const cv::Vec3d ray(ray_points[0].x,ray_points[0].y,1);
      const double denominator=normal.dot(ray);
      if (std::abs(denominator)<1e-6 || numerator/denominator<=0) continue;
      const cv::Vec3d point=R.t()*(ray*(numerator/denominator)-tv);
      const double radius=std::hypot(point[0],point[1]);
      if (radius<g.radius_m*(arm_only ? 0.05 : 0.5) || radius>1.5*g.radius_m) continue;
      const double angle=std::atan2(-point[0],point[1]);
      const int sector=(static_cast<int>(std::llround(angle*5/(2*std::numbers::pi)))+5)%5;
      if (sector==0) continue;
      const double error=std::abs(std::remainder(angle-sector*2*std::numbers::pi/5,2*std::numbers::pi));
      if (error>g.max_sector_error_rad) continue;
      const double score=error-0.03*std::clamp(double(o.keypoints.confidence),0.0,1.0);
      // Non-overlapping sector gates make the best-per-sector assignment unique.
      if (score<scores[sector]) { scores[sector]=score; slots[sector]=&candidate; }
    }
    std::array<std::size_t,5> assignment;
    assignment.fill(observations.size());
    for (int sector=0;sector<5;++sector) if (slots[sector]) assignment[sector]=slots[sector]->index;
    if (std::find(seen_assignments.begin(),seen_assignments.end(),assignment)!=seen_assignments.end()) continue;
    seen_assignments.push_back(assignment);
    std::vector<cv::Point3d> anchors;
    std::vector<cv::Point2d> pixels;
    std::vector<Field> fields;
    std::vector<BladeMatch> matches;
    for (int sector=0;sector<5;++sector) {
      const auto* p=slots[sector];
      if (!p) continue;
      const auto rotated=rotatePoints(p->anchors,sector);
      anchors.insert(anchors.end(),rotated.begin(),rotated.end());
      pixels.insert(pixels.end(),p->pixels.begin(),p->pixels.end());
      for (const auto& f : p->fields) {
        Field field=f; field.model=rotatePoints(f.model,sector); fields.push_back(std::move(field));
      }
      matches.push_back({p->index,sector,observations[p->index].keypoints.class_id==1,p->pca});
    }
    std::vector<cv::Mat> rotations,translations;
    if (!cv::solvePnPGeneric(anchors,pixels,camera,distortion,rotations,translations,false,cv::SOLVEPNP_IPPE)) continue;
    for (std::size_t i=0;i<rotations.size();++i) {
      cv::Vec3d r,t; rotations[i].copyTo(r); translations[i].copyTo(t);
      Parameters p; p<<r[0],r[1],r[2],t[0],t[1],t[2];
      if (!inFront(anchors,p)) continue;
      auto e=evaluate(p,anchors,pixels,fields,camera,distortion,g,vertical);
      cv::solvePnPRefineLM(anchors,pixels,camera,distortion,r,t,
                          cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,15,1e-8));
      Parameters polished; polished<<r[0],r[1],r[2],t[0],t[1],t[2];
      auto polished_eval=evaluate(polished,anchors,pixels,fields,camera,distortion,g,vertical);
      if (polished_eval.cost<e.cost) { p=polished; e=std::move(polished_eval); }
      if (!std::isfinite(e.cost)) continue;
      const double before=e.cost;
      if (!fields.empty() || (vertical && g.normal_vertical_weight_px>0))
        refine(p,e,anchors,pixels,fields,camera,distortion,g,vertical);
      if (e.anchor_rms>g.max_reprojection_error_px ||
          (!fields.empty() && e.chamfer_rms>g.max_chamfer_error_px)) continue;
      // Gate EACH blade as well, so many good blades cannot hide one bad match.
      const auto projected=project(anchors,p,camera,distortion);
      bool acceptable=true;
      std::size_t offset=0;
      for (int sector=0;sector<5;++sector) {
        if (!slots[sector]) continue;
        double sum=0;
        const auto& observed=slots[sector]->pixels;
        for (std::size_t j=0;j<observed.size();++j) {
          const auto delta=projected[offset+j]-observed[j]; sum+=delta.dot(delta);
        }
        if (!observed.empty() && std::sqrt(sum/observed.size())>g.max_reprojection_error_px) acceptable=false;
        offset+=observed.size();
      }
      if (!acceptable || !consistentFields(fields,p,camera,distortion,g)) continue;
      if (matches.size()>best_matches.size() ||
          (matches.size()==best_matches.size() && e.cost<best_eval.cost)) {
        best=p; best_eval=std::move(e); best_matches=matches;
        best_refined=!fields.empty() && best_eval.cost<before;
      }
    }
  }
  if (!std::isfinite(best_eval.cost)) {
    // A truncated contour can produce a finite but incorrect PCA envelope.
    // Retry NN anchors once with the SAME contour and quality gates.
    if (allow_pca && std::any_of(prepared.begin(),prepared.end(),[](const auto& p){return p.pca;}))
      return solve(observations,frame,reference,false);
    return {};
  }
  const auto r=rotation(best);
  Eigen::Isometry3d T_camera_model=Eigen::Isometry3d::Identity();
  for (int row=0;row<3;++row)
    for (int col=0;col<3;++col) T_camera_model.linear()(row,col)=r(row,col);
  T_camera_model.translation()<<best[3],best[4],best[5];
  Pose pose;
  pose.T_barrel_model=*frame.T_barrel_camera*T_camera_model;
  pose.center_barrel=pose.T_barrel_model.translation();
  pose.target_barrel=pose.T_barrel_model*Eigen::Vector3d(0,g.radius_m,0);
  pose.normal_barrel=pose.T_barrel_model.linear().col(2);
  if (pose.normal_barrel.x()<0) pose.normal_barrel=-pose.normal_barrel;
  if (world_rotation) pose.up_barrel=world_rotation->transpose()*Eigen::Vector3d::UnitZ();
  if (pose.target_barrel.x()<=0 || !pose.target_barrel.allFinite() || !pose.center_barrel.allFinite()) return {};
  pose.activated=observations[reference].keypoints.class_id==1;
  pose.captured_at=frame.captured_at;
  if (world_rotation) pose.q_world_barrel=Eigen::Quaterniond(*world_rotation);
  pose.reprojection_error_px=best_eval.anchor_rms;
  pose.chamfer_error_px=best_eval.chamfer_rms;
  pose.contour_refined=best_refined;
  pose.attitude_constrained=vertical.has_value() && g.normal_vertical_weight_px>0;
  pose.normal_vertical_component=std::abs(pose.normal_barrel.dot(pose.up_barrel));
  pose.blades=std::move(best_matches);
  for (auto& blade : pose.blades) {
    const auto point=rotatePoint({0,g.radius_m,0},blade.sector);
    blade.target_barrel=pose.T_barrel_model*Eigen::Vector3d(point.x,point.y,point.z);
  }
  const Pose raw=pose;
  plane_window_.update(pose,frame,g);
  if (pose.plane_smoothed) {
    const auto T_camera_smoothed=frame.T_barrel_camera->inverse()*pose.T_barrel_model;
    cv::Matx33d R;
    for (int row=0;row<3;++row) for (int col=0;col<3;++col) R(row,col)=T_camera_smoothed.linear()(row,col);
    cv::Vec3d rv; cv::Rodrigues(R,rv);
    Parameters parameters;
    parameters<<rv[0],rv[1],rv[2],T_camera_smoothed.translation().x(),
                T_camera_smoothed.translation().y(),T_camera_smoothed.translation().z();
    std::vector<cv::Point3d> pooled_anchors;
    std::vector<cv::Point2d> pooled_pixels;
    std::vector<Field> pooled_fields;
    bool acceptable=true;
    for (const auto& match : pose.blades) {
      const auto item=std::find_if(prepared.begin(),prepared.end(),[&](const auto& item){return item.index==match.observation_index;});
      const auto rotated=rotatePoints(item->anchors,match.sector);
      if (!rotated.empty()) {
        const auto predicted=project(rotated,parameters,camera,distortion);
        double sum=0;
        for (std::size_t j=0;j<predicted.size();++j) {
          const auto delta=predicted[j]-item->pixels[j]; sum+=delta.dot(delta);
        }
        if (std::sqrt(sum/predicted.size())>g.max_reprojection_error_px) acceptable=false;
      }
      pooled_anchors.insert(pooled_anchors.end(),rotated.begin(),rotated.end());
      pooled_pixels.insert(pooled_pixels.end(),item->pixels.begin(),item->pixels.end());
      for (const auto& f : item->fields) {
        Field field=f; field.model=rotatePoints(f.model,match.sector);
        pooled_fields.push_back(std::move(field));
      }
    }
    const auto quality=evaluate(parameters,pooled_anchors,pooled_pixels,pooled_fields,camera,distortion,g,vertical);
    if (!acceptable || !std::isfinite(quality.cost) ||
        !consistentFields(pooled_fields,parameters,camera,distortion,g) ||
        quality.anchor_rms>g.max_reprojection_error_px || quality.chamfer_rms>g.max_chamfer_error_px ||
        (pose.target_barrel-raw.target_barrel).norm()>g.plane_window_max_correction_m ||
        std::abs(pose.normal_barrel.dot(pose.up_barrel))>raw.normal_vertical_component+0.05) {
      pose=raw; plane_window_.reset(); plane_window_.update(pose,frame,g);
    } else {
      pose.reprojection_error_px=quality.anchor_rms;
      pose.chamfer_error_px=quality.chamfer_rms;
    }
  }
  pose.normal_vertical_component=std::abs(pose.normal_barrel.dot(pose.up_barrel));
  updatePhase(pose);
  if (!std::isfinite(pose.phase_rad)) return {};
  return pose;
}

} // namespace Rune
