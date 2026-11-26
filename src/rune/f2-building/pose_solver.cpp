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
#include <limits>
#include <numbers>
#include <numeric>
#include <utility>

namespace Rune {
using namespace building_detail;
namespace {
constexpr double sector_angle=2*std::numbers::pi/5;

bool armOnly(const Observation& observation, Mode mode) {
  return mode==Mode::Big && observation.keypoints.class_id==1;
}

std::optional<cv::Point2d> bladePixel(const Observation& observation, Mode mode) {
  if (!armOnly(observation,mode)) return observation.keypoints.armorCenter();
  const auto moments=cv::moments(observation.arm);
  if (std::abs(moments.m00)<1) return {};
  return cv::Point2d(moments.m10/moments.m00,moments.m01/moments.m00);
}

bool consistentBladeAnchors(const Evaluation& evaluation, const std::vector<cv::Point2d>& observed,
                            const std::array<std::size_t,5>& counts, double max_error) {
  std::size_t offset=0;
  for (const auto count : counts) {
    if (count==0) continue;
    double sum=0;
    for (std::size_t i=offset; i<offset+count; ++i) {
      const auto delta=evaluation.projected_anchors[i]-observed[i];
      sum+=delta.dot(delta);
    }
    if (std::sqrt(sum/count)>max_error) return false;
    offset+=count;
  }
  return true;
}

void collectNumberedAnchors(Prepared& blade, const DetectedContours& measured,
                            const NumberedAnchors& model, cv::Size image_size) {
  const auto append = [&](const auto& points, const auto& coordinates) {
    for (std::size_t id=0; id<points.size(); ++id) {
      if (!points[id] || !coordinates[id]) continue;
      const auto pixel=*points[id];
      if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x<0 || pixel.y<0 ||
          pixel.x>=image_size.width || pixel.y>=image_size.height) continue;
      blade.numbered_anchors.push_back(*coordinates[id]);
      blade.numbered_pixels.emplace_back(pixel);
    }
  };
  append(measured.gap_points, model.gaps);
  if (measured.fan_point_ids.size()!=measured.fan_points.size()) return;
  std::array<std::optional<cv::Point2f>,6> fan_points{};
  std::array<bool,6> seen{};
  for (std::size_t i=0; i<measured.fan_points.size(); ++i) {
    if (!measured.fanPointValid(i)) continue;
    const int id=measured.fanPointId(i);
    if (id<0 || std::size_t(id)>=fan_points.size()) continue;
    if (seen[id]) fan_points[id].reset();
    else fan_points[id]=measured.fan_points[i];
    seen[id]=true;
  }
  append(fan_points, model.fan);
}

std::size_t appendCorrespondences(const Prepared& blade, int sector,
                                  std::vector<cv::Point3d>& anchors,
                                  std::vector<cv::Point2d>& pixels) {
  const auto rotated=rotatePoints(blade.anchors,sector);
  anchors.insert(anchors.end(),rotated.begin(),rotated.end());
  pixels.insert(pixels.end(),blade.pixels.begin(),blade.pixels.end());
  for (const auto& point : blade.numbered_anchors) anchors.push_back(rotatePoint(point,sector));
  pixels.insert(pixels.end(),blade.numbered_pixels.begin(),blade.numbered_pixels.end());
  return blade.pixels.size()+blade.numbered_pixels.size();
}
} // namespace

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
    const bool arm_only=armOnly(o,frame.mode);
    if (!validObservation(o,arm_only,gimbal_only_)) continue;
    Prepared p; p.index=i;
    const int state=o.keypoints.class_id==0 ? 0 : (frame.mode==Mode::Small ? 1 : 2);
    const auto& model=models_[state];
    if (!arm_only) {
      const auto pixels=detectedAnchorPixels(o);
      // SCUT 风格的联合解算只使用一次共享 R 标观测。
      const auto anchor_count=gimbal_only_ && i!=reference ? 4U : anchors_.size();
      for (std::size_t j=0;j<anchor_count;++j) if (o.keypoints.point_valid[j]) {
        p.anchors.push_back(anchors_[j]); p.pixels.push_back(pixels[j]);
      }
      if (allow_pca && g.use_pca_anchors && model && !model->pca_box.empty() &&
          std::all_of(o.keypoints.point_valid.begin(),o.keypoints.point_valid.end(),[](bool v){return v;})) {
        if (const auto box=pcaEnvelope(o,camera,distortion)) {
          p.anchors=model->pca_box;
          p.pixels=*box;
          if (anchor_count==anchors_.size()) {
            p.anchors.emplace_back(0,0,0);
            p.pixels.emplace_back(o.keypoints.center_r);
          }
          p.pca=true;
        }
      }
    } else if (gimbal_only_ && i==reference && o.keypoints.point_valid[4]) {
      p.anchors.emplace_back(0,0,0);
      p.pixels.emplace_back(o.keypoints.center_r);
    }
    if (use_numbered_anchors_ && o.keypoints.image_contours &&
        o.keypoints.target_ring_status!=TargetRingStatus::Rejected) {
      collectNumberedAnchors(p,*o.keypoints.image_contours,g.numbered_anchors[state],frame.bgr.size());
    }
    if (g.refinement_iterations>0 && model) {
      const std::array<const std::vector<cv::Point>*,3> contours{&o.armor,&o.arm,&o.center_r};
      bool usable=true;
      std::size_t local_pixels=0;
      for (std::size_t j=0;j<3;++j) {
        // The hub is shared, not five rotated copies of the R glyph. In joint
        // mode use its shared center anchors; its in-plane orientation is unknown.
        if (j==2 && (observations.size()>1 || (gimbal_only_ && o.center_r.empty()))) continue;
        if (j==0 && gimbal_only_ && arm_only && o.armor.empty()) continue;
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
    if (p.anchors.empty() && p.numbered_anchors.empty() && p.fields.empty()) continue;
    prepared.push_back(std::move(p));
    if (prepared.size()==10) break;
  }
  const auto ref=std::find_if(prepared.begin(),prepared.end(),[&](const auto& p){return p.index==reference;});
  if (ref==prepared.end()) return {};

  using Slots=std::array<const Prepared*,5>;
  std::vector<Slots> assignments;
  const auto addAssignment=[&](const auto& angleFor) {
    Slots slots{};
    std::array<double,5> scores;
    scores.fill(std::numeric_limits<double>::infinity());
    slots[0]=&*ref;
    for (const auto& candidate : prepared) {
      if (candidate.index==reference) continue;
      const auto& observation=observations[candidate.index];
      const auto target=bladePixel(observation,frame.mode);
      if (!target) continue;
      const auto angle=angleFor(observation,*target);
      if (!angle || !std::isfinite(*angle)) continue;
      const int sector=(static_cast<int>(std::llround(*angle/sector_angle))+5)%5;
      if (sector==0) continue;
      const double error=std::abs(std::remainder(*angle-sector*sector_angle,2*std::numbers::pi));
      if (error>g.max_sector_error_rad) continue;
      const double score=error-0.03*std::clamp(double(observation.keypoints.confidence),0.0,1.0);
      if (score<scores[sector]) { scores[sector]=score; slots[sector]=&candidate; }
    }
    if (std::find(assignments.begin(),assignments.end(),slots)==assignments.end())
      assignments.push_back(slots);
  };

  if (gimbal_only_) {
    const auto& reference_observation=observations[reference];
    const cv::Point2d center=reference_observation.keypoints.center_r;
    const auto target=bladePixel(reference_observation,frame.mode);
    if (target && cv::norm(*target-center)>=2) {
      const auto reference_axis=*target-center;
      // 参考 SCUT，用图像方位角建立候选扇区，再由重投影误差选择模型朝向。
      for (double direction : {1.0,-1.0}) {
        addAssignment([&](const Observation& observation,cv::Point2d point) -> std::optional<double> {
          if (observation.keypoints.point_valid[4] &&
              cv::norm(cv::Point2d(observation.keypoints.center_r)-center)>g.max_center_disagreement_px)
            return {};
          const auto axis=point-center;
          if (cv::norm(axis)<2) return {};
          return direction*std::atan2(reference_axis.cross(axis),reference_axis.dot(axis));
        });
      }
    }
  }

  std::vector<cv::Point3d> seed_anchors=ref->anchors;
  std::vector<cv::Point2d> seed_pixels=ref->pixels;
  auto seed_method=cv::SOLVEPNP_IPPE;
  if (seed_anchors.size()<4 && gimbal_only_) {
    seed_anchors.insert(seed_anchors.end(),ref->numbered_anchors.begin(),ref->numbered_anchors.end());
    seed_pixels.insert(seed_pixels.end(),ref->numbered_pixels.begin(),ref->numbered_pixels.end());
    seed_method=cv::SOLVEPNP_SQPNP;
  }
  std::vector<cv::Mat> seeds_r,seeds_t;
  if (seed_anchors.size()>=4)
    cv::solvePnPGeneric(seed_anchors,seed_pixels,camera,distortion,seeds_r,seeds_t,false,seed_method);
  for (std::size_t seed=0;seed<seeds_r.size();++seed) {
    cv::Vec3d rv,tv;
    seeds_r[seed].copyTo(rv); seeds_t[seed].copyTo(tv);
    Parameters initial; initial<<rv[0],rv[1],rv[2],tv[0],tv[1],tv[2];
    if (!inFront(seed_anchors,initial)) continue;
    const auto R=rotation(initial);
    const cv::Vec3d normal(R(0,2),R(1,2),R(2,2));
    const double numerator=normal.dot(tv);
    const auto center_pixel=project(std::vector<cv::Point3d>{{0,0,0}},initial,camera,distortion)[0];
    addAssignment([&](const Observation& o,cv::Point2d target) -> std::optional<double> {
      if (o.keypoints.point_valid[4] &&
          cv::norm(cv::Point2d(o.keypoints.center_r)-center_pixel)>g.max_center_disagreement_px) return {};
      std::vector<cv::Point2d> ray_points;
      cv::undistortPoints(std::vector<cv::Point2d>{target},ray_points,camera,distortion);
      const cv::Vec3d ray(ray_points[0].x,ray_points[0].y,1);
      const double denominator=normal.dot(ray);
      if (std::abs(denominator)<1e-6 || numerator/denominator<=0) return {};
      const cv::Vec3d point=R.t()*(ray*(numerator/denominator)-tv);
      const double radius=std::hypot(point[0],point[1]);
      if (radius<g.radius_m*(armOnly(o,frame.mode) ? 0.05 : 0.5) || radius>1.5*g.radius_m) return {};
      return std::atan2(-point[0],point[1]);
    });
  }

  Parameters best;
  Evaluation best_eval;
  std::vector<BladeMatch> best_matches;
  std::size_t best_numbered_count=0;
  bool best_refined=false;
  for (const auto& slots : assignments) {
    std::vector<cv::Point3d> anchors;
    std::vector<cv::Point2d> pixels;
    std::array<std::size_t,5> blade_anchor_counts{};
    std::vector<Field> fields;
    std::vector<BladeMatch> matches;
    std::size_t numbered_count=0;
    for (int sector=0;sector<5;++sector) {
      const auto* p=slots[sector];
      if (!p) continue;
      blade_anchor_counts[sector]=appendCorrespondences(*p,sector,anchors,pixels);
      numbered_count+=p->numbered_anchors.size();
      for (const auto& f : p->fields) {
        Field field=f; field.model=rotatePoints(f.model,sector); fields.push_back(std::move(field));
      }
      matches.push_back({p->index,sector,observations[p->index].keypoints.class_id==1,p->pca});
    }
    std::vector<cv::Mat> rotations,translations;
    // 编号角点允许有深度差，联合解算使用支持非共面点的 SQPnP。
    const auto method=gimbal_only_ || numbered_count>0 ? cv::SOLVEPNP_SQPNP : cv::SOLVEPNP_IPPE;
    if (anchors.size()<4) continue;
    if (!cv::solvePnPGeneric(anchors,pixels,camera,distortion,rotations,translations,false,method)) continue;
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
          (!fields.empty() && e.chamfer_rms>g.max_chamfer_error_px) ||
          std::sqrt(e.worst_contour_mean_square)>g.max_chamfer_error_px ||
          !consistentBladeAnchors(e,pixels,blade_anchor_counts,g.max_reprojection_error_px)) continue;
      if (matches.size()>best_matches.size() ||
          (matches.size()==best_matches.size() && e.cost<best_eval.cost)) {
        best=p; best_eval=std::move(e); best_matches=matches;
        best_numbered_count=numbered_count;
        best_refined=!fields.empty() && best_eval.cost<before;
      }
    }
  }
  if (!std::isfinite(best_eval.cost)) {
    // A truncated contour can produce a finite but incorrect PCA envelope.
    // 使用检测锚点重试，沿用相同的轮廓约束和质量门限。
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
  pose.numbered_anchor_count=best_numbered_count;
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
    std::array<std::size_t,5> blade_anchor_counts{};
    std::size_t count_index=0;
    for (const auto& match : pose.blades) {
      const auto item=std::find_if(prepared.begin(),prepared.end(),[&](const auto& item){return item.index==match.observation_index;});
      blade_anchor_counts[count_index++]=appendCorrespondences(*item,match.sector,pooled_anchors,pooled_pixels);
      for (const auto& f : item->fields) {
        Field field=f; field.model=rotatePoints(f.model,match.sector);
        pooled_fields.push_back(std::move(field));
      }
    }
    const auto quality=evaluate(parameters,pooled_anchors,pooled_pixels,pooled_fields,camera,distortion,g,vertical);
    if (!std::isfinite(quality.cost) ||
        !consistentBladeAnchors(quality,pooled_pixels,blade_anchor_counts,g.max_reprojection_error_px) ||
        std::sqrt(quality.worst_contour_mean_square)>g.max_chamfer_error_px ||
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
