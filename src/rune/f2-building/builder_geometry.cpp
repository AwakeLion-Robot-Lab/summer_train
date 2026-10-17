// Algorithm reference: SZURPVision/RP-26Rune, PowerRunePlane.
// See README.md and RP-26Rune-LICENSE in this directory.
// R/inactive-seed layout selection also references Alliance-Algorithm/
// rmcs_auto_aim_v2 (MIT); see ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f2-building/builder_detail.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace Rune::building_detail {

bool validGeometry(const Geometry& g) {
  return std::isfinite(g.radius_m) && g.radius_m > 0 &&
    std::isfinite(g.armor_half_width_m) && g.armor_half_width_m > 0 &&
    std::isfinite(g.armor_half_height_m) && g.armor_half_height_m > 0 &&
    g.radius_m > g.armor_half_height_m &&
    std::isfinite(g.max_reprojection_error_px) && g.max_reprojection_error_px > 0 &&
    g.refinement_iterations >= 0 && g.refinement_iterations <= 100 &&
    g.max_samples_per_contour >= 4 && g.max_samples_per_contour <= 4096 &&
    g.distance_field_padding_px >= 2 && g.distance_field_padding_px <= 4096 &&
    g.max_distance_field_pixels > 0 &&
    std::isfinite(g.contour_weight) && g.contour_weight > 0 &&
    std::isfinite(g.huber_delta_px) && g.huber_delta_px > 0 &&
    std::isfinite(g.max_chamfer_error_px) && g.max_chamfer_error_px > 0 &&
    std::isfinite(g.max_sector_error_rad) && g.max_sector_error_rad>0 &&
    g.max_sector_error_rad<std::numbers::pi/5 &&
    std::isfinite(g.max_center_disagreement_px) && g.max_center_disagreement_px>0 &&
    std::isfinite(g.normal_vertical_weight_px) && g.normal_vertical_weight_px>=0 &&
    std::isfinite(g.plane_window_duration_s) && g.plane_window_duration_s>0 &&
    g.plane_window_max_frames>=0 && g.plane_window_max_frames<=300 &&
    g.plane_window_min_frames>=2 && g.plane_window_min_frames<=300 &&
    std::isfinite(g.plane_window_max_gap_s) && g.plane_window_max_gap_s>0 &&
    std::isfinite(g.plane_window_max_center_jump_m) && g.plane_window_max_center_jump_m>0 &&
    std::isfinite(g.plane_window_max_normal_angle_rad) && g.plane_window_max_normal_angle_rad>0 &&
    g.plane_window_max_normal_angle_rad<std::numbers::pi/2 &&
    std::isfinite(g.plane_window_max_rms_m) && g.plane_window_max_rms_m>0 &&
    std::isfinite(g.plane_window_max_correction_m) && g.plane_window_max_correction_m>0;
}
bool validCamera(const Frame& frame, cv::Mat& camera, cv::Mat& distortion) {
  const auto& k = frame.camera_matrix;
  if (k.rows != 3 || k.cols != 3 || k.channels() != 1 ||
      (k.depth() != CV_32F && k.depth() != CV_64F)) return false;
  k.convertTo(camera, CV_64F);
  if (!cv::checkRange(camera) || camera.at<double>(0,0) <= 0 ||
      camera.at<double>(1,1) <= 0 || std::abs(camera.at<double>(2,2)-1) > 1e-9 ||
      std::abs(camera.at<double>(0,1)) > 1e-9 ||
      std::abs(camera.at<double>(1,0)) > 1e-9 ||
      std::abs(camera.at<double>(2,0)) > 1e-9 ||
      std::abs(camera.at<double>(2,1)) > 1e-9) return false;
  const auto& d = frame.distortion;
  if (!d.empty()) {
    const auto n = d.total();
    if (d.channels() != 1 || (d.rows != 1 && d.cols != 1) ||
        (d.depth() != CV_32F && d.depth() != CV_64F) ||
        (n != 4 && n != 5 && n != 8 && n != 12 && n != 14)) return false;
    d.convertTo(distortion, CV_64F);
    if (!cv::checkRange(distortion)) return false;
  }
  if (!frame.T_barrel_camera) return false;
  const auto& t = *frame.T_barrel_camera;
  return t.matrix().allFinite() &&
    (t.linear().transpose()*t.linear()-Eigen::Matrix3d::Identity()).norm() < 1e-6 &&
    std::abs(t.linear().determinant()-1) < 1e-6 &&
    (t.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).norm() < 1e-9;
}
bool worldRotation(const Frame& frame, std::optional<Eigen::Matrix3d>& rotation) {
  if (frame.q_world_barrel) {
    const auto& q=*frame.q_world_barrel;
    if (!q.coeffs().allFinite() || std::abs(q.squaredNorm()-1)>1e-3) return false;
    rotation=q.normalized().toRotationMatrix();
  }
  if (frame.T_world_barrel) {
    if (!rigid(*frame.T_world_barrel)) return false;
    if (rotation && ((*rotation)-frame.T_world_barrel->linear()).norm()>1e-3) return false;
    rotation=frame.T_world_barrel->linear();
  }
  return true;
}
std::vector<cv::Point2d> networkPoints(const Observation& o) {
  const auto& k=o.keypoints;
  return {k.top,k.left,k.right,k.bottom,k.center_r};
}
bool validObservation(const Observation& o, bool arm_only) {
  if ((o.keypoints.class_id!=0 && o.keypoints.class_id!=1) ||
      !std::isfinite(o.keypoints.confidence) || o.arm.empty() || o.center_r.empty() ||
      (!arm_only && o.armor.empty())) return false;
  for (const auto* contour : {&o.armor,&o.arm,&o.center_r}) {
    if (contour->empty() && arm_only && contour==&o.armor) continue;
    if (contour->size()<3) return false;
    for (const auto& point : *contour)
      if (std::abs(double(point.x))>1000000 || std::abs(double(point.y))>1000000) return false;
    if (std::abs(cv::contourArea(*contour))<1) return false;
  }
  for (const auto& p : networkPoints(o))
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
  if (arm_only) return true;
  std::vector<cv::Point2f> hull;
  cv::convexHull(std::vector<cv::Point2f>{o.keypoints.top,o.keypoints.left,
                  o.keypoints.right,o.keypoints.bottom},hull);
  return std::abs(cv::contourArea(hull))>=1;
}
cv::Point3d rotatePoint(const cv::Point3d& p, int sector) {
  const double angle=sector*2*std::numbers::pi/5;
  const double c=std::cos(angle),s=std::sin(angle);
  return {c*p.x-s*p.y,s*p.x+c*p.y,p.z};
}
std::vector<cv::Point3d> rotatePoints(const std::vector<cv::Point3d>& points,int sector) {
  std::vector<cv::Point3d> result;
  result.reserve(points.size());
  for (const auto& p : points) result.push_back(rotatePoint(p,sector));
  return result;
}
void updatePhase(Pose& pose) {
  const Eigen::Vector3d up_projection=pose.up_barrel-pose.normal_barrel*pose.normal_barrel.dot(pose.up_barrel);
  if (up_projection.norm()<1e-3) { pose.phase_rad=std::numeric_limits<double>::quiet_NaN(); return; }
  const Eigen::Vector3d up=up_projection.normalized();
  const Eigen::Vector3d radial=(pose.target_barrel-pose.center_barrel).normalized();
  pose.phase_rad=std::atan2(radial.dot(pose.normal_barrel.cross(up)),radial.dot(up));
}
bool validPcaBox(const std::vector<cv::Point3d>& box) {
  if (box.empty()) return true;
  if (box.size()!=4 || !std::all_of(box.begin(),box.end(),finite)) return false;
  constexpr double tolerance=1e-8;
  for (const auto& p : box) if (std::abs(p.z)>tolerance) return false;
  return box[2].x-box[0].x>1e-5 && box[0].y-box[1].y>1e-5 &&
    std::abs(box[0].x-box[1].x)<tolerance && std::abs(box[2].x-box[3].x)<tolerance &&
    std::abs(box[0].y-box[3].y)<tolerance && std::abs(box[1].y-box[2].y)<tolerance;
}

} // namespace Rune::building_detail
