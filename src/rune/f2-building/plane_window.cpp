#include "rune/f2-building/plane_window.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace Rune {
void PlaneWindow::update(Pose& pose, const Frame& frame, const Geometry& g) {
  pose.plane_window_frames=0;
  pose.plane_smoothed=false;
  if (!frame.T_world_barrel || frame.captured_at==TimePoint{} ||
      g.plane_window_max_frames<g.plane_window_min_frames) {
    reset();
    return;
  }
  const auto T_world_model=*frame.T_world_barrel*pose.T_barrel_model;
  Sample current{frame.captured_at,T_world_model.translation(),
                 T_world_model.linear().col(2),T_world_model.linear(),frame.world_frame_id};
  if (!samples_.empty()) {
    const auto& last=samples_.back();
    const double gap=std::chrono::duration<double>(current.time-last.time).count();
    if (gap<=0 || gap>g.plane_window_max_gap_s || current.frame_id!=last.frame_id ||
        (current.center-last.center).norm()>g.plane_window_max_center_jump_m ||
        std::abs(current.normal.dot(last.normal))<std::cos(g.plane_window_max_normal_angle_rad))
      reset();
  }
  samples_.push_back(current);
  while (samples_.size()>static_cast<std::size_t>(g.plane_window_max_frames) ||
         std::chrono::duration<double>(current.time-samples_.front().time).count()>g.plane_window_duration_s)
    samples_.pop_front();
  pose.plane_window_frames=static_cast<int>(samples_.size());
  if (pose.plane_window_frames<g.plane_window_min_frames) return;

  // Five evenly spread coplanar samples per frame, with equal frame weights.
  // Accumulate moments directly instead of allocating a tall SVD matrix.
  Eigen::Vector3d centroid=Eigen::Vector3d::Zero();
  for (const auto& s : samples_) centroid+=s.center;
  centroid/=samples_.size();
  Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
  for (const auto& s : samples_) {
    const Eigen::Vector3d d=s.center-centroid;
    covariance+=d*d.transpose()+0.5*g.radius_m*g.radius_m*
      (s.basis.col(0)*s.basis.col(0).transpose()+s.basis.col(1)*s.basis.col(1).transpose());
  }
  covariance/=samples_.size();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info()!=Eigen::Success || !solver.eigenvalues().allFinite() ||
      solver.eigenvalues()[1]<1e-8 ||
      std::sqrt(std::max(0.0,solver.eigenvalues()[0]))>g.plane_window_max_rms_m) {
    reset();
    samples_.push_back(current);
    pose.plane_window_frames=1;
    return;
  }
  Eigen::Vector3d normal=solver.eigenvectors().col(0);
  if (normal.dot(current.normal)<0) normal=-normal;
  const double correction=normal.dot(current.center-centroid);
  if (std::abs(correction)>g.plane_window_max_correction_m ||
      normal.dot(current.normal)<std::cos(g.plane_window_max_normal_angle_rad)) {
    reset(); samples_.push_back(current); pose.plane_window_frames=1;
    return;
  }
  // Rigidly align the whole model rather than projecting its points separately.
  // This preserves radius, fivefold symmetry and a consistent T_barrel_model.
  Eigen::Isometry3d smoothed=Eigen::Isometry3d::Identity();
  smoothed.linear()=Eigen::Quaterniond::FromTwoVectors(current.normal,normal).toRotationMatrix()*current.basis;
  smoothed.translation()=current.center-correction*normal;
  const Eigen::Isometry3d T_barrel_model=frame.T_world_barrel->inverse()*smoothed;
  if (!T_barrel_model.matrix().allFinite()) return;
  pose.T_barrel_model=T_barrel_model;
  pose.center_barrel=T_barrel_model.translation();
  pose.target_barrel=T_barrel_model*Eigen::Vector3d(0,g.radius_m,0);
  pose.normal_barrel=T_barrel_model.linear().col(2);
  if (pose.normal_barrel.x()<0) pose.normal_barrel=-pose.normal_barrel;
  for (auto& blade : pose.blades) {
    const double angle=blade.sector*2*std::numbers::pi/5;
    blade.target_barrel=T_barrel_model*Eigen::Vector3d(-g.radius_m*std::sin(angle),
                                                      g.radius_m*std::cos(angle),0);
  }
  pose.plane_smoothed=true;
}
} // namespace Rune
