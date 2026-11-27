// Algorithm reference: Alliance-Algorithm/rmcs_auto_aim_v2 (MIT).
// Adapted to Rune's optical-camera / barrel / optional full-world transforms.
#include "rune/f3-estimator/feature_filter.hpp"
#include "rune/f3-estimator/association.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace Rune {
namespace {
constexpr double sector_angle=2*std::numbers::pi/5;
using Vector=Eigen::Matrix<double,7,1>;
using Matrix=Eigen::Matrix<double,7,7>;
using Jacobian=Eigen::Matrix<double,2,7>;
Eigen::Vector3d normal(const Vector& x) {
  return {std::cos(x[4])*std::cos(x[3]),std::cos(x[4])*std::sin(x[3]),std::sin(x[4])};
}
Eigen::Vector3d radial(const Vector& x,const Eigen::Vector3d& up,int sector) {
  const auto n=normal(x);
  const Eigen::Vector3d v=(up-n*n.dot(up)).normalized();
  const double a=x[5]+sector*sector_angle;
  return v*std::cos(a)+n.cross(v)*std::sin(a);
}
Eigen::Vector3d bladeRightAxis(const Vector& x, const Eigen::Vector3d& up, int sector,
                               const Eigen::Isometry3d& camera_tracking) {
  const auto n = normal(x);
  const Eigen::Vector3d n_cam = camera_tracking.linear() * n;
  const Eigen::Vector3d n_away = n_cam.z() >= 0.0 ? n : -n;
  const Eigen::Vector3d v = radial(x, up, sector);
  return n_away.cross(v);
}
std::optional<Eigen::Vector2d> projectModelPoint(
    const Vector& x, const Eigen::Vector3d& up, int sector,
    const cv::Point3d& anchor, const Eigen::Isometry3d& camera_tracking, const Frame& frame) {
  const auto n = normal(x);
  if ((up - n * n.dot(up)).norm() < 1e-3) return std::nullopt;
  const Eigen::Vector3d v = radial(x, up, sector);
  const Eigen::Vector3d right = bladeRightAxis(x, up, sector, camera_tracking);
  const Eigen::Vector3d n_cam = camera_tracking.linear() * n;
  const Eigen::Vector3d n_away = n_cam.z() >= 0.0 ? n : -n;
  const Eigen::Vector3d p_track = x.head<3>() + anchor.x * right + anchor.y * v - anchor.z * n_away;
  const Eigen::Vector3d p_cam = camera_tracking * p_track;
  if (!p_cam.allFinite() || p_cam.z() <= 0.1) return std::nullopt;
  std::vector<cv::Point2d> pixels;
  cv::projectPoints(std::vector<cv::Point3d>{{p_cam.x(), p_cam.y(), p_cam.z()}},
                    cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                    frame.camera_matrix, frame.distortion, pixels);
  const Eigen::Vector2d uv(pixels[0].x, pixels[0].y);
  if (!uv.allFinite()) return std::nullopt;
  return uv;
}
std::optional<Jacobian> modelPointJacobian(
    const Vector& x, const Eigen::Vector3d& up, int sector,
    const cv::Point3d& anchor, const Eigen::Isometry3d& camera_tracking, const Frame& frame) {
  Jacobian h = Jacobian::Zero();
  constexpr double step = 1e-5;
  for (int k = 0; k < 6; ++k) {
    Vector a = x, b = x;
    a[k] += step;
    b[k] -= step;
    const auto pa = projectModelPoint(a, up, sector, anchor, camera_tracking, frame);
    const auto pb = projectModelPoint(b, up, sector, anchor, camera_tracking, frame);
    if (!pa || !pb) return std::nullopt;
    h.col(k) = (*pa - *pb) / (2.0 * step);
  }
  return h;
}
void predictPhase(Vector& x,double dt) { x[5]+=x[6]*dt; }
void predictCovariance(Matrix& p, double dt, const FeatureFilterConfig& config) {
  Matrix f=Matrix::Identity();
  f(5,6)=dt;
  p=(f*p*f.transpose()).eval();
  p.diagonal().head<3>().array()+=config.center_process_variance_m2_s*dt;
  p.diagonal().segment<2>(3).array()+=config.normal_process_variance_rad2_s*dt;
  const double q=config.angular_acceleration_variance;
  p(5,5)+=q*dt*dt*dt/3;
  p(5,6)+=q*dt*dt/2;
  p(6,5)+=q*dt*dt/2;
  p(6,6)+=q*dt;
}
struct Projection {
  std::array<Eigen::Vector2d,6> pixels{};
  std::array<bool,6> visible{};
};
Projection project(const Vector& x,const Eigen::Vector3d& up,double radius,
                   const Eigen::Isometry3d& camera_tracking,const Frame& frame) {
  Projection result;
  if ((up-normal(x)*normal(x).dot(up)).norm()<1e-3) return result;
  std::vector<cv::Point3d> points;
  for (int j=0;j<6;++j) {
    Eigen::Vector3d p=x.head<3>();
    if (j) p+=radius*radial(x,up,j-1);
    p=camera_tracking*p;
    result.visible[j]=p.allFinite() && p.z()>0.1;
    points.emplace_back(p.x(),p.y(),p.z());
  }
  std::vector<cv::Point2d> pixels;
  cv::projectPoints(points,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),
                    frame.camera_matrix,frame.distortion,pixels);
  for (int j=0;j<6;++j) {
    result.pixels[j]={pixels[j].x,pixels[j].y};
    result.visible[j]=result.visible[j] && result.pixels[j].allFinite();
  }
  return result;
}
std::array<Jacobian,6> jacobians(const Vector& x,const Eigen::Vector3d& up,
    double radius,const Eigen::Isometry3d& camera_tracking,const Frame& frame) {
  std::array<Jacobian,6> result;
  for (auto& h : result) h.setZero();
  // Differentiate the actual distorted projection, including OpenCV's supported
  // distortion models. Angular speed influences phase through the prediction.
  for (int k=0;k<6;++k) {
    const double step=1e-5;
    Vector a=x,b=x; a[k]+=step; b[k]-=step;
    const auto pa=project(a,up,radius,camera_tracking,frame);
    const auto pb=project(b,up,radius,camera_tracking,frame);
    for (int j=0;j<6;++j) if (pa.visible[j] && pb.visible[j])
      result[j].col(k)=(pa.pixels[j]-pb.pixels[j])/(2*step);
  }
  return result;
}
struct Measurement {
  Eigen::Vector2d pixel;
  std::size_t index;
  bool icon, inactive;
  double variance_scale;
};
std::vector<Measurement> measurements(const std::vector<Observation>& observations,Mode mode) {
  std::vector<Measurement> result;
  // 多片叶子共享 R 标，更新前对重复的 R 标观测去重。
  for (std::size_t i=0;i<observations.size();++i) {
    const auto& k=observations[i].keypoints;
    if (!k.point_valid[4]) continue;
    const Eigen::Vector2d pixel(k.center_r.x,k.center_r.y);
    if (!pixel.allFinite() || (k.class_id!=0 && k.class_id!=1)) continue;
    const Measurement measurement{pixel, i, true, false, k.center_r_variance_scale};
    const auto duplicate = std::find_if(result.begin(), result.end(), [&](const auto& m) {
      return m.icon && (m.pixel - pixel).norm() < 2;
    });
    if (duplicate == result.end()) result.push_back(measurement);
    else if (measurement.variance_scale < duplicate->variance_scale) *duplicate = measurement;
  }
  for (std::size_t i=0;i<observations.size();++i) {
    const auto& k=observations[i].keypoints;
    if ((k.class_id!=0 && k.class_id!=1) || (mode==Mode::Big && k.class_id==1)) continue;
    const auto reliable_center=k.armorCenter();
    if (!reliable_center) continue;
    const cv::Point2d center=*reliable_center;
    const Eigen::Vector2d pixel(center.x,center.y);
    if (pixel.allFinite())
      result.push_back({pixel, i, false, k.class_id == 0, k.target_variance_scale});
  }
  return result;
}
} // namespace

FeatureFilter::FeatureFilter(Geometry geometry,FeatureFilterConfig config,bool follow_only)
  : geometry_(std::move(geometry)),config_(config),follow_only_(follow_only) {
  for (double v : {config.gate_squared,config.observation_variance_px2,
      config.center_process_variance_m2_s,config.normal_process_variance_rad2_s,
      config.angular_acceleration_variance,config.max_gap_s,
      config.max_center_jump_m,config.max_normal_jump_rad,config.max_speed_rad_s})
    if (!std::isfinite(v) || v<=0) throw std::invalid_argument("Invalid rune filter configuration");
  if (config.big_sector_hold_ms<=0 || config.big_sector_confirmation_ms<=0)
    throw std::invalid_argument("Big rune sector tracking times must be positive");
}
void FeatureFilter::noteObservationGap() {
  pending_sector_=-1;
  pending_since_={};
}

void FeatureFilter::reset() {
  state_.reset();
  previous_time_={};
  target_sector_=-1;
  target_seen_at_={};
  noteObservationGap();
}

std::optional<FeatureFilter::TargetPrediction> FeatureFilter::predictTarget(
    const Frame& frame,TimePoint at,int sector) const {
  const double dt=std::chrono::duration<double>(at-previous_time_).count();
  if (!state_ || sector<0 || sector>=5 || frame.captured_at<=previous_time_ ||
      at<frame.captured_at || dt>config_.max_gap_s || frame.world_frame_id!=world_frame_id_ ||
      frame.T_world_barrel.has_value()!=state_->world_frame) return {};
  const auto& x=state_->x;
  Vector predicted=x;
  predictPhase(predicted,dt);
  TargetPrediction result{
      x.head<3>()+geometry_.radius_m*radial(x,up_,sector),
      predicted.head<3>()+geometry_.radius_m*radial(predicted,up_,sector)};
  if (frame.T_world_barrel) {
    result.observed_direction-=previous_world_origin_;
    result.predicted_direction-=frame.T_world_barrel->translation();
  } else if (frame.q_world_barrel.has_value()!=previous_world_orientation_.has_value() ||
             (frame.q_world_barrel && (!frame.q_world_barrel->coeffs().allFinite() ||
              std::abs(frame.q_world_barrel->squaredNorm()-1)>1e-3))) {
    return {};
  }
  if (!result.observed_direction.allFinite() || !result.predicted_direction.allFinite()) return {};
  return result;
}

std::optional<FeatureFilter::GimbalPrediction> FeatureFilter::predictGimbal(
    const Frame& frame,TimePoint at,bool center_only) const {
  const double dt=std::chrono::duration<double>(at-previous_time_).count();
  if (!follow_only_ || !state_ || target_sector_<0 || frame.captured_at<previous_time_ ||
      at<frame.captured_at || dt<0 || dt>config_.max_gap_s || frame.world_frame_id!=world_frame_id_ ||
      frame.T_world_barrel.has_value()!=state_->world_frame) return {};
  if (!state_->world_frame && (!previous_world_orientation_ || !frame.q_world_barrel ||
      !frame.q_world_barrel->coeffs().allFinite() ||
      std::abs(frame.q_world_barrel->squaredNorm()-1)>1e-3)) return {};
  if (frame.mode==Mode::Big && !center_only &&
      (target_seen_at_==TimePoint{} ||
       at-target_seen_at_>std::chrono::milliseconds(config_.big_sector_hold_ms))) return {};
  Vector predicted=state_->x;
  predictPhase(predicted,dt);
  GimbalPrediction result{predicted.head<3>(),center_only ? -1 : target_sector_};
  if (!center_only) result.direction+=geometry_.radius_m*radial(predicted,up_,target_sector_);
  if (frame.T_world_barrel) result.direction-=frame.T_world_barrel->translation();
  if (!result.direction.allFinite()) return {};
  return result;
}

void FeatureFilter::updateReobservedPhase(const Frame& frame) {
  const double dt=std::chrono::duration<double>(frame.captured_at-previous_time_).count();
  if (!follow_only_ || !state_ || target_sector_<0 || dt<=0 || dt>config_.max_gap_s ||
      frame.reobserved_target_rings.empty() || !frame.T_barrel_camera ||
      frame.camera_matrix.rows!=3 || frame.camera_matrix.cols!=3 ||
      frame.world_frame_id!=world_frame_id_ || frame.T_world_barrel.has_value()!=state_->world_frame)
    return;

  Eigen::Isometry3d tracking_barrel=Eigen::Isometry3d::Identity();
  if (state_->world_frame) {
    tracking_barrel=*frame.T_world_barrel;
  } else {
    if (!frame.q_world_barrel || !previous_world_orientation_ ||
        !frame.q_world_barrel->coeffs().allFinite() ||
        std::abs(frame.q_world_barrel->squaredNorm()-1)>1e-3) return;
    tracking_barrel.linear()=frame.q_world_barrel->toRotationMatrix();
  }
  if (!tracking_barrel.matrix().allFinite() || !frame.T_barrel_camera->matrix().allFinite()) return;
  const Eigen::Isometry3d camera_tracking=frame.T_barrel_camera->inverse()*tracking_barrel.inverse();
  Vector predicted=state_->x;
  predictPhase(predicted,dt);
  const auto projection=project(predicted,up_,geometry_.radius_m,camera_tracking,frame);

  const ReobservedTargetRing* best=nullptr;
  Eigen::Vector2d innovation;
  int slot=0;
  double best_fraction=1;
  for (const auto& ring : frame.reobserved_target_rings) {
    const auto& p=ring.ellipse.center;
    const double diameter=std::min(ring.ellipse.size.width,ring.ellipse.size.height);
    if (!std::isfinite(diameter) || diameter<4 || !std::isfinite(ring.fit_rms_px)) continue;
    const Eigen::Vector2d point(p.x,p.y);
    const double gate=std::max(4.0,.18*diameter);
    int match=0;
    for (int j=1;j<=5;++j) {
      if (!projection.visible[j] || (point-projection.pixels[j]).norm()>=gate) continue;
      if (match) { match=0; break; }
      match=j;
    }
    if (!match) continue;
    const Eigen::Vector2d residual=point-projection.pixels[match];
    const double fraction=residual.norm()/gate;
    if (!std::isfinite(fraction) || fraction>=best_fraction) continue;
    best_fraction=fraction;
    innovation=residual;
    best=&ring;
    slot=match;
  }
  if (!best) return;

  // 只更新相位。单个重检靶心不提供新的符心、法向或角速度观测。
  constexpr double step=1e-5;
  Vector before=predicted, after=predicted;
  before[5]-=step;
  after[5]+=step;
  const auto a=project(before,up_,geometry_.radius_m,camera_tracking,frame);
  const auto b=project(after,up_,geometry_.radius_m,camera_tracking,frame);
  if (!a.visible[slot] || !b.visible[slot]) return;
  const Eigen::Vector2d derivative=(b.pixels[slot]-a.pixels[slot])/(2*step);
  const double pixels_per_rad=derivative.norm();
  if (pixels_per_rad<10 || !std::isfinite(pixels_per_rad)) return;
  const double phase_error=derivative.dot(innovation)/(pixels_per_rad*pixels_per_rad);
  if (std::abs(phase_error)>.12) return;
  const double sigma=std::max(4.0,3*best->fit_rms_px);
  const double variance=(sigma/pixels_per_rad)*(sigma/pixels_per_rad);
  Matrix covariance=state_->covariance;
  predictCovariance(covariance,dt,config_);
  const double gain=covariance(5,5)/(covariance(5,5)+variance);
  predicted[5]+=gain*phase_error;
  Matrix complement=Matrix::Identity();
  complement(5,5)-=gain;
  covariance=(complement*covariance*complement.transpose()).eval();
  covariance(5,5)+=gain*gain*variance;
  if (!covariance.allFinite() || !predicted.allFinite()) return;
  state_->x=predicted;
  state_->covariance=covariance;
  previous_time_=frame.captured_at;
}

std::vector<FeatureFilter::TargetPriorProjection> FeatureFilter::predictTargetPriors(
    const Frame& frame) const {
  std::vector<TargetPriorProjection> priors;
  const double dt = std::chrono::duration<double>(frame.captured_at - previous_time_).count();
  if (!state_ || dt <= 0.0 || dt > config_.max_gap_s || !frame.T_barrel_camera ||
      frame.camera_matrix.rows != 3 || frame.camera_matrix.cols != 3 ||
      frame.world_frame_id != world_frame_id_ ||
      frame.T_world_barrel.has_value() != state_->world_frame) {
    return priors;
  }
  Eigen::Isometry3d tracking_barrel = Eigen::Isometry3d::Identity();
  if (state_->world_frame) {
    tracking_barrel = *frame.T_world_barrel;
  } else if (frame.q_world_barrel.has_value() != previous_world_orientation_.has_value()) {
    return priors;
  } else if (frame.q_world_barrel) {
    if (!frame.q_world_barrel->coeffs().allFinite() ||
        std::abs(frame.q_world_barrel->squaredNorm() - 1.0) > 1e-3) {
      return priors;
    }
    tracking_barrel.linear() = frame.q_world_barrel->normalized().toRotationMatrix();
  }
  if (!tracking_barrel.matrix().allFinite() || !frame.T_barrel_camera->matrix().allFinite())
    return priors;

  cv::Mat k_mat;
  frame.camera_matrix.convertTo(k_mat, CV_64F);
  Eigen::Matrix3d K;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) K(r, c) = k_mat.at<double>(r, c);
  if (!K.allFinite()) return priors;

  const Eigen::Isometry3d camera_tracking = frame.T_barrel_camera->inverse() * tracking_barrel.inverse();
  Vector predicted = state_->x;
  predictPhase(predicted, dt);
  Matrix predicted_cov = state_->covariance;
  predictCovariance(predicted_cov, dt, config_);

  const auto hub_uv = projectModelPoint(predicted, up_, 0, {0.0, 0.0, 0.0}, camera_tracking, frame);
  if (!hub_uv) return priors;
  const double half_w = geometry_.armor_half_width_m;
  const double half_h = geometry_.armor_half_height_m;
  const double radius = geometry_.radius_m;
  if (half_w <= 0.0 || half_h <= 0.0 || radius <= 0.0) return priors;

  priors.reserve(5);
  for (int sector = 0; sector < 5; ++sector) {
    const Eigen::Vector3d v = radial(predicted, up_, sector);
    const Eigen::Vector3d right = bladeRightAxis(predicted, up_, sector, camera_tracking);
    const Eigen::Vector3d target_3d = predicted.head<3>() + radius * v;
    const Eigen::Vector3d center_cam = camera_tracking * target_3d;
    if (!center_cam.allFinite() || center_cam.z() <= 0.1) continue;

    // 构造规范化靶面圆 (u, v) ∈ [-1, 1]^2 -> 像素的单应性矩阵 H。
    const Eigen::Vector3d col_u = K * (camera_tracking.linear() * (half_w * right));
    const Eigen::Vector3d col_v = K * (camera_tracking.linear() * (half_h * v));
    const Eigen::Vector3d col_t = K * center_cam;
    Eigen::Matrix3d H_eig;
    H_eig.col(0) = col_u;
    H_eig.col(1) = col_v;
    H_eig.col(2) = col_t;
    if (!H_eig.allFinite() || std::abs(H_eig.determinant()) < 1e-6) continue;

    TargetPriorProjection prior;
    prior.sector = sector;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) prior.homography_target_to_image(r, c) = H_eig(r, c);

    const auto center_uv = projectModelPoint(predicted, up_, sector, {0.0, radius, 0.0}, camera_tracking, frame);
    const auto top_uv = projectModelPoint(predicted, up_, sector, {0.0, radius + half_h, 0.0}, camera_tracking, frame);
    const auto right_uv = projectModelPoint(predicted, up_, sector, {half_w, radius, 0.0}, camera_tracking, frame);
    if (!center_uv || !top_uv || !right_uv) continue;

    prior.center_pixel = cv::Point2f(float(center_uv->x()), float(center_uv->y()));
    prior.hub_pixel = cv::Point2f(float(hub_uv->x()), float(hub_uv->y()));
    prior.outward_direction = prior.center_pixel - prior.hub_pixel;
    if (cv::norm(prior.outward_direction) < 2.0) continue;

    // 由规范化正交半轴构造预测外环椭圆。
    const Eigen::Vector2d e1 = *right_uv - *center_uv;
    const Eigen::Vector2d e2 = *top_uv - *center_uv;
    const Eigen::Matrix2d shape = e1 * e1.transpose() + e2 * e2.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(shape);
    if (eig.info() != Eigen::Success) continue;
    const double l0 = std::max(1.0, eig.eigenvalues()[0]);
    const double l1 = std::max(1.0, eig.eigenvalues()[1]);
    const Eigen::Vector2d major_vec = eig.eigenvectors().col(1);
    prior.projected_ellipse = cv::RotatedRect(
        prior.center_pixel,
        cv::Size2f(float(2.0 * std::sqrt(l1)), float(2.0 * std::sqrt(l0))),
        float(std::atan2(major_vec.y(), major_vec.x()) * 180.0 / std::numbers::pi));

    const auto& gap_anchors = geometry_.numbered_anchors[0].gaps;
    bool all_gaps_ok = true;
    for (std::size_t k = 0; k < 8; ++k) {
      const double phi = double(int(k) - 1) * (0.25 * std::numbers::pi);
      const cv::Point3d anchor = gap_anchors[k].value_or(
          cv::Point3d(0.7037 * half_w * std::sin(phi),
                      radius + 0.7037 * half_h * std::cos(phi), 0.0));
      const auto g_uv = projectModelPoint(predicted, up_, sector, anchor, camera_tracking, frame);
      const auto g_h = modelPointJacobian(predicted, up_, sector, anchor, camera_tracking, frame);
      if (!g_uv || !g_h) {
        all_gaps_ok = false;
        break;
      }
      prior.gap_pixels[k] = cv::Point2f(float(g_uv->x()), float(g_uv->y()));
      const Eigen::Matrix2d cov_uv = (*g_h) * predicted_cov * g_h->transpose();
      const double max_var = std::max(0.25, cov_uv.trace() * 0.5);
      prior.gap_search_sigma_px[k] = std::clamp(std::sqrt(max_var), 1.0, 40.0);
    }
    if (all_gaps_ok) priors.push_back(prior);
  }
  return priors;
}

bool FeatureFilter::updateReobservedTargetGaps(
    const Frame& frame, int sector,
    const std::array<std::optional<cv::Point2f>, 8>& gap_points,
    const std::array<float, 8>& gap_variances) {
  const double dt = std::chrono::duration<double>(frame.captured_at - previous_time_).count();
  if (!state_ || sector < 0 || sector >= 5 || dt <= 0.0 || dt > config_.max_gap_s ||
      !frame.T_barrel_camera || frame.camera_matrix.rows != 3 || frame.camera_matrix.cols != 3 ||
      frame.world_frame_id != world_frame_id_ ||
      frame.T_world_barrel.has_value() != state_->world_frame) {
    return false;
  }
  Eigen::Isometry3d tracking_barrel = Eigen::Isometry3d::Identity();
  if (state_->world_frame) {
    tracking_barrel = *frame.T_world_barrel;
  } else {
    if (!frame.q_world_barrel || !previous_world_orientation_ ||
        !frame.q_world_barrel->coeffs().allFinite() ||
        std::abs(frame.q_world_barrel->squaredNorm() - 1.0) > 1e-3) {
      return false;
    }
    tracking_barrel.linear() = frame.q_world_barrel->normalized().toRotationMatrix();
  }
  if (!tracking_barrel.matrix().allFinite() || !frame.T_barrel_camera->matrix().allFinite())
    return false;
  const Eigen::Isometry3d camera_tracking = frame.T_barrel_camera->inverse() * tracking_barrel.inverse();

  Vector x = state_->x;
  Matrix p = state_->covariance;
  predictPhase(x, dt);
  predictCovariance(p, dt, config_);

  const auto& gap_anchors = geometry_.numbered_anchors[0].gaps;
  int valid_count = 0;
  for (std::size_t k = 0; k < 8; ++k) valid_count += (gap_points[k] && gap_anchors[k]);
  const double correlation_scale = std::max(3.0, double(valid_count));
  int accepted = 0;
  for (std::size_t k = 0; k < 8; ++k) {
    if (!gap_points[k] || !gap_anchors[k]) continue;
    const auto projected = projectModelPoint(x, up_, sector, *gap_anchors[k], camera_tracking, frame);
    const auto h = modelPointJacobian(x, up_, sector, *gap_anchors[k], camera_tracking, frame);
    if (!projected || !h) continue;
    const Eigen::Vector2d measured_px(gap_points[k]->x, gap_points[k]->y);
    if (!measured_px.allFinite()) continue;
    const Eigen::Vector2d innovation = measured_px - *projected;
    const double base_var = gap_variances[k] > 0.0F ? double(gap_variances[k]) : config_.observation_variance_px2;
    const Eigen::Matrix2d meas_r =
        Eigen::Matrix2d::Identity() * std::max(config_.observation_variance_px2, base_var) * correlation_scale;
    const Eigen::Matrix2d s = (*h) * p * h->transpose() + meas_r;
    const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) continue;
    const double distance = innovation.dot(ldlt.solve(innovation));
    if (!std::isfinite(distance) || distance >= config_.gate_squared) continue;
    const Eigen::Matrix<double, 7, 2> gain = p * h->transpose() * ldlt.solve(Eigen::Matrix2d::Identity());
    const Vector next = x + gain * innovation;
    if (!next.allFinite() || std::abs(next[6]) > config_.max_speed_rad_s) continue;
    const Matrix complement = Matrix::Identity() - gain * (*h);
    p = (complement * p * complement.transpose() + gain * meas_r * gain.transpose()).eval();
    p = (0.5 * (p + p.transpose())).eval();
    x = next;
    x[3] = std::remainder(x[3], 2.0 * std::numbers::pi);
    ++accepted;
  }
  if (accepted < 4 || !x.allFinite() || !p.allFinite()) return false;
  state_->x = x;
  state_->covariance = p;
  previous_time_ = frame.captured_at;
  if (sector == target_sector_) target_seen_at_ = frame.captured_at;
  return true;
}

std::optional<Pose> FeatureFilter::update(const Pose& seed,
    const std::vector<Observation>& observations,const Frame& frame) {
  const auto fail=[&]() -> std::optional<Pose> { reset(); return {}; };
  if (!frame.T_barrel_camera || observations.empty() || observations.size()>32 ||
      !seed.center_barrel.allFinite() || !seed.normal_barrel.allFinite() ||
      seed.normal_barrel.norm()<0.9 || !std::isfinite(seed.phase_rad) ||
      !std::isfinite(geometry_.radius_m) || geometry_.radius_m<=0) return fail();
  try {
    const bool world=frame.T_world_barrel.has_value();
    // 只有姿态时，状态保存在以枪管原点为基准、轴向与世界系对齐的坐标系。
    const bool gyro_frame=!world && frame.q_world_barrel.has_value();
    Eigen::Isometry3d tracking_barrel=world ? *frame.T_world_barrel : Eigen::Isometry3d::Identity();
    if (gyro_frame) {
      if (!frame.q_world_barrel->coeffs().allFinite() ||
          std::abs(frame.q_world_barrel->squaredNorm()-1)>1e-3) return fail();
      tracking_barrel.linear()=frame.q_world_barrel->normalized().toRotationMatrix();
    }
    const Eigen::Isometry3d camera_tracking=frame.T_barrel_camera->inverse()*tracking_barrel.inverse();
    const Eigen::Vector3d center=tracking_barrel*seed.center_barrel;
    const Eigen::Vector3d n=(tracking_barrel.linear()*seed.normal_barrel).normalized();
    const Eigen::Vector3d up=(tracking_barrel.linear()*seed.up_barrel).normalized();
    if ((up-n*n.dot(up)).norm()<1e-3) return fail();
    auto measured=measurements(observations,frame.mode);
    const bool pose_center_only=follow_only_ && seed.activated &&
      std::none_of(measured.begin(),measured.end(),[](const auto& m){return !m.icon;});
    // 已激活灯臂可以联合解算符心。使用这个实测几何结果更新中心，避免重复计入 R 标。
    if (pose_center_only) measured.clear();
    struct Linearization {
      Projection pixels;
      std::array<Jacobian,6> derivatives;
    };
    std::optional<Linearization> linearization;
    const auto linearize = [&](const Vector& x) -> const Linearization& {
      if (!linearization) linearization.emplace(Linearization{
          project(x,up,geometry_.radius_m,camera_tracking,frame),
          jacobians(x,up,geometry_.radius_m,camera_tracking,frame)});
      return *linearization;
    };
    const double dt=std::chrono::duration<double>(frame.captured_at-previous_time_).count();
    bool initialized=!state_ || seed.history_reset || dt<=0 || dt>config_.max_gap_s ||
      state_->world_frame!=world || world_frame_id_!=frame.world_frame_id ||
      (!world && gyro_frame!=previous_world_orientation_.has_value());
    if (!initialized && ((center-state_->x.head<3>()).norm()>config_.max_center_jump_m ||
        normal(state_->x).dot(n)<std::cos(config_.max_normal_jump_rad))) {
      // IPPE can switch between ambiguous planar poses despite continuous image
      // features. Keep the existing state when BOTH the hub and an inactive
      // target still support its projection under the normal innovation gate.
      // Real image discontinuities and missing support still reinitialize it.
      Vector predicted=state_->x;
      predictPhase(predicted,dt);
      const auto& [pixels, hs] = linearize(predicted);
      bool hub_support=false,target_support=false;
      for (const auto& measurement : measured) {
        for (int j=0;j<6;++j) {
          if (!pixels.visible[j] || measurement.icon!=(j==0) ||
              (!measurement.icon && !measurement.inactive)) continue;
          const Eigen::Vector2d innovation=measurement.pixel-pixels.pixels[j];
          const Eigen::Matrix2d s=hs[j]*state_->covariance*hs[j].transpose()+
              config_.observation_variance_px2*Eigen::Matrix2d::Identity();
          const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
          if (ldlt.info()!=Eigen::Success || !ldlt.isPositive()) continue;
          const double distance=innovation.dot(ldlt.solve(innovation));
          if (!std::isfinite(distance) || distance>=config_.gate_squared) continue;
          if (measurement.icon) hub_support=true;
          else target_support=true;
        }
      }
      initialized=!(hub_support && target_support);
    }
    const auto before_update=follow_only_ && !initialized ? state_ : std::nullopt;
    const auto previous_up=up_;
    if (initialized) {
      linearization.reset();
      state_=FilterState{};
      state_->world_frame=world;
      auto& x=state_->x;
      x.head<3>()=center;
      x[3]=std::atan2(n.y(),n.x()); x[4]=std::atan2(n.z(),n.head<2>().norm());
      const Eigen::Vector3d v=(up-n*n.dot(up)).normalized();
      const Eigen::Vector3d r=(tracking_barrel.linear()*(seed.target_barrel-seed.center_barrel)).normalized();
      x[5]=std::atan2(r.dot(n.cross(v)),r.dot(v));
      state_->covariance.diagonal()<<.04,.04,.04,.04,.04,.01,4.0;
      target_sector_=-1;
      target_seen_at_={};
      noteObservationGap();
    } else {
      auto& x=state_->x; auto& p=state_->covariance;
      predictPhase(x,dt);
      predictCovariance(p,dt,config_);
    }
    up_=up; world_frame_id_=frame.world_frame_id;
    auto& x=state_->x; auto& p=state_->covariance;
    if (pose_center_only && !initialized) {
      cv::Mat camera;
      frame.camera_matrix.convertTo(camera,CV_64F);
      if (camera.rows!=3 || camera.cols!=3 || camera.channels()!=1) return fail();
      const double focal=std::min(camera.at<double>(0,0),camera.at<double>(1,1));
      if (!std::isfinite(focal) || focal<=0) return fail();
      const double pixel_error=std::max(seed.reprojection_error_px,std::sqrt(config_.observation_variance_px2));
      const double sigma=std::max(.03,center.norm()*pixel_error/focal);
      const Eigen::Matrix3d noise=Eigen::Matrix3d::Identity()*sigma*sigma;
      const Eigen::Matrix3d innovation_covariance=p.topLeftCorner<3,3>()+noise;
      const auto inverse=innovation_covariance.ldlt();
      if (inverse.info()!=Eigen::Success || !inverse.isPositive()) return fail();
      const Eigen::Matrix<double,7,3> gain=p.leftCols<3>()*inverse.solve(Eigen::Matrix3d::Identity());
      x+=gain*(center-x.head<3>()).eval();
      Matrix complement=Matrix::Identity();
      complement.leftCols<3>()-=gain;
      p=(complement*p*complement.transpose()+gain*noise*gain.transpose()).eval();
      p=(0.5*(p+p.transpose())).eval();
    }
    const auto& [predicted, hs] = linearize(x);
    const Eigen::Matrix2d r=Eigen::Matrix2d::Identity()*config_.observation_variance_px2;
    Eigen::MatrixXd costs=Eigen::MatrixXd::Constant(measured.size(),6,
      std::numeric_limits<double>::infinity());
    for (int j=0;j<6;++j) {
      if (!predicted.visible[j]) continue;
      const Eigen::Matrix2d s=hs[j]*p*hs[j].transpose()+r;
      const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
      if (ldlt.info()!=Eigen::Success || !ldlt.isPositive()) continue;
      for (std::size_t i=0;i<measured.size();++i) {
        if (measured[i].icon!=(j==0)) continue;
        const Eigen::Vector2d innovation=measured[i].pixel-predicted.pixels[j];
        costs(i,j)=innovation.dot(ldlt.solve(innovation));
      }
    }
    const auto assignments=hungarianAssign(costs,config_.gate_squared);
    const bool confirm_sector=follow_only_ && frame.mode==Mode::Big;
    Pose output=seed;
    output.feature_matches.clear();
    std::array<int,5> inactive; inactive.fill(-1);
    std::array<int,5> matched; matched.fill(-1);
    std::array<bool,5> unambiguous{};
    for (std::size_t i=0;i<measured.size();++i) {
      const int j=assignments[i];
      if (j<0) continue;
      const auto& [projected, derivatives] = linearize(x);
      if (!projected.visible[j]) continue;
      const Jacobian& h=derivatives[j];
      const Eigen::Vector2d innovation=measured[i].pixel-projected.pixels[j];
      const Eigen::Matrix2d projected_covariance=h*p*h.transpose();
      const Eigen::Matrix2d s=projected_covariance+r;
      const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
      if (ldlt.info()!=Eigen::Success || !ldlt.isPositive()) continue;
      const double distance=innovation.dot(ldlt.solve(innovation));
      if (!std::isfinite(distance) || distance>=config_.gate_squared) continue;
      // 保持关联门限独立于观测权重，避免低质量候选因方差较大而更容易关联。
      const Eigen::Matrix2d measurement_noise = r * measured[i].variance_scale;
      const Eigen::Matrix2d inverse_covariance = measured[i].variance_scale == 1.0
          ? Eigen::Matrix2d(ldlt.solve(Eigen::Matrix2d::Identity()))
          : Eigen::Matrix2d((projected_covariance + measurement_noise).ldlt().solve(Eigen::Matrix2d::Identity()));
      const Eigen::Matrix<double,7,2> gain = p * h.transpose() * inverse_covariance;
      const Vector next=x+gain*innovation;
      if (!next.allFinite() || std::abs(next[6])>config_.max_speed_rad_s) continue;
      const Matrix complement=Matrix::Identity()-gain*h;
      p=(complement*p*complement.transpose()+gain*measurement_noise*gain.transpose()).eval();
      p=(0.5*(p+p.transpose())).eval();
      x=next; x[3]=std::remainder(x[3],2*std::numbers::pi);
      linearization.reset();
      output.feature_matches.push_back({measured[i].index,j,distance});
      if (j) {
        matched[j-1]=measured[i].index;
        if (measured[i].inactive) {
          inactive[j-1]=measured[i].index;
          if (confirm_sector) {
            double second=std::numeric_limits<double>::infinity();
            for (int k=1;k<=5;++k) if (k!=j) second=std::min(second,costs(i,k));
            unambiguous[j-1]=second-costs(i,j)>=2.0;
          }
        }
      }
    }
    // 紧耦合观测更新：将已关联未激活叶片的 8 个靶面精修关键点按异方差直接送入七状态 EKF。
    const auto& gap_anchors = geometry_.numbered_anchors[0].gaps;
    const bool has_gap_anchors = std::any_of(gap_anchors.begin(), gap_anchors.end(),
                                             [](const auto& a) { return a.has_value(); });
    if (has_gap_anchors && !pose_center_only) {
      for (int sector = 0; sector < 5; ++sector) {
        if (inactive[sector] < 0) continue;
        const auto& kpts = observations[inactive[sector]].keypoints;
        if (!kpts.image_contours || kpts.target_ring_status == TargetRingStatus::Rejected) continue;
        const auto& contours = *kpts.image_contours;
        int valid_count = 0;
        for (std::size_t k = 0; k < 8; ++k) valid_count += (contours.gap_points[k] && gap_anchors[k]);
        if (valid_count == 0) continue;
        const double correlation_scale = std::max(2.5, double(valid_count));
        for (std::size_t k = 0; k < 8; ++k) {
          if (!contours.gap_points[k] || !gap_anchors[k]) continue;
          const Eigen::Vector2d pixel(contours.gap_points[k]->x, contours.gap_points[k]->y);
          if (!pixel.allFinite()) continue;
          const auto projected = projectModelPoint(x, up_, sector, *gap_anchors[k], camera_tracking, frame);
          const auto h = modelPointJacobian(x, up_, sector, *gap_anchors[k], camera_tracking, frame);
          if (!projected || !h) continue;
          const Eigen::Vector2d innovation = pixel - *projected;
          const double point_var = contours.gap_point_variances[k] > 0.0F
              ? double(contours.gap_point_variances[k])
              : config_.observation_variance_px2;
          // 按同靶环有效点数放大单点方差，避免多点相关性导致协方差过度收窄与法向抖动。
          const Eigen::Matrix2d meas_r =
              Eigen::Matrix2d::Identity() * std::max(config_.observation_variance_px2, point_var) * correlation_scale;
          const Eigen::Matrix2d s = (*h) * p * h->transpose() + meas_r;
          const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
          if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) continue;
          const double distance = innovation.dot(ldlt.solve(innovation));
          if (!std::isfinite(distance) || distance >= config_.gate_squared) continue;
          const Eigen::Matrix<double, 7, 2> gain = p * h->transpose() * ldlt.solve(Eigen::Matrix2d::Identity());
          const Vector next = x + gain * innovation;
          if (!next.allFinite() || std::abs(next[6]) > config_.max_speed_rad_s) continue;
          const Matrix complement = Matrix::Identity() - gain * (*h);
          p = (complement * p * complement.transpose() + gain * meas_r * gain.transpose()).eval();
          p = (0.5 * (p + p.transpose())).eval();
          x = next;
          x[3] = std::remainder(x[3], 2.0 * std::numbers::pi);
          linearization.reset();
        }
      }
    }
    int selected=target_sector_;
    if (selected<0 || inactive[selected]<0) {
      selected=-1;
      double best=std::numeric_limits<double>::infinity();
      for (int j=0;j<5;++j) if (inactive[j]>=0) {
        const double distance=(tracking_barrel.inverse()*(x.head<3>()+
          geometry_.radius_m*radial(x,up_,j))-seed.target_barrel).squaredNorm();
        if (distance<best) { best=distance; selected=j; }
      }
    }
    if (confirm_sector) {
      const auto now=frame.captured_at;
      if (target_sector_>=0 && selected!=target_sector_ && selected>=0 && unambiguous[selected]) {
        if (pending_sector_!=selected) {
          pending_sector_=selected;
          pending_since_=now;
        }
        if (now-pending_since_<std::chrono::milliseconds(config_.big_sector_confirmation_ms)) selected=-1;
      } else {
        noteObservationGap();
        if (selected>=0 && !unambiguous[selected]) selected=-1;
      }
      if (selected>=0 && inactive[selected]>=0) {
        target_seen_at_=now;
        noteObservationGap();
      } else if (target_sector_>=0 && target_seen_at_!=TimePoint{} &&
                 now-target_seen_at_<=std::chrono::milliseconds(config_.big_sector_hold_ms)) {
        selected=target_sector_;
      }
    }
    if (selected<0 && follow_only_ && seed.activated)
      selected=std::max(0,target_sector_);
    if (!p.allFinite() || !x.allFinite()) return fail();
    if (selected<0) {
      if (!before_update) return fail();
      state_=before_update;
      up_=previous_up;
      return {};
    }
    output.history_reset=initialized;
    output.target_switched=target_sector_>=0 && selected!=target_sector_;
    target_sector_=selected; state_->target_sector=selected;
    output.center_barrel=tracking_barrel.inverse()*x.head<3>();
    output.normal_barrel=tracking_barrel.linear().transpose()*normal(x);
    const Eigen::Vector3d v=tracking_barrel.linear().transpose()*radial(x,up_,selected);
    output.target_barrel=output.center_barrel+geometry_.radius_m*v;
    output.phase_rad=std::remainder(x[5]+selected*sector_angle,2*std::numbers::pi);
    output.T_barrel_model.translation()=output.center_barrel;
    const Eigen::Vector3d n_cam=frame.T_barrel_camera->linear().transpose()*output.normal_barrel;
    const Eigen::Vector3d n_away=n_cam.z()>=0.0 ? output.normal_barrel : -output.normal_barrel;
    output.T_barrel_model.linear().col(0)=n_away.cross(v);
    output.T_barrel_model.linear().col(1)=v;
    output.T_barrel_model.linear().col(2)=-n_away;
    output.normal_vertical_component=std::abs(output.normal_barrel.dot(output.up_barrel));
    output.blades.clear();
    for (int j=0;j<5;++j) if (matched[j]>=0) {
      BladeMatch blade;
      blade.observation_index=matched[j]; blade.sector=(j-selected+5)%5;
      blade.activated=observations[matched[j]].keypoints.class_id==1;
      const auto original=std::find_if(seed.blades.begin(),seed.blades.end(),[&](const auto& b) {
        return b.observation_index==blade.observation_index;
      });
      if (original!=seed.blades.end()) blade.pca_anchors=original->pca_anchors;
      blade.target_barrel=tracking_barrel.inverse()*(x.head<3>()+geometry_.radius_m*radial(x,up_,j));
      output.blades.push_back(blade);
    }
    output.filter_state=state_;
    output.q_world_barrel=frame.T_world_barrel
      ? std::optional<Eigen::Quaterniond>(Eigen::Quaterniond(frame.T_world_barrel->linear()))
      : frame.q_world_barrel;
    if (output.target_barrel.x()<=0) return fail();
    previous_time_=frame.captured_at;
    previous_world_orientation_=output.q_world_barrel;
    previous_world_origin_=tracking_barrel.translation();
    return output;
  } catch (const cv::Exception&) { return fail(); }
}
} // namespace Rune
