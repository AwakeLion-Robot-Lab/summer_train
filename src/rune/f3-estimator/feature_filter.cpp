// Algorithm reference: Alliance-Algorithm/rmcs_auto_aim_v2 (MIT).
// Adapted to Rune's optical-camera / barrel / optional full-world transforms.
#include "rune/f3-estimator/feature_filter.hpp"
#include "rune/f3-estimator/association.hpp"

#include <Eigen/Cholesky>
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

FeatureFilter::FeatureFilter(Geometry geometry,FeatureFilterConfig config,bool gimbal_only)
  : geometry_(std::move(geometry)),config_(config),gimbal_only_(gimbal_only) {
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
  } else {
    if (frame.q_world_barrel.has_value()!=previous_world_orientation_.has_value()) return {};
    if (previous_world_orientation_) {
      const auto& current=*frame.q_world_barrel;
      if (!current.coeffs().allFinite() || std::abs(current.squaredNorm()-1)>1e-3) return {};
      // 相对坐标中的滤波状态属于上一次有效曝光时的云台坐标系。
      const auto rotation=previous_world_orientation_->normalized();
      result.observed_direction=rotation*result.observed_direction;
      result.predicted_direction=rotation*result.predicted_direction;
    }
  }
  if (!result.observed_direction.allFinite() || !result.predicted_direction.allFinite()) return {};
  return result;
}

std::optional<FeatureFilter::GimbalPrediction> FeatureFilter::predictGimbal(
    const Frame& frame,TimePoint at,bool center_only) const {
  const double dt=std::chrono::duration<double>(at-previous_time_).count();
  if (!gimbal_only_ || !state_ || target_sector_<0 || frame.captured_at<previous_time_ ||
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
  if (!gimbal_only_ || !state_ || target_sector_<0 || dt<=0 || dt>config_.max_gap_s ||
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

std::optional<Pose> FeatureFilter::update(const Pose& seed,
    const std::vector<Observation>& observations,const Frame& frame) {
  const auto fail=[&]() -> std::optional<Pose> { reset(); return {}; };
  if (!frame.T_barrel_camera || observations.empty() || observations.size()>32 ||
      !seed.center_barrel.allFinite() || !seed.normal_barrel.allFinite() ||
      seed.normal_barrel.norm()<0.9 || !std::isfinite(seed.phase_rad) ||
      !std::isfinite(geometry_.radius_m) || geometry_.radius_m<=0) return fail();
  try {
    const bool world=frame.T_world_barrel.has_value();
    const bool gyro_frame=gimbal_only_ && !world && frame.q_world_barrel.has_value();
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
    const bool pose_center_only=gimbal_only_ && seed.activated &&
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
      (gimbal_only_ && !world && gyro_frame!=previous_world_orientation_.has_value());
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
    const auto before_update=gimbal_only_ && !initialized ? state_ : std::nullopt;
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
    const bool confirm_sector=gimbal_only_ && frame.mode==Mode::Big;
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
    if (selected<0 && gimbal_only_ && seed.activated)
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
    output.T_barrel_model.linear().col(0)=v.cross(output.normal_barrel);
    output.T_barrel_model.linear().col(1)=v;
    output.T_barrel_model.linear().col(2)=output.normal_barrel;
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
