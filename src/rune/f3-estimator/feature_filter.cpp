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
struct Measurement { Eigen::Vector2d pixel; std::size_t index; bool icon,inactive; };
std::vector<Measurement> measurements(const std::vector<Observation>& observations,Mode mode) {
  std::vector<Measurement> result;
  // Deduplicate the shared hub before correction: five NN detections do not
  // represent five statistically independent R observations.
  for (std::size_t i=0;i<observations.size();++i) {
    const auto& k=observations[i].keypoints;
    const Eigen::Vector2d pixel(k.center_r.x,k.center_r.y);
    if (!pixel.allFinite() || (k.class_id!=0 && k.class_id!=1)) continue;
    if (std::none_of(result.begin(),result.end(),[&](const auto& m) {
          return m.icon && (m.pixel-pixel).norm()<2;
        })) result.push_back({pixel,i,true,false});
  }
  for (std::size_t i=0;i<observations.size();++i) {
    const auto& k=observations[i].keypoints;
    if ((k.class_id!=0 && k.class_id!=1) || (mode==Mode::Big && k.class_id==1)) continue;
    const cv::Point2d center=(cv::Point2d(k.top)+cv::Point2d(k.left)+
                             cv::Point2d(k.right)+cv::Point2d(k.bottom))*0.25;
    const Eigen::Vector2d pixel(center.x,center.y);
    if (pixel.allFinite()) result.push_back({pixel,i,false,k.class_id==0});
  }
  return result;
}
} // namespace

FeatureFilter::FeatureFilter(Geometry geometry,FeatureFilterConfig config)
  : geometry_(std::move(geometry)),config_(config) {
  for (double v : {config.gate_squared,config.observation_variance_px2,
      config.center_process_variance_m2_s,config.normal_process_variance_rad2_s,
      config.angular_acceleration_variance,config.max_gap_s,
      config.max_center_jump_m,config.max_normal_jump_rad,config.max_speed_rad_s})
    if (!std::isfinite(v) || v<=0) throw std::invalid_argument("Invalid rune filter configuration");
}
void FeatureFilter::reset() { state_.reset(); previous_time_={}; target_sector_=-1; }

std::optional<Pose> FeatureFilter::update(const Pose& seed,
    const std::vector<Observation>& observations,const Frame& frame) {
  const auto fail=[&]() -> std::optional<Pose> { reset(); return {}; };
  if (!frame.T_barrel_camera || observations.empty() || observations.size()>32 ||
      !seed.center_barrel.allFinite() || !seed.normal_barrel.allFinite() ||
      seed.normal_barrel.norm()<0.9 || !std::isfinite(seed.phase_rad) ||
      !std::isfinite(geometry_.radius_m) || geometry_.radius_m<=0) return fail();
  try {
    const bool world=frame.T_world_barrel.has_value();
    const Eigen::Isometry3d tracking_barrel=world ? *frame.T_world_barrel : Eigen::Isometry3d::Identity();
    const Eigen::Isometry3d camera_tracking=frame.T_barrel_camera->inverse()*tracking_barrel.inverse();
    const Eigen::Vector3d center=tracking_barrel*seed.center_barrel;
    const Eigen::Vector3d n=(tracking_barrel.linear()*seed.normal_barrel).normalized();
    const Eigen::Vector3d up=(tracking_barrel.linear()*seed.up_barrel).normalized();
    if ((up-n*n.dot(up)).norm()<1e-3) return fail();
    const double dt=std::chrono::duration<double>(frame.captured_at-previous_time_).count();
    bool initialized=!state_ || seed.history_reset || dt<=0 || dt>config_.max_gap_s ||
      state_->world_frame!=world || world_frame_id_!=frame.world_frame_id;
    if (!initialized) initialized=(center-state_->x.head<3>()).norm()>config_.max_center_jump_m ||
      normal(state_->x).dot(n)<std::cos(config_.max_normal_jump_rad);
    if (initialized) {
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
    } else {
      auto& x=state_->x; auto& p=state_->covariance;
      x[5]+=x[6]*dt;
      Matrix f=Matrix::Identity(); f(5,6)=dt;
      p=(f*p*f.transpose()).eval();
      p.diagonal().head<3>().array()+=config_.center_process_variance_m2_s*dt;
      p.diagonal().segment<2>(3).array()+=config_.normal_process_variance_rad2_s*dt;
      const double q=config_.angular_acceleration_variance;
      p(5,5)+=q*dt*dt*dt/3; p(5,6)+=q*dt*dt/2;
      p(6,5)+=q*dt*dt/2; p(6,6)+=q*dt;
    }
    up_=up; world_frame_id_=frame.world_frame_id;
    auto& x=state_->x; auto& p=state_->covariance;
    const auto measured=measurements(observations,frame.mode);
    const auto predicted=project(x,up_,geometry_.radius_m,camera_tracking,frame);
    const auto hs=jacobians(x,up_,geometry_.radius_m,camera_tracking,frame);
    const Eigen::Matrix2d r=Eigen::Matrix2d::Identity()*config_.observation_variance_px2;
    Eigen::MatrixXd costs=Eigen::MatrixXd::Constant(measured.size(),6,
      std::numeric_limits<double>::infinity());
    for (std::size_t i=0;i<measured.size();++i) for (int j=0;j<6;++j) {
      if (!predicted.visible[j] || measured[i].icon!=(j==0)) continue;
      const Eigen::Vector2d innovation=measured[i].pixel-predicted.pixels[j];
      const Eigen::Matrix2d s=hs[j]*p*hs[j].transpose()+r;
      const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
      if (ldlt.info()==Eigen::Success && ldlt.isPositive())
        costs(i,j)=innovation.dot(ldlt.solve(innovation));
    }
    const auto assignments=hungarianAssign(costs,config_.gate_squared);
    Pose output=seed;
    output.feature_matches.clear();
    std::array<int,5> inactive; inactive.fill(-1);
    std::array<int,5> matched; matched.fill(-1);
    for (std::size_t i=0;i<measured.size();++i) {
      const int j=assignments[i];
      if (j<0) continue;
      // Recompute BOTH projection and Jacobian after every feature update.
      const auto projected=project(x,up_,geometry_.radius_m,camera_tracking,frame);
      if (!projected.visible[j]) continue;
      const Jacobian h=jacobians(x,up_,geometry_.radius_m,camera_tracking,frame)[j];
      const Eigen::Vector2d innovation=measured[i].pixel-projected.pixels[j];
      const Eigen::Matrix2d s=h*p*h.transpose()+r;
      const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
      if (ldlt.info()!=Eigen::Success || !ldlt.isPositive()) continue;
      const double distance=innovation.dot(ldlt.solve(innovation));
      if (!std::isfinite(distance) || distance>=config_.gate_squared) continue;
      const Eigen::Matrix<double,7,2> gain=p*h.transpose()*ldlt.solve(Eigen::Matrix2d::Identity());
      const Vector next=x+gain*innovation;
      if (!next.allFinite() || std::abs(next[6])>config_.max_speed_rad_s) continue;
      const Matrix complement=Matrix::Identity()-gain*h;
      p=(complement*p*complement.transpose()+gain*r*gain.transpose()).eval();
      p=(0.5*(p+p.transpose())).eval();
      x=next; x[3]=std::remainder(x[3],2*std::numbers::pi);
      output.feature_matches.push_back({measured[i].index,j,distance});
      if (j) {
        matched[j-1]=measured[i].index;
        if (measured[i].inactive) inactive[j-1]=measured[i].index;
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
    if (selected<0 || !p.allFinite()) return fail();
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
    previous_time_=frame.captured_at;
    if (output.target_barrel.x()<=0) return fail();
    return output;
  } catch (const cv::Exception&) { return fail(); }
}
} // namespace Rune
