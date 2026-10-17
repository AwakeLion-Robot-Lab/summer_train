#include "rune/f3-estimator/estimator.hpp"
#include "rune/f3-estimator/association.hpp"
#include <Eigen/Eigenvalues>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {
using namespace Rune;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
TimePoint stamp(double seconds) {
  return TimePoint{}+std::chrono::seconds(10)+
    std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(seconds));
}
void assignment() {
  Eigen::MatrixXd c(2,2); c<<1,2,1.1,8;
  auto a=hungarianAssign(c,9.21);
  check(a==std::vector<int>({1,0}),"Hungarian became greedy");
  c<<100,200,300,400;
  check(hungarianAssign(c,9.21)==std::vector<int>({-1,-1}),"square rejection failed");
  c.resize(3,1); c<<.1,.2,std::numeric_limits<double>::quiet_NaN();
  check(hungarianAssign(c,9.21)==std::vector<int>({0,-1,-1}),"rectangular matching failed");
  // Exhaustive oracle over small random matrices, including unmatched rows.
  std::mt19937 rng(47); std::uniform_real_distribution<double> random(0,14);
  for (int iteration=0;iteration<100;++iteration) {
    const int rows=1+iteration%4,cols=1+(iteration/4)%4;
    Eigen::MatrixXd cost(rows,cols);
    for (int i=0;i<rows;++i) for (int j=0;j<cols;++j) cost(i,j)=random(rng);
    const auto result=hungarianAssign(cost,9.21);
    double actual=0,best=1e30;
    for (int i=0;i<rows;++i) actual+=result[i]<0 ? 9.21 : cost(i,result[i]);
    const auto solve=[&](auto&& self,int row,unsigned used,double sum)->void {
      if (row==rows) { best=std::min(best,sum); return; }
      self(self,row+1,used,sum+9.21);
      for (int j=0;j<cols;++j) if (!(used&(1U<<j)) && cost(row,j)<9.21)
        self(self,row+1,used|(1U<<j),sum+cost(row,j));
    };
    solve(solve,0,0,0);
    check(std::abs(actual-best)<1e-9,"assignment differs from exhaustive optimum");
  }
}
void fits() {
  for (double direction : {-1.0,1.0}) {
    MotionFitter fitter;
    for (int i=0;i<=300;++i) {
      const double t=.02*i;
      fitter.push(stamp(t),direction*(1.2*t-.8/1.94*std::cos(1.94*t+.7)));
    }
    const auto sine=fitter.fitSine(),linear=fitter.fitLinear();
    check(sine && linear && sine->cost<1e-12,"sinusoidal angle fit failed");
    check(std::abs(sine->omega-1.94)<1e-8,"frequency sweep failed");
    const double expected=direction*(1.2*.4+.8/1.94*(std::cos(1.94*6+.7)-std::cos(1.94*6.4+.7)));
    check(std::abs(sine->delta(0,.4)-expected)<1e-7,"signed phase integral failed");
  }
  MotionFitter weighted({.weight_half_life_s=.5}),uniform({.weight_half_life_s=1e6});
  for (int i=0;i<=300;++i) {
    const double t=.02*i,phase=t+(t<2 ? .5 : 0);
    weighted.push(stamp(t),phase); uniform.push(stamp(t),phase);
  }
  check(std::abs(weighted.fitLinear()->v-1)<std::abs(uniform.fitLinear()->v-1)*.3,
        "history is not exponentially weighted");
  MotionFitter bounded({.max_samples=100});
  for (int i=0;i<400;++i) bounded.push(stamp(i*.02),i*.02);
  const auto size=bounded.size();
  bounded.push(stamp(1),2); bounded.push(stamp(8),NAN);
  check(size==100 && bounded.size()==100,"sample cap / invalid stamps failed");
}
void selection() {
  EstimatorStage estimator;
  Pose pose; pose.center_barrel={7,0,0}; pose.normal_barrel={1,0,0};
  Motion motion;
  for (int i=0;i<=400;++i) {
    const double t=i*.02;
    pose.captured_at=stamp(t); pose.phase_rad=std::remainder(t,2*std::numbers::pi);
    motion=estimator.update(pose,Mode::Big);
  }
  check(motion.ready && motion.model==MotionModel::Linear,"constant rotation in big mode failed");
  check(std::abs(motion.angular_speed_rad_s-1)<1e-8,"linear speed incorrect");
  const auto snapshot=motion.phase_delta_rad;
  for (int i=401;i<=1200;++i) {
    const double t=i*.02;
    pose.captured_at=stamp(t);
    pose.phase_rad=std::remainder(1.2*t-.8/1.94*std::cos(1.94*t+.7),2*std::numbers::pi);
    motion=estimator.update(pose,Mode::Big);
  }
  check(motion.ready && motion.model==MotionModel::Sine,"linear-to-sine switching failed");
  check(std::abs(*snapshot(.4)-.4)<1e-8,"prediction snapshot changed after later fits");
  check(!motion.phase_delta_rad(-1) && !motion.phase_delta_rad(NAN),"invalid horizon accepted");
  check(motion.phase_delta_signed_rad && motion.phase_delta_signed_rad(-.2) &&
        !motion.phase_delta_signed_rad(NAN),"signed MPC phase snapshot missing");
  for (int i=1201;i<=2000;++i) {
    const double t=i*.02;
    pose.captured_at=stamp(t); pose.phase_rad=std::remainder(t,2*std::numbers::pi);
    motion=estimator.update(pose,Mode::Big);
  }
  check(motion.ready && motion.model==MotionModel::Linear,"sine-to-linear switching failed");
  pose.captured_at+=std::chrono::seconds(1);
  check(!estimator.update(pose,Mode::Big).ready,"time gap retained ready fit");
  estimator.reset();
  for (int i=0;i<320;++i) {
    const double t=.02*i;
    pose.captured_at=stamp(t);
    pose.phase_rad=std::remainder(1.2*t-.8/1.94*std::cos(1.94*t),2*std::numbers::pi);
    motion=estimator.update(pose,Mode::Small);
  }
  check(!motion.ready && motion.model==MotionModel::Linear,
        "small rune accepted sinusoidal rotation as a ready fit");
  check(std::isinf(motion.sine_cost_rad2) && motion.sine_frequency_rad_s==0,
        "small rune ran the big-rune sine fitter");
}
void modeSeparation() {
  for (double direction : {-1.0,1.0}) {
    EstimatorStage estimator;
    Pose pose; pose.center_barrel={7,0,0}; pose.normal_barrel={1,0,0};
    Motion motion;
    for (int i=0;i<=320;++i) {
      const double t=.02*i;
      pose.captured_at=stamp(t);
      pose.phase_rad=std::remainder(direction*.9*t,2*std::numbers::pi);
      motion=estimator.update(pose,Mode::Small);
    }
    check(motion.ready && motion.model==MotionModel::Linear,"small-rune constant fit failed");
    check(std::abs(*motion.phase_delta_rad(.4)-direction*.36)<1e-8,
          "small-rune signed prediction failed");
    const auto snapshot=motion.phase_delta_rad;
    const auto switched=estimator.update(pose,Mode::Big);
    check(!switched.ready && !switched.phase_delta_rad,
          "small-to-big switch reused small-rune history");
    for (int i=321;i<=670;++i) {
      const double t=.02*i;
      pose.captured_at=stamp(t);
      pose.phase_rad=std::remainder(direction*(1.2*t-.8/1.94*std::cos(1.94*t+.7)),
                                    2*std::numbers::pi);
      motion=estimator.update(pose,Mode::Big);
    }
    check(motion.ready && motion.model==MotionModel::Sine,
          "big-rune sine fit failed after small-rune mode");
    check(std::abs(*snapshot(.4)-direction*.36)<1e-8,
          "big-rune fit changed a returned small-rune prediction");
    const auto small=estimator.update(pose,Mode::Small);
    check(!small.ready && !small.phase_delta_rad,
          "big-to-small switch reused big-rune history");
  }
}
struct Scene {
  Frame frame;
  Pose seed;
  std::vector<Observation> observations;
  Eigen::Vector3d center{7,.1,.2},normal;
  Scene(double t,double phase,int inactive=0,bool world=true) {
    frame.mode=Mode::Small; frame.captured_at=stamp(t);
    frame.camera_matrix=(cv::Mat_<double>(3,3)<<1200,0,640,0,1200,480,0,0,1);
    frame.distortion=(cv::Mat_<double>(1,5)<<-.15,.04,.002,-.001,0);
    Eigen::Isometry3d barrel_camera=Eigen::Isometry3d::Identity();
    barrel_camera.linear()<<0,0,1,-1,0,0,0,-1,0;
    barrel_camera.translation()<<.07,.02,.03;
    frame.T_barrel_camera=barrel_camera;
    Eigen::Isometry3d world_barrel=Eigen::Isometry3d::Identity();
    if (world) {
      world_barrel.linear()=(Eigen::AngleAxisd(.08*std::sin(t),Eigen::Vector3d::UnitZ())*
        Eigen::AngleAxisd(.05*std::sin(.7*t),Eigen::Vector3d::UnitY())).toRotationMatrix();
      world_barrel.translation()<<.03*t,.02*std::sin(t),0;
      frame.T_world_barrel=world_barrel;
      frame.q_world_barrel=Eigen::Quaterniond(world_barrel.linear());
    }
    normal=Eigen::Vector3d(1,.2,.08).normalized();
    const Eigen::Vector3d up=(Eigen::Vector3d::UnitZ()-normal*normal.z()).normalized();
    const Eigen::Vector3d tangent=normal.cross(up);
    const auto image=[&](const Eigen::Vector3d& world_point) {
      const Eigen::Vector3d p=barrel_camera.inverse()*world_barrel.inverse()*world_point;
      std::vector<cv::Point2d> pixels;
      cv::projectPoints(std::vector<cv::Point3d>{{p.x(),p.y(),p.z()}},cv::Vec3d(0,0,0),
        cv::Vec3d(0,0,0),frame.camera_matrix,frame.distortion,pixels);
      return cv::Point2f(pixels[0]);
    };
    for (int sector=0;sector<5;++sector) {
      const double a=phase+sector*2*std::numbers::pi/5;
      const Eigen::Vector3d radial=up*std::cos(a)+tangent*std::sin(a);
      const Eigen::Vector3d target=center+.7*radial,side=radial.cross(normal);
      Observation o;
      o.keypoints={image(target+.06*radial),image(target-.07*side),
        image(target+.07*side),image(target-.06*radial),image(center),sector==inactive ? 0 : 1,.9F};
      observations.push_back(o);
      if (sector==inactive) seed.target_barrel=world_barrel.inverse()*target;
    }
    seed.center_barrel=world_barrel.inverse()*center;
    seed.normal_barrel=world_barrel.linear().transpose()*normal;
    seed.up_barrel=world_barrel.linear().transpose()*Eigen::Vector3d::UnitZ();
    seed.phase_rad=std::remainder(phase+inactive*2*std::numbers::pi/5,2*std::numbers::pi);
    seed.captured_at=frame.captured_at;
  }
};
void filtering() {
  FeatureFilter filter({.7,.07,.06});
  EstimatorStage estimator;
  std::optional<Pose> filtered;
  Motion motion;
  for (int i=0;i<=350;++i) {
    const double t=.02*i;
    Scene s(t,.3+t,i<140 ? 0 : 2);
    if (i%2) std::reverse(s.observations.begin(),s.observations.end());
    auto foreign=s.observations[0];
    foreign.keypoints.top.x+=400; foreign.keypoints.bottom.x+=400;
    foreign.keypoints.left.x+=400; foreign.keypoints.right.x+=400;
    s.observations.push_back(foreign);
    s.observations.push_back(s.observations[1]); // duplicate blade
    filtered=filter.update(s.seed,s.observations,s.frame);
    check(filtered && filtered->filter_state,"feature EKF failed");
    check(filtered->history_reset==(i==0),"moving camera or blade switch reset filter");
    check(filtered->target_switched==(i==140),"persistent blade identity failed");
    check(filtered->feature_matches.size()==6,"duplicate hub / blade or foreign feature accepted");
    std::array<bool,6> used{};
    for (const auto& m : filtered->feature_matches) {
      check(!used[m.feature_id],"feature updated twice"); used[m.feature_id]=true;
      check(m.mahalanobis_squared<9.21,"Mahalanobis gate ignored");
    }
    const auto& state=*filtered->filter_state;
    check((state.x.head<3>()-s.center).norm()<.02,"world-frame center drifted");
    check(std::abs(state.x[5]-(.3+t))<.02,"canonical phase drift / blade step");
    check((state.covariance-state.covariance.transpose()).norm()<1e-10,"asymmetric covariance");
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,7,7>> eigen(state.covariance);
    check(eigen.eigenvalues().minCoeff()>-1e-10,"covariance not positive semidefinite");
    motion=estimator.update(*filtered,Mode::Small);
    if (i==140) check(motion.ready,"blade switch cleared fit history");
  }
  check(motion.ready && motion.model==MotionModel::Linear,"filter-to-fit integration failed");
  check(std::abs(motion.angular_speed_rad_s-1)<.02,"filter angular velocity failed");
  Scene reset(8,8.3); reset.frame.world_frame_id=2;
  filtered=filter.update(reset.seed,reset.observations,reset.frame);
  check(filtered && filtered->history_reset,"world reset retained filter");
  reset.frame.mode=Mode::Big;
  for (auto& o : reset.observations) o.keypoints.class_id=1;
  check(!filter.update(reset.seed,reset.observations,reset.frame),"arm-only observations became target centers");
  Scene local(0,.3,0,false);
  check(filter.update(local.seed,local.observations,local.frame).has_value(),"barrel-frame fallback failed");
}
void filteredSine() {
  for (double direction : {-1.0,1.0}) {
    FeatureFilter filter({.7,.07,.06});
    EstimatorStage estimator;
    Motion motion;
    const auto angle=[&](double t) { return .3+direction*(1.2*t-.8/1.94*std::cos(1.94*t+.7)); };
    for (int i=0;i<=350;++i) {
      const double t=.02*i;
      Scene s(t,angle(t));
      s.frame.mode=Mode::Big;
      s.observations[2].keypoints.class_id=0;
      const auto pose=filter.update(s.seed,s.observations,s.frame);
      check(pose.has_value(),"big-rune sine observations rejected by EKF");
      motion=estimator.update(*pose,Mode::Big);
    }
    check(motion.ready && motion.model==MotionModel::Sine,"filtered big-rune sine model not selected");
    check(std::abs(*motion.phase_delta_rad(.4)-(angle(7.4)-angle(7)))<.02,
          "filtered sine prediction has wrong phase or direction");
  }
}
} // namespace
int main() {
  try { assignment(); fits(); selection(); modeSeparation(); filtering(); filteredSine(); std::cout<<"rune_estimator_smoke passed\n"; }
  catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
