#include "rune/f2-building/builder.hpp"
#include "sensor/camera/camera_calibration.hpp"
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
std::vector<cv::Point3d> boundary(const std::vector<cv::Point3d>& corners) {
  std::vector<cv::Point3d> samples;
  for (std::size_t i=0;i<corners.size();++i)
    for (int j=0;j<24;++j)
      samples.push_back(corners[i]+(corners[(i+1)%corners.size()]-corners[i])*(j/24.0));
  return samples;
}
struct Scene {
  Rune::Geometry g{0.7,0.12,0.10};
  Rune::Frame frame;
  Rune::Observation observation;
  Eigen::Vector3d center, target;
  Scene(cv::Vec3d r, cv::Vec3d t, bool distortion=true) {
    g.use_pca_anchors=false;
    frame.camera_matrix=(cv::Mat_<double>(3,3)<<1400,0,640,0,1380,480,0,0,1);
    if (distortion) frame.distortion=(cv::Mat_<double>(1,5)<<-0.18,0.06,0.001,-0.002,0.0);
    frame.captured_at=std::chrono::steady_clock::now();
    Eigen::Isometry3d transform=Eigen::Isometry3d::Identity();
    transform.linear()<<0,0,1,-1,0,0,0,-1,0;
    transform.translation()<<0.05,-0.02,0.03;
    frame.T_barrel_camera=transform;
    const std::vector<cv::Point3d> anchors{{0,.8,0},{-.12,.7,0},{.12,.7,0},{0,.6,0},{0,0,0}};
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(anchors,r,t,frame.camera_matrix,frame.distortion,pixels);
    observation.keypoints={pixels[0],pixels[1],pixels[2],pixels[3],pixels[4],0,.9F};
    // Synthetic test fixtures, deliberately NOT claimed to be regulation geometry.
    g.armor_boundary_m=boundary({{-.12,.6,0},{.12,.6,0},{.12,.8,0},{-.12,.8,0}});
    g.arm_boundary_m=boundary({{-.035,.15,0},{.035,.15,0},{.035,.50,0},{-.035,.50,0}});
    g.center_r_boundary_m=boundary({{-.025,-.025,0},{.025,-.025,0},{.025,.025,0},{-.025,.025,0}});
    const auto contour=[&](const std::vector<cv::Point3d>& model) {
      std::vector<cv::Point2d> projected;
      cv::projectPoints(model,r,t,frame.camera_matrix,frame.distortion,projected);
      std::vector<cv::Point> result;
      for (const auto& p : projected) result.emplace_back(cvRound(p.x),cvRound(p.y));
      return result;
    };
    observation.armor=contour(g.armor_boundary_m);
    observation.arm=contour(g.arm_boundary_m);
    observation.center_r=contour(g.center_r_boundary_m);
    cv::Matx33d R;
    cv::Rodrigues(r,R);
    auto q=R*cv::Vec3d(0,.7,0)+t;
    center=transform*Eigen::Vector3d(t[0],t[1],t[2]);
    target=transform*Eigen::Vector3d(q[0],q[1],q[2]);
  }
};
void recovery() {
  for (const auto& r : {cv::Vec3d(0,0,0),cv::Vec3d(.3,-.4,.7),cv::Vec3d(-.2,.6,-1.1)}) {
    Scene s(r,{.35,-.2,7});
    auto g=s.g; g.refinement_iterations=0;
    const auto pose=Rune::BuildingStage(g).run(s.observation,s.frame);
    check(pose.has_value(),"exact synthetic pose rejected");
    check((pose->center_barrel-s.center).norm()<1e-4,"center recovery/transform failed");
    check((pose->target_barrel-s.target).norm()<1e-4,"target recovery failed");
    check((pose->T_barrel_model*Eigen::Vector3d(0,.7,0)-pose->target_barrel).norm()<1e-9 &&
          (pose->T_barrel_model.translation()-pose->center_barrel).norm()<1e-9,
          "full model-to-barrel transform inconsistent");
    check(pose->reprojection_error_px<1e-3,"exact reprojection failed");
    check(std::abs(pose->normal_barrel.norm()-1)<1e-9,"normal not unit length");
    check(pose->captured_at==s.frame.captured_at,"capture timestamp changed");
    const auto delta=pose->target_barrel-pose->center_barrel;
    const Eigen::Vector3d up=(Eigen::Vector3d::UnitZ()-pose->normal_barrel*pose->normal_barrel.z()).normalized();
    const Eigen::Vector3d radial=up*std::cos(pose->phase_rad)+pose->normal_barrel.cross(up)*std::sin(pose->phase_rad);
    check((radial*.7-delta).norm()<1e-6,"phase convention inconsistent");
  }
  Scene s({.1,.2,-.3},{-.2,.1,4},false);
  s.g.armor_boundary_m.clear(); s.g.arm_boundary_m.clear(); s.g.center_r_boundary_m.clear();
  s.frame.camera_matrix.convertTo(s.frame.camera_matrix,CV_32F);
  s.observation.keypoints.class_id=1;
  const auto legacy=Rune::BuildingStage(s.g).run(s.observation,s.frame);
  check(legacy && legacy->activated && !legacy->contour_refined &&
        (legacy->target_barrel-s.target).norm()<1e-4,
        "legacy geometry / float calibration / activation compatibility failed");
}
void refinement() {
  Scene s({.3,-.4,.7},{.35,-.2,7});
  auto& k=s.observation.keypoints;
  k.top+=cv::Point2f(2,-2); k.left+=cv::Point2f(-2,1);
  k.right+=cv::Point2f(1,2); k.bottom+=cv::Point2f(-1,-1);
  auto baseline=s.g; baseline.refinement_iterations=0;
  const auto a=Rune::BuildingStage(baseline).run(s.observation,s.frame);
  Rune::BuildingStage builder(s.g);
  const auto b=builder.run(s.observation,s.frame);
  check(a && b && b->contour_refined,"contour refinement did not run");
  const double before=(a->target_barrel-s.target).norm();
  const double after=(b->target_barrel-s.target).norm();
  check(after<before,"contours failed to improve noisy target position");
  check(b->chamfer_error_px<1,"semantic contour mismatch");
  std::cout<<"noisy target error: "<<before*1000<<" -> "<<after*1000<<" mm\n";
  constexpr int count=40;
  const auto start=std::chrono::steady_clock::now();
  for (int i=0;i<count;++i) check(builder.run(s.observation,s.frame).has_value(),"repeated solve failed");
  std::cout<<"contour solve mean: "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/count<<" ms\n";
  s.observation.arm.clear();
  check(!builder.run(s.observation,s.frame),"missing semantic contour accepted");
}
void invalid() {
  Scene s({.3,-.4,.7},{.35,-.2,7});
  Rune::BuildingStage builder(s.g);
  auto frame=s.frame;
  frame.camera_matrix=cv::Mat::eye(3,3,CV_32S);
  check(!builder.run(s.observation,frame),"integer calibration accepted");
  frame=s.frame; frame.distortion=cv::Mat::zeros(1,6,CV_64F);
  check(!builder.run(s.observation,frame),"invalid distortion accepted");
  frame=s.frame; frame.T_barrel_camera->linear()(0,0)=2;
  check(!builder.run(s.observation,frame),"non-rigid transform accepted");
  frame=s.frame; frame.T_barrel_camera.reset();
  check(!builder.run(s.observation,frame),"missing extrinsic accepted");
  auto observation=s.observation;
  observation.keypoints.top+=cv::Point2f(80,-100);
  check(!builder.run(observation,s.frame),"inconsistent keypoints accepted");
  observation=s.observation;
  observation.keypoints.top.x=std::numeric_limits<float>::quiet_NaN();
  check(!builder.run(observation,s.frame),"NaN keypoint accepted");
  observation=s.observation;
  observation.keypoints.top=observation.keypoints.left=observation.keypoints.right=observation.keypoints.bottom={1,1};
  check(!builder.run(observation,s.frame),"collapsed anchors accepted");
  auto geometry=s.g; geometry.radius_m=0;
  check(!Rune::BuildingStage(geometry).run(s.observation,s.frame),"invalid geometry accepted");
  geometry=s.g; geometry.max_distance_field_pixels=10;
  check(!Rune::BuildingStage(geometry).run(s.observation,s.frame),"ROI memory cap ignored");
  geometry=s.g; geometry.arm_boundary_m[0].x=std::numeric_limits<double>::infinity();
  check(!Rune::BuildingStage(geometry).run(s.observation,s.frame),"invalid boundary accepted");
  observation=s.observation;
  for (auto& p : observation.arm) p.x+=100;
  check(!builder.run(observation,s.frame),"wrong semantic contour accepted");
}
void sharedCalibration() {
  const auto config=YAML::LoadFile("config/camera_config.yaml");
  const auto calibration=L1Sensor::loadCameraCalibration(config["calibration"],"camera_config.yaml");
  Rune::Frame frame;
  frame.bgr=cv::Mat(calibration.image_size,CV_8UC3,cv::Scalar(0));
  check(Rune::applyCameraCalibration(frame,calibration),"armor calibration binding failed");
  const std::vector<cv::Point3d> anchors{{0,.8,0},{-.12,.7,0},{.12,.7,0},{0,.6,0},{0,0,0}};
  const cv::Vec3d r(.2,-.3,.5),t(.2,-.1,6);
  std::vector<cv::Point2d> pixels;
  cv::projectPoints(anchors,r,t,calibration.camera_matrix,calibration.distortion_coefficients,pixels);
  Rune::Observation observation;
  observation.keypoints={pixels[0],pixels[1],pixels[2],pixels[3],pixels[4],0,.9F};
  observation.armor=observation.arm=observation.center_r={{1,1},{3,1},{1,3}};
  const auto pose=Rune::BuildingStage({.7,.12,.10}).run(observation,frame);
  check(pose.has_value(),"solve with shared armor calibration failed");
  const Eigen::Vector3d expected=calibration.T_barrel_camera->linear()*Eigen::Vector3d(t[0],t[1],t[2])+
    calibration.T_barrel_camera->translation();
  check((pose->center_barrel-expected).norm()<1e-4,"armor camera-to-barrel convention differs");
  frame.bgr=cv::Mat(10,10,CV_8UC3);
  check(!Rune::applyCameraCalibration(frame,calibration) && !frame.T_barrel_camera &&
        frame.camera_matrix.empty(),"resolution mismatch retained stale calibration");
  frame.bgr=cv::Mat(calibration.image_size,CV_8UC3);
  auto missing=calibration; missing.T_barrel_camera.reset();
  check(!Rune::applyCameraCalibration(frame,missing),"missing armor extrinsic accepted");
}
} // namespace
int main() {
  try { recovery(); refinement(); invalid(); sharedCalibration(); std::cout<<"rune_building_smoke passed\n"; }
  catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
