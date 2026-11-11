#include "rune/debug_overlay.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void checkPrediction() {
  Rune::Frame frame;
  frame.bgr=cv::Mat::zeros(800,800,CV_8UC3);
  frame.captured_at=Rune::TimePoint{}+std::chrono::seconds(1);
  frame.camera_matrix=(cv::Mat_<double>(3,3)<<600,0,400,0,600,400,0,0,1);
  frame.distortion=(cv::Mat_<double>(1,5)<<-.1,.01,.001,-.001,0);
  frame.T_barrel_camera=Eigen::Isometry3d::Identity();
  frame.T_barrel_camera->linear()<<0,0,1,-1,0,0,0,-1,0;
  frame.T_barrel_camera->translation()<<.07,.06,.04;
  Rune::Geometry geometry;
  geometry.radius_m=.7;
  geometry.armor_half_width_m=geometry.armor_half_height_m=.15;
  Rune::Motion motion;
  motion.pose.captured_at=frame.captured_at;
  motion.pose.filter_state=Rune::FilterState{};
  Eigen::Isometry3d camera_model=Eigen::Isometry3d::Identity();
  camera_model.translation().z()=4;
  motion.pose.T_barrel_model=*frame.T_barrel_camera*camera_model;
  motion.ready=true;
  motion.angular_speed_rad_s=1;
  double requested_horizon=-1;
  motion.phase_delta_rad=[&](double horizon)->std::optional<double> {
    requested_horizon=horizon;
    return .42;
  };
  Rune::AimPlan plan;
  plan.valid=true;
  plan.captured_at=frame.captured_at;
  plan.prediction_horizon_s=.37;
  auto image=frame.bgr.clone();
  require(Rune::drawPredictionOverlay(image,frame,motion,geometry,&plan),
          "ready model did not draw future targets");
  require(std::abs(requested_horizon-.37)<1e-9,"overlay ignored the ballistic prediction horizon");
  // Check one rim point of every blade using distorted projection and the
  // full extrinsic. Neither detections nor observed blade count may limit this.
  for (int sector=0;sector<5;++sector) {
    const Eigen::Vector3d corner=camera_model*(Eigen::AngleAxisd(
        .42+sector*2*std::numbers::pi/5,Eigen::Vector3d::UnitZ())*
        Eigen::Vector3d(0,.85,0));
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(std::vector<cv::Point3d>{{corner.x(),corner.y(),corner.z()}},
        cv::Vec3d{},cv::Vec3d{},frame.camera_matrix,frame.distortion,pixels);
    const cv::Point p(cvRound(pixels[0].x),cvRound(pixels[0].y));
    cv::Mat green;
    cv::inRange(image(cv::Rect(p.x-2,p.y-2,5,5)),cv::Scalar(0,180,0),
                cv::Scalar(0,255,0),green);
    require(cv::countNonZero(green)>0,"one of the five future outlines is missing or misplaced");
  }
  cv::Mat green, labels;
  cv::inRange(image(cv::Rect(250,250,300,300)),cv::Scalar(0,180,0),cv::Scalar(0,255,0),green);
  require(cv::connectedComponents(green,labels)==6,"prediction did not draw five separate outlines");
  plan.captured_at-=std::chrono::milliseconds(1);
  image=frame.bgr.clone();
  require(Rune::drawPredictionOverlay(image,frame,motion,geometry,&plan,.15) &&
          std::abs(requested_horizon-.15)<1e-9,"stale plan was used instead of the preview horizon");
  motion.ready=false;
  requested_horizon=-1;
  image=frame.bgr.clone();
  require(!Rune::drawPredictionOverlay(image,frame,motion,geometry) && requested_horizon==-1 &&
          cv::norm(image(cv::Rect(250,250,300,300)),cv::NORM_INF)==0,
          "unready motion was rendered as a future target");
  motion.ready=true;
  motion.pose.captured_at-=std::chrono::milliseconds(1);
  image=frame.bgr.clone();
  require(!Rune::drawPredictionOverlay(image,frame,motion,geometry) &&
          cv::norm(image,cv::NORM_INF)==0,"stale pose was rendered as a future target");
}
} // namespace

int main() {
  try { checkPrediction(); std::cout<<"rune_prediction_overlay_smoke passed\n"; return 0; }
  catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
