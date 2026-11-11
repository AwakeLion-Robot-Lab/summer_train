#include "rune/processor.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value,const char* message) {
  if (!value) throw std::runtime_error(message);
}

Rune::Frame frameAt(double seconds,bool detected=true) {
  Rune::Frame frame;
  frame.captured_at=Rune::TimePoint{}+std::chrono::seconds(1)+
    std::chrono::duration_cast<Rune::TimePoint::duration>(std::chrono::duration<double>(seconds));
  frame.bgr=cv::Mat::zeros(720,960,CV_8UC3);
  frame.color=Rune::Color::Blue;
  frame.camera_matrix=(cv::Mat_<double>(3,3)<<800,0,480,0,800,360,0,0,1);
  frame.T_barrel_camera=Eigen::Isometry3d::Identity();
  frame.T_barrel_camera->linear()<<0,0,1,-1,0,0,0,-1,0;
  if (!detected) return frame;
  const auto project=[&](const std::vector<cv::Point3d>& points) {
    std::vector<cv::Point2f> pixels;
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points,cv::Vec3d(0,0,.4*seconds),cv::Vec3d(0,0,4),
        frame.camera_matrix,cv::Mat{},projected);
    for (const auto& p : projected) pixels.emplace_back(p);
    return pixels;
  };
  const auto points=project({{0,.85,0},{-.15,.7,0},{.15,.7,0},{0,.55,0},{0,0,0}});
  frame.detections.push_back({points[0],points[1],points[2],points[3],points[4],0,.95F});
  for (const auto& contour : std::vector<std::vector<cv::Point3d>>{
      {{-.15,.55,0},{.15,.55,0},{.15,.85,0},{-.15,.85,0}},
      {{-.03,.12,0},{.03,.12,0},{.03,.46,0},{-.03,.46,0}},
      {{-.04,-.04,0},{.04,-.04,0},{.04,.04,0},{-.04,.04,0}}}) {
    std::vector<cv::Point> polygon;
    for (const auto& p : project(contour)) polygon.emplace_back(cvRound(p.x),cvRound(p.y));
    cv::fillConvexPoly(frame.bgr,polygon,cv::Scalar(255,0,0));
  }
  return frame;
}

void check() {
  Rune::Geometry geometry;
  geometry.radius_m=.7;
  geometry.armor_half_width_m=geometry.armor_half_height_m=.15;
  geometry.normal_vertical_weight_px=0;
  geometry.plane_window_max_frames=0;
  Rune::EstimatorConfig estimator;
  estimator.fit.min_fit_s=.3;
  Rune::Processor processor(geometry,estimator);
  Rune::ControlInput control;
  control.bullet_speed_m_s=25;
  const auto update=[&](Rune::Frame frame) {
    control.command_at=frame.captured_at;
    return processor.process(frame,control);
  };
  for (int i=0;i<=50;++i) {
    update(frameAt(i*.02));
    require(processor.lastMotion().has_value(),"synthetic visible frame failed pose/filter");
  }
  require(processor.lastMotion()->ready,"initial visible sequence did not become motion-ready");
  for (int i=51;i<56;++i) {
    const auto command=update(frameAt(i*.02,false));
    require(!command.found && !command.fire && !processor.lastMotion() && !processor.lastPlan(),
            "missing frames exposed stale measurements, plans or firing commands");
  }
  update(frameAt(1.12));
  require(processor.lastMotion() && !processor.lastMotion()->pose.history_reset &&
          processor.lastMotion()->ready,
          "a 120ms blink erased the accumulated motion history");
  for (int i=57;i<=70;++i) update(frameAt(i*.02,false));
  update(frameAt(1.42));
  require(processor.lastMotion() && processor.lastMotion()->pose.history_reset &&
          !processor.lastMotion()->ready,"long loss did not reset tracking and warmup");
  update(frameAt(1.44));
  auto missing=frameAt(1.46,false);
  missing.bgr.release();
  update(missing);
  update(frameAt(1.48));
  require(processor.lastMotion() && processor.lastMotion()->pose.history_reset,
          "explicit suspension did not force immediate history reset");
  // Missing contour is also a short observation failure, not a new measurement.
  auto no_contour=frameAt(1.5);
  no_contour.bgr.setTo(cv::Scalar::all(0));
  const auto command=update(no_contour);
  require(command.status==Rune::Status::NoContour && !command.found && !command.fire,
          "contour failure was treated as a valid observation");
  update(frameAt(1.52));
  require(processor.lastMotion() && !processor.lastMotion()->pose.history_reset,
          "a one-frame contour failure erased history");
  processor.reset();
  update(frameAt(1.54));
  require(processor.lastMotion() && processor.lastMotion()->pose.history_reset,
          "explicit reset retained history");
  const auto duplicate=update(frameAt(1.54));
  require(!duplicate.found && !duplicate.fire && !processor.lastMotion(),
          "a repeated timestamp produced a valid tracking command");
  update(frameAt(1.56));
  require(processor.lastMotion() && processor.lastMotion()->pose.history_reset,
          "a timestamp discontinuity retained stale history");
  auto big=frameAt(1.58); big.mode=Rune::Mode::Big;
  update(big);
  require(processor.lastMotion() && processor.lastMotion()->pose.history_reset,
          "mode change did not reset tracking history");
  auto red=frameAt(1.6); red.mode=Rune::Mode::Big; red.color=Rune::Color::Red;
  cv::cvtColor(red.bgr,red.bgr,cv::COLOR_BGR2RGB);
  update(red);
  require(processor.lastMotion() && processor.lastMotion()->pose.history_reset,
          "enemy color change did not reset tracking history");
}

void checkStationaryVisualPrediction() {
  Rune::Geometry geometry;
  geometry.radius_m=.7;
  geometry.armor_half_width_m=geometry.armor_half_height_m=.15;
  geometry.normal_vertical_weight_px=0;
  geometry.plane_window_max_frames=0;
  Rune::EstimatorConfig estimator;
  estimator.fit.min_fit_s=.3;
  for (const auto mode : {Rune::Mode::Small, Rune::Mode::Big}) {
    Rune::Processor processor(geometry, estimator);
    for (int i=0;i<=35;++i) {
      auto frame=frameAt(0);
      frame.mode=mode;
      frame.captured_at+=std::chrono::milliseconds(20*i);
      (void)processor.observe(frame);
      require(processor.lastMotion().has_value(),"visual-only stationary observation failed");
      require(!processor.lastPlan(),"visual-only observation generated an aiming plan");
      require(!processor.lastMotion()->pose.q_world_barrel,
              "visual-only observation fabricated world orientation");
      if (i==0) require(!processor.lastMotion()->ready,"one sample made motion ready");
    }
    const auto& motion=*processor.lastMotion();
    require(motion.ready && std::abs(motion.angular_speed_rad_s)<.01,
            "a stable stationary model never became ready");
    const auto delta=motion.phase_delta_rad(.2);
    require(delta && std::abs(*delta)<.002,"stationary prediction invented angular movement");
  }
}

void checkFilteredPhaseCorrection() {
  Rune::EstimatorConfig config;
  config.fit.min_fit_s=.3;
  Rune::EstimatorStage estimator(config);
  Rune::Pose pose;
  pose.center_barrel={4,0,0};
  pose.normal_barrel=Eigen::Vector3d::UnitX();
  pose.filter_state=Rune::FilterState{};
  Rune::Motion motion;
  for (int i=0;i<=31;++i) {
    pose.captured_at=Rune::TimePoint{}+std::chrono::seconds(1)+std::chrono::milliseconds(20*i);
    pose.phase_rad=.4*.02*i+(i==31 ? .13 : 0);
    pose.filter_state->x[5]=pose.phase_rad;
    pose.filter_state->x[6]=.4;
    motion=estimator.update(pose,Rune::Mode::Small);
  }
  require(motion.ready && motion.fit_cost_rad2<.06*.06,
          "a bounded Kalman phase correction erased valid motion fitting history");
}

void checkSmallFitWindow() {
  for (const auto mode : {Rune::Mode::Small,Rune::Mode::Big}) {
    Rune::EstimatorStage estimator;
    Rune::Pose pose;
    pose.center_barrel={4,0,0};
    pose.normal_barrel=Eigen::Vector3d::UnitX();
    Rune::Motion motion;
    for (int i=0;i<=25;++i) {
      pose.captured_at=Rune::TimePoint{}+std::chrono::seconds(1)+std::chrono::milliseconds(20*i);
      pose.phase_rad=.4*.02*i;
      motion=estimator.update(pose,mode);
      if (i<15) require(!motion.ready,"small motion became ready before its minimum fit span");
    }
    require(motion.ready==(mode==Rune::Mode::Small),
            "small constant-speed fitting incorrectly required the big-rune fit window");
  }
}
} // namespace

int main() {
  try { check(); checkStationaryVisualPrediction(); checkFilteredPhaseCorrection(); checkSmallFitWindow();
    std::cout<<"rune_tracking_continuity_smoke passed\n"; return 0; }
  catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
