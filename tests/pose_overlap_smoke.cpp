// Production scheduling with deterministic detections; no camera or serial writes.
#define main originalRuntimeMain
#include "../src/runtime/auto_aim_runtime.cpp"
#undef main
#include "sensor/serial/pose_preview.hpp"
#include <iostream>
#include <fstream>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Pose {
  mutable std::string events;
  std::optional<Eigen::Quaterniond> preview{Eigen::Quaterniond::Identity()};
  Eigen::Quaterniond aligned{Eigen::Quaterniond::Identity()};
  Clock::time_point timestamp, deadline;
  bool timeout{false};
  auto roiPosePreview(Clock::time_point t) const { check(t == timestamp, "preview time"); return preview; }
  bool waitPoseUntil(Clock::time_point t, Clock::time_point d) const {
    check(t == timestamp && d == deadline, "absolute deadline changed"); events += 'W'; return !timeout;
  }
  std::optional<Eigen::Quaterniond> gimbalPoseAt(Clock::time_point t) const {
    check(t == timestamp, "exposure time"); events += 'P'; return aligned;
  }
};
struct Detector {
  Pose* pose;
  std::vector<L2Perception::Armor> armors;
  bool empty{false};
  bool ready() const { return true; }
  double net_aspect_ratio() const { return 1; }
  L2Perception::DetectTiming lastTiming() const { return {}; }
  L2Perception::ArmorFrame detectFrame(const cv::Mat&, const std::optional<cv::Rect>& = {}) {
    pose->events += 'D';
    return {empty ? std::vector<L2Perception::Armor>{} : armors, {}, true};
  }
  void completeSideLights(const cv::Mat&, L2Perception::ArmorFrame&,
                         const std::vector<L2Perception::LightHint>&, L2Perception::ArmorColor) {
    check(pose->events.find('P') != std::string::npos, "side search before exposure pose");
    pose->events += 'S';
  }
};
void previewChecks() {
  using namespace std::chrono;
  const auto t = Clock::time_point(seconds(10));
  L1Sensor::TimedGimbalPose a{Eigen::Quaterniond::Identity(), t};
  auto b = a; b.timestamp += std::chrono::milliseconds(5);
  check(L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(8)).has_value(), "fresh stationary pose rejected");
  check(!L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(14)), "stale pose accepted");
  check(!L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(4)), "already-covered pose speculated");
  b.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(0.02,Eigen::Vector3d::UnitZ()));
  check(!L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(8)), "fast motion accepted");
  b = a;
  check(!L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(1)), "batched timestamps accepted");
  b.timestamp += std::chrono::milliseconds(30);
  check(!L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(31)), "packet gap accepted");
  b.timestamp = t+std::chrono::milliseconds(5); b.rotation.coeffs()[0] = NAN;
  check(!L1Sensor::previewRoiPose(a,b,t+std::chrono::milliseconds(8)), "nonfinite pose accepted");
  check(runtime::paddedPoseRoi({0,0,100,100},{100,100}) == cv::Rect(0,0,100,100), "ROI clipping");
  check(!runtime::coversPoseRoi({10,10,100,100},{9,10,100,100}), "escaped ROI accepted");
  runtime::PoseOverlapState schedule;
  schedule.finishAttempt(true); schedule.finishAttempt(true);
  for(int i=0;i<30;++i) check(!schedule.beginFrame(), "repair cooldown too short");
  check(schedule.beginFrame(), "repair cooldown never ends");
}
void runtimeChecks() {
  L1Sensor::CameraCalibration c;
  c.image_size = {1440,1080};
  c.camera_matrix = (cv::Mat_<double>(3,3) << 1210,0,720,0,1210,540,0,0,1);
  c.distortion_coefficients = cv::Mat::zeros(1,5,CV_64F);
  c.T_barrel_camera = Eigen::Isometry3d::Identity();
  c.T_barrel_camera->linear() = Eigen::Matrix3d{{0,0,1},{-1,0,0},{0,-1,0}};
  namespace V = L3Estimation::VehicleModel;
  Eigen::VectorXd truth = Eigen::VectorXd::Zero(V::kStateSize);
  truth[V::idx::CX]=3; truth[V::idx::LOG_R1]=truth[V::idx::LOG_R2]=std::log(.25);
  L3Estimation::ObsContext ctx{.name=L3Estimation::ArmorName::Infantry3,.armor_num=4,
    .camera_in_world=*c.T_barrel_camera,.camera_matrix=c.camera_matrix,
    .distortion_coefficients=c.distortion_coefficients};
  const auto l=ctx.project(0,true,truth), r=ctx.project(0,false,truth);
  L2Perception::Armor armor;
  armor.corners={l.first,r.first,r.second,l.second};
  armor.center=(l.first+r.first+r.second+l.second)*.25f;
  armor.class_id=3; armor.color=L2Perception::ArmorColor::Blue; armor.confidence=.99;
  std::vector<L2Perception::Armor> armors{armor};
  const auto start=Clock::time_point(std::chrono::seconds(100));
  for (int scenario=0;scenario<6;++scenario) {
    std::optional<L3Estimation::EskfTracker> tracker;
    tracker.emplace(c);
    for(int f=0;f<10;++f) tracker->track(armors,Eigen::Quaterniond::Identity(),start+std::chrono::milliseconds(f*10));
    check(tracker->state()==L3Estimation::TrackState::Tracking,"fixture did not track");
    ArmorVisionPacket packet;
    packet.image=cv::Mat::zeros(c.image_size,CV_8UC3);
    packet.timing.captured_at=start+std::chrono::milliseconds(100);
    packet.timing.ready_at=packet.timing.captured_at+std::chrono::milliseconds(2);
    packet.state=L1Sensor::RobotState{}; packet.state->enemy_color=L1Sensor::EnemyColor::Blue;
    Pose pose; pose.timestamp=packet.timing.captured_at; pose.deadline=packet.timing.ready_at+std::chrono::milliseconds(20);
    if(scenario==1) pose.preview.reset();
    if(scenario==2) pose.aligned=Eigen::Quaterniond(Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitZ()));
    if(scenario==4) pose.timeout=true;
    Detector detector{&pose,armors,scenario==3};
    runtime::PoseOverlapState schedule;
    processArmorVision(packet,detector,tracker,pose,true,20,scenario==5?nullptr:&schedule);
    const bool repair=scenario==2||scenario==3;
    const bool overlap=scenario!=1&&scenario!=5;
    check(pose.events==(overlap?(repair?"DWPDS":"DWPS"):"WPDS"),"production stage order");
    check(packet.timing.pose_overlap==overlap && packet.timing.pose_roi_repair==repair,"overlap diagnostics");
    check(packet.timing.detection_passes==(repair?2u:1u),"unbounded or missing repair");
    check(packet.image_pose && packet.image_pose->angularDistance(pose.aligned)<1e-12,"provisional pose leaked to tracking");
    check(packet.timing.pose_timeout==(scenario==4),"timeout diagnostics lost");
  }
}
// Opt-in real-image A/B. Availability is SIMULATED: 200 Hz pose samples,
// latest sample 4 ms before exposure, final exposure pose available after detect.
// This tests crop/association changes, not camera latency or serial timing.
void replayChecks(const std::string& prefix) {
  auto config=runtime::loadAutoAimConfig("config/auto_aim.yaml");
  auto c=L1Sensor::loadCameraCalibration(YAML::LoadFile("config/camera_config.yaml")["calibration"],prefix);
  std::ifstream poses(prefix+".txt");
  cv::VideoCapture video(prefix+".avi");
  struct Sample { cv::Mat image; double seconds; Eigen::Quaterniond q; };
  std::vector<Sample> samples;
  for(int i=0;i<360;++i) {
    Sample sample; double w,x,y,z;
    if(!video.read(sample.image) || !(poses>>sample.seconds>>w>>x>>y>>z)) break;
    sample.q=Eigen::Quaterniond(w,x,y,z).normalized(); samples.push_back(std::move(sample));
  }
  check(samples.size()>50,"not enough replay samples");
  struct ReplayPose {
    Eigen::Quaterniond exact;
    std::optional<Eigen::Quaterniond> provisional;
    bool waitPoseUntil(Clock::time_point,Clock::time_point)const{return true;}
    auto gimbalPoseAt(Clock::time_point)const{return std::optional{exact};}
    auto roiPosePreview(Clock::time_point)const{return provisional;}
  };
  int baseline_tracking=0;
  for(bool enabled:{false,true}) {
    auto detector=runtime::makeDetector(config);
    for(int i=0;i<10;++i) (void)detector.detect(samples[i].image);
    std::optional<L3Estimation::EskfTracker> tracker;
    tracker.emplace(c,config.armor,config.ieskf_tracker,config.ieskf_target);
    runtime::PoseOverlapState schedule;
    int tracked=0,detected=0,matched=0,attempts=0,repairs=0,passes=0;
    std::vector<double> innovations;
    for(std::size_t i=0;i<samples.size();++i) {
      const auto& sample=samples[i];
      ArmorVisionPacket packet; packet.image=sample.image;
      packet.timing.captured_at=Clock::time_point(std::chrono::seconds(1000))+
        std::chrono::microseconds(static_cast<long long>(sample.seconds*1e6));
      packet.timing.ready_at=Clock::now();
      packet.state=L1Sensor::RobotState{};packet.state->enemy_color=L1Sensor::EnemyColor::Blue;
      ReplayPose pose{sample.q,{}};
      if(i>0) {
        const double dt=sample.seconds-samples[i-1].seconds;
        if(dt>=0.009 && dt<0.1) {
          const auto before=samples[i-1].q.slerp((dt-.009)/dt,sample.q);
          const auto latest=samples[i-1].q.slerp((dt-.004)/dt,sample.q);
          pose.provisional=L1Sensor::previewRoiPose(
            {before,packet.timing.captured_at-std::chrono::milliseconds(9)},
            {latest,packet.timing.captured_at-std::chrono::milliseconds(4)},packet.timing.captured_at);
        }
      }
      processArmorVision(packet,detector,tracker,pose,true,20,enabled?&schedule:nullptr);
      tracked+=packet.track_state==L3Estimation::TrackState::Tracking;
      detected+=packet.perception.armors.size();matched+=packet.match_count;
      attempts+=packet.timing.pose_overlap;repairs+=packet.timing.pose_roi_repair;
      passes+=packet.timing.detection_passes;
      if(packet.target && packet.target->lastInnovation().timestamp==packet.timing.captured_at)
        for(const auto& corner:packet.target->lastInnovation().corners)
          if(!corner.isolated) innovations.push_back(std::abs(corner.dv));
    }
    std::sort(innovations.begin(),innovations.end());
    check(!innovations.empty(),"replay has no updates");
    std::cout<<"pose_replay,"<<prefix<<",enabled,"<<enabled<<",frames,"<<samples.size()
      <<",tracking,"<<tracked<<",detected,"<<detected<<",matched,"<<matched
      <<",attempts,"<<attempts<<",repairs,"<<repairs<<",passes,"<<passes
      <<",corner_abs_dv_p90,"<<innovations[innovations.size()*9/10]<<'\n';
    if(!enabled) baseline_tracking=tracked;
    else {
      check(attempts>0,"replay did not exercise speculation");
      check(tracked>=baseline_tracking*.98,"tracking availability regressed by more than 2%");
    }
  }
}

}
int main(int argc,char** argv) {
  try { previewChecks(); runtimeChecks(); if(argc>1) replayChecks(argv[1]); std::cout<<"pose overlap gates, stage ordering, deadlines, repair and cooldown passed\n"; }
  catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
