#include "rune/processor.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {
using namespace Rune;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
std::vector<cv::Point3d> rectangle(double w,double lo,double hi) {
  const std::array<cv::Point3d,4> corners{{{-w,lo,0},{w,lo,0},{w,hi,0},{-w,hi,0}}};
  std::vector<cv::Point3d> points;
  for(int i=0;i<4;++i) for(int j=0;j<24;++j)
    points.push_back(corners[i]+(corners[(i+1)%4]-corners[i])*(j/24.0));
  return points;
}
struct Scene {
  Geometry g{.7,.12,.10};
  Frame frame;
  std::vector<Observation> observations;
  Eigen::Isometry3d world_model=Eigen::Isometry3d::Identity();
  Scene(Mode mode=Mode::Small,double phase=.3,double yaw=.3,
        Eigen::Isometry3d world_barrel=Eigen::Isometry3d::Identity(),bool full_pose=true,double depth_noise=0) {
    frame.mode=mode;
    frame.camera_matrix=(cv::Mat_<double>(3,3)<<1500,0,720,0,1500,540,0,0,1);
    frame.distortion=(cv::Mat_<double>(1,5)<<-.12,.03,.001,-.001,0);
    frame.captured_at=TimePoint{}+std::chrono::seconds(10);
    frame.bgr=cv::Mat(1080,1440,CV_8UC3,cv::Scalar(0));
    Eigen::Isometry3d barrel_camera=Eigen::Isometry3d::Identity();
    barrel_camera.linear()<<0,0,1,-1,0,0,0,-1,0;
    barrel_camera.translation()<<.07,.06,.04;
    frame.T_barrel_camera=barrel_camera;
    frame.q_world_barrel=Eigen::Quaterniond(world_barrel.linear());
    if(full_pose) frame.T_world_barrel=world_barrel;
    world_model.linear()=barrel_camera.linear()*
      Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitY()).toRotationMatrix()*
      Eigen::AngleAxisd(phase,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    world_model.translation()<<7+depth_noise,.15,.3;
    const auto model_camera=barrel_camera.inverse()*world_barrel.inverse()*world_model;
    cv::Matx33d rotation;
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) rotation(r,c)=model_camera.linear()(r,c);
    cv::Vec3d rvec; cv::Rodrigues(rotation,rvec);
    cv::Vec3d tvec(model_camera.translation().x(),model_camera.translation().y(),model_camera.translation().z());
    g.armor_boundary_m=rectangle(.12,.6,.8);
    g.arm_boundary_m=rectangle(.035,.15,.50);
    g.center_r_boundary_m=rectangle(.025,-.025,.025);
    g.small_activated_model=BladeModel{g.armor_boundary_m,g.arm_boundary_m,g.center_r_boundary_m,{}};
    g.big_activated_model=BladeModel{{},rectangle(.065,.15,.35),g.center_r_boundary_m,{}};
    for(int sector=0;sector<5;++sector) {
      const bool inactive=sector==0 || (mode==Mode::Big && sector==2);
      const bool arm_only=mode==Mode::Big && !inactive;
      const double angle=sector*2*std::numbers::pi/5;
      const auto project=[&](const std::vector<cv::Point3d>& model) {
        std::vector<cv::Point3d> rotated;
        for(const auto& p : model) rotated.emplace_back(std::cos(angle)*p.x-std::sin(angle)*p.y,
                                                        std::sin(angle)*p.x+std::cos(angle)*p.y,p.z);
        std::vector<cv::Point2d> pixels;
        cv::projectPoints(rotated,rvec,tvec,frame.camera_matrix,frame.distortion,pixels);
        return pixels;
      };
      const auto contour=[&](const std::vector<cv::Point3d>& model) {
        std::vector<cv::Point> points;
        for(const auto& p : project(model)) points.emplace_back(cvRound(p.x),cvRound(p.y));
        return points;
      };
      Observation o;
      const std::vector<cv::Point3d> anchors=arm_only
        ? std::vector<cv::Point3d>{{0,.35,0},{-.065,.25,0},{.065,.25,0},{0,.15,0},{0,0,0}}
        : std::vector<cv::Point3d>{{0,.8,0},{-.12,.7,0},{.12,.7,0},{0,.6,0},{0,0,0}};
      const auto pixels=project(anchors);
      o.keypoints={pixels[0],pixels[1],pixels[2],pixels[3],pixels[4],inactive ? 0 : 1,sector==0 ? .99F : .9F};
      if(!arm_only) o.armor=contour(g.armor_boundary_m);
      o.arm=contour(arm_only ? g.big_activated_model->arm_boundary_m : g.arm_boundary_m);
      std::vector<cv::Point3d> shared_hub;
      for(const auto& p : g.center_r_boundary_m)
        shared_hub.emplace_back(std::cos(angle)*p.x+std::sin(angle)*p.y,
                                -std::sin(angle)*p.x+std::cos(angle)*p.y,p.z);
      o.center_r=contour(shared_hub);
      const auto paint=[&](const auto& c) { if(!c.empty()) cv::fillPoly(frame.bgr,std::vector<std::vector<cv::Point>>{c},cv::Scalar(0,0,255)); };
      paint(o.armor); paint(o.arm); paint(o.center_r);
      observations.push_back(o);
      frame.detections.push_back(o.keypoints);
    }
  }
  Eigen::Vector3d expectedTarget() const {
    return frame.T_world_barrel->inverse()*(world_model*Eigen::Vector3d(0,.7,0));
  }
};
void joint() {
  Scene s(Mode::Small,.6,1.0);
  s.g.use_pca_anchors=false; s.g.refinement_iterations=0; s.g.normal_vertical_weight_px=0;
  auto duplicate=s.observations[1]; duplicate.keypoints.confidence=.2F;
  s.observations.push_back(duplicate);
  auto foreign=s.observations[2]; foreign.keypoints.center_r.x+=70;
  s.observations.push_back(foreign);
  std::swap(s.observations[0],s.observations[3]);
  BuildingStage building(s.g);
  const auto pose=building.run(s.observations,s.frame);
  check(pose && pose->blades.size()==5,"five-blade perspective matching failed");
  check((pose->target_barrel-s.expectedTarget()).norm()<1e-4,"joint pose recovery failed");
  std::array<bool,5> used{};
  for(const auto& m : pose->blades) {
    check(!used[m.sector],"duplicate sector accepted"); used[m.sector]=true;
    check(m.observation_index<5,"duplicate/foreign blade accepted");
  }
  // Partial occlusion is allowed: the visible inactive leaf remains usable.
  s.observations.erase(s.observations.begin()+1,s.observations.begin()+3);
  s.frame.captured_at+=std::chrono::milliseconds(20);
  check(building.run(s.observations,s.frame).has_value(),"partial observation failed");
}
void pcaAndBig() {
  Scene s(Mode::Big,.1,.15);
  BuildingStage building(s.g);
  const auto pose=building.run(s.observations,s.frame);
  check(pose && pose->blades.size()==5,"big rune arm-only support failed");
  check(std::count_if(pose->blades.begin(),pose->blades.end(),[](const auto& m){return m.pca_anchors;})==2,
        "PCA anchors not used on both inactive blades");
  check((pose->target_barrel-s.expectedTarget()).norm()<.10,"PCA target error excessive");
  check(pose->chamfer_error_px<2,"big rune state-specific semantic model failed");
  // Confidence reversal must not switch between the two visible inactive blades.
  s.observations[0].keypoints.confidence=.7F;
  s.observations[2].keypoints.confidence=1;
  s.frame.captured_at+=std::chrono::milliseconds(20);
  const auto stable=building.run(s.observations,s.frame);
  check(stable && !stable->target_switched && stable->blades.front().observation_index==0,
        "confidence reversal switched physical target");
  s.observations.erase(s.observations.begin());
  s.frame.captured_at+=std::chrono::milliseconds(20);
  const auto switched=building.run(s.observations,s.frame);
  check(switched && switched->target_switched,"physical target switch was not signaled");
  const auto start=std::chrono::steady_clock::now();
  for(int i=0;i<10;++i) { s.frame.captured_at+=std::chrono::milliseconds(20); check(building.run(s.observations,s.frame).has_value(),"repeated joint solve failed"); }
  std::cout<<"four-blade PCA/contour solve: "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/10<<" ms\n";
}
void window() {
  Scene initial;
  auto g=initial.g; g.use_pca_anchors=false; g.refinement_iterations=0;
  BuildingStage building(g);
  for(int i=0;i<7;++i) {
    Eigen::Isometry3d world_barrel=Eigen::Isometry3d::Identity();
    world_barrel.linear()=(Eigen::AngleAxisd(.015*i,Eigen::Vector3d::UnitZ())*
                           Eigen::AngleAxisd(.02*i,Eigen::Vector3d::UnitY())).toRotationMatrix();
    world_barrel.translation()<<.02*i,.015*i,0;
    Scene s(Mode::Small,.3+.01*i,.3,world_barrel);
    s.frame.captured_at=initial.frame.captured_at+std::chrono::milliseconds(20*i);
    auto pose=building.run(s.observations,s.frame);
    check(pose && pose->plane_window_frames==i+1,"moving-camera window did not accumulate");
    check((pose->target_barrel-s.expectedTarget()).norm()<1e-4,"world-frame compensation failed");
    check(pose->attitude_constrained && pose->normal_vertical_component<1e-5,"gravity constraint frame wrong");
    if(i>=2) check(pose->plane_smoothed,"plane fit not applied");
  }
  Scene missing; missing.frame.captured_at+=std::chrono::milliseconds(200);
  missing.frame.T_world_barrel.reset();
  auto pose=building.run(missing.observations,missing.frame);
  check(pose && pose->plane_window_frames==0 && pose->attitude_constrained,"orientation-only frame accumulated history");
  missing.frame.q_world_barrel.reset(); missing.frame.captured_at+=std::chrono::milliseconds(20);
  pose=building.run(missing.observations,missing.frame);
  check(pose && !pose->attitude_constrained && !pose->plane_smoothed,"missing attitude reused stale prior");
  Scene reset;
  reset.frame.captured_at+=std::chrono::seconds(1);
  pose=building.run(reset.observations,reset.frame);
  check(pose && pose->plane_window_frames==1,"time gap did not reset window");
  reset.frame.world_frame_id=2; reset.frame.captured_at+=std::chrono::milliseconds(20);
  pose=building.run(reset.observations,reset.frame);
  check(pose && pose->plane_window_frames==1,"world reset did not clear history");
  // Duplicate/out-of-order stamps cannot grow a window.
  pose=building.run(reset.observations,reset.frame);
  check(pose && pose->plane_window_frames==1,"duplicate timestamp accumulated");
  auto invalid=reset.frame; invalid.T_world_barrel->linear()(0,0)=2;
  check(!building.run(reset.observations,invalid),"invalid world transform accepted");
  reset.frame.captured_at+=std::chrono::milliseconds(20);
  pose=building.run(reset.observations,reset.frame);
  check(pose && pose->plane_window_frames==1,"failed frame left stale history");
  Scene jumped(Mode::Small,.3,.3,Eigen::Isometry3d::Identity(),true,.4);
  jumped.frame.captured_at=reset.frame.captured_at+std::chrono::milliseconds(20);
  pose=building.run(jumped.observations,jumped.frame);
  check(pose && pose->plane_window_frames==1,"center jump did not reset plane samples");
  Scene tilted(Mode::Small,.3,1.0);
  tilted.frame.captured_at=jumped.frame.captured_at+std::chrono::milliseconds(20);
  pose=building.run(tilted.observations,tilted.frame);
  check(pose && pose->plane_window_frames==1,"normal jump did not reset plane samples");
  tilted.frame.color=Color::Blue; tilted.frame.captured_at+=std::chrono::milliseconds(20);
  pose=building.run(tilted.observations,tilted.frame);
  check(pose && pose->history_reset && pose->plane_window_frames==1,"color change retained context");
  tilted.frame.T_barrel_camera->translation().x()+=.01;
  tilted.frame.captured_at+=std::chrono::milliseconds(20);
  pose=building.run(tilted.observations,tilted.frame);
  check(pose && pose->history_reset && pose->plane_window_frames==1,"new extrinsics retained context");
  tilted.frame.q_world_barrel=Eigen::Quaterniond(Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitX()));
  check(!building.run(tilted.observations,tilted.frame),"inconsistent orientation and full pose accepted");
}
void denoising() {
  Scene initial;
  auto g=initial.g; g.use_pca_anchors=false; g.refinement_iterations=0;
  auto raw_geometry=g; raw_geometry.plane_window_max_frames=0;
  BuildingStage smooth(g),raw(raw_geometry);
  double before=0,after=0;
  for(int i=0;i<16;++i) {
    const double noise=i%2 ? .015 : -.015;
    Eigen::Isometry3d camera_motion=Eigen::Isometry3d::Identity();
    camera_motion.translation()<<.01*i,.005*i,0;
    Scene s(Mode::Small,.3+.01*i,0,camera_motion,true,noise);
    s.frame.captured_at=initial.frame.captured_at+std::chrono::milliseconds(20*i);
    const auto a=raw.run(s.observations,s.frame),b=smooth.run(s.observations,s.frame);
    check(a && b,"noisy window solve failed");
    if(i>=2) {
      before+=std::abs((*s.frame.T_world_barrel*a->center_barrel).x()-7);
      after+=std::abs((*s.frame.T_world_barrel*b->center_barrel).x()-7);
    }
  }
  check(after<before*.7,"plane window did not reduce world-depth jitter");
  std::cout<<"world-depth MAE: "<<before/14*1000<<" -> "<<after/14*1000<<" mm\n";
}
void pcaNoise() {
  Scene s(Mode::Small,.1,0);
  s.observations.resize(1);
  auto& k=s.observations[0].keypoints;
  k.top+=cv::Point2f(2,3); k.left+=cv::Point2f(-3,1);
  k.right+=cv::Point2f(3,-1); k.bottom+=cv::Point2f(-1,-3);
  s.g.refinement_iterations=0; s.g.normal_vertical_weight_px=0;
  auto nn=s.g; nn.use_pca_anchors=false;
  const auto a=BuildingStage(nn).run(s.observations,s.frame);
  const auto b=BuildingStage(s.g).run(s.observations,s.frame);
  check(a && b && b->blades.front().pca_anchors,"PCA anchor solve failed");
  const double before=(a->target_barrel-s.expectedTarget()).norm();
  const double after=(b->target_barrel-s.expectedTarget()).norm();
  check(after<before,"PCA did not improve noisy-anchor pose");
  std::cout<<"PCA noisy-target error: "<<before*1000<<" -> "<<after*1000<<" mm\n";
  auto bad_box=s.g;
  bad_box.refinement_iterations=15;
  bad_box.pca_box_anchors_m={{-.5,.8,0},{-.5,.15,0},{.5,.15,0},{.5,.8,0}};
  const auto fallback=BuildingStage(bad_box).run(s.observations,s.frame);
  check(fallback && !fallback->blades.front().pca_anchors,
        "inconsistent PCA envelope did not fall back to validated network anchors");
}
void attitude() {
  Scene s;
  s.g.use_pca_anchors=false; s.g.refinement_iterations=15;
  s.g.armor_boundary_m.clear(); s.g.arm_boundary_m.clear(); s.g.center_r_boundary_m.clear();
  s.g.small_activated_model.reset();
  // One noisy inactive leaf is the difficult planar ambiguity case.
  s.observations.resize(1);
  s.observations[0].keypoints.top+=cv::Point2f(2,-1);
  s.observations[0].keypoints.left+=cv::Point2f(-1,1);
  auto unconstrained=s.g; unconstrained.normal_vertical_weight_px=0;
  auto constrained=s.g; constrained.normal_vertical_weight_px=100;
  const auto a=BuildingStage(unconstrained).run(s.observations,s.frame);
  const auto b=BuildingStage(constrained).run(s.observations,s.frame);
  check(a && b && b->normal_vertical_component<a->normal_vertical_component,
        "gravity prior did not reduce vertical normal error");
  std::cout<<"normal vertical component: "<<a->normal_vertical_component<<" -> "<<b->normal_vertical_component<<'\n';
}
void observationPipeline() {
  Scene s(Mode::Big,.1,.05);
  DetectionStage detection;
  const auto regions=detection.runAll(s.frame);
  check(regions.size()==5,"f1 discarded supporting blades");
  ObservationStage observation;
  std::vector<Observation> observations;
  for(const auto& region : regions)
    if(auto o=observation.run(region,s.frame.color,s.frame.mode)) observations.push_back(*o);
  check(observations.size()==5,"f1 failed to extract all semantic observations");
  const auto pose=BuildingStage(s.g).run(observations,s.frame);
  check(pose && pose->blades.size()==5,"f1-to-f2 joint integration failed");
  Processor processor(s.g);
  ControlInput control; control.command_at=s.frame.captured_at; control.bullet_speed_m_s=28;
  const auto command=processor.process(s.frame,control);
  check(command.status==Status::MotionNotReady && !command.fire,"processor did not run joint building stage");
  for(auto& detection : s.frame.detections) detection.class_id=1;
  check(processor.process(s.frame,control).status==Status::NoDetection,
        "activated-only frame was used as an aiming target");
}
void initializationCandidates() {
  Scene s(Mode::Big,.25,.4);
  s.g.use_pca_anchors=false; s.g.refinement_iterations=0;
  // A higher-confidence, self-consistent single leaf from another rune must
  // lose to the two inactive centers of the actual rune. With contour
  // refinement disabled, the three arm-only leaves carry no constraints.
  auto foreign=s.observations[0];
  foreign.keypoints.confidence=1;
  for (auto* p : {&foreign.keypoints.top,&foreign.keypoints.left,&foreign.keypoints.right,
                 &foreign.keypoints.bottom,&foreign.keypoints.center_r}) *p+=cv::Point2f(250,150);
  for (auto* contour : {&foreign.armor,&foreign.arm,&foreign.center_r})
    for (auto& p : *contour) p+=cv::Point(250,150);
  s.observations.insert(s.observations.begin(),foreign);
  const auto pose=BuildingStage(s.g).run(s.observations,s.frame);
  check(pose && pose->blades.size()==2 && (pose->target_barrel-s.expectedTarget()).norm()<1e-3,
        "initialization used confidence instead of layout");
  s.observations.erase(s.observations.begin());
  // The R attached to the selected NN leaf is wrong; another observed R still
  // generates the correct five-anchor PnP hypothesis.
  s.observations[0].keypoints.center_r+=cv::Point2f(100,60);
  const auto cross=BuildingStage(s.g).run(s.observations,s.frame);
  check(cross && cross->blades.size()==2 && (cross->target_barrel-s.expectedTarget()).norm()<1e-3,
        "R / inactive target candidate enumeration failed");
}
void processorMotion() {
  Scene initial(Mode::Big,.1,.15);
  auto g=initial.g; g.use_pca_anchors=false; g.refinement_iterations=0;
  PlanningConfig planning;
  planning.small_warmup_s=planning.big_warmup_s=0;
  planning.mpc.enable_mpc=true;
  Processor processor(g,{},planning);
  Command command;
  for (int i=0;i<=100;++i) {
    Scene s(Mode::Big,.1+.02*i,.15);
    s.frame.captured_at=initial.frame.captured_at+std::chrono::milliseconds(20*i);
    ControlInput control;
    control.command_at=s.frame.captured_at; control.bullet_speed_m_s=28;
    command=processor.process(s.frame,control);
    check(command.status!=Status::PoseFailed && command.status!=Status::NoContour,
          "processor feature filter integration failed");
  }
  check(command.found && !command.center_hold && command.using_mpc && !command.fire,"default big-rune fit did not reach planner");
}
} // namespace
int main() {
  try { joint(); pcaAndBig(); window(); denoising(); pcaNoise(); attitude(); observationPipeline(); initializationCandidates(); processorMotion(); std::cout<<"rune_joint_building_smoke passed\n"; }
  catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
