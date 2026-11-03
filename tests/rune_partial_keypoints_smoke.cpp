#include "rune/f2-building/builder.hpp"
#include "rune/f3-estimator/feature_filter.hpp"

#include <opencv2/calib3d.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void checkPartialGeometry() {
  Rune::Geometry geometry;
  geometry.radius_m=.7;
  geometry.armor_half_width_m=geometry.armor_half_height_m=.15;
  geometry.normal_vertical_weight_px=0;
  geometry.plane_window_max_frames=0;
  Rune::Frame frame;
  frame.captured_at=std::chrono::steady_clock::now();
  frame.camera_matrix=(cv::Mat_<double>(3,3)<<1000,0,320,0,1000,240,0,0,1);
  frame.T_barrel_camera=Eigen::Isometry3d::Identity();
  frame.T_barrel_camera->linear()<<0,0,1,-1,0,0,0,-1,0;
  const cv::Vec3d rotation(.1,.2,.3), translation(0,0,4);
  const auto project=[&](const std::vector<cv::Point3d>& points) {
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(points,rotation,translation,frame.camera_matrix,cv::Mat{},pixels);
    std::vector<cv::Point2f> result;
    for (const auto& pixel : pixels) result.emplace_back(pixel);
    return result;
  };
  const auto points=project({{0,.85,0},{-.15,.7,0},{.15,.7,0},{0,.55,0},{0,0,0}});
  Rune::Observation observation;
  observation.keypoints={points[0],points[1],points[2],points[3],points[4],0,.95F};
  const auto contour=[&](const std::vector<cv::Point3d>& model) {
    std::vector<cv::Point> result;
    for (const auto& p : project(model)) result.emplace_back(cvRound(p.x),cvRound(p.y));
    return result;
  };
  observation.armor=contour({{-.15,.55,0},{.15,.55,0},{.15,.85,0},{-.15,.85,0}});
  observation.arm=contour({{-.03,.1,0},{.03,.1,0},{.03,.5,0},{-.03,.5,0}});
  observation.center_r=contour({{-.05,-.05,0},{.05,-.05,0},{.05,.05,0},{-.05,.05,0}});
  Rune::BuildingStage full_builder(geometry);
  const auto full=full_builder.run(observation,frame);
  require(full.has_value(),"synthetic full-keypoint pose must solve");
  for (int weak=0;weak<5;++weak) {
    auto partial=observation;
    auto& k=partial.keypoints;
    k.point_valid[weak]=false;
    k.point_confidences[weak]=.1F;
    const std::array<cv::Point2f*,5> coordinates{&k.top,&k.left,&k.right,&k.bottom,&k.center_r};
    *coordinates[weak]+=cv::Point2f(120,90);
    Rune::BuildingStage builder(geometry);
    const auto pose=builder.run(partial,frame);
    require(pose.has_value(),"four reliable anchors must remain usable for PnP");
    require((pose->center_barrel-full->center_barrel).norm()<.001 &&
            (pose->target_barrel-full->target_barrel).norm()<.001,
            "a corrupted weak keypoint changed the solved pose");
    Rune::FeatureFilter filter(geometry,{});
    const auto filtered=filter.update(*pose,{partial},frame);
    require(filtered.has_value(),"partial observation must support feature filtering");
    if (weak==4) {
      for (const auto& match : filtered->feature_matches)
        require(match.feature_id!=0,"weak R must not become a hub measurement");
    }
  }
  auto three=observation;
  three.keypoints.point_valid={true,false,false,true,true};
  Rune::BuildingStage builder(geometry);
  require(!builder.run(three,frame),"three detected points must not fabricate a four-point PnP seed");

  // A second detection can supply a reliable shared R. Copy its confidence and
  // validity with its coordinate, so the three-point candidate can use it.
  auto borrowed=observation;
  borrowed.keypoints.point_valid={true,true,true,false,false};
  borrowed.keypoints.point_confidences={.9F,.9F,.9F,.1F,.1F};
  borrowed.keypoints.bottom+=cv::Point2f(120,90);
  borrowed.keypoints.center_r+=cv::Point2f(120,90);
  auto supporting=observation;
  supporting.keypoints.confidence=.8F;
  Rune::BuildingStage joint_builder(geometry);
  const auto joint=joint_builder.run(std::vector<Rune::Observation>{borrowed,supporting},frame);
  require(joint.has_value() && !joint->blades.empty() && joint->blades[0].observation_index==0 &&
          (joint->target_barrel-full->target_barrel).norm()<.001,
          "a shared reliable R did not repair the partial candidate's pose anchors");
}
}

int main() {
  try {
    checkPartialGeometry();
    std::cout<<"rune_partial_keypoints_smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<"rune_partial_keypoints_smoke: "<<error.what()<<'\n';
    return 1;
  }
}
