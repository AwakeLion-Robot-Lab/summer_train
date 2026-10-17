// Algorithm reference: SZURPVision/RP-26Rune, PowerRunePlane.
// See README.md and RP-26Rune-LICENSE in this directory.
// R/inactive-seed layout selection also references Alliance-Algorithm/
// rmcs_auto_aim_v2 (MIT); see ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f2-building/builder.hpp"
#include "rune/f2-building/builder_detail.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace Rune::building_detail {

std::optional<std::vector<cv::Point2d>> pcaEnvelope(const Observation& o,
                                                   const cv::Mat& camera,const cv::Mat& distortion) {
  if (o.armor.size()<3 || o.arm.size()<3) return {};
  std::vector<cv::Point2d> samples;
  appendContourSamples(o.armor,samples); appendContourSamples(o.arm,samples);
  if (samples.size()<6) return {};
  cv::undistortPoints(samples,samples,camera,distortion,cv::noArray(),camera);
  const auto nn=networkPoints(o);
  std::vector<cv::Point2d> undistorted;
  cv::undistortPoints(nn,undistorted,camera,distortion,cv::noArray(),camera);
  const cv::Point2d radial=(undistorted[0]+undistorted[1]+undistorted[2]+undistorted[3])*0.25-undistorted[4];
  if (cv::norm(radial)<2) return {};
  cv::Mat data(static_cast<int>(samples.size()),2,CV_64F);
  for (int i=0;i<data.rows;++i) { data.at<double>(i,0)=samples[i].x; data.at<double>(i,1)=samples[i].y; }
  cv::PCA pca(data,cv::Mat(),cv::PCA::DATA_AS_ROW);
  cv::Point2d y(pca.eigenvectors.at<double>(0,0),pca.eigenvectors.at<double>(0,1));
  cv::Point2d alternate(-y.y,y.x);
  if (std::abs(alternate.dot(radial))>std::abs(y.dot(radial))) y=alternate;
  if (y.dot(radial)<0) y=-y;
  cv::Point2d x(y.y,-y.x);
  if (x.dot(undistorted[2]-undistorted[1])<0) x=-x;
  const cv::Point2d center(pca.mean.at<double>(0,0),pca.mean.at<double>(0,1));
  double xmin=1e30,xmax=-1e30,ymin=1e30,ymax=-1e30;
  for (const auto& point : samples) {
    const auto d=point-center;
    xmin=std::min(xmin,d.dot(x)); xmax=std::max(xmax,d.dot(x));
    ymin=std::min(ymin,d.dot(y)); ymax=std::max(ymax,d.dot(y));
  }
  if (xmax-xmin<2 || ymax-ymin<2) return {};
  const std::array<cv::Point2d,4> box{center+x*xmin+y*ymax,center+x*xmin+y*ymin,
                                   center+x*xmax+y*ymin,center+x*xmax+y*ymax};
  std::vector<cv::Point3d> rays;
  for (const auto& p : box)
    rays.emplace_back((p.x-camera.at<double>(0,2))/camera.at<double>(0,0),
                      (p.y-camera.at<double>(1,2))/camera.at<double>(1,1),1);
  return project(rays,Parameters::Zero(),camera,distortion);
}

} // namespace Rune::building_detail

namespace Rune {
using namespace building_detail;

BuildingStage::BuildingStage(Geometry geometry) : geometry_(std::move(geometry)) {
  const auto& g=geometry_;
  anchors_={{0,g.radius_m+g.armor_half_height_m,0},
            {-g.armor_half_width_m,g.radius_m,0},{g.armor_half_width_m,g.radius_m,0},
            {0,g.radius_m-g.armor_half_height_m,0},{0,0,0}};
  const auto cache=[&](const BladeModel& source) {
    CachedModel model;
    const std::array<const std::vector<cv::Point3d>*,3> boundaries{
      &source.armor_boundary_m,&source.arm_boundary_m,&source.center_r_boundary_m};
    for (std::size_t j=0;j<3;++j) {
      const auto& points=*boundaries[j];
      if (!std::all_of(points.begin(),points.end(),finite)) { model.boundaries[j]=points; continue; }
      const auto n=std::min(points.size(),static_cast<std::size_t>(std::clamp(g.max_samples_per_contour,4,4096)));
      for (std::size_t i=0;i<n;++i)
        model.boundaries[j].push_back(points[n==1 ? 0 : i*(points.size()-1)/(n-1)]);
    }
    model.pca_box=source.pca_box_anchors_m;
    // In this model frame the envelope is aligned with tangential x / radial y.
    // Derive metric corners only from caller-supplied measured boundaries.
    if (model.pca_box.empty() && !source.armor_boundary_m.empty() && !source.arm_boundary_m.empty()) {
      double xmin=1e30,xmax=-1e30,ymin=1e30,ymax=-1e30;
      for (const auto* contour : {&source.armor_boundary_m,&source.arm_boundary_m})
        for (const auto& p : *contour) {
          xmin=std::min(xmin,p.x); xmax=std::max(xmax,p.x);
          ymin=std::min(ymin,p.y); ymax=std::max(ymax,p.y);
        }
      model.pca_box={{xmin,ymax,0},{xmin,ymin,0},{xmax,ymin,0},{xmax,ymax,0}};
    }
    return model;
  };
  models_[0]=cache({g.armor_boundary_m,g.arm_boundary_m,g.center_r_boundary_m,g.pca_box_anchors_m});
  if (g.small_activated_model) models_[1]=cache(*g.small_activated_model);
  if (g.big_activated_model) models_[2]=cache(*g.big_activated_model);
}

} // namespace Rune
