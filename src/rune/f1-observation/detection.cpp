#include "rune/f1-observation/detection.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace Rune {

std::optional<Region> DetectionStage::run(const Frame& frame) const {
  auto regions=runAll(frame);
  for (auto& region : regions)
    if (region.keypoints.class_id==0) return std::move(region);
  return {};
}

std::vector<Region> DetectionStage::runAll(const Frame& frame) const {
  if (frame.bgr.empty() || frame.bgr.type()!=CV_8UC3 ||
      !std::isfinite(roi_scale_) || roi_scale_<1 || roi_scale_>10) return {};
  std::vector<Keypoints> candidates;
  for (const auto& candidate : frame.detections) {
    if ((candidate.class_id!=0 && candidate.class_id!=1) ||
        !std::isfinite(candidate.confidence)) continue;
    bool valid=true;
    for (const auto& point : {candidate.top,candidate.left,candidate.right,
                              candidate.bottom,candidate.center_r}) {
      valid &= std::isfinite(point.x) && std::isfinite(point.y) &&
        point.x>=0 && point.y>=0 && point.x<frame.bgr.cols && point.y<frame.bgr.rows;
    }
    if (valid) candidates.push_back(candidate);
  }
  std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b) {
    return a.confidence>b.confidence;
  });
  std::vector<Region> regions;
  for (const auto& candidate : candidates) {
    const std::array<cv::Point2f,5> points{candidate.top,candidate.left,candidate.right,
                                        candidate.bottom,candidate.center_r};
    float min_x=points[0].x,max_x=min_x,min_y=points[0].y,max_y=min_y;
    for (const auto& p : points) {
      min_x=std::min(min_x,p.x); max_x=std::max(max_x,p.x);
      min_y=std::min(min_y,p.y); max_y=std::max(max_y,p.y);
    }
    if (max_x-min_x<2 || max_y-min_y<2) continue;
    const double cx=(min_x+max_x)/2.0,cy=(min_y+max_y)/2.0;
    const double half_w=(max_x-min_x)*roi_scale_/2,half_h=(max_y-min_y)*roi_scale_/2;
    // Clamp before casting: even a very large source image cannot overflow Rect.
    const int x0=static_cast<int>(std::max(0.0,std::floor(cx-half_w)));
    const int y0=static_cast<int>(std::max(0.0,std::floor(cy-half_h)));
    const int x1=static_cast<int>(std::min(double(frame.bgr.cols),std::ceil(cx+half_w)));
    const int y1=static_cast<int>(std::min(double(frame.bgr.rows),std::ceil(cy+half_h)));
    const cv::Rect bounds(x0,y0,x1-x0,y1-y0);
    if (!bounds.empty()) regions.push_back({candidate,bounds,frame.bgr(bounds).clone()});
    // Permit duplicates/false positives for joint matching to resolve, but bound work.
    if (regions.size()==10) break;
  }
  return regions;
}

} // namespace Rune
