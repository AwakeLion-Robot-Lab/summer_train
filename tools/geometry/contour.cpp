// Extracted from the rune builder; algorithm reference: SZURPVision/RP-26Rune.
// Provenance and MIT license: ../../src/rune/f2-building/README.md and RP-26Rune-LICENSE.
#include "contour.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace tools::geometry {

std::optional<ContourDistanceField> makeContourDistanceField(
    const std::vector<cv::Point>& contour, int padding_px, std::int64_t max_pixels) {
  if (padding_px<0 || padding_px>4096 || max_pixels<=0) return {};
  if (contour.size() < 3 || std::abs(cv::contourArea(contour)) < 1) return {};
  // Bound coordinates BEFORE boundingRect/drawing to avoid overflow and huge allocations.
  constexpr int limit = 1000000;
  for (const auto& p : contour)
    if (std::abs(static_cast<double>(p.x)) > limit ||
        std::abs(static_cast<double>(p.y)) > limit) return {};
  cv::Rect roi = cv::boundingRect(contour);
  roi.x -= padding_px;
  roi.y -= padding_px;
  roi.width += 2*padding_px;
  roi.height += 2*padding_px;
  if (static_cast<int64_t>(roi.width)*roi.height > max_pixels) return {};
  std::vector<cv::Point> local;
  local.reserve(contour.size());
  for (const auto& p : contour) local.push_back(p-roi.tl());
  cv::Mat mask(roi.size(), CV_8U, cv::Scalar(255)), distances;
  cv::polylines(mask, std::vector<std::vector<cv::Point>>{std::move(local)}, true,
                cv::Scalar(0), 1, cv::LINE_8);
  cv::distanceTransform(mask, distances, cv::DIST_L2, cv::DIST_MASK_PRECISE);
  return ContourDistanceField{roi, std::move(distances)};
}
double sampleDistanceField(const ContourDistanceField& f, cv::Point2d p, double cap) {
  const double x = p.x-f.bounds.x, y = p.y-f.bounds.y;
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 ||
      x >= f.distances.cols-1 || y >= f.distances.rows-1) return cap;
  const int ix = static_cast<int>(x), iy = static_cast<int>(y);
  const double dx = x-ix, dy = y-iy;
  return std::min(cap,
    (1-dy)*((1-dx)*f.distances.at<float>(iy,ix)+dx*f.distances.at<float>(iy,ix+1))+
    dy*((1-dx)*f.distances.at<float>(iy+1,ix)+dx*f.distances.at<float>(iy+1,ix+1)));
}
void appendContourSamples(const std::vector<cv::Point>& contour, std::vector<cv::Point2d>& out, int count) {
  if (contour.size()<3 || count<=0) return;
  // Uniform perimeter sampling makes PCA independent of CHAIN_APPROX_SIMPLE.
  double length=cv::arcLength(contour,true);
  if (!std::isfinite(length) || length<3) return;
  double traversed=0;
  std::size_t segment=0;
  for (int i=0;i<count;++i) {
    const double distance=i*length/count;
    while (segment+1<contour.size()) {
      const double edge=cv::norm(contour[(segment+1)%contour.size()]-contour[segment]);
      if (traversed+edge>=distance) break;
      traversed+=edge; ++segment;
    }
    const cv::Point2d a=contour[segment],b=contour[(segment+1)%contour.size()];
    const double edge=cv::norm(b-a);
    out.push_back(edge>1e-9 ? a+(b-a)*std::clamp((distance-traversed)/edge,0.0,1.0) : a);
  }
}

} // namespace tools::geometry
