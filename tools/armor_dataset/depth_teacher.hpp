#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace armor_dataset {

struct TeacherPose {
  cv::Vec3d rotation;
  cv::Vec3d translation;
  double rmse_px{0.0};
  double alternative_depth_relative_gap{0.0};
};

// Camera-frame translation only: no IMU, barrel transform, tracking lag or frame association.
inline std::optional<TeacherPose> solveTeacher(
  const std::array<cv::Point2f, 4>& corners, const cv::Mat& camera,
  const cv::Mat& distortion, double width_m, double height_m)
{
  if (!(width_m > 0 && height_m > 0)) return std::nullopt;
  const std::vector<cv::Point3d> points{
    {-width_m/2, -height_m/2, 0}, {width_m/2, -height_m/2, 0},
    {width_m/2, height_m/2, 0}, {-width_m/2, height_m/2, 0}};
  const std::vector<cv::Point2f> pixels(corners.begin(), corners.end());
  for (const auto& p : pixels) if (!std::isfinite(p.x) || !std::isfinite(p.y)) return std::nullopt;
  std::vector<cv::Mat> rotations, translations;
  try {
    cv::solvePnPGeneric(points, pixels, camera, distortion, rotations, translations,
                        false, cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception&) { return std::nullopt; }
  std::vector<TeacherPose> candidates;
  for (std::size_t i=0; i<rotations.size(); ++i) {
    const cv::Vec3d r(rotations[i]), t(translations[i]);
    if (!cv::checkRange(rotations[i]) || !cv::checkRange(translations[i]) || t[2] <= 0) continue;
    cv::Matx33d rotation;
    cv::Rodrigues(r, rotation);
    bool positive = true;
    for (const auto& p : points) positive &= (rotation*cv::Vec3d(p.x,p.y,p.z)+t)[2] > 1e-6;
    if (!positive) continue;
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points, r, t, camera, distortion, projected);
    double error = 0;
    for (std::size_t j=0;j<4;++j) {
      const cv::Point2d delta = projected[j]-cv::Point2d(pixels[j]);
      error += delta.dot(delta);
    }
    candidates.push_back({r,t,std::sqrt(error/4),0});
  }
  if (candidates.empty()) return std::nullopt;
  std::sort(candidates.begin(), candidates.end(), [](const auto& a,const auto& b){return a.rmse_px<b.rmse_px;});
  if (candidates.size()>1) candidates[0].alternative_depth_relative_gap =
    std::abs(candidates[1].translation[2]-candidates[0].translation[2])/candidates[0].translation[2];
  return candidates[0];
}

inline std::array<cv::Point2f,4> digitGuide(const std::array<cv::Point2f,4>& c)
{
  const auto at = [&](float u,float v) {
    return ((1-u)*c[0]+u*c[1])*(1-v)+((1-u)*c[3]+u*c[2])*v;
  };
  return {at(.08F,-.85F),at(.92F,-.85F),at(.92F,1.85F),at(.08F,1.85F)};
}

struct DigitRegion { cv::Rect box; double contrast{0}; };

// Teacher corners only guide the search. The saved crop is an unrectified raw-image
// box around the actual white glyph; neither lightbar pixels nor teacher corners
// are student inputs. A deployed digit detector can provide this same box contract.
inline std::optional<DigitRegion> findDigit(
  const cv::Mat& image, const std::array<cv::Point2f,4>& corners)
{
  const auto guide = digitGuide(corners);
  const std::vector<cv::Point2f> polygon(guide.begin(),guide.end());
  cv::Rect roi = cv::boundingRect(polygon);
  if ((roi & cv::Rect(0,0,image.cols,image.rows)) != roi || roi.area() <= 0) return std::nullopt;
  std::vector<cv::Mat> channels;
  cv::split(image(roi),channels);
  cv::Mat white;
  cv::min(channels[0],channels[1],white); cv::min(white,channels[2],white);
  cv::GaussianBlur(white,white,cv::Size(3,3),.5);
  cv::Mat allowed = cv::Mat::zeros(roi.size(),CV_8U);
  std::vector<cv::Point> local;
  for (const auto& p : guide) local.emplace_back(cvRound(p.x-roi.x),cvRound(p.y-roi.y));
  cv::fillConvexPoly(allowed,local,255);
  cv::Mat selected = cv::Mat::zeros(white.size(),CV_8U);
  white.copyTo(selected,allowed);
  // Measure contrast in the central part of the plate. Bright posters just
  // beyond its upper/lower edge must not set the glyph's binarization level.
  const auto at = [&](float u,float v) {
    return ((1-u)*corners[0]+u*corners[1])*(1-v)+((1-u)*corners[3]+u*corners[2])*v;
  };
  std::vector<cv::Point> core;
  for(const auto& p:{at(.25F,-.20F),at(.75F,-.20F),at(.75F,1.20F),at(.25F,1.20F)})
    core.emplace_back(cvRound(p.x-roi.x),cvRound(p.y-roi.y));
  cv::Mat core_mask=cv::Mat::zeros(roi.size(),CV_8U);
  cv::fillConvexPoly(core_mask,core,255);
  double low,high;
  cv::minMaxLoc(selected,&low,&high,nullptr,nullptr,core_mask);
  if (high-low<35) return std::nullopt;
  cv::Mat binary;
  cv::threshold(selected,binary,std::max(40.0,low+.40*(high-low)),255,cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
  const double light_length = .5*(cv::norm(corners[3]-corners[0])+cv::norm(corners[2]-corners[1]));
  const cv::Point2f center = (corners[0]+corners[1]+corners[2]+corners[3])*.25F;
  double best_score = 0;
  std::optional<DigitRegion> result;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    cv::Rect box = cv::boundingRect(contour);
    if (area<12 || box.width<5 || box.height<std::max(12.0,.75*light_length) || box.height>3.2*light_length) continue;
    const double ratio = static_cast<double>(box.width)/box.height;
    if (ratio<.10 || ratio>1.5) continue;
    const cv::Point2f candidate(box.x+roi.x+box.width*.5F,box.y+roi.y+box.height*.5F);
    const cv::Point2f left=(corners[0]+corners[3])*.5F;
    const cv::Point2f across=(corners[1]+corners[2])*.5F-left;
    const double u=(candidate-left).dot(across)/across.dot(across);
    // A wider search keeps the full sticker glyph; its centre must still be
    // between the bars, so a saturated fragment of either bar cannot win.
    if(u<.30 || u>.70)continue;
    const double offset = cv::norm(candidate-center)/light_length;
    if (offset>1.0) continue;
    const int pad = std::max(1,cvRound(box.height*.05));
    box = cv::Rect(box.x+roi.x-pad,box.y+roi.y-pad,box.width+2*pad,box.height+2*pad);
    if((box&cv::Rect(0,0,image.cols,image.rows))!=box)continue;
    bool clipped=false;
    for(const auto& p:contour)
      clipped |= cv::pointPolygonTest(polygon,cv::Point2f(p.x+roi.x,p.y+roi.y),true)<.8;
    if(clipped)continue;
    // An axis-aligned glyph box can have empty corners outside a tilted search
    // polygon. Check the actual strokes for clipping, and exclude lightbar
    // centre-lines with a thickness margin instead of rejecting those corners.
    const int light_margin=std::max(2,cvRound(light_length*.10));
    const cv::Rect safe=box+cv::Size(2*light_margin,2*light_margin)-cv::Point(light_margin,light_margin);
    bool includes_light=false;
    for(const auto pair:{std::array<int,2>{0,3},std::array<int,2>{1,2}}){
      cv::Point a(cvRound(corners[pair[0]].x),cvRound(corners[pair[0]].y));
      cv::Point b(cvRound(corners[pair[1]].x),cvRound(corners[pair[1]].y));
      includes_light |= cv::clipLine(safe,a,b);
    }
    if(includes_light)continue;
    const double score=area/(1+3*offset);
    if (score>best_score) {best_score=score;result=DigitRegion{box,high-low};}
  }
  return result;
}

} // namespace armor_dataset
