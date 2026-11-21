#include "rune/f1-observation/scut_feature_points.hpp"
#include "rune/debug_overlay.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
using Contour = std::vector<cv::Point>;
std::vector<Contour> contours(const cv::Mat& mask) {
  std::vector<Contour> result;
  cv::findContours(mask, result, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  return result;
}

void gaps() {
  const cv::Point center{200, 200};
  const cv::RotatedRect ellipse(center, {180, 180}, 0);
  cv::Mat disk = cv::Mat::zeros(500, 500, CV_8U);
  cv::circle(disk, center, 90, cv::Scalar(255), -1);
  Rune::DetectedContours measured;
  measured.armor = contours(disk).front();
  std::vector<Contour> holes;
  for (int quadrant = 0; quadrant < 4; ++quadrant) {
    Contour polygon;
    for (int a = 5; a <= 85; ++a) {
      const double theta = (a + quadrant * 90) * CV_PI / 180;
      polygon.push_back(center + cv::Point(cvRound(72 * std::cos(theta)), cvRound(72 * std::sin(theta))));
    }
    for (int a = 85; a >= 5; --a) {
      const double theta = (a + quadrant * 90) * CV_PI / 180;
      polygon.push_back(center + cv::Point(cvRound(50 * std::cos(theta)), cvRound(50 * std::sin(theta))));
    }
    cv::Mat mask = cv::Mat::zeros(disk.size(), CV_8U);
    cv::fillPoly(mask, std::vector<Contour>{polygon}, cv::Scalar(255));
    holes.push_back(contours(mask).front());
  }
  Rune::extractScutFeaturePoints(measured, ellipse, {200, 450}, true, holes, {});
  for (const auto& p : measured.gap_points)
    require(p && cv::norm(*p - cv::Point2f(center)) > 55 && cv::norm(*p - cv::Point2f(center)) < 65,
            "four measured gaps must produce eight inner-ring points");
  holes.pop_back();
  Rune::extractScutFeaturePoints(measured, ellipse, {200, 450}, true, holes, {});
  int available = 0;
  for (const auto& p : measured.gap_points) available += p.has_value();
  require(available == 5, "a missing gap must remove its center and both adjacent endpoints");
  Rune::extractScutFeaturePoints(measured, ellipse, {200, 450}, true, {}, {});
  for (const auto& p : measured.gap_points) require(!p, "absent gaps were fabricated");
}

void fan() {
  cv::Mat mask = cv::Mat::zeros(600, 600, CV_8U);
  for (const auto& path : std::vector<Contour>{
      {{200, 250}, {200, 380}, {300, 520}}, {{300, 250}, {300, 520}},
      {{400, 250}, {400, 380}, {300, 520}}, {{180, 380}, {200, 380}},
      {{420, 380}, {400, 380}}})
    cv::polylines(mask, path, false, cv::Scalar(255), 10);
  Rune::DetectedContours measured;
  const cv::RotatedRect ellipse({300, 150}, {100, 100}, 0);
  Rune::extractScutFeaturePoints(measured, ellipse, {300, 570}, false, {}, contours(mask));
  require(measured.fan_points.size() == 6, "connected fan must expose six measured feature points");
  for (const auto expected : {cv::Point2f(200, 245), cv::Point2f(300, 245), cv::Point2f(400, 245),
                             cv::Point2f(200, 380), cv::Point2f(400, 380), cv::Point2f(300, 525)}) {
    double nearest = 1e6;
    for (const auto& p : measured.fan_points) nearest = std::min(nearest, cv::norm(p - expected));
    require(nearest < 10, "fan tips do not follow measured strip ends");
  }
  mask.rowRange(360, mask.rows).setTo(0);
  Rune::extractScutFeaturePoints(measured, ellipse, {300, 570}, false, {}, contours(mask));
  require(measured.fan_points.size() == 3, "occluded fan must keep only its three measured tips");
  mask.setTo(0);
  cv::circle(mask, {300, 380}, 30, cv::Scalar(255), -1);
  Rune::extractScutFeaturePoints(measured, ellipse, {300, 570}, false, {}, contours(mask));
  require(measured.fan_points.empty(), "a round blob was converted into a fan polygon");
}

void overlay() {
  auto measured = std::make_shared<Rune::DetectedContours>();
  measured->target_ellipse = cv::RotatedRect({180, 180}, {100, 80}, 0);
  measured->fan_points = {{140, 280}, {180, 280}, {220, 280}};
  Rune::Keypoints detection;
  detection.class_id = 1; detection.confidence = 1;
  detection.top = {400, 400}; detection.left = {390, 410};
  detection.right = {410, 410}; detection.bottom = {400, 420};
  detection.center_r = {180, 400}; detection.image_contours = measured;
  cv::Mat image = cv::Mat::zeros(500, 500, CV_8UC3);
  Rune::drawDetectionOverlay(image, {detection}, Rune::Mode::Big);
  require(cv::norm(image(cv::Rect(125, 135, 110, 90)), cv::NORM_INF) > 0,
          "overlay failed to draw the measured target ellipse");
  require(cv::norm(image(cv::Rect(135, 265, 95, 20)), cv::NORM_INF) > 0,
          "overlay failed to draw the measured fan endpoints");
  require(cv::norm(image(cv::Rect(380, 380, 50, 50)), cv::NORM_INF) == 0,
          "SCUT overlay still draws the compatibility five-point box");
}

void rectangleIdentity() {
  cv::Mat mask=cv::Mat::zeros(600,600,CV_8U);
  cv::rectangle(mask,{285,140},{315,230},cv::Scalar(255),-1);
  const cv::Point2f hub(300,300),target(300,100);
  Rune::DetectedContours source;
  Rune::extractScutFeaturePoints(source,cv::RotatedRect(target,{100,100},0),hub,true,{},contours(mask));
  require(source.fan_points.size()==4 && source.fan_points[0].x<source.fan_points[1].x &&
          source.fan_points[0].y<source.fan_points[3].y,
          "inactive fan corners do not follow the target/hub semantic frame");
  for (const double degrees : {-1.0,1.0,89.0,91.0}) {
    const auto warp=cv::getRotationMatrix2D(hub,degrees,1);
    const auto apply=[&](cv::Point2f p) {
      return cv::Point2f(float(warp.at<double>(0,0)*p.x+warp.at<double>(0,1)*p.y+warp.at<double>(0,2)),
                         float(warp.at<double>(1,0)*p.x+warp.at<double>(1,1)*p.y+warp.at<double>(1,2)));
    };
    cv::Mat rotated;cv::warpAffine(mask,rotated,warp,mask.size());
    Rune::DetectedContours current;
    Rune::extractScutFeaturePoints(current,cv::RotatedRect(apply(target),{100,100},0),hub,true,{},contours(rotated));
    require(current.fan_points.size()==4,"rotated inactive fan lost its rectangle");
    for (std::size_t i=0;i<4;++i)
      require(current.fanPointId(i)==int(i) && cv::norm(current.fan_points[i]-apply(source.fan_points[i]))<2.5,
              "inactive fan corner ID changed across an OpenCV rectangle angle boundary");
  }
}
} // namespace

int main() {
  try {
    gaps(); fan(); overlay();rectangleIdentity();
    std::cout << "rune_scut_features_smoke passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
