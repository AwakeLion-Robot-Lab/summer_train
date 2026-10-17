#include "rune/f1-observation/reobservation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace Rune {

std::optional<Observation> ObservationStage::run(const Region& region,
                                                 Color color, Mode mode) const {
  if (region.bgr.empty() || region.bgr.type() != CV_8UC3) return std::nullopt;
  std::vector<cv::Mat> channels;
  cv::split(region.bgr, channels);
  cv::Mat difference, smooth, mask;
  if (color == Color::Red) cv::subtract(channels[2], channels[0], difference);
  else cv::subtract(channels[0], channels[2], difference);
  cv::GaussianBlur(difference, smooth, {5, 5}, 0);
  cv::threshold(smooth, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  const cv::Point2f origin(static_cast<float>(region.bounds.x),
                           static_cast<float>(region.bounds.y));
  const cv::Point2f armor_center =
    (region.keypoints.top + region.keypoints.left + region.keypoints.right +
     region.keypoints.bottom) * 0.25F - origin;
  const cv::Point2f r = region.keypoints.center_r - origin;

  int armor_index = -1, r_index = -1, arm_index = -1;
  double armor_area = 0, r_area = 0, arm_area = 0;
  for (std::size_t i = 0; i < contours.size(); ++i) {
    const auto& contour = contours[i];
    const double area = cv::contourArea(contour);
    if (area < 5) continue;
    const bool contains_armor = cv::pointPolygonTest(contour, armor_center, false) >= 0;
    const bool contains_r = cv::pointPolygonTest(contour, r, false) >= 0;
    if (contains_armor && !contains_r && area > armor_area) {
      armor_index = static_cast<int>(i); armor_area = area;
    }
    if (contains_r && !contains_armor && (r_index < 0 || area < r_area)) {
      r_index = static_cast<int>(i); r_area = area;
    }
    if (!contains_armor && !contains_r && area > arm_area) {
      // A usable arm intersects the center-to-target segment.
      bool crosses = false;
      for (int step = 1; step < 10; ++step) {
        const cv::Point2f p = r + (armor_center - r) * (step / 10.0F);
        crosses |= cv::pointPolygonTest(contour, p, false) >= 0;
      }
      if (crosses) { arm_index = static_cast<int>(i); arm_area = area; }
    }
  }
  const bool arm_only = mode == Mode::Big && region.keypoints.class_id == 1;
  if (arm_only && arm_index < 0 && armor_index >= 0) {
    arm_index = armor_index;
    armor_index = -1;
  }
  if (r_index < 0 || arm_index < 0 || (!arm_only && armor_index < 0))
    return std::nullopt;

  const auto to_full_image = [&](int index) {
    if (index < 0) return std::vector<cv::Point>{};
    auto result = contours[index];
    for (auto& point : result) point += region.bounds.tl();
    return result;
  };
  return Observation{region.keypoints, to_full_image(armor_index),
                     to_full_image(arm_index), to_full_image(r_index)};
}

}  // namespace Rune
