#include "l1_perception/armor/armor_refiner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

float distance(const cv::Point2f& left, const cv::Point2f& right) noexcept
{
  return static_cast<float>(cv::norm(left - right));
}

struct Lightbar
{
  cv::Point2f center{};
  cv::Point2f axis{};
  double length{0.0};
  double width{0.0};
  double ratio{0.0};
};

Lightbar makeLightbar(const cv::RotatedRect& rotated_rect)
{
  std::array<cv::Point2f, 4> corners{};
  rotated_rect.points(corners.data());
  const cv::Point2f edge0 = corners[1] - corners[0];
  const cv::Point2f edge1 = corners[2] - corners[1];
  const double length0 = cv::norm(edge0);
  const double length1 = cv::norm(edge1);

  Lightbar lightbar;
  lightbar.center = rotated_rect.center;
  lightbar.length = std::max(length0, length1);
  lightbar.width = std::min(length0, length1);
  lightbar.ratio = lightbar.width > 0.0 ? lightbar.length / lightbar.width : 0.0;
  if (lightbar.length > 0.0) {
    // 只取长轴；轴的正负由各侧网络端点决定，不按图像 y 或矩形 angle 字段定。
    lightbar.axis = (length0 >= length1 ? edge0 : edge1) /
                   static_cast<float>(lightbar.length);
  }
  return lightbar;
}

bool finiteCorners(const std::array<cv::Point2f, 4>& corners) noexcept
{
  return std::all_of(corners.begin(), corners.end(), [](const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
  });
}

// 严格凸四边形返回绕向；交叉、凹陷或退化返回 0。
int quadWinding(const std::array<cv::Point2f, 4>& corners) noexcept
{
  int winding = 0;
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const cv::Point2f a = corners[(index + 1) % 4] - corners[index];
    const cv::Point2f b = corners[(index + 2) % 4] - corners[(index + 1) % 4];
    const double cross = static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x;
    if (!std::isfinite(cross) || std::abs(cross) < 1e-4) {
      return 0;
    }
    const int sign = cross > 0 ? 1 : -1;
    if (winding != 0 && winding != sign) {
      return 0;
    }
    winding = sign;
  }
  return winding;
}

struct LightMatch
{
  std::size_t contour_id;
  cv::Point2f top;
  cv::Point2f bottom;
  float endpoint_distance;
};

}  // namespace

ArmorRefiner::ArmorRefiner(ArmorRefinerConfig config) : config_(std::move(config))
{
}

bool ArmorRefiner::detect(Armor& armor, const cv::Mat& bgr_img) const
{
  if (!config_.enable || bgr_img.empty() || bgr_img.type() != CV_8UC3) {
    return false;
  }
  return detectOne(armor, bgr_img, nullptr);
}

bool ArmorRefiner::detectOne(Armor& armor, const cv::Mat& bgr_img, RefineRecord* record) const
{
  // TL、TR、BR、BL 表示网络角点身份；大倾角时不一定是图像的上下左右。
  const std::array<cv::Point2f, 4> input_corners = armor.corners;
  if (!finiteCorners(input_corners)) {
    return false;
  }
  const int input_winding = quadWinding(input_corners);
  if (input_winding == 0) {
    return false;
  }

  const cv::Point2f tl = input_corners[0];
  const cv::Point2f tr = input_corners[1];
  const cv::Point2f br = input_corners[2];
  const cv::Point2f bl = input_corners[3];
  const float left_length = distance(tl, bl);
  const float right_length = distance(tr, br);
  const cv::Point2f left_direction = (bl - tl) / left_length;
  const cv::Point2f right_direction = (br - tr) / right_length;

  if (record != nullptr) {
    record->lightbar_length = 0.5F * (left_length + right_length);
    const cv::Point2f left_center = (tl + bl) * 0.5F;
    const cv::Point2f right_center = (tr + br) * 0.5F;
    record->aspect_ratio = record->lightbar_length > 0.0F
                               ? distance(left_center, right_center) / record->lightbar_length
                               : 0.0F;
  }

  // 以下 ROI 外扩公式原样对应 SP-Vision Detector::detect(Armor&, image)。
  const cv::Point2f left_top_to_bottom = bl - tl;
  const cv::Point2f right_top_to_bottom = br - tr;
  const cv::Point2f tl1 = (tl + bl) * 0.5F - left_top_to_bottom;
  const cv::Point2f bl1 = (tl + bl) * 0.5F + left_top_to_bottom;
  const cv::Point2f br1 = (tr + br) * 0.5F + right_top_to_bottom;
  const cv::Point2f tr1 = (tr + br) * 0.5F - right_top_to_bottom;
  const cv::Point2f top_left_to_right = tr1 - tl1;
  const cv::Point2f bottom_left_to_right = br1 - bl1;
  const cv::Point2f tl2 = (tl1 + tr) * 0.5F - 0.75F * top_left_to_right;
  const cv::Point2f tr2 = (tl1 + tr) * 0.5F + 0.75F * top_left_to_right;
  const cv::Point2f bl2 = (bl1 + br) * 0.5F - 0.75F * bottom_left_to_right;
  const cv::Point2f br2 = (bl1 + br) * 0.5F + 0.75F * bottom_left_to_right;

  // SP 先转为整数 Point 再求 minAreaRect；保留该取整步骤，避免 ROI 边界相差 1 px。
  const std::vector<cv::Point> roi_points{tl2, tr2, br2, bl2};
  const cv::Rect bounding_box = cv::minAreaRect(roi_points).boundingRect();
  if (record != nullptr) {
    record->roi = bounding_box;
  }

  // 与 SP 一样：ROI 只要越界就放弃传统矫正，不裁切，也不改动网络结果。
  if (bounding_box.x < 0 || bounding_box.y < 0 ||
      bounding_box.x + bounding_box.width > bgr_img.cols ||
      bounding_box.y + bounding_box.height > bgr_img.rows || bounding_box.empty()) {
    return false;
  }

  const cv::Mat armor_roi = bgr_img(bounding_box);
  cv::Mat gray_img;
  cv::cvtColor(armor_roi, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, config_.binary_threshold, 255.0, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::vector<Lightbar> lightbars;
  lightbars.reserve(contours.size());
  for (const auto& contour : contours) {
    const Lightbar lightbar = makeLightbar(cv::minAreaRect(contour));
    const bool ratio_ok =
        lightbar.ratio > config_.min_lightbar_ratio && lightbar.ratio < config_.max_lightbar_ratio;
    const bool length_ok = lightbar.length > config_.min_lightbar_length_px;
    if (ratio_ok && length_ok) {
      lightbars.emplace_back(lightbar);
    }
  }

  if (lightbars.size() < 2) {
    if (record != nullptr) {
      record->size_skipped = record->lightbar_length < config_.min_lightbar_length_px;
    }
    return false;
  }

  const cv::Point2f roi_offset(static_cast<float>(bounding_box.x),
                               static_cast<float>(bounding_box.y));
  const double min_direction_dot =
    std::cos(static_cast<double>(config_.max_angle_error_deg) * CV_PI / 180.0);
  std::vector<LightMatch> left_matches;
  std::vector<LightMatch> right_matches;
  const auto match = [&](std::size_t id, cv::Point2f direction, cv::Point2f top,
                          cv::Point2f bottom, std::vector<LightMatch>& matches) {
    const auto& lightbar = lightbars[id];
    const float dot = lightbar.axis.dot(direction);
    if (std::abs(dot) < min_direction_dot) {
      return;
    }
    const cv::Point2f axis = dot < 0 ? -lightbar.axis : lightbar.axis;
    const cv::Point2f half = axis * static_cast<float>(lightbar.length * 0.5);
    const cv::Point2f found_top = lightbar.center + roi_offset - half;
    const cv::Point2f found_bottom = lightbar.center + roi_offset + half;
    matches.push_back({id, found_top, found_bottom,
                       distance(top, found_top) + distance(bottom, found_bottom)});
  };
  for (std::size_t id = 0; id < lightbars.size(); ++id) {
    match(id, left_direction, tl, bl, left_matches);
    match(id, right_direction, tr, br, right_matches);
  }

  if (record != nullptr) {
    record->left_found = !left_matches.empty();
    record->right_found = !right_matches.empty();
  }

  // 联合选出总端点偏差最小的有效配对，不能让两侧各自选中同一根亮条。
  std::array<cv::Point2f, 4> refined_corners{};
  float best_distance = std::numeric_limits<float>::max();
  float min_endpoint_distance = std::numeric_limits<float>::max();
  bool found_pair = false;
  for (const auto& left : left_matches) {
    for (const auto& right : right_matches) {
      if (left.contour_id == right.contour_id) {
        continue;
      }
      const float endpoint_distance = left.endpoint_distance + right.endpoint_distance;
      min_endpoint_distance = std::min(min_endpoint_distance, endpoint_distance);
      if (endpoint_distance >= config_.max_endpoint_distance_px ||
          endpoint_distance >= best_distance) {
        continue;
      }
      const std::array<cv::Point2f, 4> corners{left.top, right.top, right.bottom, left.bottom};
      if (quadWinding(corners) != input_winding) {
        continue;
      }
      refined_corners = corners;
      best_distance = endpoint_distance;
      found_pair = true;
    }
  }
  if (!found_pair) {
    if (record != nullptr) {
      record->shift_rejected = true;
      record->corner_shift = min_endpoint_distance;
    }
    return false;
  }

  float max_shift = 0.0F;
  for (std::size_t index = 0; index < refined_corners.size(); ++index) {
    max_shift = std::max(max_shift, distance(refined_corners[index], input_corners[index]));
  }

  // 只在全部条件满足后一次性写回，保证 false 时 Armor 完整保持原样。
  armor.corners = refined_corners;
  // SP 只覆盖 armor.points，不重算构造时由网络角点得到的 center。中心仍按
  // 网络结果保留，Tracker 的中心距离排序才能与 SP 完全同口径。

  if (record != nullptr) {
    record->corner_shift = max_shift;
  }
  return true;
}

RefineStats ArmorRefiner::refine(const cv::Mat& image, std::vector<Armor>& armors,
                                 std::vector<RefineRecord>* records) const
{
  RefineStats stats;
  if (records != nullptr) {
    records->clear();
    records->reserve(armors.size());
  }
  if (!config_.enable || image.empty() || image.type() != CV_8UC3) {
    return stats;
  }

  for (Armor& armor : armors) {
    RefineRecord record;
    record.network_corners = armor.corners;
    RefineRecord* record_ptr = records != nullptr ? &record : nullptr;

    if (detectOne(armor, image, record_ptr)) {
      ++stats.refined;
      record.verdict = RefineVerdict::Refined;
    } else {
      ++stats.network_kept;
      record.verdict = RefineVerdict::NetworkKept;
    }

    if (records != nullptr) {
      record.corners = armor.corners;
      records->push_back(record);
    }
  }

  return stats;
}

}  // namespace L2Perception
