#include "l2_perception/armor/armor_refiner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <list>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

[[nodiscard]] float distance(const cv::Point2f& left, const cv::Point2f& right) noexcept
{
  return static_cast<float>(cv::norm(left - right));
}

// 字段与构造方式逐项对应 SP-Vision auto_aim::Lightbar。
struct Lightbar
{
  cv::Point2f center{};
  cv::Point2f top{};
  cv::Point2f bottom{};
  double length{0.0};
  double width{0.0};
  double ratio{0.0};
  double angle_error{0.0};
};

[[nodiscard]] Lightbar makeLightbar(const cv::RotatedRect& rotated_rect)
{
  std::array<cv::Point2f, 4> corners{};
  rotated_rect.points(corners.data());
  std::sort(corners.begin(), corners.end(),
            [](const cv::Point2f& left, const cv::Point2f& right) { return left.y < right.y; });

  Lightbar lightbar;
  lightbar.center = rotated_rect.center;
  lightbar.top = (corners[0] + corners[1]) * 0.5F;
  lightbar.bottom = (corners[2] + corners[3]) * 0.5F;

  const cv::Point2f top_to_bottom = lightbar.bottom - lightbar.top;
  lightbar.width = cv::norm(corners[0] - corners[1]);
  lightbar.length = cv::norm(top_to_bottom);
  lightbar.ratio = lightbar.width > 0.0 ? lightbar.length / lightbar.width : 0.0;

  const double angle = std::atan2(top_to_bottom.y, top_to_bottom.x);
  lightbar.angle_error = std::abs(angle - CV_PI * 0.5);
  return lightbar;
}

[[nodiscard]] bool finiteCorners(const std::array<cv::Point2f, 4>& corners) noexcept
{
  return std::all_of(corners.begin(), corners.end(), [](const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
  });
}

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
  // SP-Vision 的点顺序为 TL、TR、BR、BL，newvision::Armor::corners 使用相同约定。
  const std::array<cv::Point2f, 4> input_corners = armor.corners;
  if (!finiteCorners(input_corners)) {
    return false;
  }

  const cv::Point2f tl = input_corners[0];
  const cv::Point2f tr = input_corners[1];
  const cv::Point2f br = input_corners[2];
  const cv::Point2f bl = input_corners[3];

  if (record != nullptr) {
    const float left_length = distance(tl, bl);
    const float right_length = distance(tr, br);
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

  const double max_angle_error = static_cast<double>(config_.max_angle_error_deg) / 57.3;
  std::list<Lightbar> lightbars;
  for (const auto& contour : contours) {
    const Lightbar lightbar = makeLightbar(cv::minAreaRect(contour));
    const bool angle_ok = lightbar.angle_error < max_angle_error;
    const bool ratio_ok =
        lightbar.ratio > config_.min_lightbar_ratio && lightbar.ratio < config_.max_lightbar_ratio;
    const bool length_ok = lightbar.length > config_.min_lightbar_length_px;
    if (angle_ok && ratio_ok && length_ok) {
      lightbars.emplace_back(lightbar);
    }
  }

  if (lightbars.size() < 2) {
    if (record != nullptr) {
      record->size_skipped = record->lightbar_length < config_.min_lightbar_length_px;
    }
    return false;
  }

  lightbars.sort([](const Lightbar& left, const Lightbar& right) {
    return left.center.x < right.center.x;
  });

  const cv::Point2f roi_offset(static_cast<float>(bounding_box.x),
                               static_cast<float>(bounding_box.y));
  const Lightbar* closest_left_lightbar = nullptr;
  const Lightbar* closest_right_lightbar = nullptr;
  float min_distance_tl_bl = std::numeric_limits<float>::max();
  float min_distance_br_tr = std::numeric_limits<float>::max();
  for (const Lightbar& lightbar : lightbars) {
    const float distance_tl_bl = static_cast<float>(
      cv::norm(tl - (lightbar.top + roi_offset)) +
      cv::norm(bl - (lightbar.bottom + roi_offset)));
    if (distance_tl_bl < min_distance_tl_bl) {
      min_distance_tl_bl = distance_tl_bl;
      closest_left_lightbar = &lightbar;
    }

    const float distance_br_tr = static_cast<float>(
      cv::norm(br - (lightbar.bottom + roi_offset)) +
      cv::norm(tr - (lightbar.top + roi_offset)));
    if (distance_br_tr < min_distance_br_tr) {
      min_distance_br_tr = distance_br_tr;
      closest_right_lightbar = &lightbar;
    }
  }

  if (record != nullptr) {
    record->left_found = closest_left_lightbar != nullptr;
    record->right_found = closest_right_lightbar != nullptr;
  }

  const float endpoint_distance = min_distance_tl_bl + min_distance_br_tr;
  if (closest_left_lightbar == nullptr || closest_right_lightbar == nullptr ||
      endpoint_distance >= config_.max_endpoint_distance_px) {
    if (record != nullptr) {
      record->shift_rejected = true;
      record->corner_shift = endpoint_distance;
    }
    return false;
  }

  const std::array<cv::Point2f, 4> refined_corners{
      closest_left_lightbar->top + roi_offset, closest_right_lightbar->top + roi_offset,
      closest_right_lightbar->bottom + roi_offset, closest_left_lightbar->bottom + roi_offset};

  const std::array<cv::Point2f, 4> network_corners =
      armor.corner_source == CornerSource::Refined ? armor.network_corners : input_corners;
  float max_shift = 0.0F;
  for (std::size_t index = 0; index < refined_corners.size(); ++index) {
    max_shift = std::max(max_shift, distance(refined_corners[index], network_corners[index]));
  }

  // 只在全部条件满足后一次性写回，保证 false 时 Armor 完整保持原样。
  armor.network_corners = network_corners;
  armor.corners = refined_corners;
  armor.corner_source = CornerSource::Refined;
  armor.corner_shift = max_shift;
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
