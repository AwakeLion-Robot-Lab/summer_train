#include "l2_perception/armor/armor_refiner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <list>
#include <limits>
#include <optional>
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

Lightbar makeLightbar(const cv::RotatedRect& rotated_rect)
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


// 灯条端点的亮度梯度修正。照搬 awakening 的 correct_corners（其注释标为
// copy from sp_vision_25）。
//
// 与 makeLightbar 用 minAreaRect 取端点的区别：minAreaRect 的端点由二值化
// 边界决定，阈值偏低斑点膨胀、偏高收缩，**误差方向正好沿灯条轴向**——那正是
// UVL 观测里灯条长度和左右间距两个深度线索最怕的方向。这里改用亮度突变定位：
// 阈值只用来圈 ROI，端点由灰度梯度找，对曝光变化不敏感。
//
// 流程：ROI 扩 7% → 归一化到固定亮度范围 → 亮度矩求质心 → 对非零亮度点做
// PCA 求主轴 → 沿主轴正反向搜索亮度突变点（diff 最大且前一点亮于均值）→
// 多条平行搜索线取平均。任一步失败都保留原端点，不劣化。
bool correctLightbarEndpoints(Lightbar& lightbar, const cv::Mat& gray)
{
  constexpr float kMaxBrightness = 25.0F;  // 归一化上限
  constexpr float kRoiScale = 0.07F;       // ROI 扩展比例
  constexpr float kSearchStart = 0.4F;     // 从质心出发的搜索起点（灯条长度比例）
  constexpr float kSearchEnd = 0.6F;       // 搜索终点

  if (gray.empty() || lightbar.length < 2.0 || lightbar.width < 1.0) {
    return false;
  }

  // 以端点连线为主轴构造外接框，再按比例外扩。
  const cv::Rect2f raw_box = cv::RotatedRect(
    lightbar.center, cv::Size2f(static_cast<float>(lightbar.width),
                                static_cast<float>(lightbar.length)),
    static_cast<float>(std::atan2(
      lightbar.bottom.y - lightbar.top.y,
      lightbar.bottom.x - lightbar.top.x) * 180.0 / CV_PI - 90.0)).boundingRect2f();

  cv::Rect2f roi_box = raw_box;
  roi_box.x -= roi_box.width * kRoiScale;
  roi_box.y -= roi_box.height * kRoiScale;
  roi_box.width += 2.0F * roi_box.width * kRoiScale;
  roi_box.height += 2.0F * roi_box.height * kRoiScale;
  roi_box &= cv::Rect2f(0.0F, 0.0F, static_cast<float>(gray.cols),
                        static_cast<float>(gray.rows));
  if (roi_box.width <= 1.0F || roi_box.height <= 1.0F) {
    return false;
  }

  const cv::Rect roi_rect(roi_box);
  if (roi_rect.width <= 1 || roi_rect.height <= 1) {
    return false;
  }
  cv::Mat roi;
  gray(roi_rect).convertTo(roi, CV_32F);
  const float mean_value = static_cast<float>(cv::mean(gray(roi_rect))[0]);
  cv::normalize(roi, roi, 0.0F, kMaxBrightness, cv::NORM_MINMAX);

  const cv::Moments moments = cv::moments(roi);
  if (std::abs(moments.m00) < 1e-6) {
    return false;
  }
  const cv::Point2f centroid(
    static_cast<float>(moments.m10 / moments.m00) + static_cast<float>(roi_rect.x),
    static_cast<float>(moments.m01 / moments.m00) + static_cast<float>(roi_rect.y));

  // 亮度非零的点参与 PCA，权重体现在点的分布上。
  std::vector<cv::Point2f> points;
  points.reserve(static_cast<std::size_t>(roi.rows) * static_cast<std::size_t>(roi.cols));
  for (int row = 0; row < roi.rows; ++row) {
    for (int col = 0; col < roi.cols; ++col) {
      if (roi.at<float>(row, col) > 1e-3F) {
        points.emplace_back(static_cast<float>(col), static_cast<float>(row));
      }
    }
  }
  if (points.size() < 2) {
    return false;
  }

  const cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
  cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
  const float axis_norm = static_cast<float>(cv::norm(axis));
  if (axis_norm < 1e-6F) {
    return false;
  }
  axis /= axis_norm;
  if (axis.y > 0.0F) {
    axis = -axis;  // 统一指向图像上方
  }

  const auto findCorner = [&](int direction, cv::Point2f fallback) -> cv::Point2f {
    const float dx = axis.x * static_cast<float>(direction);
    const float dy = axis.y * static_cast<float>(direction);
    const float search_length =
      static_cast<float>(lightbar.length) * (kSearchEnd - kSearchStart);
    const int half_width =
      std::max(0, static_cast<int>((lightbar.width - 2.0) * 0.5));

    std::vector<cv::Point2f> candidates;
    for (int offset = -half_width; offset <= half_width; ++offset) {
      const cv::Point2f start(
        centroid.x + static_cast<float>(lightbar.length) * kSearchStart * dx +
          static_cast<float>(offset),
        centroid.y + static_cast<float>(lightbar.length) * kSearchStart * dy);

      cv::Point2f corner = start;
      float max_diff = 0.0F;
      bool found = false;
      for (float step = 0.0F; step < search_length; step += 1.0F) {
        const cv::Point2f current(start.x + dx * step, start.y + dy * step);
        const cv::Point current_pixel(
          static_cast<int>(current.x), static_cast<int>(current.y));
        const cv::Point previous_pixel(
          static_cast<int>(current.x - dx), static_cast<int>(current.y - dy));
        if (current_pixel.x < 0 || current_pixel.x >= gray.cols ||
            current_pixel.y < 0 || current_pixel.y >= gray.rows) {
          break;
        }
        if (previous_pixel.x < 0 || previous_pixel.x >= gray.cols ||
            previous_pixel.y < 0 || previous_pixel.y >= gray.rows) {
          continue;
        }
        const float previous_value =
          static_cast<float>(gray.at<uchar>(previous_pixel));
        const float current_value = static_cast<float>(gray.at<uchar>(current_pixel));
        const float diff = previous_value - current_value;
        // 只认"由亮到暗"的下降沿，且下降前那点必须亮于 ROI 均值，
        // 否则会在背景噪声上找到假端点。
        if (diff > max_diff && previous_value > mean_value) {
          max_diff = diff;
          corner = cv::Point2f(static_cast<float>(previous_pixel.x),
                               static_cast<float>(previous_pixel.y));
          found = true;
        }
      }
      if (found) {
        candidates.push_back(corner);
      }
    }

    if (candidates.empty()) {
      return fallback;  // 搜索失败保留原端点，不劣化
    }
    cv::Point2f sum(0.0F, 0.0F);
    for (const cv::Point2f& candidate : candidates) {
      sum += candidate;
    }
    return sum / static_cast<float>(candidates.size());
  };

  lightbar.top = findCorner(1, lightbar.top);
  lightbar.bottom = findCorner(-1, lightbar.bottom);
  return true;
}

Light makeIndependentLight(
  const Lightbar& lightbar, ArmorColor color, std::size_t id)
{
  Light light;
  light.top = lightbar.top;
  light.bottom = lightbar.bottom;
  light.center = (light.top + light.bottom) * 0.5F;
  light.length = cv::norm(light.top - light.bottom);
  light.width = lightbar.width;
  light.tilt_angle_deg = static_cast<float>(
    std::atan2(
      std::abs(light.top.x - light.bottom.x),
      std::abs(light.top.y - light.bottom.y)) *
    180.0 / CV_PI);
  light.color = color;
  light.id = id;
  return light;
}

bool finiteCorners(const std::array<cv::Point2f, 4>& corners) noexcept;

std::optional<double> referenceLightThreshold(
  const cv::Mat& gray, const cv::Rect& roi,
  const std::vector<Armor>& armors, ArmorColor target_color)
{
  std::vector<double> values;
  values.reserve(armors.size() * 2);

  const auto sample = [&](cv::Point2f top, cv::Point2f bottom) {
    top -= cv::Point2f(
      static_cast<float>(roi.x), static_cast<float>(roi.y));
    bottom -= cv::Point2f(
      static_cast<float>(roi.x), static_cast<float>(roi.y));
    const cv::Point2f delta = bottom - top;
    const int samples = std::max(2, static_cast<int>(cv::norm(delta)));
    double sum = 0.0;
    int count = 0;
    for (int index = 0; index < samples; ++index) {
      const float ratio = static_cast<float>(index) /
                          static_cast<float>(samples - 1);
      const cv::Point2f point = top + ratio * delta;
      const int x = static_cast<int>(std::round(point.x));
      const int y = static_cast<int>(std::round(point.y));
      if (x < 0 || x >= gray.cols || y < 0 || y >= gray.rows) {
        continue;
      }
      sum += static_cast<double>(gray.at<unsigned char>(y, x));
      ++count;
    }
    if (count > 0) {
      values.push_back(sum / static_cast<double>(count));
    }
  };

  for (const Armor& armor : armors) {
    if (armor.color == ArmorColor::Unknown ||
        (target_color != ArmorColor::Unknown && armor.color != target_color) ||
        !finiteCorners(armor.corners)) {
      continue;
    }
    sample(armor.corners[0], armor.corners[3]);
    sample(armor.corners[1], armor.corners[2]);
  }

  if (values.empty()) {
    return std::nullopt;
  }
  double sum = 0.0;
  for (const double value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

cv::Mat independentLightColorMask(
  const cv::Mat& hsv, ArmorColor color,
  const ArmorRefinerConfig& config)
{
  cv::Mat mask;
  if (color == ArmorColor::Blue) {
    cv::inRange(
      hsv,
      cv::Scalar(
        config.independent_light_blue_h_min,
        config.independent_light_blue_s_min,
        config.independent_light_blue_v_min),
      cv::Scalar(
        config.independent_light_blue_h_max, 255, 255), mask);
    return mask;
  }

  if (color == ArmorColor::Red) {
    cv::Mat low;
    cv::Mat high;
    cv::inRange(
      hsv,
      cv::Scalar(
        config.independent_light_red_h_min_low,
        config.independent_light_red_s_min,
        config.independent_light_red_v_min),
      cv::Scalar(
        config.independent_light_red_h_max_low, 255, 255), low);
    cv::inRange(
      hsv,
      cv::Scalar(
        config.independent_light_red_h_min_high,
        config.independent_light_red_s_min,
        config.independent_light_red_v_min),
      cv::Scalar(
        config.independent_light_red_h_max_high, 255, 255), high);
    cv::bitwise_or(low, high, mask);
  }
  return mask;
}

bool finiteCorners(const std::array<cv::Point2f, 4>& corners) noexcept
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
    Lightbar lightbar = makeLightbar(cv::minAreaRect(contour));
    const bool angle_ok = lightbar.angle_error < max_angle_error;
    const bool ratio_ok =
        lightbar.ratio > config_.min_lightbar_ratio && lightbar.ratio < config_.max_lightbar_ratio;
    const bool length_ok = lightbar.length > config_.min_lightbar_length_px;
    if (angle_ok && ratio_ok && length_ok) {
      // 亮度梯度修正端点。放在筛选之后是因为它比 minAreaRect 贵得多，
      // 没必要在明显不是灯条的轮廓上花；放在配对之前是为了让端点距离门限
      // 也用上修正后的位置。
      if (config_.pca_corner_correction) {
        correctLightbarEndpoints(lightbar, gray_img);
      }
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

std::vector<Light> ArmorRefiner::detectLights(
  const cv::Mat& image, const cv::Rect& roi,
  const std::vector<Armor>& reference_armors,
  ArmorColor target_color) const
{
  std::vector<Light> lights;
  if (!config_.independent_light_enable || image.empty() ||
      image.type() != CV_8UC3) {
    return lights;
  }

  const cv::Rect image_rect(0, 0, image.cols, image.rows);
  const cv::Rect clipped_roi = roi & image_rect;
  if (clipped_roi.empty()) {
    return lights;
  }

  const cv::Mat detect_roi = image(clipped_roi);
  cv::Mat gray;
  cv::cvtColor(detect_roi, gray, cv::COLOR_BGR2GRAY);
  cv::Mat hsv;
  cv::cvtColor(detect_roi, hsv, cv::COLOR_BGR2HSV);

  double threshold = config_.independent_light_binary_threshold;
  if (config_.independent_light_threshold_tolerance > 0.0) {
    if (const auto reference =
          referenceLightThreshold(
            gray, clipped_roi, reference_armors, target_color)) {
      // 网络完整板只允许把门限抬高，不能把基础门限拉低；后者正是暗背景纹理
      // 在低曝光帧突然大批进入候选集的主要原因。
      threshold = std::max(
        threshold,
        *reference - config_.independent_light_threshold_tolerance);
    }
  }
  threshold = std::clamp(threshold, 0.0, 254.0);

  cv::Mat brightness_mask;
  cv::threshold(
    gray, brightness_mask, threshold, 255.0, cv::THRESH_BINARY);

  std::size_t light_id = 0;
  const cv::Point2f offset(
    static_cast<float>(clipped_roi.x), static_cast<float>(clipped_roi.y));

  const auto detectColor = [&](ArmorColor color) {
    cv::Mat binary = independentLightColorMask(hsv, color, config_);
    if (binary.empty()) {
      return;
    }
    // dx_vision 的 HSV 色域负责排除白光和错误颜色；亮度掩码继续保留
    // Awakening 随本帧完整板自适应曝光的能力。
    cv::bitwise_and(binary, brightness_mask, binary);
    if (config_.independent_light_use_morphology) {
      const cv::Size kernel_size(
        std::max(1, config_.independent_light_morphology_width),
        std::max(1, config_.independent_light_morphology_height));
      const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, kernel_size);
      cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
      cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
      binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    lights.reserve(lights.size() + contours.size());
    for (const auto& contour : contours) {
      if (contour.size() < 2) {
        continue;
      }
      const double contour_area = std::abs(cv::contourArea(contour));
      if (contour_area < config_.independent_light_min_contour_area_px) {
        continue;
      }

      const cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
      const double rect_area = static_cast<double>(rotated_rect.size.area());
      if (!(rect_area > 1e-6) ||
          contour_area / rect_area < config_.independent_light_min_fill_ratio) {
        continue;
      }

      Lightbar lightbar = makeLightbar(rotated_rect);
      if (!(lightbar.length > 0.0) ||
          lightbar.length < config_.independent_light_min_length_px) {
        continue;
      }
      const double width_length_ratio = lightbar.width / lightbar.length;
      const double tilt_angle_deg = lightbar.angle_error * 180.0 / CV_PI;
      if (width_length_ratio <=
            config_.independent_light_min_width_length_ratio ||
          width_length_ratio >=
            config_.independent_light_max_width_length_ratio ||
          tilt_angle_deg >= config_.independent_light_max_tilt_angle_deg) {
        continue;
      }

      // dx_vision 会按轮廓质量在三种端点策略间切换；newvision 已有同目的的
      // PCA + 灰度下降沿修正，沿用它可避免二值阈值改变 UVL 灯条长度。
      if (config_.pca_corner_correction) {
        correctLightbarEndpoints(lightbar, gray);
      }
      Light light = makeIndependentLight(lightbar, color, light_id);
      if (light.length < config_.independent_light_min_length_px ||
          light.tilt_angle_deg >=
            config_.independent_light_max_tilt_angle_deg) {
        continue;
      }

      // 在整个轮廓内部取平均色，而不是只采轮廓边界。背景色块即使色相落入
      // HSV 范围，内部夹杂的大量无色像素也会把红蓝通道差拉低并被拒绝。
      cv::Mat contour_mask = cv::Mat::zeros(binary.size(), CV_8UC1);
      std::vector<std::vector<cv::Point>> one_contour{contour};
      cv::drawContours(
        contour_mask, one_contour, 0, cv::Scalar{255}, cv::FILLED);
      const cv::Scalar mean_bgr = cv::mean(detect_roi, contour_mask);
      if (std::abs(mean_bgr[2] - mean_bgr[0]) <=
          config_.independent_light_color_diff_threshold) {
        continue;
      }

      light.center += offset;
      light.top += offset;
      light.bottom += offset;
      light.id = light_id++;
      lights.push_back(light);
    }
  };

  if (target_color == ArmorColor::Red ||
      target_color == ArmorColor::Unknown) {
    detectColor(ArmorColor::Red);
  }
  if (target_color == ArmorColor::Blue ||
      target_color == ArmorColor::Unknown) {
    detectColor(ArmorColor::Blue);
  }
  return lights;
}

}  // namespace L2Perception
