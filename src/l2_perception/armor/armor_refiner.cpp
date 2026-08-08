#include "l2_perception/armor/armor_refiner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

[[nodiscard]] float distance(const cv::Point2f& left, const cv::Point2f& right) noexcept
{
  return static_cast<float>(cv::norm(left - right));
}

// 复刻 SP-Vision 的 Lightbar：按 y 排序 minAreaRect 的四角，上两点均值为顶端、
// 下两点均值为底端，width 取顶部两角间距，angle_error 是主轴与竖直方向的夹角。
// 各量定义必须与其一致，否则 standard3.yaml 里那套阈值搬过来就不成立了。
struct Lightbar
{
  cv::Point2f center{};
  cv::Point2f top{};
  cv::Point2f bottom{};
  float length{0.0F};
  float width{0.0F};
  float ratio{0.0F};
  float angle_error{0.0F};
};

[[nodiscard]] Lightbar makeLightbar(const cv::RotatedRect& rect)
{
  std::array<cv::Point2f, 4> corners{};
  rect.points(corners.data());
  std::sort(corners.begin(), corners.end(),
            [](const cv::Point2f& left, const cv::Point2f& right) { return left.y < right.y; });

  Lightbar bar;
  bar.center = rect.center;
  bar.top = (corners[0] + corners[1]) * 0.5F;
  bar.bottom = (corners[2] + corners[3]) * 0.5F;

  const cv::Point2f top2bottom = bar.bottom - bar.top;
  bar.width = distance(corners[0], corners[1]);
  bar.length = static_cast<float>(cv::norm(top2bottom));
  bar.ratio = bar.width > 0.0F ? bar.length / bar.width : 0.0F;

  const float angle = std::atan2(top2bottom.y, top2bottom.x);
  bar.angle_error = std::abs(angle - static_cast<float>(CV_PI) * 0.5F);
  return bar;
}

}  // namespace

ArmorRefiner::ArmorRefiner(ArmorRefinerConfig config) : config_(std::move(config))
{
}

RefineVerdict ArmorRefiner::refineOne(const cv::Mat& image, Armor& armor,
                                     RefineRecord* record) const
{
  // 期望的左右灯条中轴直接取自网络角点：左 = TL->BL，右 = TR->BR。
  // 有了这个先验就不需要 SP-Vision 那种全图两两配对，也就不存在共用灯条的歧义。
  const cv::Point2f left_center = (armor.corners[0] + armor.corners[3]) * 0.5F;
  const cv::Point2f right_center = (armor.corners[1] + armor.corners[2]) * 0.5F;
  const float separation = distance(left_center, right_center);
  const float lightbar_length =
      0.5F * (distance(armor.corners[0], armor.corners[3]) +
              distance(armor.corners[1], armor.corners[2]));

  if (record != nullptr) {
    record->lightbar_length = lightbar_length;
    record->aspect_ratio = lightbar_length > 0.0F ? separation / lightbar_length : 0.0F;
  }

  // 尺寸门槛：太小时传统检测的结论只是阈值噪声，必须跳过而不是当作证据。
  if (separation < 1.0F || lightbar_length < config_.min_lightbar_length_px) {
    if (record != nullptr) {
      record->size_skipped = true;
    }
    return RefineVerdict::NetworkKept;
  }

  const std::vector<cv::Point2f> corner_points(armor.corners.begin(), armor.corners.end());
  cv::Rect roi = cv::boundingRect(corner_points);
  const int expand_x = static_cast<int>(static_cast<float>(roi.width) * config_.roi_expand_ratio);
  const int expand_y = static_cast<int>(static_cast<float>(roi.height) * config_.roi_expand_ratio);
  roi.x -= expand_x;
  roi.y -= expand_y;
  roi.width += 2 * expand_x;
  roi.height += 2 * expand_y;
  roi &= cv::Rect(0, 0, image.cols, image.rows);
  if (record != nullptr) {
    record->roi = roi;
  }
  if (roi.width < 3 || roi.height < 3) {
    return RefineVerdict::NetworkKept;
  }

  // image(roi) 是零拷贝视图，由此得到的坐标都是 ROI 局部的。偏移在生成
  // RotatedRect 时一次性补回原图坐标，避免后续每处都要记得加。
  const cv::Mat roi_image = image(roi);

  // 与 SP-Vision Detector::detect 完全一致：转灰度 → 固定阈值二值化 → 外轮廓。
  cv::Mat gray;
  cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);
  cv::Mat binary;
  cv::threshold(gray, binary, config_.binary_threshold, 255.0, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  const cv::Point2f roi_offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
  const float max_angle_error =
      config_.max_angle_error_deg * static_cast<float>(CV_PI) / 180.0F;
  std::optional<Lightbar> left_bar;
  std::optional<Lightbar> right_bar;
  float left_best = std::numeric_limits<float>::max();
  float right_best = std::numeric_limits<float>::max();
  bool merged_blob = false;

  for (const auto& contour : contours) {
    cv::RotatedRect rect = cv::minAreaRect(contour);
    // ROI 是零拷贝视图，轮廓坐标是局部的。在这里一次性补回原图坐标，
    // 后续所有几何量就都在原图系里，不必每处再记得加偏移。
    rect.center += roi_offset;
    const Lightbar bar = makeLightbar(rect);

    // 复刻 SP-Vision Detector::check_geometry(const Lightbar&)。
    if (bar.angle_error >= max_angle_error) {
      continue;
    }
    if (bar.ratio <= config_.min_lightbar_ratio || bar.ratio >= config_.max_lightbar_ratio) {
      continue;
    }
    if (bar.length <= config_.min_lightbar_length_px) {
      continue;
    }

    const float to_left = distance(bar.center, left_center);
    const float to_right = distance(bar.center, right_center);

    // 连通域落在两条中轴正中间，说明过曝把两根灯条粘成了一块。这是检查失效，
    // 不是遮挡证据，必须和"真的只有一根"区分开。
    if (std::abs(to_left - to_right) < separation * config_.merged_blob_ratio) {
      merged_blob = true;
      continue;
    }

    if (to_left < to_right) {
      if (to_left < separation * config_.max_assign_ratio && to_left < left_best) {
        left_best = to_left;
        left_bar = bar;
      }
    } else {
      if (to_right < separation * config_.max_assign_ratio && to_right < right_best) {
        right_best = to_right;
        right_bar = bar;
      }
    }
  }

  const bool has_left = left_bar.has_value();
  const bool has_right = right_bar.has_value();
  if (record != nullptr) {
    record->left_found = has_left;
    record->right_found = has_right;
    record->merged_blob = merged_blob;
  }

  if (!has_left && !has_right) {
    // 一根都没找到：阈值或 ROI 没起作用，属于检查失效。
    return RefineVerdict::NetworkKept;
  }

  if (has_left != has_right) {
    // 只匹配到一侧。粘连情形已经说明检查不可靠，不能据此拒绝。
    if (merged_blob) {
      return RefineVerdict::NetworkKept;
    }
    // 侧对门槛：正对时只找到一根灯条通常是曝光或阈值问题，不是遮挡。
    // 只有明显侧对时，单灯条才是真实的物理遮挡。
    const float aspect_ratio = separation / lightbar_length;
    if (aspect_ratio >= config_.edge_on_aspect_ratio) {
      return RefineVerdict::NetworkKept;
    }
    return RefineVerdict::Rejected;
  }

  // 顺序必须是左上、右上、右下、左下，与 Armor::corners 的约定一致。
  const std::array<cv::Point2f, 4> refined{left_bar->top, right_bar->top, right_bar->bottom,
                                           left_bar->bottom};

  const float diagonal = distance(armor.corners[0], armor.corners[2]);
  float max_shift = 0.0F;
  for (std::size_t index = 0; index < refined.size(); ++index) {
    if (!std::isfinite(refined[index].x) || !std::isfinite(refined[index].y)) {
      return RefineVerdict::NetworkKept;
    }
    max_shift = std::max(max_shift, distance(refined[index], armor.corners[index]));
  }

  // 位移过大说明两条通路互相矛盾，此时哪一个都不可信，保留网络角点。
  if (diagonal > 0.0F && max_shift > diagonal * config_.max_corner_shift_ratio) {
    if (record != nullptr) {
      record->shift_rejected = true;
      record->corner_shift = max_shift;
    }
    return RefineVerdict::NetworkKept;
  }

  armor.corners = refined;
  armor.corner_shift = max_shift;
  armor.corner_source = CornerSource::Refined;
  armor.center = {};
  for (const cv::Point2f& corner : armor.corners) {
    armor.center += corner;
  }
  armor.center = armor.center * 0.25F;
  return RefineVerdict::Refined;
}

RefineStats ArmorRefiner::refine(const cv::Mat& image, std::vector<Armor>& armors,
                                std::vector<RefineRecord>* records) const
{
  RefineStats stats;
  if (records != nullptr) {
    records->clear();
  }
  if (!config_.enable || image.empty() || image.type() != CV_8UC3) {
    return stats;
  }

  std::vector<Armor> kept;
  kept.reserve(armors.size());
  for (Armor& armor : armors) {
    // 精修前先留档原始角点，供离线评估和回放叠加对比。
    armor.network_corners = armor.corners;

    RefineRecord record;
    RefineRecord* record_ptr = records != nullptr ? &record : nullptr;
    if (record_ptr != nullptr) {
      record.network_corners = armor.corners;
    }

    const RefineVerdict verdict = refineOne(image, armor, record_ptr);

    if (record_ptr != nullptr) {
      record.verdict = verdict;
      // 被拒的检出马上就要从 armors 中消失，这里保存的角点是画面上唯一还能
      // 标出它位置的依据。
      record.corners = armor.corners;
      record.corner_shift = armor.corner_shift > 0.0F ? armor.corner_shift : record.corner_shift;
      records->push_back(record);
    }

    switch (verdict) {
      case RefineVerdict::Refined:
        ++stats.refined;
        kept.push_back(armor);
        break;
      case RefineVerdict::NetworkKept:
        ++stats.network_kept;
        kept.push_back(armor);
        break;
      case RefineVerdict::Rejected:
        // 单灯条：网络给出的另一侧角点是凭空生成的，丢弃整块检出。
        ++stats.rejected;
        break;
    }
  }

  armors = std::move(kept);
  return stats;
}

}  // namespace L2Perception
