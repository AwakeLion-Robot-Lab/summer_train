#include "l3_estimation/armor/track_roi.hpp"

#include <Eigen/Core>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace L3Estimation::Roi {

namespace {

// 保持中心不动按比例放大矩形，再裁回图像范围内。
cv::Rect expandAndClip(const cv::Rect & rect, double ratio, const cv::Rect & image_rect)
{
  const double center_x = rect.x + rect.width * 0.5;
  const double center_y = rect.y + rect.height * 0.5;
  const int width = std::max(1, static_cast<int>(std::round(rect.width * ratio)));
  const int height = std::max(1, static_cast<int>(std::round(rect.height * ratio)));
  cv::Rect expanded(
    static_cast<int>(std::round(center_x - width * 0.5)),
    static_cast<int>(std::round(center_y - height * 0.5)), width, height);
  return expanded & image_rect;
}

}  // namespace

std::optional<cv::Rect> bounds(const Focus & focus, const cv::Size & image_size)
{
  if (focus.target == nullptr || image_size.width <= 0 || image_size.height <= 0) {
    return std::nullopt;
  }

  // 与滤波器同一个外推截止：超过 temp_lost_predict_time 没更新就只推到截止时刻。
  EskfTarget predicted = focus.target->snapshot();
  if (focus.motion_end > predicted.t()) {
    predicted.predict(focus.motion_end);
  }
  const Eigen::VectorXd state = predicted.rawState();

  std::vector<cv::Point2f> points;
  points.reserve(static_cast<std::size_t>(predicted.armor_num()) * 4);
  for (int id = 0; id < predicted.armor_num(); ++id) {
    for (const bool is_left : {true, false}) {
      const auto endpoints = focus.ctx.project(id, is_left, state);
      for (const cv::Point2f & point : {endpoints.first, endpoints.second}) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
          points.push_back(point);
        }
      }
    }
  }
  if (points.empty()) {
    return std::nullopt;
  }

  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const cv::Rect box = cv::boundingRect(points);
  if ((box & image_rect).empty()) {
    return std::nullopt;
  }
  return box;
}

cv::Rect light(const cv::Rect & box, const cv::Size & image_size)
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  constexpr double kExpandRatio = 1.6;
  const cv::Rect expanded = expandAndClip(box, kExpandRatio, image_rect);
  return expanded.empty() ? image_rect : expanded;
}

cv::Rect net(
  const Focus & focus, const cv::Rect & box, const cv::Size & image_size,
  double target_wh_ratio)
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  if (focus.target == nullptr) {
    return image_rect;
  }

  // 基地尺寸大、整车模型退化，ROI 放得更宽。
  constexpr double kExpandRatio = 1.4;
  constexpr double kExpandRatioBase = 3.0;
  cv::Rect rect = expandAndClip(
    box, VehicleModel::isBase(focus.target->name) ? kExpandRatioBase : kExpandRatio,
    image_rect);
  if (rect.empty()) {
    return image_rect;
  }

  // ① 按网络输入宽高比修正形状：相机图像的长宽比通常与网络输入不一致，直接
  //    letterbox 会整体缩小，先把 ROI 修成同一比例能少填不少边。
  const double ratio =
    (std::isfinite(target_wh_ratio) && target_wh_ratio > 0.0) ? target_wh_ratio : 1.0;
  double target_width = std::max(rect.width, 1);
  double target_height = std::max(rect.height, 1);
  if (target_width / target_height < ratio) {
    target_width = target_height * ratio;
  } else {
    target_height = target_width / ratio;
  }
  const double center_x = rect.x + rect.width * 0.5;
  const double center_y = rect.y + rect.height * 0.5;
  cv::Rect ratio_rect(
    static_cast<int>(std::round(center_x - target_width * 0.5)),
    static_cast<int>(std::round(center_y - target_height * 0.5)),
    static_cast<int>(std::round(target_width)),
    static_cast<int>(std::round(target_height)));
  ratio_rect &= image_rect;
  if (ratio_rect.empty()) {
    return image_rect;
  }

  // ② 按距上次更新的时长线性膨胀，超时直接退化成整图：越久没更新预测越不可
  //    信，搜索范围就该越大，跟丢时 ROI 会自动放手，而不是把网络锁死在一个
  //    错误的小窗口里。
  const int base_side = std::max(ratio_rect.width, ratio_rect.height);
  const int max_side = std::max(image_size.width, image_size.height);
  int side = max_side;
  if (focus.lost_thres > 0.0 && focus.lost_time < focus.lost_thres) {
    const double fraction = std::clamp(focus.lost_time / focus.lost_thres, 0.0, 1.0);
    side = static_cast<int>(std::round(base_side + (max_side - base_side) * fraction));
  }
  side = std::clamp(side, 1, max_side);

  // ③ 扩成方形，适配常见的方形网络输入。
  const int square_center_x = ratio_rect.x + ratio_rect.width / 2;
  const int square_center_y = ratio_rect.y + ratio_rect.height / 2;
  cv::Rect square(square_center_x - side / 2, square_center_y - side / 2, side, side);
  square &= image_rect;
  return square.empty() ? image_rect : square;
}

}  // namespace L3Estimation::Roi
