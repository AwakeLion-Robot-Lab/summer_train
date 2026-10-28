#include "l2_perception/buff/buff_detector.hpp"
#include "l6_telemetry/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace L2Perception::Buff {
namespace {

constexpr double kBullseyeRadius = 0.15;
constexpr double kIconRadius = 0.05;

void extractEnemyChannel(
  const cv::Mat& image, L1Sensor::EnemyColor color, cv::Mat& output)
{
  if (image.empty() || image.type() != CV_8UC3 ||
      color == L1Sensor::EnemyColor::Unknown) {
    output.release();
    return;
  }

  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  const int positive_channel = color == L1Sensor::EnemyColor::Red ? 2 : 0;
  const int other_channel = color == L1Sensor::EnemyColor::Red ? 0 : 2;

  cv::Mat difference;
  cv::Mat purity;
  cv::Mat mask;
  cv::Mat purity_mask;
  cv::Mat brightness_mask;
  cv::subtract(channels[positive_channel], channels[other_channel], difference,
               cv::noArray(), CV_16S);
  difference.convertTo(difference, CV_8U);
  cv::threshold(difference, mask, 40, 255, cv::THRESH_BINARY);

  cv::subtract(channels[positive_channel], channels[1], purity,
               cv::noArray(), CV_16S);
  purity.convertTo(purity, CV_8U);
  cv::threshold(purity, purity_mask, 20, 255, cv::THRESH_BINARY);
  cv::bitwise_and(mask, purity_mask, mask);

  cv::threshold(channels[positive_channel], brightness_mask, 80, 255,
                cv::THRESH_BINARY);
  cv::bitwise_and(mask, brightness_mask, output);
  cv::dilate(output, output,
             cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)),
             cv::Point(-1, -1), 1);
}

double iconScore(const cv::Mat& roi)
{
  cv::Mat binary;
  cv::threshold(roi, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  cv::Mat skeleton;
  cv::ximgproc::thinning(binary, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
  cv::Mat labels;
  if (cv::connectedComponents(skeleton, labels, 8) - 1 != 1) return 0.0;

  const int center_y = skeleton.rows / 2;
  int total_endpoints = 0;
  int lower_endpoints = 0;
  int branch_points = 0;
  for (int y = 1; y + 1 < skeleton.rows; ++y) {
    for (int x = 1; x + 1 < skeleton.cols; ++x) {
      if (skeleton.at<std::uint8_t>(y, x) == 0) continue;
      int neighbours = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          neighbours += skeleton.at<std::uint8_t>(y + dy, x + dx) > 0;
        }
      }
      if (neighbours == 1) {
        ++total_endpoints;
        if (y >= center_y) ++lower_endpoints;
      } else if (neighbours >= 3) {
        ++branch_points;
      }
    }
  }

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);
  int holes = 0;
  if (!contours.empty() && !hierarchy.empty()) {
    std::size_t outer = 0;
    double largest_area = 0.0;
    for (std::size_t i = 0; i < hierarchy.size(); ++i) {
      if (hierarchy[i][3] != -1) continue;
      const double area = cv::contourArea(contours[i]);
      if (area > largest_area) {
        largest_area = area;
        outer = i;
      }
    }
    for (std::size_t i = 0; i < hierarchy.size(); ++i) {
      if (hierarchy[i][3] == static_cast<int>(outer)) ++holes;
    }
  }
  if (total_endpoints >= 1 && lower_endpoints >= 1 && branch_points >= 8 &&
      branch_points <= 50 && holes <= 2) {
    return 0.5 + 0.1 * total_endpoints;
  }
  return 0.0;
}

bool bullseyeFeature(
  const cv::Mat& roi, cv::Point2f center, std::array<cv::Point2f, 4>& endpoints,
  double& score, double active_threshold)
{
  constexpr int kRadialBins = 32;
  constexpr int kThetaBins = 180;
  constexpr double kArmLengthStddevMax = 0.3;

  const int square_size = std::max(roi.cols, roi.rows);
  cv::Mat square(square_size, square_size, roi.type(), cv::Scalar::all(0));
  roi.copyTo(square(cv::Rect(0, 0, roi.cols, roi.rows)));
  cv::Mat gray;
  if (square.channels() == 3) cv::cvtColor(square, gray, cv::COLOR_BGR2GRAY);
  else gray = square;

  const double max_radius = std::min(square.cols, square.rows) * 0.5;
  std::array<float, kRadialBins * kThetaBins> polar{};
  for (int y = 0; y < gray.rows; ++y) {
    for (int x = 0; x < gray.cols; ++x) {
      const double dx = static_cast<double>(x) - center.x;
      const double dy = static_cast<double>(y) - center.y;
      const double radius = std::hypot(dx, dy);
      if (radius >= max_radius) continue;
      const double theta = std::atan2(dy, dx);
      const int radial = std::clamp(
        static_cast<int>(radius / max_radius * kRadialBins), 0, kRadialBins - 1);
      const int angular = std::clamp(
        static_cast<int>((theta + std::numbers::pi) /
                         (2.0 * std::numbers::pi) * kThetaBins),
        0, kThetaBins - 1);
      polar[radial * kThetaBins + angular] +=
        static_cast<float>(gray.at<std::uint8_t>(y, x));
    }
  }

  std::array<double, kThetaBins> angular{};
  for (int theta = 0; theta < kThetaBins; ++theta) {
    for (int radius = 0; radius < kRadialBins; ++radius) {
      angular[theta] += polar[radius * kThetaBins + theta] * (radius + 1);
    }
  }
  double sum_cos = 0.0;
  double sum_sin = 0.0;
  for (int theta = 0; theta < kThetaBins; ++theta) {
    const double angle = 4.0 * theta / kThetaBins * 2.0 * std::numbers::pi;
    sum_cos += angular[theta] * std::cos(angle);
    sum_sin += angular[theta] * std::sin(angle);
  }
  const double phase = std::atan2(sum_sin, sum_cos) / 4.0;

  constexpr int kWindow = 2;
  std::array<double, 4> peaks{};
  std::array<double, 4> valleys{};
  for (std::size_t page = 0; page < 4; ++page) {
    const double peak_angle = phase + page * std::numbers::pi * 0.5;
    const double valley_angle = peak_angle + std::numbers::pi * 0.25;
    const auto to_index = [](double angle) {
      const int raw = static_cast<int>(std::llround(
        (angle + std::numbers::pi) / (2.0 * std::numbers::pi) * kThetaBins));
      return ((raw % kThetaBins) + kThetaBins) % kThetaBins;
    };
    const int peak_index = to_index(peak_angle);
    const int valley_index = to_index(valley_angle);
    for (int offset = -kWindow; offset <= kWindow; ++offset) {
      peaks[page] += angular[(peak_index + offset + kThetaBins) % kThetaBins];
      valleys[page] += angular[(valley_index + offset + kThetaBins) % kThetaBins];
    }
    peaks[page] /= 2 * kWindow + 1;
    valleys[page] /= 2 * kWindow + 1;
  }

  double mean_peak = 0.0;
  double mean_valley = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    mean_peak += peaks[i] * 0.25;
    mean_valley += valleys[i] * 0.25;
  }
  const double pvd = (mean_peak - mean_valley) /
                     (mean_peak + mean_valley + 1e-10);
  score = pvd;

  if (pvd < active_threshold) {
    std::array<double, kRadialBins> radial{};
    for (int radius = 0; radius < kRadialBins; ++radius) {
      for (int theta = 0; theta < kThetaBins; ++theta) {
        radial[radius] += polar[radius * kThetaBins + theta];
      }
    }
    const auto maximum = std::max_element(radial.begin(), radial.end());
    const auto radius_bin = static_cast<std::size_t>(maximum - radial.begin());
    const double radius = max_radius * (radius_bin + 1) / kRadialBins;
    endpoints = {
      center + cv::Point2f{0.0F, static_cast<float>(-radius)},
      center + cv::Point2f{static_cast<float>(radius), 0.0F},
      center + cv::Point2f{0.0F, static_cast<float>(radius)},
      center + cv::Point2f{static_cast<float>(-radius), 0.0F}};
    return true;
  }

  std::array<cv::Point2f, 4> tips{};
  for (std::size_t page = 0; page < 4; ++page) {
    const double angle = phase + page * std::numbers::pi * 0.5;
    const int theta_index = static_cast<int>(std::llround(
      (angle + std::numbers::pi) / (2.0 * std::numbers::pi) * kThetaBins)) %
      kThetaBins;
    const double dx = std::cos(angle);
    const double dy = std::sin(angle);
    std::array<double, kRadialBins> profile{};
    for (int radius = 0; radius < kRadialBins; ++radius) {
      for (int dt = -3; dt <= 3; ++dt) {
        profile[radius] +=
          polar[radius * kThetaBins + (theta_index + dt + kThetaBins) % kThetaBins];
      }
    }
    const double max_energy = *std::max_element(profile.begin(), profile.end());
    const double threshold = max_energy * 0.25;
    int boundary = -1;
    for (int radius = kRadialBins - 1; radius >= 0; --radius) {
      if (profile[radius] >= threshold) {
        boundary = radius;
        break;
      }
    }
    double exact_radius = 0.0;
    if (boundary >= kRadialBins - 1) exact_radius = 1.0;
    else if (boundary < 0) {
      exact_radius = static_cast<double>(
        std::max_element(profile.begin(), profile.end()) - profile.begin()) /
        kRadialBins;
    } else {
      const double inside = profile[boundary];
      const double outside = profile[boundary + 1];
      if (inside > outside) {
        const double fraction = (threshold - outside) / (inside - outside);
        exact_radius = (boundary + 1.0 - fraction) / kRadialBins;
      } else exact_radius = static_cast<double>(boundary + 1) / kRadialBins;
    }
    const double arm_length = max_radius * exact_radius;
    tips[page] = {
      center.x + static_cast<float>(arm_length * dx),
      center.y + static_cast<float>(arm_length * dy)};
  }

  double mean = 0.0;
  std::array<double, 4> lengths{};
  for (std::size_t i = 0; i < 4; ++i) {
    lengths[i] = cv::norm(tips[i] - center);
    mean += lengths[i] * 0.25;
  }
  double variance = 0.0;
  for (const double length : lengths) {
    const double difference = length - mean;
    variance += difference * difference * 0.25;
  }
  if (std::sqrt(variance) > mean * kArmLengthStddevMax) return false;
  endpoints = tips;
  return true;
}

void findContours(
  const DetectorConfig& config, const cv::Mat& binary,
  std::vector<std::vector<cv::Point>>& icons,
  std::vector<std::vector<cv::Point>>& bullseyes,
  const cv::Mat& camera_matrix)
{
  icons.clear();
  bullseyes.clear();
  if (binary.empty() || binary.type() != CV_8UC1 || camera_matrix.empty()) return;
  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  if (fx <= 0.0 || fy <= 0.0) return;
  const double focal = (fx + fy) * 0.5;
  const double perspective = config.max_perspective_deg * std::numbers::pi / 180.0;
  const double cos_theta = std::cos(perspective);
  if (cos_theta <= 0.0) return;

  const auto radius_range = [&](double physical_radius) {
    constexpr double margin = 0.1;
    return std::array<double, 2>{
      focal * physical_radius * cos_theta / config.max_distance * (1.0 - margin),
      focal * physical_radius / (config.min_distance * cos_theta) * (1.0 + margin)};
  };
  const auto area_range = [&](double physical_radius, double fill_ratio) {
    constexpr double margin = 0.1;
    return std::array<double, 2>{
      focal * focal * physical_radius * physical_radius * cos_theta /
        (config.max_distance * config.max_distance) * (1.0 - margin) *
        std::numbers::pi * fill_ratio,
      focal * focal * physical_radius * physical_radius /
        (config.min_distance * config.min_distance * cos_theta) * (1.0 + margin) *
        std::numbers::pi};
  };

  const auto bullseye_range = radius_range(kBullseyeRadius);
  const auto icon_area_range = area_range(kIconRadius, 0.2);
  const double min_circularity = 2.0 * cos_theta / (1.0 + cos_theta * cos_theta);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  struct Ellipse { cv::Point center; double major; double minor; std::size_t index; };
  std::vector<Ellipse> visited;
  for (std::size_t i = 0; i < contours.size(); ++i) {
    const auto& contour = contours[i];
    if (contour.size() < 5) continue;
    const double area = cv::contourArea(contour);
    if (area <= 0.0) continue;
    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0) continue;
    const double radius = std::sqrt(area / std::numbers::pi);
    if (radius >= bullseye_range[0] && radius <= bullseye_range[1]) {
      constexpr double area_ratio_min = 0.7;
      constexpr double area_ratio_max = 1.3;
      constexpr double circularity_margin = 0.7;
      const auto ellipse = cv::fitEllipse(contour);
      const double major = std::max(ellipse.size.width, ellipse.size.height);
      const double minor = std::min(ellipse.size.width, ellipse.size.height);
      if (major <= 0.0 || minor <= 0.0 || minor / major < cos_theta) continue;
      const double ellipse_area = std::numbers::pi * (major * 0.5) * (minor * 0.5);
      const double area_ratio = area / ellipse_area;
      if (area_ratio < area_ratio_min || area_ratio > area_ratio_max) continue;
      const double circularity = 4.0 * std::numbers::pi * area /
                                 (perimeter * perimeter);
      if (circularity < min_circularity * circularity_margin) continue;
      const auto moments = cv::moments(contour);
      visited.push_back({
        {static_cast<int>(moments.m10 / moments.m00),
         static_cast<int>(moments.m01 / moments.m00)},
        major, minor, i});
    } else if (area >= icon_area_range[0] && area <= icon_area_range[1]) {
      const auto ellipse = cv::fitEllipse(contour);
      const double major = std::max(ellipse.size.width, ellipse.size.height);
      const double minor = std::min(ellipse.size.width, ellipse.size.height);
      if (major <= 0.0 || minor <= 0.0 || minor / major < cos_theta) continue;
      icons.push_back(contour);
    }
  }

  std::vector<bool> seen(visited.size(), false);
  for (std::size_t i = 0; i < visited.size(); ++i) {
    if (seen[i]) continue;
    seen[i] = true;
    std::vector<std::size_t> group{i};
    for (std::size_t j = i + 1; j < visited.size(); ++j) {
      if (seen[j]) continue;
      const auto& a = visited[i];
      const auto& b = visited[j];
      const double distance = std::hypot(
        static_cast<double>(a.center.x - b.center.x),
        static_cast<double>(a.center.y - b.center.y));
      if (distance < std::max(a.major, b.major) * 0.5 * 0.3) {
        seen[j] = true;
        group.push_back(j);
      }
    }
    const auto largest = std::max_element(
      group.begin(), group.end(), [&](std::size_t lhs, std::size_t rhs) {
        return visited[lhs].major < visited[rhs].major;
      });
    bullseyes.push_back(contours[visited[*largest].index]);
  }

  std::vector<std::vector<cv::Point>> filtered_icons;
  for (const auto& icon : icons) {
    const auto moments = cv::moments(icon);
    const cv::Point2f center{
      static_cast<float>(moments.m10 / moments.m00),
      static_cast<float>(moments.m01 / moments.m00)};
    bool inside = false;
    for (const auto& bullseye : bullseyes) {
      if (cv::pointPolygonTest(bullseye, center, false) >= 0) {
        inside = true;
        break;
      }
    }
    if (!inside) filtered_icons.push_back(icon);
  }
  icons = std::move(filtered_icons);
}

// 网络只负责选 ROI/颜色/激活状态。该模型的 R 类关键点落在内圈，不能
// 当作半径 0.15 m 的十字臂端点交给 PnP；在 ROI 内沿用传统端点提取。
std::optional<Bullseye> refineModelDetection(
  const cv::Mat& image, const cv::Mat& binary, const ModelDetection& detection,
  const DetectorConfig& config)
{
  if (binary.empty()) return std::nullopt;
  const int margin = std::max(8, static_cast<int>(
    std::max(detection.box.width, detection.box.height) * 0.15F));
  cv::Rect roi{static_cast<int>(std::floor(detection.box.x)) - margin,
               static_cast<int>(std::floor(detection.box.y)) - margin,
               static_cast<int>(std::ceil(detection.box.width)) + 2 * margin + 1,
               static_cast<int>(std::ceil(detection.box.height)) + 2 * margin + 1};
  roi &= cv::Rect{0, 0, image.cols, image.rows};
  if (roi.width < 3 || roi.height < 3) return std::nullopt;
  const cv::Point2f center = detection.center - cv::Point2f{roi.tl()};
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary(roi).clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  const std::vector<cv::Point>* selected = nullptr;
  double largest_area = 0;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area > largest_area && cv::pointPolygonTest(contour, center, false) >= 0) {
      selected = &contour;
      largest_area = area;
    }
  }
  if (!selected || largest_area < 4) return std::nullopt;
  cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);
  cv::drawContours(mask, std::vector<std::vector<cv::Point>>{*selected}, -1,
                   cv::Scalar{255}, cv::FILLED);
  cv::Mat masked;
  image(roi).copyTo(masked, mask);
  Bullseye bullseye;
  double feature_score = 0;
  if (!bullseyeFeature(masked, center, bullseye.corners, feature_score,
                       config.active_threshold)) return std::nullopt;
  // 待打叶片必须观测到十字，不能把圆环上的任意四点伪造成 PnP 端点。
  if (!detection.active && feature_score < config.active_threshold) return std::nullopt;
  for (auto& corner : bullseye.corners) corner += cv::Point2f{roi.tl()};
  bullseye.center = detection.center;
  bullseye.active = detection.active;
  bullseye.score = detection.confidence;
  return bullseye;
}

}  // namespace

Detector::Detector() : Detector(DetectorConfig{}) {}

Detector::Detector(DetectorConfig config)
  : config_(std::move(config)), decoder_(config_.decoder)
{
  if (!config_.use_model) return;
  backend_ = makeInferenceBackend(InferenceBackendKind::OpenVino);
  try {
    backend_->load(config_.model);
  } catch (const std::exception& error) {
    if (config_.model.device == "CPU") throw;
    L6Telemetry::logWarn("buff model failed on", config_.model.device,
                         error.what(), "; falling back to CPU");
    config_.model.device = "CPU";
    backend_ = makeInferenceBackend(InferenceBackendKind::OpenVino);
    backend_->load(config_.model);
  }
  const auto outputs = probeOutputSpecs(*backend_);
  if (outputs.size() != 1 || outputs[0].shape.size() != 3 ||
      outputs[0].shape[0] != 1 || outputs[0].shape[1] != 16 || outputs[0].shape[2] == 0) {
    throw std::runtime_error("buff model requires [1,16,N] output (YOLOv8 pose BR/BW/RR/RW)");
  }
  L6Telemetry::logInfo("buff model loaded openvino", config_.model.model_path.string(),
                       config_.model.device, "layout yolov8_pose_16; traditional R/endpoint refinement");
}

void Detector::setCalibration(const L1Sensor::CameraCalibration& calibration)
{
  calibration_ = calibration;
  calibration_ready_ = calibration.camera_matrix.rows == 3 &&
                       calibration.camera_matrix.cols == 3;
}

Detections Detector::detect(const cv::Mat& image) const
{
  Detections result;
  if (image.empty() || !calibration_ready_) return result;
  cv::Mat binary;
  extractEnemyChannel(image, color_, binary);
  std::vector<std::vector<cv::Point>> icons;
  std::vector<std::vector<cv::Point>> bullseyes;
  findContours(config_, binary, icons, bullseyes, calibration_.camera_matrix);

  if (config_.use_model && color_ != L1Sensor::EnemyColor::Unknown) {
    try {
      const auto input = ImagePreprocessor::run(
        image, backend_->inputSpec(),
        {.padding_color = cv::Scalar{114, 114, 114}, .alignment = LetterboxAlignment::Centered});
      const auto output = backend_->infer(input.input);
      result.model_detections = decoder_.decode(output, input.transform, color_);
      for (const auto& detection : result.model_detections) {
        if (auto refined = refineModelDetection(image, binary, detection, config_)) {
          result.bullseyes.push_back(*refined);
        }
      }
    } catch (const std::exception& error) {
      L6Telemetry::logError("buff inference failed", error.what());
      return {}; // 失败帧不伪装成传统检测成功，由上层进入保持状态。
    }
  }

  if (!config_.use_model) for (const auto& bullseye : bullseyes) {
    const auto bounds = cv::boundingRect(bullseye);
    cv::Rect roi(bounds.x - 20, bounds.y - 20,
                 bounds.width + 40, bounds.height + 40);
    roi &= cv::Rect(0, 0, image.cols, image.rows);
    if (roi.width <= 0 || roi.height <= 0) continue;
    const auto moments = cv::moments(bullseye);
    const cv::Point2f center{
      static_cast<float>(moments.m10 / moments.m00),
      static_cast<float>(moments.m01 / moments.m00)};
    const cv::Point2f center_roi{
      center.x - static_cast<float>(roi.x), center.y - static_cast<float>(roi.y)};
    std::vector<cv::Point> local = bullseye;
    for (auto& point : local) {
      point.x -= roi.x;
      point.y -= roi.y;
    }
    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{local}, -1,
                     cv::Scalar{255}, cv::FILLED);
    cv::Mat masked_roi;
    image(roi).copyTo(masked_roi, mask);

    std::array<cv::Point2f, 4> endpoints{};
    double score = 0.0;
    if (!bullseyeFeature(masked_roi, center_roi, endpoints, score,
                         config_.active_threshold)) continue;
    for (auto& point : endpoints) {
      point.x += static_cast<float>(roi.x);
      point.y += static_cast<float>(roi.y);
    }
    Bullseye item;
    item.center = center;
    item.corners = endpoints;
    // 源提交就是用 score < threshold 写入 active；不要按命名自行翻转。
    item.active = score < config_.active_threshold;
    item.score = score;
    result.bullseyes.push_back(item);
  }

  for (const auto& icon : icons) {
    cv::Rect box = cv::boundingRect(icon) & cv::Rect(0, 0, image.cols, image.rows);
    cv::Rect roi(box.x - 5, box.y - 5, box.width + 10, box.height + 10);
    roi &= cv::Rect(0, 0, image.cols, image.rows);
    if (roi.width <= 0 || roi.height <= 0) continue;
    cv::Mat gray;
    cv::cvtColor(image(roi), gray, cv::COLOR_BGR2GRAY);
    const double score = iconScore(gray);
    if (score < config_.match_threshold) continue;
    const auto rectangle = cv::minAreaRect(icon);
    result.icons.push_back({rectangle.center, score});
  }
  return result;
}

}  // namespace L2Perception::Buff
