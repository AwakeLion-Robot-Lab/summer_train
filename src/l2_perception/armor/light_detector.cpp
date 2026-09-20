#include "l2_perception/armor/light_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

// 端点连线与图像竖直方向的夹角，单位为度。分母取绝对值下限避免灯条水平时除零。
float tiltDegrees(const cv::Point2f& top, const cv::Point2f& bottom)
{
  return static_cast<float>(
    std::atan2(std::abs(top.x - bottom.x), std::max(1e-6F, std::abs(top.y - bottom.y))) *
    180.0 / CV_PI);
}

}  // namespace

ArmorColor lightColor(
  const cv::Mat& image, const cv::Point2f& top, const cv::Point2f& bottom,
  double ratio_threshold)
{
  // 取样框：横向按灯条长度的 0.45 倍外扩，灯条宽约为长的 1/6，这样能吃到两侧光晕。
  const cv::Point2f center = (top + bottom) * 0.5F;
  const float length = static_cast<float>(cv::norm(top - bottom));
  const float half_width = std::max(3.0F, length * 0.45F);
  const float half_height = std::max(3.0F, length * 0.60F);
  cv::Rect roi(
    static_cast<int>(std::lround(center.x - half_width)),
    static_cast<int>(std::lround(center.y - half_height)),
    static_cast<int>(std::lround(half_width * 2.0F)),
    static_cast<int>(std::lround(half_height * 2.0F)));
  roi &= cv::Rect(0, 0, image.cols, image.rows);
  if (roi.width < 3 || roi.height < 3) {
    return ArmorColor::Unknown;
  }

  double red = 0.0;
  double blue = 0.0;
  long long counted = 0;
  const cv::Mat patch = image(roi);
  for (int y = 0; y < patch.rows; ++y) {
    const cv::Vec3b* row = patch.ptr<cv::Vec3b>(y);
    for (int x = 0; x < patch.cols; ++x) {
      const cv::Vec3b& pixel = row[x];
      // 三通道都接近饱和的像素没有颜色信息，留着只会把比值往 1 拉。
      if (pixel[0] >= 245 && pixel[1] >= 245 && pixel[2] >= 245) {
        continue;
      }
      // 同理丢掉过暗的背景像素，它们数量远多于灯条本身。
      if (pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40) {
        continue;
      }
      blue += pixel[0];
      red += pixel[2];
      ++counted;
    }
  }
  // 样本太少时不猜颜色，交给上层当未知灯条丢弃。
  if (counted < 8) {
    return ArmorColor::Unknown;
  }

  const double ratio = red / std::max(1.0, blue);
  if (ratio > ratio_threshold) {
    return ArmorColor::Red;
  }
  if (ratio < 1.0 / ratio_threshold) {
    return ArmorColor::Blue;
  }
  return ArmorColor::Unknown;
}

std::vector<Light> findLights(
  const cv::Mat& image, const cv::Rect& roi, const LightFinderConfig& config,
  ArmorColor color)
{
  const cv::Rect area = roi & cv::Rect(0, 0, image.cols, image.rows);
  if (image.empty() || image.type() != CV_8UC3 || area.area() <= 0) {
    return {};
  }

  // 底图与阈值必须配套，见 LightFinderConfig 的两个键。
  const bool color_diff = config.color_channel_diff && color != ArmorColor::Unknown;
  cv::Mat base;
  if (color_diff) {
    cv::Mat blue;
    cv::Mat red;
    cv::extractChannel(image(area), blue, 0);
    cv::extractChannel(image(area), red, 2);
    // 饱和减法：白色背景和过曝白核都被压到 0。
    if (color == ArmorColor::Blue) {
      cv::subtract(blue, red, base);
    } else {
      cv::subtract(red, blue, base);
    }
  } else {
    cv::cvtColor(image(area), base, cv::COLOR_BGR2GRAY);
  }
  cv::Mat binary;
  cv::threshold(
    base, binary, color_diff ? config.color_diff_threshold : config.binary_threshold, 255,
    cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  const cv::Point2f offset(static_cast<float>(area.x), static_cast<float>(area.y));
  std::vector<Light> lights;
  for (const std::vector<cv::Point>& contour : contours) {
    // 点数太少的轮廓拟合不出可信的方向。
    if (contour.size() < 6) {
      continue;
    }
    // 角点按 y 排序后，前两个是上边、后两个是下边，取中点作端点并补 roi 偏移。
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    std::array<cv::Point2f, 4> corners;
    rect.points(corners.data());
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
      return a.y < b.y;
    });
    const cv::Point2f top = (corners[0] + corners[1]) * 0.5F + offset;
    const cv::Point2f bottom = (corners[2] + corners[3]) * 0.5F + offset;
    const float length = static_cast<float>(cv::norm(top - bottom));
    if (!(length >= config.min_length)) {
      continue;
    }
    // 宽也按排序后的角点取，不用外接矩形的尺寸：横躺的细条按 y 排序后「宽」
    // 是它的长边，比值远大于 1 会被挡掉；用尺寸算则会把它当成竖直的短灯条。
    const float ratio = static_cast<float>(cv::norm(corners[0] - corners[1])) / length;
    if (!(config.min_ratio < ratio && ratio < config.max_ratio)) {
      continue;
    }
    const float tilt_deg = tiltDegrees(top, bottom);
    if (!(tilt_deg < config.max_angle_deg)) {
      continue;
    }

    Light light;
    light.top = top;
    light.bottom = bottom;
    light.center = (top + bottom) * 0.5F;
    light.length = length;
    light.tilt_angle_deg = tilt_deg;
    light.color = lightColor(image, top, bottom, config.color_ratio_threshold);
    light.id = lights.size();
    lights.push_back(light);
  }
  return lights;
}

}  // namespace L2Perception
